import os
import re
from pathlib import Path
import secrets
from datetime import datetime, date
from urllib.parse import urlencode
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel
from dotenv import load_dotenv

from agent.graph import blog_app
from agent.schemas import Plan

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY           = os.getenv("SECRET_KEY", secrets.token_hex(32))
BASE_URL             = os.getenv("BASE_URL", "http://localhost:8000")

ALLOWED_EMAILS = set(filter(None, os.getenv("ALLOWED_EMAILS", "").split(",")))

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"

COOKIE_NAME    = "blogforge_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

serializer = URLSafeTimedSerializer(SECRET_KEY)
sessions: dict[str, dict] = {}

# ─────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────
def create_session_token(data: dict) -> str:
    return serializer.dumps(data)

def verify_session_token(token: str, max_age: int = COOKIE_MAX_AGE) -> Optional[dict]:
    try:
        return serializer.loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None

def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    data = verify_session_token(token)
    if not data:
        return None
    return sessions.get(data.get("session_id"))

def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# ─────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────
class GenerateRequest(BaseModel):
    topic:    str
    tone:     str = "Technical & Deep-dive"
    length:   str = "Long (~2000 words)"
    pub_date: str = ""
    tags:     str = ""

# ─────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if get_current_user(request):
        return RedirectResponse("/dashboard")
    return RedirectResponse("/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    if get_current_user(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
        "google_client_id": GOOGLE_CLIENT_ID,
    })


@app.get("/auth/google")
async def auth_google():
    state = secrets.token_urlsafe(32)
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  f"{BASE_URL}/auth/callback",
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")
    response.set_cookie(
        "oauth_state",
        serializer.dumps({"state": state}),
        max_age=600, httponly=True, samesite="lax", secure=False,
    )
    return response


@app.get("/auth/callback")
async def auth_callback(
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
):
    if error:
        return RedirectResponse("/login?error=access_denied")
    if not code or not state:
        return RedirectResponse("/login?error=invalid_callback")

    state_cookie = request.cookies.get("oauth_state")
    if not state_cookie:
        return RedirectResponse("/login?error=state_missing")

    try:
        state_data = serializer.loads(state_cookie, max_age=600)
        if state_data.get("state") != state:
            return RedirectResponse("/login?error=state_mismatch")
    except (BadSignature, SignatureExpired):
        return RedirectResponse("/login?error=state_expired")

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  f"{BASE_URL}/auth/callback",
                "grant_type":    "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        return RedirectResponse("/login?error=token_exchange_failed")

    access_token = token_resp.json().get("access_token")
    if not access_token:
        return RedirectResponse("/login?error=no_access_token")

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            GOOGLE_USERINFO,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if user_resp.status_code != 200:
        return RedirectResponse("/login?error=userinfo_failed")

    google_user = user_resp.json()
    email = google_user.get("email")

    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        return RedirectResponse("/login?error=unauthorized_email")

    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "session_id":  session_id,
        "email":       email,
        "name":        google_user.get("name"),
        "picture":     google_user.get("picture"),
        "given_name":  google_user.get("given_name", google_user.get("name", "").split()[0]),
        "logged_in_at": datetime.utcnow().isoformat(),
    }

    response = RedirectResponse("/dashboard")
    response.set_cookie(
        COOKIE_NAME,
        create_session_token({"session_id": session_id}),
        max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax", secure=False,
    )
    response.delete_cookie("oauth_state")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login")

    blogs_dir = Path("blogs")
    blogs_dir.mkdir(exist_ok=True)

    # Get latest blogs sorted by modified time
    blog_files = sorted(
        blogs_dir.glob("*.html"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    recent_blogs = [
        {
            "title": f.stem.replace("_", " "),
            "file": f.name
        }
        for f in blog_files[:10]
    ]

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "recent_blogs": recent_blogs
    })


@app.get("/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        data = verify_session_token(token)
        if data:
            sessions.pop(data.get("session_id"), None)
    response = RedirectResponse("/login")
    response.delete_cookie(COOKIE_NAME)
    return response



app.mount("/blogs", StaticFiles(directory="blogs"), name="blogs")

# ─────────────────────────────────────────
# API — /api/generate
# ─────────────────────────────────────────
@app.post("/api/generate")
async def api_generate(request: Request, body: GenerateRequest):
    require_user(request)

    # Default as_of
    as_of = body.pub_date or date.today().isoformat()

    # Call your LangGraph app
    out = blog_app.invoke(
        {
            "topic": body.topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "final": "",
        }
    )

    plan: Plan = out["plan"]

    # Recreate safe filename (must match reducer logic)
    safe_title = re.sub(r"[^\w\- ]", "", plan.blog_title).strip()
    safe_title = safe_title.replace(" ", "_")
    filename = f"{safe_title}.html"

    file_path = Path("blogs") / filename

    return {
        "success": True,
        "title": plan.blog_title,
        "blog_post": out.get("final", ""),  # full HTML content
        "linkedin_post": f"🚀 {plan.blog_title}\n\nRead more on our blog.",
        "tags": ", ".join(body.tags) if isinstance(body.tags, list) else (body.tags or ""),
        "pub_date": as_of,
        "tone": body.tone,
        "file_name": filename,
        "file_path": str(file_path),
    }

# ─────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}