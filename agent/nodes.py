from langchain.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.types import Send
from typing import List, Optional
from agent.state import State
from agent.schemas import Task, Plan, EvidenceItem, RouterDecision, EvidencePack
from agent.prompts import ROUTER_SYSTEM, RESEARCH_SYSTEM, ORCH_SYSTEM, WORKER_SYSTEM
from agent.llm import get_writer_llm, get_generic_llm
from datetime import date, timedelta
from pathlib import Path
import markdown
import re
from dotenv import load_dotenv
load_dotenv()


writer_llm = get_writer_llm()
generic_llm = get_generic_llm()

def router_node(state:State)-> dict:
    topic = state["topic"]
    decider = generic_llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content = ROUTER_SYSTEM),
            HumanMessage(content = f"Topic:{topic}\nAs-of date:{state['as_of']}"),
        ]
    )

    # setting default recensy window based on mode
    if decision.mode == "open_book":
        recency_days =  7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650
    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State)-> str:
    return "research" if state["needs_research"] else "orchestrator"


# RESEARCH TOOL(TAVILY)

def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})

    # Handle different return shapes
    if isinstance(results, dict) and "results" in results:
        results = results["results"]
    elif isinstance(results, str):
        return []  # fallback safely
    elif not isinstance(results, list):
        return []

    normalized: List[dict] = []

    for r in results:
        if not isinstance(r, dict):
            continue

        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )

    return normalized


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None



#  RESEARCH NODE

def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])[:10]
    max_results = 6

    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = generic_llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw_results}"
                )
            ),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # HARD RECENCY FILTER for open_book weekly roundup:
    # keep only items with a parseable ISO date and within the window.
    mode = state.get("mode", "closed_book")
    if mode == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        fresh: List[EvidenceItem] = []
        for e in evidence:
            d = _iso_to_date(e.published_at)
            if d and d >= cutoff:
                fresh.append(e)
        evidence = fresh

    return {"evidence": evidence}




# Planner Node

def orchestrator_node(state: State) -> dict:
    planner = generic_llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    # Force blog_kind for open_book
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}\n\n"
                    f"Instruction: If mode=open_book, your plan must NOT drift into a tutorial."
                )
            ),
        ]
    )

    # Ensure open_book forces the kind even if model forgets
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# Fanout Node

def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]



# Worker Node

def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")
    as_of = payload.get("as_of")
    recency_days = payload.get("recency_days")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # Provide a compact evidence list for citation use
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = writer_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {as_of} (recency_days={recency_days})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    # deterministic ordering
    return {"sections": [(task.id, section_md)]}


# REDUCER NODE

def reducer_node(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("Reducer called without a plan.")

    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]

    body_md = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body_md}\n"

    # Convert Markdown → HTML
    body_html = markdown.markdown(
        final_md,
        extensions=["fenced_code", "tables"]
    )

    # Load template
    template_path = Path("templates/blog_template.html")
    template_html = template_path.read_text(encoding="utf-8")

    # Inject content
    final_html = (
        template_html
        .replace("{{title}}", plan.blog_title)
        .replace("{{content}}", body_html)
    )

    # Safe filename
    safe_title = re.sub(r"[^\w\- ]", "", plan.blog_title).strip()
    safe_title = safe_title.replace(" ", "_")

    # Ensure blogs directory exists
    blogs_dir = Path("blogs")
    blogs_dir.mkdir(parents=True, exist_ok=True)

    # Save HTML
    html_path = blogs_dir / f"{safe_title}.html"
    html_path.write_text(final_html, encoding="utf-8")

    # Save Markdown
    md_path = blogs_dir / f"{safe_title}.md"
    md_path.write_text(final_md, encoding="utf-8")

    return {"final": final_html}