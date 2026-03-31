/* ═══════════════════════════════════════════════
   BlogForge Dashboard — Client Logic
   File: static/dashboard.js
═══════════════════════════════════════════════ */

/* ─────────────────────────────────────────────
   DRAWER (mobile input panel)
───────────────────────────────────────────── */
const drawer   = () => document.getElementById('panel-left');
const backdrop = () => document.getElementById('drawer-backdrop');

function openDrawer() {
  drawer().classList.add('open');
  backdrop().classList.add('open');
  document.body.style.overflow = 'hidden'; // prevent bg scroll while sheet open
  document.getElementById('drawer-toggle').classList.add('active');
}

function closeDrawer() {
  drawer().classList.remove('open');
  backdrop().classList.remove('open');
  document.body.style.overflow = '';
  document.getElementById('drawer-toggle').classList.remove('active');
}

// close on backdrop tap
document.addEventListener('DOMContentLoaded', () => {
  const bd = backdrop();
  if (bd) bd.addEventListener('click', closeDrawer);

  // swipe-down to close drawer
  let touchStartY = 0;
  const panel = drawer();
  panel.addEventListener('touchstart', e => { touchStartY = e.touches[0].clientY; }, { passive: true });
  panel.addEventListener('touchmove', e => {
    const dy = e.touches[0].clientY - touchStartY;
    if (dy > 60) closeDrawer();
  }, { passive: true });
});

/* ─────────────────────────────────────────────
   TABS
───────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(`pane-${name}`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');

  // update bottom nav highlight
  document.querySelectorAll('.bn-item[data-tab]').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
}

/* ─────────────────────────────────────────────
   TOAST
───────────────────────────────────────────── */
function toast(msg, type = 'info') {
  const icons = { info: 'ℹ️', success: '✅', error: '❌', loading: '⏳' };
  const el = document.getElementById('toast');
  el.querySelector('.toast-icon').textContent = icons[type] ?? 'ℹ️';
  el.querySelector('.toast-msg').textContent  = msg;
  el.className = `toast show${type === 'error' ? ' error' : type === 'success' ? ' success' : ''}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 3500);
}

/* ─────────────────────────────────────────────
   GENERATE
───────────────────────────────────────────── */
async function generate() {
  const topic = document.getElementById('inp-topic').value.trim();
  if (!topic) {
    document.getElementById('inp-topic').focus();
    toast('Please enter a topic first', 'error');
    return;
  }

  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  btn.innerHTML = '<span class="dots"><span></span><span></span><span></span></span> Generating…';

  toast('Calling your graph…', 'loading');

  try {
    const res = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ topic }),
    });

    if (!res.ok) throw new Error(await res.text());

    const data = await res.json();

    closeDrawer();

    populateBlog(data);
    populateLinkedIn(data);

    switchTab('blog');
    toast('Post generated! Review below.', 'success');

  } catch (err) {
    toast('Generation failed: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.2">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
      </svg>
      Generate Blog Post`;
  }
}


/* ─────────────────────────────────────────────
   POPULATE — BLOG TAB
───────────────────────────────────────────── */
function populateBlog(data) {
  document.getElementById('blog-empty').style.display = 'none';
  document.getElementById('blog-post').style.display  = '';

  document.getElementById('post-tags').innerHTML = (data.tags || '')
    .split(',')
    .map(t => t.trim())
    .filter(Boolean)
    .map(t => `<span class="tag">${t}</span>`)
    .join('');

  document.getElementById('post-meta').textContent =
    [data.pub_date, data.tone].filter(Boolean).join('  ·  ');

  document.getElementById('post-title').textContent = data.title || '';

  const contentEl = document.getElementById('post-content');
contentEl.innerHTML = data.blog_post || '';

if (window.MathJax) {
  MathJax.typesetPromise([contentEl]).catch(err =>
    console.error("MathJax typeset failed:", err)
  );
}
}

/* ─────────────────────────────────────────────
   POPULATE — LINKEDIN TAB
───────────────────────────────────────────── */
function populateLinkedIn(data) {
  document.getElementById('li-empty').style.display   = 'none';
  document.getElementById('li-content').style.display = '';

  const text = data.linkedin_post || '';
  document.getElementById('li-preview').textContent  = text;
  document.getElementById('li-textarea').value       = text;
}

/* keep preview in sync with textarea edits */
function syncLinkedIn() {
  document.getElementById('li-preview').textContent =
    document.getElementById('li-textarea').value;
}

/* ─────────────────────────────────────────────
   USER MENU
───────────────────────────────────────────── */
function toggleUserMenu() {
  document.getElementById('user-menu').classList.toggle('open');
}

document.addEventListener('click', e => {
  const menu = document.getElementById('user-menu');
  if (
    menu &&
    !e.target.closest('#user-menu') &&
    !e.target.closest('#sb-avatar-btn') &&
    !e.target.closest('#topbar-avatar-btn')
  ) {
    menu.classList.remove('open');
  }
});

/* ─────────────────────────────────────────────
   UTILITY
───────────────────────────────────────────── */
function slugify(str) {
  return (str || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
}



/* ─────────────────────────────────────────────
   LOAD EXISTING BLOG FROM SERVER
───────────────────────────────────────────── */
async function loadBlog(filename) {
  try {
    const res = await fetch(`/blogs/${filename}`);
    if (!res.ok) throw new Error("Unable to load blog");

    const html = await res.text();

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");

    // 🔥 Extract ONLY the article content
    const article = doc.querySelector("article");
    const content = article ? article.innerHTML : "";

    const title = doc.querySelector("title")?.textContent ||
      filename.replace(".html", "").replaceAll("_", " ");

    populateBlog({
      title: title,
      blog_post: content,
      linkedin_post: "",
      tags: "",
      pub_date: "",
      tone: ""
    });

    switchTab("blog");
    toast("Loaded blog", "info");

  } catch (err) {
    toast("Failed to load blog", "error");
  }
}

window.loadBlog = loadBlog;
/* ─────────────────────────────────────────────
   INIT
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // default date = today
  const d = document.getElementById('inp-date');
  if (d && !d.value) d.value = new Date().toISOString().split('T')[0];

  // start on blog tab
  switchTab('blog');
});