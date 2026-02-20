/**
 * APEX Dashboard — Main Application (ES Module)
 *
 * Hash router, global data loader, lazy page module imports.
 */

// Global data store
window.APEX = {
  summary: null,
  scoreHistory: null,
  manifest: null,
  indicators: null,
  _cache: {},  // per-symbol JSON cache
  navigateTo: () => {},  // replaced below
};

const PAGES = ['overview', 'signals', 'screeners', 'regime', 'backtest'];
const pageModules: Record<string, any> = {};
const pageInitialized = new Set<string>();

// ─── Data Loading ────────────────────────────────────────────────────────────

async function fetchJSON(url: string): Promise<any> {
  const resp = await fetch(url);
  if (!resp.ok) return null;
  return resp.json();
}

async function loadGlobalData(): Promise<void> {
  const [summary, scoreHistory, manifest, indicators] = await Promise.all([
    fetchJSON('data/summary.json'),
    fetchJSON('data/score_history.json'),
    fetchJSON('data/manifest.json'),
    fetchJSON('data/indicators.json'),
  ]);

  window.APEX.summary = summary;
  window.APEX.scoreHistory = scoreHistory;
  window.APEX.manifest = manifest;
  window.APEX.indicators = indicators;

  // Populate generation timestamp
  if (summary?.generated_at) {
    const ts = new Date(summary.generated_at);
    const el = document.getElementById('gen-timestamp');
    if (el) el.textContent = `Updated: ${ts.toLocaleString()}`;
  }
}

// ─── Lazy Page Loading ───────────────────────────────────────────────────────

async function initPage(name: string, params: Record<string, string>): Promise<void> {
  if (!pageModules[name]) {
    try {
      pageModules[name] = await import(`./pages/${name}.js`);
    } catch (err) {
      console.error(`Failed to load page module: ${name}`, err);
      return;
    }
  }

  const mod = pageModules[name];
  if (pageInitialized.has(name)) {
    if (mod.update) mod.update(window.APEX, params);
  } else {
    if (mod.init) await mod.init(window.APEX, params);
    pageInitialized.add(name);
  }
}

// ─── Hash Router ─────────────────────────────────────────────────────────────

function parseHash(): { page: string; params: Record<string, string> } {
  const hash = window.location.hash || '#/overview';
  const [path, query] = hash.slice(1).split('?');
  const page = path.replace('/', '') || 'overview';
  const params: Record<string, string> = {};

  if (query) {
    for (const part of query.split('&')) {
      const [k, v] = part.split('=');
      if (k) params[decodeURIComponent(k)] = decodeURIComponent(v || '');
    }
  }

  return { page: PAGES.includes(page) ? page : 'overview', params };
}

function navigateTo(page: string, params: Record<string, string> = {}): void {
  let hash = `#/${page}`;
  const qs = Object.entries(params)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');
  if (qs) hash += `?${qs}`;
  window.location.hash = hash;
}

// Make navigateTo available globally for page modules
window.APEX.navigateTo = navigateTo;

async function handleRoute(): Promise<void> {
  const { page, params } = parseHash();

  // Toggle active page section
  for (const p of PAGES) {
    const section = document.getElementById(`page-${p}`);
    if (section) section.classList.toggle('active', p === page);
  }

  // Toggle active nav link
  for (const link of document.querySelectorAll('.nav-link')) {
    (link as HTMLElement).classList.toggle('active', (link as HTMLElement).dataset.page === page);
  }

  // Initialize/update page module
  await initPage(page, params);
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────

async function boot(): Promise<void> {
  try {
    await loadGlobalData();
  } catch (err) {
    console.error('Failed to load dashboard data:', err);
    document.querySelector('.page.active .page-content')?.insertAdjacentHTML(
      'afterbegin',
      '<div class="empty-state">Failed to load dashboard data. Ensure pipeline has been run.</div>'
    );
    return;
  }

  window.addEventListener('hashchange', handleRoute);
  await handleRoute();
}

boot();
