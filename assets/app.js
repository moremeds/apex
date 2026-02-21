window.APEX = {
  summary: null,
  scoreHistory: null,
  manifest: null,
  indicators: null,
  _cache: {},
  // per-symbol JSON cache
  navigateTo: () => {
  }
  // replaced below
};
const PAGES = ["overview", "signals", "screeners", "regime", "backtest"];
const pageModules = {};
const pageInitialized = /* @__PURE__ */ new Set();
async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) return null;
  return resp.json();
}
async function loadGlobalData() {
  const [summary, scoreHistory, manifest, indicators] = await Promise.all([
    fetchJSON("data/summary.json"),
    fetchJSON("data/score_history.json"),
    fetchJSON("data/manifest.json"),
    fetchJSON("data/indicators.json")
  ]);
  window.APEX.summary = summary;
  window.APEX.scoreHistory = scoreHistory;
  window.APEX.manifest = manifest;
  window.APEX.indicators = indicators;
  if (summary?.generated_at) {
    const ts = new Date(summary.generated_at);
    const el = document.getElementById("gen-timestamp");
    if (el) el.textContent = `Updated: ${ts.toLocaleString()}`;
  }
}
async function initPage(name, params) {
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
function parseHash() {
  const hash = window.location.hash || "#/overview";
  const [path, query] = hash.slice(1).split("?");
  const page = path.replace("/", "") || "overview";
  const params = {};
  if (query) {
    for (const part of query.split("&")) {
      const [k, v] = part.split("=");
      if (k) params[decodeURIComponent(k)] = decodeURIComponent(v || "");
    }
  }
  return { page: PAGES.includes(page) ? page : "overview", params };
}
function navigateTo(page, params = {}) {
  let hash = `#/${page}`;
  const qs = Object.entries(params).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
  if (qs) hash += `?${qs}`;
  window.location.hash = hash;
}
window.APEX.navigateTo = navigateTo;
async function handleRoute() {
  const { page, params } = parseHash();
  for (const p of PAGES) {
    const section = document.getElementById(`page-${p}`);
    if (section) section.classList.toggle("active", p === page);
  }
  for (const link of document.querySelectorAll(".nav-link")) {
    link.classList.toggle("active", link.dataset.page === page);
  }
  await initPage(page, params);
}
async function boot() {
  try {
    await loadGlobalData();
  } catch (err) {
    console.error("Failed to load dashboard data:", err);
    document.querySelector(".page.active .page-content")?.insertAdjacentHTML(
      "afterbegin",
      '<div class="empty-state">Failed to load dashboard data. Ensure pipeline has been run.</div>'
    );
    return;
  }
  window.addEventListener("hashchange", handleRoute);
  await handleRoute();
}
boot();
