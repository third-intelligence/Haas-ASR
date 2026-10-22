// Shared helpers for the demo pages.

export const LANG = new URLSearchParams(location.search).get("lang") === "en" ? "en" : "ja";

/** Fill every [data-t] element from the page's strings and set the language. */
export function applyStrings(strings) {
  document.documentElement.lang = LANG;
  for (const el of document.querySelectorAll("[data-t]")) {
    if (typeof strings[el.dataset.t] === "string") el.textContent = strings[el.dataset.t];
  }
  document.title = `Haas-ja-v1.0 · ${strings.title}`;
  for (const a of document.querySelectorAll("nav a")) if (LANG === "en") a.search = "?lang=en";
}

export function setStatus(el, text, kind) {
  el.querySelector("span").textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

/** Show whether the server answers. */
export async function showServerInfo(el, strings) {
  try {
    if (!(await fetch("/api/info")).ok) throw new Error("no server info");
    setStatus(el, strings.connected, "live");
  } catch {
    setStatus(el, strings.unavailable, "error");
  }
}

export function socketUrl(path) {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;
}

export function fmtClock(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return (h ? `${h}:` : "") + `${mm}:${String(r).padStart(2, "0")}`;
}

/** A canvas sized for the device pixel ratio, with its CSS width and height. */
export function canvas2d(canvas) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  return [g, w, h];
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** One transcript row: a timestamp and a line of text. */
export function makeRow(start, text) {
  const row = document.createElement("div");
  const time = document.createElement("time");
  const body = document.createElement("div");
  row.className = "row";
  time.textContent = fmtClock(start);
  body.textContent = text;
  row.append(time, body);
  return row;
}

/** Briefly highlight a row that just appeared. */
export function flash(row) {
  row.classList.add("fresh");
  requestAnimationFrame(() => requestAnimationFrame(() => row.classList.remove("fresh")));
}
