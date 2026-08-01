/* Theme: default follows prefers-color-scheme; an explicit toggle
   overrides and persists (localStorage "mc-theme"). The pre-paint stamp
   lives in index.html so the first frame is already correct. */

export type Theme = "light" | "dark";

export function effectiveTheme(): Theme {
  const t = document.documentElement.dataset.theme;
  if (t === "light" || t === "dark") return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function toggleTheme(): Theme {
  const next: Theme = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("mc-theme", next);
  } catch {
    /* private mode: the override just doesn't persist */
  }
  return next;
}

/* Re-render hook for the toggle button: system preference changes flip the
   effective theme only while no explicit override is stamped. */
export function onSystemThemeChange(fn: () => void): () => void {
  const mq = matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", fn);
  return () => mq.removeEventListener("change", fn);
}
