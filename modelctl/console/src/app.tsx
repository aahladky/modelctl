/* The console shell, after the 2026-08-04 demolition.
 *
 * The previous surface -- sidebar, nine pages, ~5.5k lines -- was taken
 * down deliberately; the mechanisms stayed (lib/api.ts, lib/types.ts,
 * the SSE client, the warm-ink sheet) and screens come back one at a
 * time, each against a contract that can be checked.
 *
 * Two screens exist: home (the stack as bars) and placement (tick
 * devices, drag ceilings, apply). Navigation is a query parameter, not
 * a router -- ?place=<model> -- because two screens do not need more
 * IA than a link each way, and the IA that will is still being
 * designed. The ?bars gallery this shell used to carry went when the
 * placement surface landed, as its own comment promised. */
import { useEffect, useState } from "preact/hooks";
import { effectiveTheme, onSystemThemeChange, toggleTheme } from "./theme";
import { ToastHost } from "./lib/toasts";
import { Home } from "./home";
import { Placement } from "./placement";

function ThemeButton() {
  const [, bump] = useState(0);
  useEffect(() => onSystemThemeChange(() => bump((n) => n + 1)), []);
  const dark = effectiveTheme() === "dark";
  return (
    /* aria-pressed: the button is a toggle with a state, not an action
       that happens to change appearance. Without it the control
       announces the same way whichever theme is in force. */
    <button class="theme-btn" type="button" aria-pressed={dark}
            title={`Switch to ${dark ? "light" : "dark"} theme (persists)`}
            onClick={() => {
              toggleTheme();
              bump((n) => n + 1);
            }}>
      {dark ? "☀ light" : "☾ dark"}
    </button>
  );
}

export function App() {
  const place = typeof location !== "undefined"
    ? new URLSearchParams(location.search).get("place")
    : null;
  return (
    <>
      <ToastHost />
      <main class="shell">
        <header class="shell-head">
          <h1><a href="/v2/">modelctl<span class="dot-accent">.</span></a></h1>
          <ThemeButton />
        </header>
        {place ? <Placement name={place} /> : <Home />}
      </main>
    </>
  );
}
