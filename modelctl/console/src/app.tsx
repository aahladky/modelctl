/* The console shell, after the 2026-08-04 demolition.

   The previous surface -- sidebar, nine pages, ~5.5k lines -- was taken
   down deliberately. It was not being used and was not considered
   usable, and the defects behind that decision were mostly not in the
   pages: an infinity taken on trust from a worker took the whole console
   down, a probe reported "absent" when it meant "busy", presence had two
   states for four situations, values arrived with no indication of which
   machine or which moment they described. A repaint would have inherited
   every one of those.

   So the mechanisms stayed and the surface went. What is kept here is
   binding, not layout: lib/api.ts and lib/types.ts describe what the
   backend actually returns, lib/stream.ts is the SSE client, theme.ts
   and tokens.css are the warm-ink sheet (the look was never the
   complaint). Everything visual gets rebuilt one screen at a time, each
   against a contract that can be checked.

   This file stays small on purpose. Routing and navigation are IA, and
   IA is exactly what is being redesigned -- there is nothing yet to
   navigate between. */
import { useEffect, useState } from "preact/hooks";
import { effectiveTheme, onSystemThemeChange, toggleTheme } from "./theme";
import { ToastHost } from "./lib/toasts";
import { ProbeBars } from "./probe-bars";
import { Home } from "./home";

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
  /* ?bars renders the device primitive on its own, before any screen
     composes it. Not routing -- routing is IA and IA is what is being
     redesigned -- just a way to look at one control while it is being
     built. It goes when the placement surface lands. */
  const probing = typeof location !== "undefined"
    && new URLSearchParams(location.search).has("bars");
  return (
    <>
      <ToastHost />
      <main class="shell">
        <header class="shell-head">
          <h1>modelctl<span class="dot-accent">.</span></h1>
          <ThemeButton />
        </header>
        {probing ? <ProbeBars /> : <Home />}
        <div class="widget">
          <p>The console surface is being rebuilt, one screen at a time.</p>
          <p class="sub">
            The API, the planner, the job lanes and the fleet are
            untouched and still serving. Nothing here stands in for a
            screen that exists elsewhere — there is no other screen yet.
          </p>
        </div>
      </main>
    </>
  );
}
