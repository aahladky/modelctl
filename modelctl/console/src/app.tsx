import { LocationProvider, Router, Route, useLocation } from "preact-iso";
import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import { effectiveTheme, onSystemThemeChange, toggleTheme } from "./theme";
import { useStream } from "./lib/stream";
import { ToastHost } from "./lib/toasts";
import { Operate } from "./pages/operate";
import { Jobs } from "./pages/jobs";

function ThemeButton() {
  const [, bump] = useState(0);
  useEffect(() => onSystemThemeChange(() => bump((n) => n + 1)), []);
  const dark = effectiveTheme() === "dark";
  return (
    <button class="theme-btn" type="button"
            title={`Switch to ${dark ? "light" : "dark"} theme (persists)`}
            onClick={() => {
              toggleTheme();
              bump((n) => n + 1);
            }}>
      {dark ? "☾ dark" : "☀ light"}
    </button>
  );
}

function Side() {
  const { path } = useLocation();
  const { tick } = useStream();
  const running = tick ? tick.jobs.filter((j) => j.status === "running").length : 0;
  const here = (p: string) => (path === p ? "item here" : "item");
  return (
    <aside class="side">
      <div class="brand">modelctl</div>
      <a class={here("/v2/")} href="/v2/">▣ operate</a>
      {/* model hub / add / settings still live in the old console until
          their phases land -- plain hrefs leave the SPA on purpose */}
      <a class="item" href="/">◇ model hub</a>
      <a class="item" href="/add">＋ add</a>
      <a class={here("/v2/jobs")} href="/v2/jobs">
        ≡ jobs{running > 0 && <span class="badge">{running}</span>}
      </a>
      <span class="spacer"></span>
      <a class="item" href="/settings">⚙ settings</a>
      <a class="item" href="/logout">← logout</a>
    </aside>
  );
}

export function LiveBadge({ label }: { label: string }) {
  const { stale, retryIn, lastAt } = useStream();
  if (!stale) {
    return (
      <span class="live"><span class="dot"></span>{label}</span>
    );
  }
  return (
    <span class="live stale"><span class="dot"></span>
      stream dropped{retryIn != null ? ` · retrying in ${retryIn}s` : " · reconnecting…"}
      {lastAt != null && ""}
    </span>
  );
}

function Shell({ title, live, children }:
               { title: string; live: string; children: ComponentChildren }) {
  return (
    <div class="frame">
      <header class="top">
        <h1>{title}</h1>
        <span class="grow"></span>
        <LiveBadge label={live} />
        <ThemeButton />
      </header>
      <main>{children}</main>
    </div>
  );
}

export function App() {
  return (
    <LocationProvider scope="/v2">
      <Side />
      <Router>
        <Route path="/v2/" component={() => (
          <Shell title="operate" live="telemetry live"><Operate /></Shell>
        )} />
        <Route path="/v2/jobs" component={() => (
          <Shell title="jobs" live="job stream live"><Jobs /></Shell>
        )} />
        <Route default component={() => (
          <Shell title="not found" live="">
            <div class="widget">
              <p>Nothing at this address. <a href="/v2/">Back to operate.</a></p>
            </div>
          </Shell>
        )} />
      </Router>
      <ToastHost />
    </LocationProvider>
  );
}
