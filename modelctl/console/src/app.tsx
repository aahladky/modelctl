import { LocationProvider, Router, Route, useLocation, useRoute } from "preact-iso";
import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import { effectiveTheme, onSystemThemeChange, toggleTheme } from "./theme";
import { useStream } from "./lib/stream";
import { fmtClock } from "./lib/api";
import { ToastHost } from "./lib/toasts";
import { Operate } from "./pages/operate";
import { Job, Jobs } from "./pages/jobs";
import { Models } from "./pages/models";
import { Model } from "./pages/model";
import { Add } from "./pages/add";
import { Wizard } from "./pages/wizard";
import { Settings } from "./pages/settings";

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
  const under = (p: string) =>
    (path === p || path.startsWith(p + "/") ? "item here" : "item");
  return (
    <aside class="side">
      <div class="brand">modelctl</div>
      <a class={here("/v2/")} href="/v2/">▣ operate</a>
      <a class={under("/v2/models")} href="/v2/models">◇ model hub</a>
      <a class={under("/v2/add")} href="/v2/add">＋ add</a>
      {/* `under`, not `here`: a per-job page is still the jobs section */}
      <a class={under("/v2/jobs")} href="/v2/jobs">
        ≡ jobs{running > 0 && <span class="badge">{running}</span>}
      </a>
      <span class="spacer"></span>
      <a class={here("/v2/settings")} href="/v2/settings">⚙ settings</a>
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
      stream dropped
      {lastAt != null ? ` · last known ${fmtClock(lastAt)}` : ""}
      {retryIn != null ? ` · retrying in ${retryIn}s` : " · reconnecting…"}
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

function ModelShell() {
  // preact-iso already URL-decodes params.
  const { params } = useRoute();
  const name = params.name ?? "";
  return (
    <Shell title={`model hub · ${name}`} live="telemetry live">
      <Model name={name} />
    </Shell>
  );
}

function JobShell() {
  const { params } = useRoute();
  const id = params.id ?? "";
  return (
    <Shell title={`job · ${id.slice(0, 12)}`} live="job stream live">
      <Job id={id} />
    </Shell>
  );
}

function WizardShell() {
  const { params } = useRoute();
  const id = params.id ?? "";
  return (
    <Shell title="add model" live="wizard live">
      <Wizard id={id} />
    </Shell>
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
        <Route path="/v2/jobs/:id" component={JobShell} />
        <Route path="/v2/models" component={() => (
          <Shell title="model hub" live="telemetry live"><Models /></Shell>
        )} />
        <Route path="/v2/models/:name" component={ModelShell} />
        <Route path="/v2/add" component={() => (
          <Shell title="add model" live="wizard live"><Add /></Shell>
        )} />
        <Route path="/v2/add/:id" component={WizardShell} />
        <Route path="/v2/settings" component={() => (
          <Shell title="settings" live="telemetry live"><Settings /></Shell>
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
