import { LocationProvider, Router, Route, useLocation, useRoute } from "preact-iso";
import type { ComponentChildren } from "preact";
import { useEffect, useState } from "preact/hooks";
import { effectiveTheme, onSystemThemeChange, toggleTheme } from "./theme";
import { useStream } from "./lib/stream";
import { fmtClock } from "./lib/api";
import { ToastHost } from "./lib/toasts";
import { Operate } from "./pages/operate";
import { Fleet } from "./pages/fleet";
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

function Side() {
  const { path } = useLocation();
  const { tick } = useStream();
  const running = tick ? tick.jobs.filter((j) => j.status === "running").length : 0;
  /* preact-iso strips trailing slashes before it hands `path` over
     (router.js: `u.pathname.replace(/\/+$/g, '') || '/'`), so on /v2/ it
     reports "/v2". An exact match against "/v2/" therefore never fired
     and the operate item never highlighted. Match both spellings. */
  const same = (p: string) => path === p || path === p.replace(/\/+$/, "");
  const here = (p: string) => (same(p) ? "item here" : "item");
  const under = (p: string) =>
    (same(p) || path.startsWith(p + "/") ? "item here" : "item");
  /* aria-current marks the active item as the current page rather than
     leaving the distinction to background colour alone. */
  const cur = (p: string, fn: (s: string) => string) =>
    (fn(p) === "item here" ? "page" : undefined);
  return (
    <aside class="side">
      <div class="brand">modelctl</div>
      <a class={here("/v2/")} aria-current={cur("/v2/", here)} href="/v2/">
        overview</a>
      {/* next to operate, not under settings: where a model can run is an
          operational question, and the rig is one of the nodes on it */}
      <a class={under("/v2/fleet")} aria-current={cur("/v2/fleet", under)}
         href="/v2/fleet">fleet</a>
      <a class={under("/v2/models")} aria-current={cur("/v2/models", under)}
         href="/v2/models">models</a>
      <a class={under("/v2/add")} aria-current={cur("/v2/add", under)}
         href="/v2/add">add</a>
      {/* `under`, not `here`: a per-job page is still the jobs section */}
      <a class={under("/v2/jobs")} aria-current={cur("/v2/jobs", under)}
         href="/v2/jobs">
        jobs{running > 0 && <span class="badge">{running}</span>}
      </a>
      <span class="spacer"></span>
      <a class={here("/v2/settings")} aria-current={cur("/v2/settings", here)}
         href="/v2/settings">settings</a>
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
    <Shell title={`models · ${name}`} live="updating live">
      <Model name={name} />
    </Shell>
  );
}

function JobShell() {
  const { params } = useRoute();
  const id = params.id ?? "";
  return (
    <Shell title={`job · ${id.slice(0, 12)}`} live="updating live">
      <Job id={id} />
    </Shell>
  );
}

function WizardShell() {
  const { params } = useRoute();
  const id = params.id ?? "";
  return (
    <Shell title="add model" live="updating live">
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
          <Shell title="overview" live="updating live"><Operate /></Shell>
        )} />
        <Route path="/v2/fleet" component={() => (
          <Shell title="fleet" live="updating live"><Fleet /></Shell>
        )} />
        <Route path="/v2/jobs" component={() => (
          <Shell title="jobs" live="updating live"><Jobs /></Shell>
        )} />
        <Route path="/v2/jobs/:id" component={JobShell} />
        <Route path="/v2/models" component={() => (
          <Shell title="models" live="updating live"><Models /></Shell>
        )} />
        <Route path="/v2/models/:name" component={ModelShell} />
        <Route path="/v2/add" component={() => (
          <Shell title="add model" live="updating live"><Add /></Shell>
        )} />
        <Route path="/v2/add/:id" component={WizardShell} />
        <Route path="/v2/settings" component={() => (
          <Shell title="settings" live="updating live"><Settings /></Shell>
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
