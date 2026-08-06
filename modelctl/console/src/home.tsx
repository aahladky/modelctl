/* The stack, as one look.
 *
 * Models are launched by API call and mostly without anyone opening this
 * (Aaron, 2026-08-05): llama-swap loads them on demand from configs
 * already written, and the console is not in the request path at all.
 * So the home screen's job is to answer "is my stack fine" -- and the
 * first cut answered it with a sentence and a model list, which Aaron
 * correctly refused as not a home screen at all. The design rule it
 * broke: the picture IS the control. A home with no picture is a notice
 * board.
 *
 * This is the picture. Abnormal still breaks through first and
 * unconditionally (redesign guide rule 8; an empty trouble band is
 * itself the answer). Then the machine: every device the planner can
 * spend, as a bar -- what live models hold on it, who holds it, what
 * room is left, what policy withholds. Then what is serving, with the
 * model's own share bar drawn from the same live holdings the fleet
 * answer carries. No controls here: home shows, placement decides.
 *
 * The DETAIL of a failure is deliberately not here. Aaron's call: the
 * record lives on the logs/jobs page, so forensics stay in one place
 * instead of bleeding across every surface.
 */
import { useEffect, useState } from "preact/hooks";

import {
  fetchFleet, fetchJobs, fetchModels, fmtGiB, stateLabel,
} from "./lib/api";
import { pct } from "./lib/device";
import { ModelShare } from "./lib/modelshare";
import { modelShares, stackBars } from "./lib/stack";
import type { StackBar } from "./lib/stack";
import { useStream } from "./lib/stream";
import { troubles } from "./lib/troubles";
import type { TroubleItem } from "./lib/troubles";
import type { FleetView, JobRow, ModelRow } from "./lib/types";

function Trouble({ items }: { items: TroubleItem[] }) {
  if (!items.length) {
    return (
      <div class="calm"
           title="every registered model is servable and every enabled node is reachable">
        <span class="calm-mark">✓</span>
        <span>all clear</span>
        {/* title is mouse-only and unreliably spoken; the sentence the
            pill compressed stays for screen readers. */}
        <span class="sr-only">— every registered model is servable and
              every enabled node is reachable</span>
      </div>
    );
  }
  return (
    <div class="widget trouble">
      <div class="label">
        <span>{items.length} thing{items.length === 1 ? "" : "s"} to look at</span>
        <a href="/v2/jobs">the record is on jobs &amp; logs →</a>
      </div>
      {items.map((t, i) => (
        <p key={i} class="trouble-row">
          <span class="trouble-what">{t.what}</span>
          <span class="trouble-detail">{t.detail}</span>
        </p>
      ))}
    </div>
  );
}

/* One device, read-only. Held is a solid fill, free-to-plan is a wash
   beside it, and everything right of the wash is hardware the policy is
   withholding or the desktop is using. The dashed line still marks
   where the planner's room ends -- at held+free, which for a GPU is its
   budget and for RAM is held beside live free room; stack.ts owns that
   rule so this component cannot re-derive it wrong.

   The numbers live in the bar and the legend lives on the section
   (Aaron 2026-08-05): a caption per row narrating held/free/fitted five
   times over is text describing the picture instead of the picture
   saying it. The only words left on a row are a holder's name, in the
   share it holds, and an absence -- the two facts a bar cannot draw. */
function StackRow({ bar }: { bar: StackBar }) {
  const held = bar.heldBytes;
  const free = bar.freeBytes;
  const span = Math.max(bar.capacity, (held ?? 0) + (free ?? 0), 1);
  const heldPct = pct(held ?? 0, span);
  const freePct = pct(free ?? 0, span);
  const absent = bar.state !== "PRESENT";
  const holders = (bar.held ?? [])
    .map((h) => `${h.profile} ${fmtGiB(h.bytes)}`)
    .join(", ");
  const withheld = bar.capacity - (held ?? 0) - (free ?? 0);
  const spoken = (held == null
    ? `what is held could not be read; ${free == null
        ? "" : `${fmtGiB(free)} GiB free to plan, `}${fmtGiB(bar.capacity)} GiB fitted`
    : `${fmtGiB(held)} GiB held${holders ? ` (${holders})` : ""}, `
      + `${fmtGiB(free ?? 0)} GiB free to plan, `
      + `${fmtGiB(bar.capacity)} GiB fitted`)
    /* The legend names "withheld" for sighted users; the spoken row
       must not know less. */
    + (withheld > 0 ? `; ${fmtGiB(withheld)} GiB withheld by policy` : "");
  /* The right edge only IS capacity when nothing overflows past it, and
     the number only fits when some withheld region exists to hold it --
     drawn over a labeled segment it collides and lies. */
  const showCap = span === bar.capacity && heldPct + freePct <= 90;

  return (
    <div class={`dev${absent ? " dev-absent" : ""}`}>
      <div class="dev-head">
        <span class="dev-name">{bar.name}</span>
        {absent
          ? <span class="dev-note dev-note-absent">
              {bar.state === "PIN_MISMATCH"
                ? "up, but built from a different commit"
                : "not reachable"}
              {bar.detail ? ` — ${bar.detail}` : ""}
            </span>
          : null}
      </div>
      <span class="sr-only">{spoken}</span>
      <div class="dev-track" aria-hidden="true" title={spoken}>
        {held != null && held > 0
          ? <div class="stack-held" style={`width:${heldPct}%`}>
              <span class="bar-num bar-num-held">
                {holders || fmtGiB(held)}
              </span>
            </div>
          : null}
        {free != null && free > 0
          ? <div class="stack-free"
                 style={`left:${heldPct}%;width:${freePct}%`}>
              {/* Suppressed when held is unknown: the unknown badge
                  anchors at the same left edge, and two labels on one
                  spot are both unreadable. */}
              {held != null
                ? <span class="bar-num bar-num-free">{fmtGiB(free)} free</span>
                : null}
            </div>
          : null}
        {held == null
          ? <span class="bar-num bar-num-unknown">held unreadable</span>
          : null}
        <div class="dev-limit" style={`left:${heldPct + freePct}%`} />
        {showCap
          ? <span class="bar-num bar-num-cap">{fmtGiB(bar.capacity)}</span>
          : null}
      </div>
    </div>
  );
}

function Machine({ bars, holdingsError }:
                 { bars: StackBar[]; holdingsError: string }) {
  if (!bars.length) return null;
  return (
    <div class="widget">
      <div class="label">
        <span>the machine</span>
        {/* One legend for every row; the calm alternative is a caption
            per bar saying the same three words five times. */}
        <span class="bar-legend" aria-hidden="true">
          <span class="bar-key">
            <span class="bar-swatch bar-swatch-held" />held</span>
          <span class="bar-key">
            <span class="bar-swatch bar-swatch-free" />free to plan</span>
          <span class="bar-key">
            <span class="bar-swatch bar-swatch-withheld" />withheld</span>
        </span>
      </div>
      {holdingsError
        ? <p class="sub">what is held could not be read — {holdingsError}</p>
        : null}
      {bars.map((b) => <StackRow key={b.key} bar={b} />)}
    </div>
  );
}

/* Why a model has no share bar decides what may be said about it. "This
   launch wrote no reservation" is only true once the fleet has answered
   and its holdings were readable; asserting it while the fleet is still
   loading, or when holdings could not be read, is a confident lie about
   a gap in our own reading -- the exact failure the null-spill rule
   above exists to prevent. */
function ShareOrWhy({ m, bars, fleetReady }:
                    { m: ModelRow; bars: StackBar[]; fleetReady: boolean }) {
  const { shares, partial } = modelShares(m.name, bars);
  if (shares.length) {
    return (
      <>
        <ModelShare shares={shares} spill={null} />
        {partial
          ? <p class="sub">may be incomplete — what is held on some
                           devices could not be read</p>
          : null}
      </>
    );
  }
  if (!fleetReady) {
    return <p class="sub">reading what it holds…</p>;
  }
  if (partial) {
    return <p class="sub">its memory cannot be attributed — what models
                          hold could not be read</p>;
  }
  return (
    <p class="sub">
      where it sits: {m.placement || "not recorded"} — this launch wrote
      no reservation, so its memory cannot be attributed here
    </p>
  );
}

function Serving({ models, bars, fleetReady }:
                 { models: ModelRow[]; bars: StackBar[];
                   fleetReady: boolean }) {
  const live = models.filter((m) => m.running);
  return (
    <div class="widget">
      <div class="label"><span>serving now</span></div>
      {live.length
        ? live.map((m) => (
            <div key={m.name}>
              <p class="srv">
                <a class="srv-name"
                   href={`/v2/?place=${encodeURIComponent(m.name)}`}
                   title="open this model's placement">{m.name}</a>
                <span class="srv-state">{stateLabel(m.state)}</span>
                <span class="sub">
                  {m.tok_s ? `${m.tok_s.toFixed(1)} tok/s` : ""}
                  {m.size_bytes ? ` · ${fmtGiB(m.size_bytes)} GiB` : ""}
                </span>
              </p>
              <ShareOrWhy m={m} bars={bars} fleetReady={fleetReady} />
            </div>
          ))
        : <p class="sub"
             title="models load on demand: the next API call loads whatever it asks for">
            idle
            <span class="sr-only">— models load on demand; the next API
                  call loads whatever it asks for</span>
          </p>}
    </div>
  );
}

function Ready({ models }: { models: ModelRow[] }) {
  const idle = models.filter((m) => m.registered && m.enabled && !m.running);
  return (
    <div class="widget">
      <div class="label">
        <span title="registered and enabled; the API loads them on demand">
          ready · {idle.length}
          <span class="sr-only">models, registered and enabled; the API
                loads them on demand</span>
        </span>
      </div>
      <p class="ready">
        {idle.map((m) => (
          <a key={m.name} class="ready-chip"
             href={`/v2/?place=${encodeURIComponent(m.name)}`}
             title="open this model's placement">{m.name}</a>
        ))}
      </p>
    </div>
  );
}

export function Home() {
  const [models, setModels] = useState<ModelRow[]>([]);
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [fleet, setFleet] = useState<FleetView | null>(null);
  const [failed, setFailed] = useState<string>("");
  const { tick, stale, lastAt, retryIn } = useStream();

  useEffect(() => {
    let live = true;
    /* Each source is awaited on its own: one that is down must not blank
       the two that are up. The first paint comes from these; every
       paint after comes from the tick below. */
    fetchModels().then((m) => { if (live) setModels(m); })
                 .catch((e) => { if (live) setFailed(String(e)); });
    fetchJobs().then((j) => { if (live) setJobs(j); }).catch(() => {});
    fetchFleet().then((f) => { if (live) setFleet(f); }).catch(() => {});
    return () => { live = false; };
  }, []);

  /* The tick already carries every model and job row, pushed; the fleet
     (held bytes, holder names, presence) is not on it, so that one is
     re-read per tick -- a local read that opens no socket. While the
     stream is down nothing overwrites the last-known picture; the
     staleness band below says how old it is instead. */
  useEffect(() => {
    if (!tick) return;
    let live = true;
    setModels(tick.models);
    setJobs(tick.jobs);
    fetchFleet().then((f) => { if (live) setFleet(f); }).catch(() => {});
    return () => { live = false; };
  }, [tick]);

  if (failed) {
    return (
      <div class="widget trouble">
        <div class="label">
          <span>the console cannot read its own state</span>
        </div>
        <p class="trouble-detail">{failed}</p>
      </div>
    );
  }
  const bars = fleet ? stackBars(fleet) : [];
  const holdingsError = String(fleet?.errors?.holdings ?? "");
  const asOf = lastAt
    ? new Date(lastAt).toLocaleTimeString([], { hour: "2-digit",
                                                minute: "2-digit" })
    : "";
  return (
    <>
      {stale
        ? <div class="widget warnband">
            <p class="sub">
              live updates dropped — showing the stack
              {asOf ? ` as of ${asOf}` : " from page open"};{" "}
              {retryIn ? `retrying in ${retryIn}s` : "reconnecting…"}
            </p>
          </div>
        : null}
      <Trouble items={troubles(models, jobs, fleet, Date.now() / 1000)} />
      <Machine bars={bars} holdingsError={holdingsError} />
      <Serving models={models} bars={bars} fleetReady={fleet != null} />
      <Ready models={models} />
    </>
  );
}
