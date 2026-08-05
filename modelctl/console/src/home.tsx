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
import { troubles } from "./lib/troubles";
import type { TroubleItem } from "./lib/troubles";
import type { FleetView, JobRow, ModelRow } from "./lib/types";

function Trouble({ items }: { items: TroubleItem[] }) {
  if (!items.length) {
    return (
      <div class="calm">
        <span class="calm-mark">✓</span>
        <span>Nothing is wrong. Every registered model is servable and every
              enabled node is reachable.</span>
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
   rule so this component cannot re-derive it wrong. */
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
          : <span class="dev-note">
              {held == null
                ? "cannot read what is held"
                : holders || "nothing held"}
            </span>}
      </div>
      {/* Decorative: the figures line below carries the same numbers as
          text, so the track is hidden from assistive tech rather than
          read out as three unlabeled boxes. */}
      <div class="dev-track" aria-hidden="true">
        {held != null && held > 0
          ? <div class="dev-fill" style={`width:${heldPct}%`} />
          : null}
        {free != null && free > 0
          ? <div class="stack-free"
                 style={`left:${heldPct}%;width:${freePct}%`} />
          : null}
        <div class="dev-limit" style={`left:${heldPct + freePct}%`} />
      </div>
      <div class="dev-figures">
        <span>{held == null ? "held unknown" : `${fmtGiB(held)} GiB held`}</span>
        <span class="sub">
          {free == null ? "" : `${fmtGiB(free)} free to plan · `}
          {fmtGiB(bar.capacity)} fitted
        </span>
      </div>
    </div>
  );
}

function Machine({ bars, holdingsError }:
                 { bars: StackBar[]; holdingsError: string }) {
  if (!bars.length) return null;
  return (
    <div class="widget">
      <div class="label"><span>the machine</span></div>
      {holdingsError
        ? <p class="sub">what models hold could not be read — {holdingsError}</p>
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
                <span class="srv-name">{m.name}</span>
                <span class="srv-state">{stateLabel(m.state)}</span>
                <span class="sub">
                  {m.tok_s ? `${m.tok_s.toFixed(1)} tok/s` : ""}
                  {m.size_bytes ? ` · ${fmtGiB(m.size_bytes)} GiB` : ""}
                </span>
              </p>
              <ShareOrWhy m={m} bars={bars} fleetReady={fleetReady} />
            </div>
          ))
        : <p class="sub">Nothing is loaded. The next API call will load
                         whatever it asks for.</p>}
    </div>
  );
}

function Ready({ models }: { models: ModelRow[] }) {
  const idle = models.filter((m) => m.registered && m.enabled && !m.running);
  return (
    <div class="widget">
      <div class="label">
        <span>{idle.length} more the API can ask for</span>
      </div>
      <p class="ready">
        {idle.map((m) => <span key={m.name} class="ready-chip">{m.name}</span>)}
      </p>
    </div>
  );
}

export function Home() {
  const [models, setModels] = useState<ModelRow[]>([]);
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [fleet, setFleet] = useState<FleetView | null>(null);
  const [failed, setFailed] = useState<string>("");

  useEffect(() => {
    let live = true;
    /* Each source is awaited on its own: one that is down must not blank
       the two that are up. */
    fetchModels().then((m) => { if (live) setModels(m); })
                 .catch((e) => { if (live) setFailed(String(e)); });
    fetchJobs().then((j) => { if (live) setJobs(j); }).catch(() => {});
    fetchFleet().then((f) => { if (live) setFleet(f); }).catch(() => {});
    return () => { live = false; };
  }, []);

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
  return (
    <>
      <Trouble items={troubles(models, jobs, fleet, Date.now() / 1000)} />
      <Machine bars={bars} holdingsError={holdingsError} />
      <Serving models={models} bars={bars} fleetReady={fleet != null} />
      <Ready models={models} />
    </>
  );
}
