/* "where it runs" -- the model page's placement surface.

   You place a model by ticking the devices it may use and dragging each
   one's memory ceiling; the planner emits the split, the device list and
   the -ot rules from that. There is no ranked list of compiled
   candidates here on purpose: "plans" is a code artifact, not something
   an operator picks from.

   Every number on this screen comes from the planner. The choice goes to
   GET /placement and the answer says how many bytes landed on each
   device and how many had nowhere to go; nothing here works out a split
   of its own. An earlier draft did, and that is the same class of
   failure as silent SSD streaming relocated into the browser -- the
   moment the two disagree, the screen is lying about where the weights
   went. The apply posts the SAME selection the preview asked about, so
   what was shown is what runs.

   Rules this screen encodes, each one settled in the 2026-08-04 design
   session:

     * Memory reads first and throughput second. Memory is the
       constraint; speed is its consequence, which is what lets the
       screen EXPLAIN slowness ("21 GB has nowhere to go") instead of
       merely reporting it.
     * The picture is the control. The memory bar IS the ceiling slider
       -- a real range input painted over the fill -- so keyboard and
       touch work without a second form.
     * One bar colour. Tightness is said in words ("0.2 GB spare") and
       in the banner; recolouring the bar said it a third time.
     * A measured number stops being claimed the moment the layout
       changes, because the measurement no longer describes what would
       run.
     * A machine is a machine. The laptop is two fleet nodes
       (ph16-71-cuda0, ph16-71-cpu0) and one computer, so rows group by
       host.
*/
import { useEffect, useMemo, useState } from "preact/hooks";
import {
  ApiError, applyPlacement, fetchFleet, fetchModelPlans, fetchPlacement,
  fmtGiB,
} from "../lib/api";
import { toast } from "../lib/toasts";
import { Info } from "../lib/info";
import type {
  FleetDeviceRow, FleetNodeRow, FleetView, Gate, PlacementSelection,
  PlacementView, PlanRow,
} from "../lib/types";

const GIB = 2 ** 30;

/* The planner is asked again on every change; a drag would otherwise be
   one request per pixel. Long enough to coalesce a drag, short enough
   that letting go feels immediate. */
const REPLAN_DEBOUNCE_MS = 200;

/* A device the model may be placed on, flattened out of the fleet view
   and tagged with the machine it physically lives in. */
interface Slot {
  key: string;          // admission key -- what the gate charges
  node: string;         // fleet node
  device: string;
  machine: string;      // grouping id (the host)
  name: string;
  kind: string;
  budget: number;       // bytes the planner may spend here
  total: number;
  editable: boolean;
  editNote: string;
  present: boolean;
}

interface Machine {
  id: string;
  title: string;
  sub: string;
  local: boolean;
  slots: Slot[];
}

/* Display order: fastest first, and local memory last because it is the
   last stop before the SSD. Within a kind, the bigger budget leads. This
   orders the ROWS only -- the planner decides the fill. */
function rank(s: Slot, local: boolean): number {
  if (s.kind === "gpu") return local ? 0 : 1;
  if (!local) return 2;
  return 3;
}

/* "ph16-71-cuda0" + "ph16-71-cpu0" -> "ph16-71": the operator's laptop
   is one computer even though the fleet registers a node per device. */
function sharedName(nodes: FleetNodeRow[]): string {
  if (nodes.length === 1) return nodes[0].name;
  const parts = nodes.map((n) => n.name.split("-"));
  const head: string[] = [];
  for (let i = 0; i < parts[0].length; i++) {
    const seg = parts[0][i];
    if (parts.every((p) => p[i] === seg)) head.push(seg);
    else break;
  }
  return head.length ? head.join("-") : nodes[0].name;
}

function deviceLabel(d: FleetDeviceRow): string {
  if (d.kind === "ram") return "Memory";
  return d.label || d.name;
}

function machinesFrom(view: FleetView): Machine[] {
  const byHost = new Map<string, FleetNodeRow[]>();
  for (const n of view.nodes) {
    const list = byHost.get(n.host) ?? [];
    list.push(n);
    byHost.set(n.host, list);
  }
  const out: Machine[] = [];
  byHost.forEach((nodes, host) => {
    const local = nodes.some((n) => n.location === "local");
    const slots: Slot[] = [];
    for (const n of nodes) {
      const present = n.presence.state === "PRESENT";
      for (const d of n.devices) {
        slots.push({
          key: d.admission_key,
          node: n.name,
          device: d.name,
          machine: host,
          name: deviceLabel(d),
          kind: d.kind,
          budget: d.budget_bytes,
          total: d.total_bytes || d.budget_bytes,
          editable: d.editable,
          editNote: d.edit_note,
          present,
        });
      }
    }
    slots.sort((a, b) =>
      rank(a, local) - rank(b, local) || b.budget - a.budget);
    out.push({
      id: host,
      title: local ? "Desktop" : sharedName(nodes),
      sub: local ? "this rig" : host,
      local,
      slots,
    });
  });
  // the rig leads; it is the machine that plans and launches
  out.sort((a, b) => Number(b.local) - Number(a.local));
  return out;
}

/* Which devices a plan actually puts weights on, as a stable key. Two
   layouts are the same layout when they use the same devices, which is
   what lets a measurement outlive the plan id it was recorded against. */
function planKey(p: PlanRow): string {
  const devices = p.admission?.devices ?? {};
  return Object.keys(devices)
    .filter((k) => (devices[k]?.demand_bytes ?? 0) > 0)
    .sort()
    .join("+");
}

/* The devices a placement answer actually used, in the same shape, so a
   measurement recorded for one layout is claimed only by that layout. */
function placedKey(p: PlacementView | null): string {
  if (!p) return "";
  return Object.entries(p.devices)
    .filter(([, d]) => d.bytes > 0 && d.backing !== "SSD via mmap")
    .map(([k]) => k)
    .sort()
    .join("+");
}

/* on[key] === false and a ceiling are the only things worth sending: an
   absent key means "untouched", so an empty selection is the automatic
   placement -- what the machine would do on its own. */
function selectionOf(slots: Slot[], on: Record<string, boolean> | null,
                     cap: Record<string, number | null>): PlacementSelection {
  const out: PlacementSelection = {};
  for (const s of slots) {
    if (!s.present) continue;   // never offered to the planner anyway
    const isOn = on?.[s.key] !== false;
    if (!isOn) { out[s.key] = { on: false }; continue; }
    const c = cap[s.key];
    if (c != null) out[s.key] = { ceiling_bytes: Math.round(c) };
  }
  return out;
}

/* "2026-08-01T17:57:49" -> "1 Aug, 17:57". The planner spends a recorded
   picture of the machine while the rows above show it as it is now; how
   old that picture is decides whether a disagreement between them is a
   problem or just an old snapshot. */
function snapshotWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

/* The tick/ceiling state that means "what is set to run right now". Used
   both to open the screen and to reset it, so those two can never drift
   apart. */
function stateFromApplied(slots: Slot[], applied: PlacementSelection) {
  const on: Record<string, boolean> = {};
  const cap: Record<string, number | null> = {};
  for (const s of slots) {
    on[s.key] = applied[s.key]?.on !== false && s.present;
    const c = applied[s.key]?.ceiling_bytes;
    if (c != null) cap[s.key] = c;
  }
  return { on, cap };
}

export function WhereItRuns({ name, revision, onChanged }: {
  name: string; revision: number; onChanged: () => void;
}) {
  const [view, setView] = useState<FleetView | null>(null);
  const [plans, setPlans] = useState<PlanRow[] | null>(null);
  const [err, setErr] = useState("");
  /* on[key]: is this device allowed. cap[key]: the ceiling the operator
     set in bytes, or null for "as much as it needs". A ceiling only ever
     takes room away -- it can never grant more than the device has. */
  const [on, setOn] = useState<Record<string, boolean> | null>(null);
  const [cap, setCap] = useState<Record<string, number | null>>({});
  const [place, setPlace] = useState<PlacementView | null>(null);
  const [placeErr, setPlaceErr] = useState("");
  const [replanning, setReplanning] = useState(false);
  const [gate, setGate] = useState<Gate | null>(null);
  const [busy, setBusy] = useState(false);
  /* Sticky once set: a re-read is a statement that the recorded snapshot
     is out of date, and every replan after it -- and the apply -- must be
     against the machine as it now is, or the screen would show one
     machine and save another. */
  const [fresh, setFresh] = useState(false);

  useEffect(() => {
    setErr("");
    Promise.all([fetchFleet(), fetchModelPlans(name)])
      .then(([f, p]) => { setView(f); setPlans(p); })
      .catch((e) => setErr(String(e)));
  }, [name, revision]);

  const machines = useMemo(() => view ? machinesFrom(view) : [], [view]);
  const slots = useMemo(() => machines.flatMap((m) => m.slots), [machines]);

  const selection = useMemo(
    () => selectionOf(slots, on, cap), [slots, on, cap]);
  const selKey = JSON.stringify(selection);

  /* The applied selection put through the SAME round trip the live one
     takes, so the two are comparable: a recorded choice can name a device
     that has since gone away, and only normalising both sides through
     selectionOf makes "unchanged" mean unchanged. */
  const appliedKey = useMemo(() => {
    if (!place) return null;
    const seeded = stateFromApplied(slots, place.applied_selection ?? {});
    return JSON.stringify(selectionOf(slots, seeded.on, seeded.cap));
  }, [slots, place]);

  /* The first answer, before any choice exists: the automatic placement.
     It is also what seeds the opening ticks below, so the screen opens on
     what the machine is set to run rather than on a guess. */
  useEffect(() => {
    if (!slots.length || on !== null) return;
    fetchPlacement(name, {})
      .then(setPlace)
      .catch((e) => setPlaceErr(String(e)));
  }, [name, revision, slots.length, on]);

  useEffect(() => {
    if (!slots.length || on !== null || !place) return;
    const seeded = stateFromApplied(slots, place.applied_selection ?? {});
    setOn(seeded.on);
    setCap(seeded.cap);
  }, [slots, place, on]);

  /* Ask the planner where this selection puts the weights. The previous
     answer stays on screen while a new one is in flight, so dragging a
     ceiling does not blank the bars it is dragging. */
  useEffect(() => {
    if (!slots.length || on === null) return;
    let live = true;
    setReplanning(true);
    const timer = setTimeout(() => {
      fetchPlacement(name, selection, fresh)
        .then((p) => { if (live) { setPlace(p); setPlaceErr(""); } })
        .catch((e) => { if (live) setPlaceErr(String(e)); })
        .finally(() => { if (live) setReplanning(false); });
    }, REPLAN_DEBOUNCE_MS);
    return () => { live = false; clearTimeout(timer); };
  }, [name, revision, selKey, slots.length, on === null, fresh]);

  /* Every layout that has ever been RUN on this machine, by device set.
     Everything else is honestly unknown until it is measured. */
  const measured = useMemo(() => {
    const out = new Map<string, { tps: number; id: string }>();
    for (const p of plans ?? []) {
      const tps = p.measured?.generation_tps;
      const k = planKey(p);
      if (!k || tps == null) continue;
      const prev = out.get(k);
      if (!prev || tps > prev.tps) out.set(k, { tps, id: p.id });
    }
    return out;
  }, [plans]);

  if (err) {
    return (
      <div class="widget">
        <p class="sub">where it runs unavailable: {err}</p>
      </div>
    );
  }
  if (!view || !place || !on) {
    return (
      <div class="widget">
        <p class="sub">
          {placeErr ? `the planner could not answer: ${placeErr}`
                    : "reading the machines…"}
        </p>
      </div>
    );
  }

  const bytesOn = (key: string) => place.devices[key]?.bytes ?? 0;
  const spill = place.spill_bytes;
  const hit = measured.get(placedKey(place));
  /* Worst first, so the banner names the device in most trouble rather
     than whichever one happens to be listed earliest. */
  const tight = slots
    .filter((s) => on[s.key] && s.present && bytesOn(s.key) > 0
                   && s.budget - bytesOn(s.key) < GIB)
    .sort((a, b) => (a.budget - bytesOn(a.key)) - (b.budget - bytesOn(b.key)));
  const dirty = appliedKey !== null && selKey !== appliedKey;
  const anyOn = slots.some((s) => on[s.key] && s.present);
  /* An empty selection IS automatic placement -- it is what the planner
     does when nobody has told it otherwise. So "automatic" needs no widget
     of its own, only a way back to it. */
  const isAutomatic = selKey === "{}";
  const appliedAutomatic =
    Object.keys(place.applied_selection ?? {}).length === 0;
  const snapshot = place.planned_against;

  const toAutomatic = () => {
    const next: Record<string, boolean> = {};
    for (const s of slots) next[s.key] = s.present;
    setOn(next);
    setCap({});
    setGate(null);
  };

  const apply = async (accept: boolean) => {
    setBusy(true);
    setGate(null);
    try {
      const r = await applyPlacement(name, selection, accept, fresh);
      toast("ok", `placing ${name}`,
            `job ${r.job_id} · watch it on the jobs page`);
      onChanged();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.body.gate) {
        setGate(e.body.gate as Gate);
      } else if (e instanceof ApiError && e.status === 405) {
        toast("err", "✗ placement refused",
              String(e.body.reason ?? e.message), 9000);
      } else if (e instanceof ApiError) {
        toast("err", `✗ placing ${name} failed`,
              String(e.body.error ?? e.message), 9000);
      } else {
        toast("err", `✗ placing ${name} failed`, String(e), 9000);
      }
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    const seeded = stateFromApplied(slots, place.applied_selection ?? {});
    setOn(seeded.on);
    setCap(seeded.cap);
    setGate(null);
  };

  return (
    <>
      <div class="machines">
        {machines.map((m) => (
          <div class="mach" key={m.id}>
            <div class="mach-head">
              <h2>{m.title}</h2><span class="sub">{m.sub}</span>
            </div>
            {m.slots.map((s) => {
              const used = bytesOn(s.key);
              const isOn = !!on[s.key] && s.present;
              const pct = s.total ? Math.min(100, used / s.total * 100) : 0;
              const spare = s.budget - used;
              const limited = isOn && cap[s.key] != null;
              const backing = place.devices[s.key]?.backing;
              return (
                <div class="dev" key={s.key}
                     aria-pressed={isOn ? "true" : "false"}>
                  <button type="button" class="box" disabled={!s.present}
                          aria-label={`${isOn ? "Stop using" : "Use"} ${s.name}`}
                          onClick={() => setOn({ ...on, [s.key]: !on[s.key] })} />
                  <span>
                    <span class="drow">
                      <span class="dname">{s.name}</span>
                      <span class="dgb">
                        {isOn ? `${fmtGiB(used)} / ` : ""}{fmtGiB(s.total)} GB
                      </span>
                    </span>
                    <span class="bar">
                      <i style={{ width: `${isOn ? pct : 0}%` }} />
                      {isOn && (
                        /* Every device the model may use gets a ceiling,
                           the rig's included. `editable` is a FLEET flag
                           -- it says whether this device's declared budget
                           can be written from the fleet page, and the
                           rig's cannot because it is derived from the
                           settings VRAM limit. A ceiling here writes no
                           budget: it is a per-model planner input, and
                           select_inputs applies it to any device key. An
                           earlier draft dragged the node budget, which is
                           where that gate came from.

                           step 0.1 GiB: on a coarser step a capacity like
                           24.7 never lands on a boundary, so the far right
                           -- and with it "no ceiling" -- is unreachable by
                           dragging. */
                        <input type="range" min={0}
                               max={(s.total / GIB).toFixed(1)} step={0.1}
                               value={((cap[s.key] ?? s.total) / GIB).toFixed(1)}
                               aria-label={`Most memory ${s.name} may use`}
                               onInput={(e) => {
                                 const v = parseFloat(
                                   (e.target as HTMLInputElement).value);
                                 const atMax = v >= s.total / GIB - 0.001;
                                 setCap({
                                   ...cap, [s.key]: atMax ? null : v * GIB,
                                 });
                               }} />
                      )}
                    </span>
                    {limited && (
                      <span class="capnote">
                        limited to {fmtGiB(cap[s.key])} GB
                      </span>
                    )}
                    {!s.present
                      ? <span class="dnote warnnote">not reachable</span>
                      : !isOn
                        ? <span class="dnote">not used</span>
                        : backing === "SSD via mmap"
                          ? <span class="dnote warnnote">
                              streamed from disk
                            </span>
                          : used > 0 && spare < 0
                            ? (
                              /* The planner put more here than this device
                                 is currently offering. Printing that as
                                 "-1.4 GB spare" is arithmetic, not an
                                 answer: say what it means. The two numbers
                                 come from different clocks -- the planner
                                 spends the recorded machine snapshot, the
                                 row shows what the device offers right now
                                 -- so this is exactly the disagreement
                                 worth surfacing rather than smoothing. */
                              <span class="dnote warnnote">
                                {fmtGiB(-spare)} GB more than this device
                                {" "}is offering now
                              </span>
                            )
                            : used > 0 && spare < GIB
                              ? <span class="dnote warnnote">
                                  {fmtGiB(spare)} GB spare
                                </span>
                              : !s.editable
                                ? <span class="dnote">{s.editNote}</span>
                                : null}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Two clocks meet here. The bars above show each device as it is
          right now; the fill in them is what the planner intends, computed
          from a recorded picture of this machine. Saying which picture is
          the difference between "this layout is broken" and "that snapshot
          is three days old". */}
      <div class="placeline">
        <span>
          {appliedAutomatic
            ? "Placed automatically."
            : "Placed by hand."}
          {dirty && <span class="sub"> · unsaved changes</span>}
        </span>
        <span class="rgrow" />
        <span class="sub">
          {fresh
            ? "Reading this machine as it is now."
            : snapshot.recorded_at
              ? `Planned against this machine as it was on `
                + `${snapshotWhen(snapshot.recorded_at)}.`
              : "Planned against this machine as it is now."}
        </span>
        {!fresh && snapshot.source === "stored" && (
          <button type="button" onClick={() => setFresh(true)}>
            Re-read the machine
          </button>
        )}
        {fresh && (
          <button type="button" onClick={() => setFresh(false)}>
            Use the saved snapshot
          </button>
        )}
      </div>

      {spill > 0.05 * GIB && (
        <div class="spill">
          <span>⚠</span>
          <span>
            <b>{fmtGiB(spill, 0)} GB has nowhere to go.</b> It gets streamed
            off the SSD while the model runs, which is why this is slow.
          </span>
        </div>
      )}
      {!!tight.length && spill <= 0.05 * GIB && (
        <div class="watch">
          <span>⚠</span>
          <span>
            <b>{tight[0].name} is nearly full.</b> If that machine needs the
            memory back, this model is killed rather than slowed.
          </span>
        </div>
      )}
      {placeErr && (
        <div class="watch">
          <span>⚠</span>
          <span>
            <b>Showing the last answer the planner gave.</b> {placeErr}
          </span>
        </div>
      )}
      {gate && (
        <div class="spill">
          <span>⚠</span>
          <span>
            <b>This changes where the weights live.</b>
            <ul>{gate.changes.map((c) => <li key={c}>{c}</li>)}</ul>
            <button type="button" class="btn-primary" disabled={busy}
                    onClick={() => apply(true)}>
              {busy ? "applying…" : "Yes, place it this way"}
            </button>
          </span>
        </div>
      )}

      <div class="result">
        <div class="rspeed">
          {hit
            ? <span class="v">{hit.tps.toFixed(1)}</span>
            : <span class="v unk">{anyOn ? "not measured" : "—"}</span>}
          <span class="u">tokens<br />per sec</span>
        </div>
        {hit
          ? <span class="tag measured">measured</span>
          : <span class="tag">{anyOn ? "untested" : "nothing on"}</span>}
        <span class="rgrow" />
        <span class="rnote">
          {replanning
            ? "asking the planner…"
            : !anyOn
              ? "Turn on at least one device."
              : hit
                ? "Run on this exact layout."
                : "This combination has never been run. Apply it, then "
                  + "measure it on the overview tab to find out."}
        </span>
        <button type="button" onClick={toAutomatic}
                disabled={busy || isAutomatic}
                title="Let the planner use every device it can">
          Automatic
        </button>
        <button type="button" onClick={reset}
                disabled={busy || !dirty}>Reset</button>
        <button type="button" class="btn-primary" onClick={() => apply(false)}
                disabled={busy || !dirty || !anyOn}>
          {busy ? "applying…" : "Apply"}
        </button>
      </div>

      <details>
        <summary>
          What the console will run{" "}
          <Info label="about this layout">
            The console writes the split, the device list and the -ot
            rules from these choices. You never type them.
          </Info>
        </summary>
        <pre>
          {place.layout.length
            ? place.layout
              .map((r) => `${r.label}  ${r.gib.toFixed(1)} GB  ${r.detail}`)
              .join("\n")
            : "Nothing selected."}
        </pre>
        {!!place.warnings.length && (
          <ul class="sub">
            {place.warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        )}
      </details>
    </>
  );
}
