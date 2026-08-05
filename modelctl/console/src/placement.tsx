/* Place a model by picking machines, not plans.
 *
 * The screen the redesign was settled around (Aaron 2026-08-04): tick
 * the devices this model may use, drag a ceiling on any of them, and
 * the planner answers where the weights land -- the split, the share
 * bar, the spill. Compose state, never pick from a computed list;
 * "plans" appears nowhere on this surface.
 *
 * Every number is the planner's own, fetched through the same
 * plan_for_selection the apply computes, so the layout on screen is the
 * layout that runs. The screen holds exactly one piece of state of its
 * own: the selection being edited. Everything else is the wire's.
 *
 * Preview requests race (a drag fires many), so responses carry a
 * sequence and only the newest lands. The apply is gated server-side:
 * a structural change answers 409 with its changes list, rendered here
 * as the confirm panel, and accepting re-sends the same selection with
 * accept_tier_change -- the gate is read against the selected plan.
 */
import { useEffect, useRef, useState } from "preact/hooks";

import {
  ApiError, applyPlacement, fetchAdoptPin, fetchPlacement, fmtGiB,
} from "./lib/api";
import { DeviceRow } from "./lib/devicebar";
import { ModelShare } from "./lib/modelshare";
import {
  deviceOrder, fmtWhen, selectionsEqual, sharesOf, updateChoice,
} from "./lib/placement";
import { toast } from "./lib/toasts";
import type {
  DeviceChoice, Gate, PlacementSelection, PlacementView,
} from "./lib/types";

const PREVIEW_DEBOUNCE_MS = 250;

export function Placement({ name }: { name: string }) {
  const [view, setView] = useState<PlacementView | null>(null);
  const [sel, setSel] = useState<PlacementSelection>({});
  const [failed, setFailed] = useState("");
  const [planning, setPlanning] = useState(false);
  const [applying, setApplying] = useState(false);
  const [gate, setGate] = useState<Gate | null>(null);
  const [adoptCaveat, setAdoptCaveat] = useState("");
  /* Sticky once the operator re-reads: previews and the eventual apply
     must spend the same machine snapshot they were shown. A ref, not
     state, and latched BEFORE the fetch -- as state it latched in the
     response handler, so an edit racing the re-read superseded that
     response and silently un-stuck the operator's explicit choice
     (review, 2026-08-05). */
  const fresh = useRef(false);
  const seq = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  /* The screen can outlive a debounced preview the moment richer
     navigation lands; a response for an unmounted screen must land
     nowhere. */
  const alive = useRef(true);
  useEffect(() => () => {
    alive.current = false;
    clearTimeout(timer.current);
  }, []);
  const gatePanel = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (gate) gatePanel.current?.focus();
  }, [gate]);

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        /* The bare read answers the AUTOMATIC layout plus what is
           applied; the screen opens on the applied selection, so a
           by-hand model needs the second read to show what runs. */
        const bare = await fetchPlacement(name, {});
        if (!live) return;
        const applied = bare.applied_selection ?? {};
        if (Object.keys(applied).length) {
          setSel(applied);
          const v = await fetchPlacement(name, applied);
          if (live) setView(v);
        } else {
          setView(bare);
        }
      } catch (e) {
        if (live) setFailed(String(e));
      }
    })();
    return () => { live = false; };
  }, [name]);

  function preview(next: PlacementSelection,
                   opts: { fresh?: boolean; immediate?: boolean } = {}) {
    setSel(next);
    setGate(null);
    /* The adoption caveat describes the selection adopt() just loaded;
       any further edit makes it about a selection nobody is looking at.
       adopt() re-sets it after calling here. */
    setAdoptCaveat("");
    if (opts.fresh) fresh.current = true;
    const mine = ++seq.current;
    const run = async () => {
      if (!alive.current) return;
      setPlanning(true);
      try {
        const v = await fetchPlacement(name, next, fresh.current);
        if (!alive.current || seq.current !== mine) return;
        setView(v);
      } catch (e) {
        if (alive.current && seq.current === mine) {
          toast("err", "the planner refused this preview", String(e));
        }
      } finally {
        if (alive.current && seq.current === mine) setPlanning(false);
      }
    };
    clearTimeout(timer.current);
    if (opts.immediate) void run();
    else timer.current = setTimeout(run, PREVIEW_DEBOUNCE_MS);
  }

  function change(key: string, patch: DeviceChoice) {
    preview(updateChoice(sel, key, patch));
  }

  async function apply(accept: boolean) {
    setApplying(true);
    try {
      const r = await applyPlacement(name, sel, accept, fresh.current);
      setGate(null);
      toast("ok", "placement queued",
            `job ${r.job_id} — the applied selection updates when it lands`);
    } catch (e) {
      const gated = e instanceof ApiError && e.status === 409
        && (e.body as { gate?: Gate }).gate;
      if (gated) setGate(gated as Gate);
      else toast("err", "the apply was refused", String(e));
    } finally {
      setApplying(false);
    }
  }

  async function adopt() {
    try {
      const a = await fetchAdoptPin(name);
      /* After preview(), which clears the caveat of any prior edit --
         this way the caveat on screen always describes the selection on
         screen. */
      preview(a.selection, { immediate: true });
      setAdoptCaveat(a.caveat);
    } catch (e) {
      toast("err", "nothing to adopt", String(e));
    }
  }

  if (failed) {
    return (
      <div class="widget trouble">
        <div class="label"><span>placement cannot be read</span></div>
        <p class="trouble-detail">{failed}</p>
      </div>
    );
  }
  if (!view) {
    return <div class="widget"><p class="sub">asking the planner…</p></div>;
  }

  const applied = view.applied_selection ?? {};
  const automatic = Object.keys(applied).length === 0;
  const dirty = !selectionsEqual(sel, applied);
  const against = view.planned_against;
  const rows = Object.entries(view.devices ?? {})
    .sort(([a], [b]) => deviceOrder(a) - deviceOrder(b)
                     || a.localeCompare(b));

  return (
    <>
      <div class="widget">
        <div class="label">
          <span>placement · {name}</span>
          <a href="/v2/">home →</a>
        </div>
        <div class="actions">
          <span class="sub">
            {automatic ? "placed automatically" : "placed by hand"}
            {dirty ? " · previewing an unapplied change" : ""}
            {planning ? " · planning…" : ""}
          </span>
          <button type="button"
                  disabled={applying || Object.keys(sel).length === 0}
                  title="clear every choice; the planner places as the machine offers"
                  onClick={() => preview({})}>
            automatic
          </button>
          <button type="button" disabled={applying}
                  title="re-read the machine instead of the recorded snapshot; applying afterwards records the machine you were shown"
                  onClick={() => preview(sel, { fresh: true, immediate: true })}>
            re-read the machine
          </button>
        </div>
        <p class="sub">
          {against.source === "live"
            ? "planned against the machine as it is now"
            : `planned against this machine as it was on ${
                fmtWhen(against.recorded_at) || "an unrecorded date"}`}
          {` · ${fmtGiB(against.ram_available_bytes)} GiB RAM free then`}
          {view.storage_floor && !view.storage_floor.allows_disk
            ? " · this model is set never to run from storage"
            : ""}
        </p>
      </div>

      {view.legacy_pin
        ? <div class="widget warnband">
            <div class="label"><span>a pinned plan still decides launches</span></div>
            <p class="sub">
              Launches follow pinned plan {view.legacy_pin.plan_id}, not a
              device selection. Adopting copies which devices that plan
              uses into a selection you can edit and apply.
            </p>
            <div class="actions">
              <button type="button" onClick={adopt}>
                adopt the pinned plan's devices
              </button>
            </div>
            {adoptCaveat ? <p class="sub">{adoptCaveat}</p> : null}
          </div>
        : null}

      <div class="widget">
        <div class="label"><span>where the weights land</span></div>
        <ModelShare shares={sharesOf(view.devices)}
                    spill={view.spill_bytes} />
        {(view.warnings ?? []).map((w, i) => (
          <p key={i} class="share-warn">{w}</p>
        ))}
      </div>

      <div class="widget">
        <div class="label">
          <span>the devices — tick to allow, drag to cap</span>
        </div>
        {rows.map(([key, row]) => (
          row.usable_bytes == null || row.capacity_bytes == null
            /* Absent is not zero: a missing bound drawn as 0 reads as
               "no room" and disables the over-budget check. Say the
               wire said nothing, and offer no drag against a bound
               that does not exist. */
            ? <div key={key} class="dev">
                <div class="dev-head">
                  <span class="dev-name">{key}</span>
                  <span class="dev-note dev-note-absent">
                    the planner reported no bound for this device — it
                    cannot be capped from here
                  </span>
                </div>
              </div>
            : <DeviceRow key={key}
                         name={key}
                         committed={row.bytes}
                         usable={row.usable_bytes}
                         capacity={row.capacity_bytes}
                         backing={row.backing}
                         state={row.state ?? ""}
                         detail={row.detail ?? ""}
                         on={sel[key]?.on ?? true}
                         ceiling={sel[key]?.ceiling_bytes ?? null}
                         onToggle={(on) => change(key, { on })}
                         onCeiling={(bytes) =>
                           change(key, { ceiling_bytes: bytes })} />
        ))}
      </div>

      {gate
        /* role/alert + focus: the one place a click grows a REQUIRED
           decision elsewhere on the page; without both, a keyboard or
           screen-reader user watches "apply" appear to do nothing. */
        ? <div class="widget trouble" role="alert" tabIndex={-1}
               ref={gatePanel}>
            <div class="label"><span>this change needs a second look</span></div>
            {(gate.changes ?? []).map((c, i) => (
              <p key={i} class="trouble-row"><span>{c}</span></p>
            ))}
            <div class="actions">
              <button class="btn-danger" type="button" disabled={applying}
                      onClick={() => apply(true)}>
                accept and apply
              </button>
              <button type="button" onClick={() => setGate(null)}>
                keep editing
              </button>
            </div>
          </div>
        : <div class="widget">
            <div class="actions">
              <button class="btn-primary" type="button"
                      disabled={applying || planning}
                      onClick={() => apply(false)}>
                {applying ? "applying…" : "apply this placement"}
              </button>
              {dirty
                ? <span class="sub">the layout above is a preview until
                        applied</span>
                : <span class="sub">this is what is applied</span>}
            </div>
          </div>}
    </>
  );
}
