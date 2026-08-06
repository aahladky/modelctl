/* The stack as bars: the home screen's arithmetic.
 *
 * Read-only sibling of device.ts. That file is the math of a CONTROL --
 * a committed amount under a draggable ceiling on one placement screen.
 * This is the math of a PICTURE: every device the planner can spend,
 * what live models hold on it right now, and what room is left. No
 * ceilings, no toggles; home shows, placement decides.
 *
 * The one real rule here is the two-budgets trap that kept RAM
 * illegible for a week. A GPU row's budget is policy (limit_pct of the
 * card minus reserve): what is held sits INSIDE it, so free room is
 * budget minus held. The RAM row's budget is free-minus-reserve read
 * live: a running model is already subtracted, so its budget IS the
 * free room and held sits BESIDE it. Same for a remote ceiling, which
 * no reading ever reduces -- held inside, like a GPU. Encoding this
 * once, tested, is the point of the module; a screen that re-derives it
 * per row is a screen that double-counts a running model again.
 *
 * held is tri-state on purpose: a list (possibly empty) when the
 * runtime DB answered, null when it could not be read. Null renders as
 * "unknown", never as zero -- zero is a claim.
 */
import type {
  FleetDeviceRow, FleetNodeRow, FleetView, GpuRow, RamRow,
} from "./types";
import type { Share } from "./sharemath";

export interface Holder {
  profile: string;
  bytes: number;
}

export interface StackBar {
  /* The admission key -- the same key the gate charges. */
  key: string;
  /* Display name: the device's own on the rig, node-qualified remote. */
  name: string;
  kind: string;
  local: boolean;
  /* Live models holding this device; null = runtime DB unreadable. */
  held: Holder[] | null;
  heldBytes: number | null;
  /* Room the planner may still spend, after the budget rule above.
     null only when held is unknown on a row where held sits inside
     the budget. */
  freeBytes: number | null;
  /* The planner's bound (budget_bytes as the wire carries it). */
  usable: number;
  /* What the hardware has. */
  capacity: number;
  /* Driver-reported bytes in use right now, from the telemetry tick.
     A MEASUREMENT, not an attribution: it includes the desktop, and it
     includes a model that wrote no reservation -- which is exactly why
     it exists (laguna, 2026-08-05: loaded, card full, holdings empty).
     null = no reading for this device (remote, or no tick yet). */
  usedBytes: number | null;
  state: string;
  detail: string;
}

function heldOf(row: FleetDeviceRow): Holder[] | null {
  /* Absent means the view could not read holdings (errors.holdings says
     why); it must not collapse into "nobody holds anything". */
  if (row.held == null || row.held_bytes == null) return null;
  return row.held.map((h) => ({ profile: h.profile, bytes: h.bytes }));
}

function freeOf(row: FleetDeviceRow, heldBytes: number | null): number | null {
  const budget = row.budget_bytes;
  /* RAM: budget is live free-minus-reserve -- a running model already
     subtracted itself, so held sits beside the budget, not inside. */
  if (row.kind === "ram") return budget;
  /* GPU limits and remote ceilings contain what is held. */
  if (heldBytes == null) return null;
  return Math.max(0, budget - heldBytes);
}

function barOf(node: FleetNodeRow, row: FleetDeviceRow): StackBar {
  const local = node.location === "local";
  /* One decision for both fields: held and heldBytes are null together
     or known together. Deriving heldBytes from held_bytes alone let a
     payload with one field but not the other render a sized fill
     beside a "nothing held" note -- a self-contradictory row. */
  const held = heldOf(row);
  const heldBytes = held == null ? null : (row.held_bytes ?? 0);
  return {
    key: row.admission_key,
    name: local ? row.name : `${node.name} · ${row.name}`,
    kind: row.kind,
    local,
    held,
    heldBytes,
    freeBytes: freeOf(row, heldBytes),
    usable: row.budget_bytes,
    capacity: row.total_bytes || row.ceiling_bytes,
    usedBytes: null,
    state: node.presence.state,
    detail: node.presence.detail,
  };
}

/* Every device of every enabled node, local first (the wire already
   orders it so). A disabled node is an operator's "not part of my
   stack" and stays off the picture; an unreachable one keeps its row
   with its state showing, because a closed laptop has not stopped
   having a GPU. */
export function stackBars(fleet: FleetView): StackBar[] {
  return fleet.nodes
    .filter((n) => n.enabled)
    .flatMap((n) => n.devices.map((d) => barOf(n, d)));
}

/* The telemetry tick's driver readings, written onto the local bars
   that match by device name. Remote devices have no local driver and
   stay null -- unknown, not zero. Returns new bars; never mutates.

   A GPU's free-to-plan is also CLAMPED by the driver's free bytes: the
   budget is a policy bound (limit_pct minus reserve) that knows nothing
   about a launch that wrote no reservation, so with laguna filling the
   card it read "28.7 free" over a strip showing 4 left (live,
   2026-08-05). The gate admits against driver-free; the bar must not
   promise more than the gate would. RAM needs no clamp -- its budget
   is already live free-minus-reserve. */
export function withUsage(bars: StackBar[], gpus: GpuRow[] | null,
                          ram: RamRow | null): StackBar[] {
  return bars.map((bar) => {
    if (!bar.local) return bar;
    if (bar.kind === "ram") {
      return ram == null ? bar : { ...bar, usedBytes: ram.used_bytes };
    }
    const gpu = (gpus ?? []).find((g) => g.device === bar.key);
    if (gpu == null) return bar;
    const freeBytes = Math.max(0, Math.min(
      bar.freeBytes ?? Number.POSITIVE_INFINITY, gpu.free_bytes));
    return { ...bar, usedBytes: gpu.used_bytes, freeBytes };
  });
}

export interface ModelSharesView {
  shares: Share[];
  /* True when any bar's holdings were unreadable: the shares that ARE
     here are real, but the total may be missing devices. A bar drawn
     from a partial read must say so, or it under-reports the model. */
  partial: boolean;
  /* Bytes addressed off the SSD, or null when the fleet has not said.
     Passed straight to ModelShare, whose spill badge is tri-state for
     the same reason: zero is a claim, and "nothing on disk" is the
     worst thing to print about a model that is mostly on disk. */
  spill: number | null;
}

/* What one model streams off the SSD instead of holding.
   Tri-state, and the three states are genuinely different:
     null  the fleet has not answered, or its holdings threw -- the key
           is absent and we never got to look
     0     the fleet answered and this model streams nothing
     n     this model addresses n bytes off the disk

   These bytes have no device to sit on, which is exactly why no bar
   could ever show them: an mmap'd expert occupies no budget anywhere
   (the planner books it outside RAM admission, and the RAM row's
   free-minus-reserve budget already counts those pages as reclaimable).
   So they ride the view, not a row. */
export function streamedBytes(fleet: FleetView | null,
                              profile: string): number | null {
  const map = fleet?.mmap_bytes;
  if (map == null) return null;
  /* The map comes off JSON.parse, so it inherits Object.prototype and a
     bare lookup answers for "constructor" or "toString" with a
     function. Profile names are operator-chosen, and a function fails
     every `> 0` test silently -- the model would render as holding
     nothing on disk. Typing the value covers the prototype keys and a
     malformed payload in one check. */
  const bytes = map[profile];
  return typeof bytes === "number" && bytes > 0 ? bytes : 0;
}

/* Where one model's bytes sit, as shares for the model bar.
   Backing follows the bar, not the plan: the device shares are bytes
   actually held right now. The SSD tail is the one share no bar can
   carry -- streamed bytes are precisely the ones nothing holds -- so it
   arrives separately and is appended last, where rankOf already sorts
   it. Without it the bar accounted for a fraction of the model and said
   nothing about the rest (laguna, 2026-08-05: 43.2 GiB of bars under a
   54.7 GiB heading, with 18.7 GiB on a disk that nothing named). */
export function modelShares(profile: string, bars: StackBar[],
                            streaming: number | null = null):
                            ModelSharesView {
  const shares: Share[] = [];
  let partial = false;
  for (const bar of bars) {
    if (bar.held == null) {
      partial = true;
      continue;
    }
    const mine = bar.held
      .filter((h) => h.profile === profile)
      .reduce((sum, h) => sum + h.bytes, 0);
    if (mine <= 0) continue;
    const backing = bar.kind === "ram" ? "RAM"
      : bar.local ? "VRAM" : "over RPC";
    shares.push({ name: bar.name, bytes: mine, backing });
  }
  if (streaming != null && streaming > 0) {
    shares.push({ name: "this machine's SSD", bytes: streaming,
                  backing: "SSD via mmap" });
  }
  return { shares, partial, spill: streaming };
}
