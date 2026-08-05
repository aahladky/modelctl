/* The arithmetic behind one device row.
 *
 * Kept out of the component so it can be tested without a DOM, which is
 * the whole shape of the console's test story: no runner, no framework,
 * no new dependency.
 *
 * The row draws three facts the wire now carries for every device --
 * what the hardware is, what the planner may spend, and what this layout
 * committed -- plus the one thing the operator sets, a ceiling. Before
 * the backend alignment work the host row shipped capacity 0 and usable
 * 0 while holding 39.5 GiB, so there was no bar to draw at all.
 *
 *   0                              usable            capacity
 *   |###############|······|          |                    |
 *    committed       headroom         ^ceiling
 *
 * The track spans capacity so the operator can see what policy is
 * withholding; the handle stops at usable, because a ceiling above it
 * changes nothing -- select_inputs clamps to min(total * limit_pct, cap)
 * -- and a control whose top third does nothing teaches that the machine
 * is lying.
 */

/* The quantum IS the control (redesign guide rule 5: steppers over free
   number fields). A quarter-GiB is fine enough to land on a real ceiling
   and coarse enough that a drag never produces 27.40218 GiB. */
export const CEILING_STEP_BYTES = 256 * 1024 * 1024;

export interface DeviceBar {
  /* Bytes this layout put on the device. */
  committed: number;
  /* What the planner is allowed to spend here. The handle's ceiling. */
  usable: number;
  /* What the hardware has. The track's full width. */
  capacity: number;
}

export function pct(part: number, whole: number): number {
  if (!(whole > 0)) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}

/* The widest number the track has to hold. Normally capacity, but a
   layout can commit more than the planner's budget (that is exactly what
   an over-committed row is), and a bar that clipped it would hide the
   overflow the screen exists to show. */
export function trackSpan(bar: DeviceBar): number {
  return Math.max(bar.capacity, bar.usable, bar.committed);
}

/* Where each mark sits on the track, as percentages of the span.
   A device with no capacity yields zeros rather than NaN: the row still
   renders, it just has no bar to fill. */
export function barMarks(bar: DeviceBar, ceiling: number | null) {
  const span = trackSpan(bar);
  return {
    committed: pct(bar.committed, span),
    usable: pct(bar.usable, span),
    ceiling: ceiling == null ? null : pct(ceiling, span),
  };
}

/* A ceiling the machine can honour: never negative, never above what the
   planner may spend, always on the quantum.

   Snapping DOWN, not to nearest: a ceiling is a cap, so rounding up
   would hand back room the operator just took away. */
export function clampCeiling(bytes: number, usable: number,
                             step = CEILING_STEP_BYTES): number {
  if (!Number.isFinite(bytes)) return usable;
  const bounded = Math.max(0, Math.min(bytes, usable));
  return Math.floor(bounded / step) * step;
}

/* Pointer position (0..1 along the track) to a ceiling in bytes. The
   track spans capacity, so a drag past `usable` lands exactly on usable
   rather than running off -- an impossible value cannot be expressed,
   which is the machine setting the bound rather than a validator
   refusing afterwards. */
export function ceilingFromFraction(fraction: number, bar: DeviceBar,
                                    step = CEILING_STEP_BYTES): number {
  const raw = Math.max(0, Math.min(1, fraction)) * trackSpan(bar);
  return clampCeiling(raw, bar.usable, step);
}

/* Whether this row is over its bound. Read off the numbers rather than
   trusting a `fits` flag: that flag was hardcoded true on the host and
   remote rows until 2026-08-04, printing "fits" beside a share streaming
   off the SSD. */
export function isOver(bar: DeviceBar): boolean {
  return bar.usable > 0 && bar.committed > bar.usable;
}

/* What a device is doing, in the fewest words that are still true.
   `state` is the fleet's own vocabulary (PRESENT / STALE /
   PIN_MISMATCH); a device that is not present says so rather than
   silently drawing an empty bar, because a node nobody can reach and a
   node holding nothing look identical otherwise. */
export function deviceNote(backing: string, state: string,
                           detail: string): string {
  if (state && state !== "PRESENT") {
    const why = detail ? ` — ${detail}` : "";
    return state === "PIN_MISMATCH"
      ? `up, but built from a different commit${why}`
      : `not reachable${why}`;
  }
  return backing || "nothing here";
}
