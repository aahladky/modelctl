/* One device, as a bar you drag.
 *
 * The primitive the placement surface is built from, and the reason it
 * is a component rather than a page: the same object -- a thing with a
 * capacity, an amount committed, and a ceiling you set -- appears at
 * model scope, fleet scope, overview and the wizard. Four pages each
 * drawing their own bars is four ways to disagree with the machine.
 *
 * Mouse first. The ceiling is a drag; there is no number field. The
 * range input IS the picture -- its thumb is the handle sitting on the
 * track -- rather than a control beside a readout, and its bounds come
 * from the machine, so a value the planner would refuse cannot be
 * expressed in the first place.
 *
 * Every number here is read off the wire. Nothing is derived a second
 * time: a screen that works out the split for itself is a screen that
 * can disagree with what runs.
 */
import {
  CEILING_STEP_BYTES, barMarks, ceilingMax, clampCeiling, deviceNote,
  isOver,
} from "./device";
import { fmtGiB } from "./api";

export interface DeviceRowProps {
  name: string;
  committed: number;
  usable: number;
  capacity: number;
  backing: string;
  state: string;
  detail: string;
  /* null means no ceiling set -- the device is on, as the machine
     offers it. Distinct from a ceiling of zero, which is a choice. */
  ceiling: number | null;
  on: boolean;
  onToggle: (on: boolean) => void;
  onCeiling: (bytes: number | null) => void;
}

export function DeviceRow(p: DeviceRowProps) {
  const bar = { committed: p.committed, usable: p.usable,
                capacity: p.capacity };
  const marks = barMarks(bar, p.ceiling);
  const over = isOver(bar);
  const note = deviceNote(p.backing, p.state, p.detail);
  const absent = Boolean(p.state) && p.state !== "PRESENT";
  const ceiling = p.ceiling ?? p.usable;

  return (
    <div class={`dev${p.on ? "" : " dev-off"}${absent ? " dev-absent" : ""}`}>
      <div class="dev-head">
        {/* The whole name is the switch: a tick box beside a label makes
            the operator aim at the smaller of two targets. */}
        <button class="dev-toggle" type="button"
                aria-pressed={p.on}
                title={p.on ? "stop using this device" : "use this device"}
                onClick={() => p.onToggle(!p.on)}>
          <span class="dev-dot" />
          <span class="dev-name">{p.name}</span>
        </button>
        <span class={`dev-note${absent ? " dev-note-absent" : ""}`}>{note}</span>
      </div>

      <div class="dev-track">
        <div class={`dev-fill${over ? " dev-fill-over" : ""}`}
             style={`width:${marks.committed}%`} />
        {/* Where the planner's budget ends. Everything to its right is
            hardware the policy is withholding, which is a fact worth
            seeing rather than a bar that simply stops. */}
        <div class="dev-limit" style={`left:${marks.usable}%`} />
        {/* The slider spans exactly the part of the track it can
            address. At full width its scale was 0..usable drawn across
            0..capacity, so a ceiling of "all of the budget" put its
            handle out at the hardware's edge -- the control
            misreporting its own range.
            Both numbers come from the machine now: the max is the last
            value a stepped input can actually land on, and the width is
            that same value as a share of the track, so every ceiling
            draws where it sits. The handle stops just short of the
            dashed line whenever the budget is off the 256 MiB grid, and
            the hardware beyond it is visibly not draggable. */}
        <input class="dev-ceiling" type="range"
               style={`width:${marks.handle}%`}
               min={0} max={ceilingMax(p.usable)} step={CEILING_STEP_BYTES}
               value={ceiling}
               disabled={!p.on}
               aria-label={`${p.name} ceiling`}
               title="drag to cap what may go here"
               onInput={(e) => {
                 const raw = Number((e.target as HTMLInputElement).value);
                 p.onCeiling(clampCeiling(raw, p.usable));
               }} />
      </div>

      <div class="dev-figures">
        <span class={over ? "dev-fig-over" : ""}>
          {fmtGiB(p.committed)} GiB
        </span>
        <span class="sub">
          {p.ceiling == null
            ? `of ${fmtGiB(p.usable)} usable`
            : `capped at ${fmtGiB(p.ceiling)} of ${fmtGiB(p.usable)}`}
          {p.capacity > p.usable ? ` · ${fmtGiB(p.capacity)} fitted` : ""}
        </span>
        {p.ceiling != null && p.on
          ? <button class="dev-clear" type="button"
                    title="remove the cap and let the planner use the budget"
                    onClick={() => p.onCeiling(null)}>clear cap</button>
          : null}
      </div>
    </div>
  );
}
