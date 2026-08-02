/* The one meter bar. Every `.meterbar` on the surface renders through
   this, so the ARIA that makes a bar readable by something other than an
   eye is written once instead of six times by hand -- which is how five
   of the six came to be missing it.

   `valuetext` is not optional decoration. `aria-valuenow` on its own
   announces a bare number ("7.8"), and the only reason the number means
   anything is the unit and the ceiling beside it, which are in the
   markup next to the bar and not in the bar. Callers pass the sentence
   they would say out loud: "7.8 of 12.0 GiB VRAM". */

interface Props {
  /* null means "not read", never "zero" — the bar draws empty and
     aria-valuenow is omitted rather than announced as 0 */
  value: number | null | undefined;
  max: number | null | undefined;
  /* what the bar measures, for a screen reader that lands on the bar
     itself with no surrounding context */
  label: string;
  valuetext: string;
  /* passed through for the few callers that need to constrain the bar's
     width in a table cell */
  style?: string;
}

export function Meter({ value, max, label, valuetext, style }: Props) {
  const known = value != null && max != null
    && Number.isFinite(value) && Number.isFinite(max) && max > 0;
  /* A missing ceiling is a missing reading, not a full bar: draw it
     empty. The number beside the bar is already an em dash in that case,
     and a bar drawn full next to an em dash reads as "full". */
  const pct = known ? Math.max(0, Math.min(100, (value! / max!) * 100)) : 0;
  return (
    <div class="meterbar" style={style} role="progressbar"
         aria-label={label} aria-valuemin={0}
         aria-valuemax={max ?? undefined}
         aria-valuenow={known ? value! : undefined}
         aria-valuetext={valuetext}>
      <div class="fill" style={`width:${pct}%`}></div>
    </div>
  );
}
