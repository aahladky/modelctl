/* Hand-rolled sparkline: a stroked polyline over a soft filled area,
   the mockup's mcSpark() as a component. Styling comes entirely from
   .spark in tokens.css (stroke/fill are CSS vars, so both themes work). */

interface Props {
  data: number[];
  /* Absolute bounds the series can never leave (0..total VRAM, 0..100%).
     They clamp the auto-range; they are NOT the drawn window. Passing the
     full range as the window compressed a +-0.5 GiB fluctuation on a
     32 GiB card into well under one pixel, and every meter drew a flat
     line -- the mockup scales each spark to a margin around its own data. */
  min?: number;
  max?: number;
  height?: number;
  label: string;
}

export function Spark({ data, min, max, height, label }: Props) {
  const W = 200, H = 34, P = 2;
  /* role="img" + a <title>: an <svg> is a generic grouping element to a
     screen reader, and aria-label alone on one is inconsistently
     announced. The title is the accessible name browsers agree on. */
  if (data.length < 2) {
    return (
      <svg class="spark" style={height ? { height } : undefined}
           role="img" aria-label={label}><title>{label}</title></svg>
    );
  }
  const dLo = Math.min(...data);
  const dHi = Math.max(...data);
  // 15% headroom either side of the observed window, then clamped into
  // whatever absolute bounds the caller declared.
  const pad = Math.max((dHi - dLo) * 0.15, Math.abs(dHi) * 0.002, 1e-6);
  let lo = dLo - pad;
  let hi = dHi + pad;
  if (min != null) lo = Math.max(min, lo);
  if (max != null) hi = Math.min(max, hi);
  if (hi - lo < 1e-9) hi = lo + 1;
  const pts = data.map((v, i) => {
    const x = P + (i / (data.length - 1)) * (W - 2 * P);
    const y = H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg class="spark" style={height ? { height } : undefined}
         viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
         role="img" aria-label={label}>
      <title>{label}</title>
      <polygon class="fillarea"
               points={`${P},${H - P} ${pts.join(" ")} ${W - P},${H - P}`} />
      <polyline points={pts.join(" ")} />
    </svg>
  );
}

/* Fixed-length ring buffer for series fed from the tick stream. */
export function push(buf: number[], v: number, cap = 40): number[] {
  const next = [...buf, v];
  return next.length > cap ? next.slice(next.length - cap) : next;
}
