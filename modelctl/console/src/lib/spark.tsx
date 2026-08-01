/* Hand-rolled sparkline: a stroked polyline over a soft filled area,
   the mockup's mcSpark() as a component. Styling comes entirely from
   .spark in tokens.css (stroke/fill are CSS vars, so both themes work). */

interface Props {
  data: number[];
  min?: number;
  max?: number;
  height?: number;
  label: string;
}

export function Spark({ data, min, max, height, label }: Props) {
  const W = 200, H = 34, P = 2;
  if (data.length < 2) return <svg class="spark" style={height ? { height } : undefined} aria-label={label} />;
  let lo = min ?? Math.min(...data);
  let hi = max ?? Math.max(...data);
  if (hi - lo < 1e-9) hi = lo + 1;
  const pts = data.map((v, i) => {
    const x = P + (i / (data.length - 1)) * (W - 2 * P);
    const y = H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg class="spark" style={height ? { height } : undefined}
         viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-label={label}>
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
