/* Shared presentation primitives (rework phase 2). Pure rendering, no
   data fetching — pages compose these. Introduced ahead of the overview
   reframe (phase 3) so the pieces exist before the page that arranges
   them. */
import type { ComponentChildren } from "preact";
import type { PlacementSummary } from "./types";

/* A titled page section: a ruled line, not another box. The optional
   children render as the small annotation beside the title ("2 of 19
   profiles", a link to the full list). */
export function SectionHead({ title, children }: {
  title: string; children?: ComponentChildren;
}) {
  return (
    <div class="section-head">
      <h2>{title}</h2>
      {children != null && <span class="sub">{children}</span>}
    </div>
  );
}

/* The resulting allocation as chips instead of plan grammar. Falls back
   to the raw placement string until the tick serializes the structured
   summary — absence renders the truth we have, never nothing. */
export function PlacementChips({ summary, fallback }: {
  summary?: PlacementSummary | null; fallback: string;
}) {
  if (!summary || (summary.devices.length === 0 && summary.flags.length === 0)) {
    return <span class="sub">{fallback}</span>;
  }
  return (
    <span class="pchips">
      {summary.devices.map((d) => (
        <span class="pchip" key={d.name}><b>{d.name}</b> {d.text}</span>
      ))}
      {summary.flags.map((f) => (
        <span class="pchip cpu" key={f}>{f}</span>
      ))}
    </span>
  );
}

/* Dots on a thread: the load pipeline. `at` is the index of the stage
   in progress; everything before it is done. Announced as one image —
   a screen reader hears "stage 4 of 6: loading tensors", not six dots. */
export function Pipeline({ stages, at, label }: {
  stages: string[]; at: number; label?: string;
}) {
  const fallback = at >= 0 && at < stages.length
    ? `stage ${at + 1} of ${stages.length}: ${stages[at]}`
    : `${stages.length} stages`;
  return (
    <div class="pipe" role="img" aria-label={label ?? fallback}>
      {stages.map((s, i) => (
        <>
          <div key={s}
               class={i < at ? "node done" : i === at ? "node now" : "node"}>
            <i></i><span>{s}</span>
          </div>
          {i < stages.length - 1 && (
            <div class={i < at ? "link done" : "link"}></div>
          )}
        </>
      ))}
    </div>
  );
}

/* One big number with its unit under it. The number the rig exists to
   produce deserves to be the biggest thing near it. */
export function StatBlock({ value, unit }: { value: string; unit: string }) {
  return (
    <div class="stat">
      <div class="v">{value}</div>
      <div class="u">{unit}</div>
    </div>
  );
}

/* Absence shrinks: a quiet line, not a full-height widget explaining
   that there is nothing to explain. */
export function EmptyState({ children }: { children: ComponentChildren }) {
  return <p class="empty">{children}</p>;
}
