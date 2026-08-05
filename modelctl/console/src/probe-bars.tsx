/* A scratch page for the device primitive. Not a screen.
 *
 * Reachable at ?bars while the placement surface is being built, so the
 * control can be looked at and dragged before anything composes it.
 * Three passes were rejected before the design settled, and this is the
 * cheapest point to catch drift from what was approved.
 *
 * The numbers are the live rig's, read off the placement endpoint on
 * 2026-08-05, so what is on screen is the shape the real data makes
 * rather than round figures chosen to flatter the layout.
 */
import { useState } from "preact/hooks";

import { DeviceRow } from "./lib/devicebar";

const GIB = 2 ** 30;

interface Row {
  name: string;
  committed: number;
  usable: number;
  capacity: number;
  backing: string;
  state: string;
  detail: string;
}

/* qwen3-5-122b-a10b-ud, automatic placement, as the API answers today. */
const LIVE: Row[] = [
  { name: "SYCL0", committed: 27.40 * GIB, usable: 28.70 * GIB,
    capacity: 31.89 * GIB, backing: "VRAM", state: "PRESENT", detail: "" },
  { name: "SYCL1", committed: 10.36 * GIB, usable: 10.74 * GIB,
    capacity: 11.93 * GIB, backing: "VRAM", state: "PRESENT", detail: "" },
  { name: "RPC:ph16-71-cuda0:CUDA0", committed: 9.52 * GIB,
    usable: 10.50 * GIB, capacity: 11.60 * GIB, backing: "over RPC",
    state: "PRESENT", detail: "" },
  { name: "RPC:ph16-71-cpu0:CPU", committed: 24.47 * GIB,
    usable: 24.70 * GIB, capacity: 30.57 * GIB, backing: "over RPC",
    state: "PRESENT", detail: "" },
  { name: "RAM", committed: 5.54 * GIB, usable: 20.44 * GIB,
    capacity: 31.08 * GIB, backing: "RAM", state: "PRESENT", detail: "" },
];

/* The states that have to survive contact with a bad day. */
const EDGES: Row[] = [
  { name: "RAM", committed: 39.52 * GIB, usable: 20.44 * GIB,
    capacity: 31.08 * GIB, backing: "SSD via mmap", state: "PRESENT",
    detail: "" },
  { name: "RPC:ph16-71-cpu0:CPU", committed: 0, usable: 24.70 * GIB,
    capacity: 30.57 * GIB, backing: "", state: "STALE",
    detail: "connection refused" },
  { name: "RPC:ph16-71-cuda0:CUDA0", committed: 0, usable: 10.50 * GIB,
    capacity: 11.60 * GIB, backing: "", state: "PIN_MISMATCH", detail: "" },
];

function Gallery({ title, note, rows }:
                 { title: string; note: string; rows: Row[] }) {
  const [on, setOn] = useState<Record<string, boolean>>({});
  const [caps, setCaps] = useState<Record<string, number | null>>({});
  return (
    <div class="widget">
      <div class="label"><span>{title}</span></div>
      <p class="sub">{note}</p>
      {rows.map((r) => (
        <DeviceRow key={r.name} {...r}
                   on={on[r.name] ?? true}
                   ceiling={caps[r.name] ?? null}
                   onToggle={(v) => setOn((s) => ({ ...s, [r.name]: v }))}
                   onCeiling={(b) => setCaps((s) => ({ ...s, [r.name]: b }))} />
      ))}
    </div>
  );
}

export function ProbeBars() {
  return (
    <>
      <Gallery
        title="qwen3-5-122b-a10b-ud — automatic placement"
        note={"Live numbers. Drag a thumb to cap a device; click a name to "
              + "switch it off. The dashed line is where the planner's budget "
              + "ends — everything right of it is hardware the policy is "
              + "withholding."}
        rows={LIVE} />
      <Gallery
        title="the states that have to survive a bad day"
        note={"Over its budget and streaming from disk; a node nobody can "
              + "reach; a node that is up but built from a different commit. "
              + "None of them vanishes — an unreachable device keeps its row "
              + "and the room it offers."}
        rows={EDGES} />
    </>
  );
}
