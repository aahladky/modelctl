import { test } from "node:test";
import assert from "node:assert/strict";

import { modelShares, stackBars, withUsage } from "./stack.ts";
import type { FleetView } from "./types.ts";

const GIB = 2 ** 30;

function device(over = {}) {
  return {
    name: "SYCL0", kind: "gpu", budget_bytes: 28 * GIB,
    total_bytes: 32 * GIB, cap_bytes: 0, ceiling_bytes: 32 * GIB,
    ceiling_basis: "reported device total", admission_key: "SYCL0",
    editable: false, edit_note: "",
    held: [] as { profile: string; bytes: number }[], held_bytes: 0,
    ...over,
  };
}

function node(over = {}) {
  return {
    name: "rig (this machine)", location: "local" as const,
    host: "127.0.0.1", port: null, endpoint: "local", variant: "local",
    enabled: true, note: "",
    pin: { node: "p", expected: "p", agrees: true },
    presence: { state: "PRESENT" as const, detail: "", reachable: true,
                protocol: "", probed_at: null, ttl_seconds: 900 },
    devices: [device()],
    ...over,
  };
}

function fleet(nodes: ReturnType<typeof node>[]): FleetView {
  return { nodes, night_lane: [], stale_profiles: [], local_pin: "",
           errors: {} } as unknown as FleetView;
}

test("a gpu's held sits inside its budget: free is budget minus held", () => {
  const f = fleet([node({ devices: [device({
    held: [{ profile: "laguna", bytes: 10 * GIB }], held_bytes: 10 * GIB,
  })] })]);
  const [bar] = stackBars(f);
  assert.equal(bar.freeBytes, 18 * GIB);
  assert.equal(bar.heldBytes, 10 * GIB);
});

test("ram's budget is already free room: held sits beside it", () => {
  /* The two-budgets trap. The RAM budget is free-minus-reserve read
     live, so a running model has already subtracted itself -- free must
     NOT subtract it again. */
  const f = fleet([node({ devices: [device({
    name: "RAM", kind: "ram", admission_key: "RAM",
    budget_bytes: 15 * GIB, total_bytes: 31 * GIB,
    held: [{ profile: "q122b", bytes: 5 * GIB }], held_bytes: 5 * GIB,
  })] })]);
  const [bar] = stackBars(f);
  assert.equal(bar.freeBytes, 15 * GIB);
  assert.equal(bar.heldBytes, 5 * GIB);
});

test("unknown held is null everywhere, never zero", () => {
  /* held/held_bytes absent = the runtime DB was unreadable. A zero
     would claim "nothing is held" on no evidence. */
  const f = fleet([node({ devices: [device({
    held: undefined, held_bytes: undefined,
  })] })]);
  const [bar] = stackBars(f);
  assert.equal(bar.held, null);
  assert.equal(bar.heldBytes, null);
  assert.equal(bar.freeBytes, null);
});

test("unknown held still leaves ram's free readable", () => {
  /* RAM's free room never depended on attribution. */
  const f = fleet([node({ devices: [device({
    name: "RAM", kind: "ram", admission_key: "RAM",
    budget_bytes: 15 * GIB, held: undefined, held_bytes: undefined,
  })] })]);
  assert.equal(stackBars(f)[0].freeBytes, 15 * GIB);
});

test("remote devices are node-qualified and keep their presence", () => {
  const f = fleet([node(), node({
    name: "ph16-71-cpu0", location: "remote" as const,
    presence: { state: "STALE" as const, detail: "connection refused",
                reachable: false, protocol: "", probed_at: null,
                ttl_seconds: 900 },
    devices: [device({ name: "CPU", admission_key: "RPC:ph16-71-cpu0:CPU",
                       budget_bytes: 24 * GIB, total_bytes: 30 * GIB })],
  })]);
  const bars = stackBars(f);
  assert.equal(bars.length, 2);
  assert.equal(bars[1].name, "ph16-71-cpu0 · CPU");
  assert.equal(bars[1].state, "STALE");
  assert.equal(bars[1].detail, "connection refused");
});

test("a disabled node is off the picture; an unreachable one is not", () => {
  const f = fleet([node(), node({ name: "off", enabled: false })]);
  assert.equal(stackBars(f).length, 1);
});

test("a held amount over the budget clamps free to zero, not negative", () => {
  const f = fleet([node({ devices: [device({
    budget_bytes: 8 * GIB,
    held: [{ profile: "big", bytes: 10 * GIB }], held_bytes: 10 * GIB,
  })] })]);
  assert.equal(stackBars(f)[0].freeBytes, 0);
});

test("capacity falls back to the ceiling when total is unreported", () => {
  const f = fleet([node({ devices: [device({
    total_bytes: 0, ceiling_bytes: 30 * GIB,
  })] })]);
  assert.equal(stackBars(f)[0].capacity, 30 * GIB);
});

test("held_bytes without held collapses to unknown, not a phantom fill", () => {
  /* The wire declares the two fields independently optional; a payload
     carrying one but not the other must not render a sized fill beside
     a "nothing held" note. One decision for both. */
  const f = fleet([node({ devices: [device({
    held: undefined, held_bytes: 10 * GIB,
  })] })]);
  const [bar] = stackBars(f);
  assert.equal(bar.held, null);
  assert.equal(bar.heldBytes, null);
  assert.equal(bar.freeBytes, null);
});

test("model shares pick out one profile's bytes with the bar's backing", () => {
  const f = fleet([
    node({ devices: [
      device({ held: [{ profile: "q122b", bytes: 27 * GIB },
                      { profile: "other", bytes: 1 * GIB }],
               held_bytes: 28 * GIB }),
      device({ name: "RAM", kind: "ram", admission_key: "RAM",
               budget_bytes: 15 * GIB,
               held: [{ profile: "q122b", bytes: 5 * GIB }],
               held_bytes: 5 * GIB }),
    ] }),
    node({ name: "ph16-71-cpu0", location: "remote" as const,
           devices: [device({ name: "CPU",
                              admission_key: "RPC:ph16-71-cpu0:CPU",
                              held: [{ profile: "q122b", bytes: 24 * GIB }],
                              held_bytes: 24 * GIB })] }),
  ]);
  const view = modelShares("q122b", stackBars(f));
  assert.equal(view.partial, false);
  assert.deepEqual(view.shares, [
    { name: "SYCL0", bytes: 27 * GIB, backing: "VRAM" },
    { name: "RAM", bytes: 5 * GIB, backing: "RAM" },
    { name: "ph16-71-cpu0 · CPU", bytes: 24 * GIB, backing: "over RPC" },
  ]);
});

test("an unreadable device makes the share view partial, not smaller", () => {
  /* The shares that are here are real; the flag says the total may be
     missing devices. Silently under-reporting a model is the same lie
     as zero-for-unknown. */
  const f = fleet([node({ devices: [
    device({ held: [{ profile: "q122b", bytes: 27 * GIB }],
             held_bytes: 27 * GIB }),
    device({ name: "SYCL1", admission_key: "SYCL1",
             held: undefined, held_bytes: undefined }),
  ] })]);
  const view = modelShares("q122b", stackBars(f));
  assert.equal(view.partial, true);
  assert.equal(view.shares.length, 1);
});

test("driver usage lands on the matching local bar, by its own name", () => {
  /* The laguna case (2026-08-05): an unmanaged launch writes no
     reservation, so holdings are empty while the card is visibly full.
     The driver's used bytes are the fact that fills that blind spot --
     as usage, never as attribution. */
  const f = fleet([node({ devices: [
    device(),
    device({ name: "RAM", kind: "ram", admission_key: "RAM",
             budget_bytes: 15 * GIB }),
  ] })]);
  const bars = withUsage(stackBars(f),
                         [{ device: "SYCL0", name: "B70",
                            total_bytes: 32 * GIB, free_bytes: 4 * GIB,
                            used_bytes: 28 * GIB }],
                         { total_bytes: 31 * GIB,
                           available_bytes: 16 * GIB,
                           used_bytes: 15 * GIB });
  assert.equal(bars[0].usedBytes, 28 * GIB);
  assert.equal(bars[1].usedBytes, 15 * GIB);
});

test("a gpu's free-to-plan never promises more than the driver's free", () => {
  /* The budget is policy; an unmanaged launch is invisible to it. The
     gate admits against driver-free, so the bar must not offer room
     the gate would refuse. */
  const f = fleet([node()]);
  const bars = withUsage(stackBars(f),
                         [{ device: "SYCL0", name: "B70",
                            total_bytes: 32 * GIB,
                            free_bytes: 4 * GIB,
                            used_bytes: 28 * GIB }], null);
  assert.equal(bars[0].freeBytes, 4 * GIB);
});

test("a gpu with unknown held still gets the driver's free as its bound", () => {
  const f = fleet([node({ devices: [device({
    held: undefined, held_bytes: undefined,
  })] })]);
  const bars = withUsage(stackBars(f),
                         [{ device: "SYCL0", name: "B70",
                            total_bytes: 32 * GIB,
                            free_bytes: 4 * GIB,
                            used_bytes: 28 * GIB }], null);
  assert.equal(bars[0].freeBytes, 4 * GIB);
});

test("a device the driver did not report reads unknown usage, not zero", () => {
  const f = fleet([
    node(),
    node({ name: "ph16-71-cpu0", location: "remote" as const,
           devices: [device({ name: "CPU",
                              admission_key: "RPC:ph16-71-cpu0:CPU" })] }),
  ]);
  const bars = withUsage(stackBars(f), [], null);
  assert.equal(bars[0].usedBytes, null);
  assert.equal(bars[1].usedBytes, null);
});

test("no tick at all leaves usage unknown everywhere", () => {
  const f = fleet([node()]);
  const bars = withUsage(stackBars(f), null, null);
  assert.equal(bars[0].usedBytes, null);
});

test("a model holding nothing has no shares, and unknown is not a share", () => {
  const f = fleet([node({ devices: [
    device(),
    device({ name: "SYCL1", admission_key: "SYCL1",
             held: undefined, held_bytes: undefined }),
  ] })]);
  const view = modelShares("q122b", stackBars(f));
  assert.deepEqual(view.shares, []);
  assert.equal(view.partial, true);
});
