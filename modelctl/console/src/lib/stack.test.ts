import { test } from "node:test";
import assert from "node:assert/strict";

import {
  modelShares, stackBars, streamedBytes, withUsage,
} from "./stack.ts";
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

/* ---- what a model streams off the SSD instead of holding ---- */

test("streamed bytes read as unknown until the fleet answers", () => {
  /* Three different silences that must not collapse into zero: no fleet
     yet, a fleet whose holdings threw (the key is absent), and both. */
  assert.equal(streamedBytes(null, "laguna"), null);
  assert.equal(streamedBytes(fleet([node()]), "laguna"), null);
});

test("a fleet that answered says zero for a model that streams nothing", () => {
  /* The map is present, so we DID look. Absent from it is a reading,
     not a gap -- this is what earns the "nothing on disk" badge. */
  const f = { ...fleet([node()]), mmap_bytes: { other: 4 * GIB } };
  assert.equal(streamedBytes(f, "laguna"), 0);
});

test("a model's streamed bytes come back keyed by its own name", () => {
  const f = { ...fleet([node()]),
              mmap_bytes: { laguna: 18 * GIB, other: 4 * GIB } };
  assert.equal(streamedBytes(f, "laguna"), 18 * GIB);
});

test("a profile named after an Object member reads zero, not a function", () => {
  /* The map arrives from JSON.parse, so it inherits Object.prototype
     and a bare lookup on "constructor" answers with a function. That
     then fails every `> 0` test silently and the model renders as
     holding nothing on disk. Profile names are operator-chosen; the
     lookup has to answer about the map's OWN keys only. */
  const f = { ...fleet([node()]), mmap_bytes: { laguna: 18 * GIB } };
  for (const name of ["constructor", "toString", "hasOwnProperty",
                      "valueOf"]) {
    assert.equal(streamedBytes(f, name), 0, name);
  }
});

test("a malformed streamed value reads zero rather than reaching the bar", () => {
  const f = { ...fleet([node()]),
              mmap_bytes: { laguna: "lots" } as unknown as
                          Record<string, number> };
  assert.equal(streamedBytes(f, "laguna"), 0);
});

test("a known streamed share becomes the bar's SSD tail and its spill", () => {
  /* The laguna case: 25.1 held on a card, 18 GiB addressed off the SSD.
     Without the tail the bar accounts for a fraction of the model and
     says nothing about the rest -- the exact reading Aaron refused. */
  const f = fleet([node({ devices: [
    device({ held: [{ profile: "laguna", bytes: 25 * GIB }],
             held_bytes: 25 * GIB }),
  ] })]);
  const view = modelShares("laguna", stackBars(f), 18 * GIB);
  assert.deepEqual(view.shares, [
    { name: "SYCL0", bytes: 25 * GIB, backing: "VRAM" },
    { name: "this machine's SSD", bytes: 18 * GIB,
      backing: "SSD via mmap" },
  ]);
  assert.equal(view.spill, 18 * GIB);
});

test("streaming nothing adds no segment but still reports a zero spill", () => {
  const f = fleet([node({ devices: [
    device({ held: [{ profile: "laguna", bytes: 25 * GIB }],
             held_bytes: 25 * GIB }),
  ] })]);
  const view = modelShares("laguna", stackBars(f), 0);
  assert.equal(view.shares.length, 1);
  assert.equal(view.spill, 0);
});

test("unknown streaming adds no segment and refuses to claim zero", () => {
  const f = fleet([node({ devices: [
    device({ held: [{ profile: "laguna", bytes: 25 * GIB }],
             held_bytes: 25 * GIB }),
  ] })]);
  const view = modelShares("laguna", stackBars(f), null);
  assert.equal(view.shares.length, 1);
  assert.equal(view.spill, null);
});

test("a model that only streams is still a model with a share", () => {
  /* Every expert on disk, no device reservation. This is the shape the
     runtime fold used to drop entirely, and the one where "runs without
     a reservation" would be the most wrong thing to print. */
  const f = fleet([node()]);
  const view = modelShares("allstream", stackBars(f), 18 * GIB);
  assert.deepEqual(view.shares, [
    { name: "this machine's SSD", bytes: 18 * GIB,
      backing: "SSD via mmap" },
  ]);
  assert.equal(view.partial, false);
});

test("the SSD tail sorts last, after the slowest real device", () => {
  const f = fleet([
    node({ devices: [device({ held: [{ profile: "laguna", bytes: 25 * GIB }],
                              held_bytes: 25 * GIB })] }),
    node({ name: "ph16-71-cpu0", location: "remote" as const,
           devices: [device({ name: "CPU",
                              admission_key: "RPC:ph16-71-cpu0:CPU",
                              held: [{ profile: "laguna", bytes: 9 * GIB }],
                              held_bytes: 9 * GIB })] }),
  ]);
  const view = modelShares("laguna", stackBars(f), 18 * GIB);
  assert.deepEqual(view.shares.map((s) => s.backing),
                   ["VRAM", "over RPC", "SSD via mmap"]);
});
