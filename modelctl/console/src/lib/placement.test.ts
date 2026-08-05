import { test } from "node:test";
import assert from "node:assert/strict";

import {
  deviceOrder, fmtWhen, selectionsEqual, sharesOf, updateChoice,
} from "./placement.ts";
import type { PlacementSelection } from "./types.ts";

const GIB = 2 ** 30;

test("updating a choice never mutates the selection it was given", () => {
  const before: PlacementSelection = { SYCL0: { on: false } };
  const after = updateChoice(before, "SYCL0", { on: true });
  assert.deepEqual(before, { SYCL0: { on: false } });
  assert.deepEqual(after, { SYCL0: { on: true } });
});

test("a ceiling merges beside an existing on/off decision", () => {
  const sel = updateChoice({ SYCL0: { on: false } }, "SYCL0",
                           { ceiling_bytes: 8 * GIB });
  assert.deepEqual(sel, { SYCL0: { on: false, ceiling_bytes: 8 * GIB } });
});

test("clearing a ceiling keeps the on/off decision", () => {
  const sel = updateChoice(
    { SYCL0: { on: false, ceiling_bytes: 8 * GIB } },
    "SYCL0", { ceiling_bytes: null });
  assert.deepEqual(sel, { SYCL0: { on: false } });
});

test("a choice that says nothing is removed, not stored", () => {
  /* {on: undefined, no ceiling} is 'as the machine offers' -- exactly
     what an absent key means, so storing it would make the same intent
     compare unequal to itself. */
  const sel = updateChoice(
    { SYCL0: { ceiling_bytes: 8 * GIB } }, "SYCL0",
    { ceiling_bytes: null });
  assert.deepEqual(sel, {});
});

test("an explicit on:true survives -- forcing a device is an intent", () => {
  const sel = updateChoice({}, "RPC:ph16-71-cpu0:CPU", { on: true });
  assert.deepEqual(sel, { "RPC:ph16-71-cpu0:CPU": { on: true } });
});

test("selections compare by meaning, not by key presence", () => {
  assert.ok(selectionsEqual({}, {}));
  assert.ok(selectionsEqual({ SYCL0: {} }, {}));
  assert.ok(selectionsEqual(
    { SYCL0: { on: true, ceiling_bytes: undefined } },
    { SYCL0: { on: true } }));
  assert.ok(!selectionsEqual({ SYCL0: { on: true } }, {}));
  assert.ok(!selectionsEqual({ SYCL0: { on: false } },
                             { SYCL0: { on: true } }));
  assert.ok(!selectionsEqual({ SYCL0: { ceiling_bytes: 8 * GIB } },
                             { SYCL0: { ceiling_bytes: 9 * GIB } }));
});

test("shares come from the rows that hold bytes, with their backing", () => {
  const shares = sharesOf({
    SYCL0: { bytes: 27 * GIB, backing: "VRAM", fits: true },
    SYCL1: { bytes: 0, backing: "", fits: true },
    RAM: { bytes: 5 * GIB, backing: "SSD via mmap", fits: false },
  });
  assert.deepEqual(shares, [
    { name: "SYCL0", bytes: 27 * GIB, backing: "VRAM" },
    { name: "RAM", bytes: 5 * GIB, backing: "SSD via mmap" },
  ]);
});

test("devices order local GPUs, then remote, then memory", () => {
  const keys = ["RAM", "RPC:ph16-71-cpu0:CPU", "SYCL1", "SYCL0",
                "RPC:ph16-71-cuda0:CUDA0"];
  keys.sort((a, b) => deviceOrder(a) - deviceOrder(b) || a.localeCompare(b));
  assert.deepEqual(keys, [
    "SYCL0", "SYCL1",
    "RPC:ph16-71-cuda0:CUDA0", "RPC:ph16-71-cpu0:CPU",
    "RAM",
  ]);
});

test("a recorded moment reads as a date, and its absence stays empty", () => {
  const s = fmtWhen("2026-08-01T17:57:49");
  assert.ok(s.includes("Aug"), s);
  assert.ok(s.includes("17:57"), s);
  assert.equal(fmtWhen(null), "");
  assert.equal(fmtWhen("not a date"), "not a date");
});
