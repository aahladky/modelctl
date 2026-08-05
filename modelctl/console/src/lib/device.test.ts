import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CEILING_STEP_BYTES, barMarks, ceilingFromFraction, clampCeiling,
  deviceNote, isOver, pct, trackSpan,
} from "./device.ts";

const GIB = 2 ** 30;

/* The live rig's big card, as the placement endpoint reports it. */
const SYCL0 = { committed: 27.40 * GIB, usable: 28.70 * GIB,
                capacity: 31.89 * GIB };

test("the track spans the hardware, not the budget", () => {
  /* So the operator can see what policy is withholding: 31.89 GiB of
     card against the 28.70 the planner may spend. */
  assert.equal(trackSpan(SYCL0), 31.89 * GIB);
  const marks = barMarks(SYCL0, null);
  assert.ok(marks.usable < 100, "the budget is not the whole track");
  assert.ok(marks.committed < marks.usable);
});

test("an over-committed row is not clipped to look like it fits", () => {
  /* The overflow is the thing the screen exists to show. */
  const over = { committed: 40 * GIB, usable: 28.70 * GIB,
                 capacity: 31.89 * GIB };
  assert.equal(trackSpan(over), 40 * GIB);
  assert.equal(barMarks(over, null).committed, 100);
  assert.ok(isOver(over));
});

test("a device holding what it may is not over", () => {
  assert.equal(isOver(SYCL0), false);
});

test("a device with no budget is never reported as over", () => {
  /* usable 0 means nothing is known about the bound, not that everything
     exceeds it -- the host row shipped exactly this until 2026-08-04. */
  assert.equal(isOver({ committed: 5 * GIB, usable: 0, capacity: 0 }), false);
});

test("a device with nothing to draw yields zeros, not NaN", () => {
  const marks = barMarks({ committed: 0, usable: 0, capacity: 0 }, null);
  assert.deepEqual(marks, { committed: 0, usable: 0, ceiling: null });
  assert.equal(pct(1, 0), 0);
});

test("a ceiling cannot be dragged above what the planner may spend", () => {
  /* The machine sets the bound, so an impossible value cannot be
     expressed -- rather than a validator refusing it afterwards. */
  assert.equal(clampCeiling(999 * GIB, SYCL0.usable), 28.5 * GIB);
  assert.equal(ceilingFromFraction(1, SYCL0), 28.5 * GIB);
});

test("a ceiling cannot be dragged below zero", () => {
  assert.equal(clampCeiling(-5, SYCL0.usable), 0);
  assert.equal(ceilingFromFraction(-2, SYCL0), 0);
});

test("a ceiling snaps down to the quantum, never up", () => {
  /* Up would hand back room the operator just took away. */
  assert.equal(clampCeiling(12 * GIB + 1, SYCL0.usable), 12 * GIB);
  assert.equal(clampCeiling(12 * GIB - 1, SYCL0.usable),
               12 * GIB - CEILING_STEP_BYTES);
});

test("dragging to the middle of the track lands on a real byte count", () => {
  const ceiling = ceilingFromFraction(0.5, SYCL0);
  assert.equal(ceiling % CEILING_STEP_BYTES, 0, "off the quantum");
  assert.ok(ceiling > 15 * GIB && ceiling < 16.5 * GIB, `${ceiling / GIB}`);
});

test("a non-finite drag falls back to the full budget", () => {
  assert.equal(clampCeiling(NaN, SYCL0.usable), SYCL0.usable);
});

test("the ceiling mark sits on the same track as the fill", () => {
  const marks = barMarks(SYCL0, 14 * GIB);
  assert.ok(marks.ceiling !== null);
  assert.ok(marks.ceiling! < marks.committed,
            "a ceiling below what is committed must read as below it");
});

test("a present device says what it is doing", () => {
  assert.equal(deviceNote("VRAM", "PRESENT", ""), "VRAM");
  assert.equal(deviceNote("SSD via mmap", "PRESENT", ""), "SSD via mmap");
});

test("an unused device says so rather than showing nothing", () => {
  assert.equal(deviceNote("", "PRESENT", ""), "nothing here");
});

test("an unreachable device says so, and why", () => {
  /* A node nobody can reach and a node holding nothing draw the same
     empty bar; only the words tell them apart. */
  assert.equal(deviceNote("", "STALE", "connection refused"),
               "not reachable — connection refused");
});

test("a pin mismatch reads as up-but-wrong, never as merely absent", () => {
  /* It answers a handshake immediately, and placing a graph across two
     ggml builds gives wrong numbers rather than an error. */
  assert.equal(deviceNote("", "PIN_MISMATCH", ""),
               "up, but built from a different commit");
});
