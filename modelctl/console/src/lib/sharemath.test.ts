import { test } from "node:test";
import assert from "node:assert/strict";

import {
  orderShares, rankOf, shownTotal, totalBytes,
} from "./sharemath.ts";
import type { Share } from "./sharemath.ts";

const GIB = 2 ** 30;

function share(name: string, bytes: number, backing: string): Share {
  return { name, bytes, backing };
}

/* ---- the ramp's order ---- */

test("shares read fastest first, so the bar's shape is the placement", () => {
  const ordered = orderShares([
    share("SSD", 4 * GIB, "SSD via mmap"),
    share("RAM", 3 * GIB, "RAM"),
    share("laptop", 2 * GIB, "over RPC"),
    share("SYCL0", 1 * GIB, "VRAM"),
  ]);
  assert.deepEqual(ordered.map((s) => s.backing),
                   ["VRAM", "over RPC", "RAM", "SSD via mmap"]);
});

test("a tie in backing breaks on size, biggest first", () => {
  const ordered = orderShares([
    share("SYCL1", 8 * GIB, "VRAM"),
    share("SYCL0", 25 * GIB, "VRAM"),
  ]);
  assert.deepEqual(ordered.map((s) => s.name), ["SYCL0", "SYCL1"]);
});

test("a share of nothing is not drawn", () => {
  assert.deepEqual(orderShares([share("SYCL0", 0, "VRAM")]), []);
});

test("an unknown backing sorts as RAM rather than falling off the ramp", () => {
  assert.equal(rankOf("something new"), rankOf("RAM"));
});

test("ordering never mutates what it was handed", () => {
  const input = [share("RAM", 3 * GIB, "RAM"),
                 share("SYCL0", 1 * GIB, "VRAM")];
  orderShares(input);
  assert.deepEqual(input.map((s) => s.name), ["RAM", "SYCL0"]);
});

/* ---- the total, and why it is the sum of what is SHOWN ---- */

test("the header equals the legend, on the layout that broke it", () => {
  /* Live placement answer for laguna-s2.1, 2026-08-05. Every part
     rounds to a tenth on its own and the exact total rounds on its own,
     so the legend read 27.8 + 9.9 + 9.6 + 17.6 = 64.9 under a header of
     65.0. A screen whose own numbers do not add up is the complaint
     this whole thread started with. */
  const shares = [
    share("SYCL0", 29863717273, "VRAM"),
    share("SYCL1", 10631090072, "VRAM"),
    share("ph16-71-cuda0 · CUDA0", 10305404928, "over RPC"),
    share("ph16-71-cpu0 · CPU", 18949865472, "over RPC"),
  ];
  assert.equal(shownTotal(shares).toFixed(1), "64.9");
  /* And deliberately NOT the exactly-rounded total. */
  assert.equal((totalBytes(shares) / GIB).toFixed(1), "65.0");
});

test("the total tracks the parts even when every part rounds up", () => {
  const shares = [share("a", 1.06 * GIB, "VRAM"),
                  share("b", 1.06 * GIB, "VRAM"),
                  share("c", 1.06 * GIB, "VRAM")];
  assert.equal(shownTotal(shares).toFixed(1), "3.3");
});

test("one share alone is its own total, exactly", () => {
  assert.equal(shownTotal([share("SYCL0", 26.34 * GIB, "VRAM")]).toFixed(1),
               "26.3");
});

test("nothing placed totals nothing", () => {
  assert.equal(shownTotal([]).toFixed(1), "0.0");
});

test("totalBytes still answers in exact bytes, for callers that need it", () => {
  assert.equal(totalBytes([share("a", 5, "VRAM"), share("b", 7, "RAM")]), 12);
});

test("a negative byte count never subtracts from a total", () => {
  assert.equal(totalBytes([share("a", 5, "VRAM"), share("b", -7, "RAM")]), 5);
  assert.equal(shownTotal([share("a", -7, "RAM")]).toFixed(1), "0.0");
});
