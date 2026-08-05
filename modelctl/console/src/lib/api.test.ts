/* Pure-logic tests for the API bindings.

   The console shipped with 2,102 backend tests and no JS test runner at
   all, so every visible defect of 2026-08-04 landed in the untested half.
   This is the cheap half of closing that: no DOM, no framework, no new
   dependency -- Node 24 strips the types itself and `node --test` runs
   them.

   What is worth testing here is the arithmetic and the encoding, because
   that is where a defect is silent. placementQuery in particular has to
   round-trip against the server's own parser (_selection_from_query in
   modelctl_web/app.py): a selection that encodes wrong asks the planner
   for a layout the operator did not choose, which is the exact failure
   the placement endpoint exists to prevent. */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  admissionQuery, fmtAgo, fmtGiB, fmtUp, placementQuery, stateLabel,
} from "./api.ts";

const GIB = 2 ** 30;

test("fmtGiB renders bytes as GiB", () => {
  assert.equal(fmtGiB(0), "0.0");
  assert.equal(fmtGiB(GIB), "1.0");
  assert.equal(fmtGiB(27.4 * GIB), "27.4");
  assert.equal(fmtGiB(GIB, 2), "1.00");
});

test("fmtGiB says nothing rather than zero when it has no number", () => {
  /* A missing byte count and an empty device are different facts, and a
     row that renders 0.0 for the first one claims the second. */
  assert.equal(fmtGiB(null), "—");
  assert.equal(fmtGiB(undefined), "—");
});

test("placementQuery encodes a device switched off", () => {
  assert.equal(placementQuery({ SYCL0: { on: false } }), "?on.SYCL0=0");
  assert.equal(placementQuery({ SYCL0: { on: true } }), "?on.SYCL0=1");
});

test("placementQuery encodes a ceiling as whole bytes", () => {
  /* The server refuses a ceiling it cannot read as an integer, so a
     fractional byte count would be a 422 the operator never caused. */
  assert.equal(placementQuery({ SYCL0: { ceiling_bytes: 12.7 } }),
               "?ceiling.SYCL0=13");
});

test("placementQuery carries a remote key's colons verbatim", () => {
  /* The admission key IS the identifier the gate charges. Mangling the
     colons would name a device the machine does not have, which the
     server now answers with 422 rather than a silent no-op. */
  const q = placementQuery({ "RPC:ph16-71-cuda0:CUDA0": { on: true } });
  assert.equal(
    new URLSearchParams(q.slice(1)).get("on.RPC:ph16-71-cuda0:CUDA0"), "1",
    "the server must read back the key it was sent");
});

test("placementQuery sends both terms for one device as one entry", () => {
  const q = placementQuery({ SYCL0: { on: true, ceiling_bytes: 4 * GIB } });
  const params = new URLSearchParams(q.slice(1));
  assert.equal(params.get("on.SYCL0"), "1");
  assert.equal(params.get("ceiling.SYCL0"), String(4 * GIB));
});

test("an empty selection is the automatic placement, with no query", () => {
  /* Not "?" -- a bare question mark is a different URL, and the whole
     point of the empty selection is that it asks for nothing. */
  assert.equal(placementQuery({}), "");
});

test("placementQuery omits a term the operator did not set", () => {
  /* `on` is boolean | undefined in the contract -- tsc caught the first
     draft of this test asserting null, which the runtime tolerates but
     the type does not promise. */
  assert.equal(placementQuery({ SYCL0: {} }), "");
  assert.equal(placementQuery({ SYCL0: { ceiling_bytes: null } }), "");
});

test("admissionQuery omits what was not asked", () => {
  assert.equal(admissionQuery(), "");
  assert.equal(admissionQuery(8192), "?ctx=8192");
});

test("admissionQuery rounds per-device budgets to whole bytes", () => {
  const q = admissionQuery(null, { SYCL0: 1.5, SYCL1: 2 * GIB });
  const params = new URLSearchParams(q.slice(1));
  assert.equal(params.get("budget_bytes.SYCL0"), "2");
  assert.equal(params.get("budget_bytes.SYCL1"), String(2 * GIB));
});

test("stateLabel says what a worker state means in words", () => {
  assert.equal(stateLabel("ready"), "running");
  assert.equal(stateLabel("unregistered"), "not serving");
});

test("stateLabel passes an unknown state through untranslated", () => {
  /* Inventing a friendly word for a state nobody has seen would be the
     screen claiming to know something it does not. */
  assert.equal(stateLabel("reticulating"), "reticulating");
});

test("fmtAgo is empty for a moment that never happened", () => {
  assert.equal(fmtAgo(0), "");
  assert.equal(fmtAgo(null), "");
});

test("fmtAgo and fmtUp scale by unit", () => {
  const now = Math.floor(Date.now() / 1000);
  assert.equal(fmtAgo(now - 30), "30s ago");
  assert.equal(fmtAgo(now - 120), "2m ago");
  assert.equal(fmtAgo(now - 7200), "2h ago");
  assert.equal(fmtUp(now - 600), "10m");
});

test("fmtAgo never reports the future as negative", () => {
  /* Clocks disagree; a row reading "-4s ago" is a bug report the
     operator cannot act on. */
  assert.equal(fmtAgo(Math.floor(Date.now() / 1000) + 60), "0s ago");
});
