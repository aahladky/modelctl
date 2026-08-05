import { test } from "node:test";
import assert from "node:assert/strict";

import {
  JOB_TROUBLE_WINDOW_S, failureSummary, troubles,
} from "./troubles.ts";
import type { FleetView, JobRow, ModelRow } from "./types.ts";

const NOW = 1_800_000_000;

function job(over: Partial<JobRow>): JobRow {
  return {
    id: "j1", type: "bench", title: "", lane: "default", status: "failed",
    progress: 0, detail: "", error: "", result_tail: "", cancellable: false,
    router_reloaded: null, created: NOW - 60, started: NOW - 50,
    finished: NOW - 40, ...over,
  } as JobRow;
}

function model(over: Partial<ModelRow>): ModelRow {
  return {
    name: "m", registered: true, enabled: true, running: false, ...over,
  } as ModelRow;
}

/* The live defect, 2026-08-05: the band rendered the job error whole,
   and the error is a multi-line Python traceback. */
const TRACEBACKED =
  "this plan needs 10.2 GiB on RPC:ph16-71-cuda0:CUDA0 but only 0.0 GiB "
  + "is free Traceback (most recent call last): File \"/home/aaron/"
  + "workspace/moe-serving/modelctl/modelctl_web/jobs.py\", line 405, in "
  + "_run_one result = fn(ctx) RuntimeError: reservation conflict";

test("a traceback never reaches the band — the sentence before it does", () => {
  const detail = failureSummary(TRACEBACKED);
  assert.ok(!detail.includes("Traceback"), "traceback stripped");
  assert.ok(!detail.includes("jobs.py"), "file paths stripped");
  assert.ok(detail.startsWith("this plan needs 10.2 GiB"), "the words survive");
});

test("an empty-message exception never leaks the traceback header", () => {
  /* Backend format is `f"{e}\n{traceback...}"`; a bare AssertionError
     has no message, so the error starts at the traceback. */
  const onlyTrace = "\nTraceback (most recent call last):\n  File \"x.py\""
    + ", line 1, in f\nAssertionError";
  const detail = failureSummary(onlyTrace);
  assert.ok(!detail.includes("Traceback"), "header must not surface");
  assert.equal(detail, "failed — the job says why");
});

test("a long sentence is capped at 200 characters", () => {
  const detail = failureSummary("x".repeat(400));
  assert.ok(detail.length <= 200);
});

test("an empty error still says something", () => {
  assert.equal(failureSummary("   "), "failed — the job says why");
});

test("a fresh failure shows; the same failure a day later does not", () => {
  const fresh = job({ title: "bench ornith", error: TRACEBACKED });
  const stale = job({ title: "bench ornith", error: TRACEBACKED,
                      finished: NOW - JOB_TROUBLE_WINDOW_S - 1 });
  assert.equal(troubles([], [fresh], null, NOW).length, 1);
  assert.equal(troubles([], [stale], null, NOW).length, 0);
});

test("a job that never started expires on its created time", () => {
  const stale = job({ started: null, finished: null,
                      created: NOW - JOB_TROUBLE_WINDOW_S - 1 });
  assert.equal(troubles([], [stale], null, NOW).length, 0);
});

test("router-did-not-reload expires like any other job trouble", () => {
  const fresh = job({ status: "done", router_reloaded: false });
  const stale = job({ status: "done", router_reloaded: false,
                      finished: NOW - JOB_TROUBLE_WINDOW_S - 1 });
  assert.equal(troubles([], [fresh], null, NOW).length, 1);
  assert.equal(troubles([], [stale], null, NOW).length, 0);
});

test("current-state facts never expire: they clear when fixed", () => {
  const disabled = model({ registered: true, enabled: false });
  const rows = troubles([disabled], [], null, NOW);
  assert.equal(rows.length, 1);
  assert.ok(rows[0].detail.includes("disabled"));
});

test("an absent enabled remote node is trouble; a present one is not", () => {
  const fleet = {
    nodes: [
      { name: "ph16-71", enabled: true, location: "remote",
        presence: { state: "ABSENT", detail: "no probe in 12 min" } },
      { name: "rig", enabled: true, location: "local",
        presence: { state: "ABSENT", detail: "" } },
    ],
    errors: {},
  } as unknown as FleetView;
  const rows = troubles([], [], fleet, NOW);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].what, "ph16-71");
});

test("nothing wrong means an empty list — the calm line's contract", () => {
  assert.deepEqual(troubles([model({})], [job({ status: "done" })], null, NOW),
                   []);
});
