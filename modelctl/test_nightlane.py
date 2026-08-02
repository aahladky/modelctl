"""The night lane: what is pre-registered, and what is allowed to run it.

Two halves, and the rails differ.

The registry half is a discipline, not an algorithm: the value is that a
criterion was written before any numbers existed and that nothing runs by
accident. Those tests are about the shipped file's content.

The lane half is a gate. Its tests are about refusal -- what it declines
to dispatch, and whether it fails closed when it cannot see the machine.
A gate that only gets tested on the path where it opens is not a gate.
"""
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modelctl_nightlane as nl

_GPU_LOCK_TMP = None


def setUpModule():
    """Every run_job() here takes the real GPU lock unless one is
    injected. Redirect it: a single-file run of this test must not flock
    the path the live night lane and `modelctl lane gpu-lock` use."""
    global _GPU_LOCK_TMP
    _GPU_LOCK_TMP = tempfile.mkdtemp(prefix="nightlane-gpu-lock-")
    os.environ["MODELCTL_GPU_LOCK"] = str(Path(_GPU_LOCK_TMP) / "gpu.lock")


def tearDownModule():
    os.environ.pop("MODELCTL_GPU_LOCK", None)
    shutil.rmtree(_GPU_LOCK_TMP, ignore_errors=True)

RPC_PAIRS = ["ornith-rpc-criterion-2026-08-02",
             "qwen122b-remote-experts-hypothesis-2026-08-02"]
MAIDEN = ["determinism-cost-c1-static-2026-08-02",
          "determinism-cost-c2-cache-2026-08-02",
          "re-anchor-c1-c2-c3-2026-08-02",
          "sdpa-reproducibility-2026-08-02"]


def _job(job_id, enabled=True, **kw):
    """A synthetic pre-registration.

    The dispatcher's behaviour is a separate fact from what the shipped
    registry happens to contain, and tests that conflated the two broke
    the moment the maiden jobs were retired after running.
    """
    fields = dict(id=job_id, title=job_id, question="q", criterion="c",
                  measures=("m",), enabled=enabled,
                  arms=(nl.Arm(name="a", profile="p"),
                        nl.Arm(name="b", profile="p")))
    fields.update(kw)
    return nl.NightLaneJob(**fields)


class TestShippedRegistry(unittest.TestCase):
    def setUp(self):
        self.jobs = nl.load_jobs()
        self.by_id = {j.id: j for j in self.jobs}

    def test_the_registry_holds_the_rpc_pairs_and_the_maiden_jobs(self):
        self.assertEqual(sorted(self.by_id), sorted(RPC_PAIRS + MAIDEN))

    def test_the_rpc_pairs_stay_disabled(self):
        # Pre-registered by the RPC enablement session and explicitly not
        # released. Enabling one is a diff someone has to make on purpose.
        for job_id in RPC_PAIRS:
            self.assertFalse(self.by_id[job_id].enabled, job_id)

    def test_the_maiden_jobs_are_retired_now_that_they_have_run(self):
        # A pre-registration that has been executed must not stay enabled,
        # or the next open window runs it again and the registry stops
        # describing what is outstanding.
        for job_id in MAIDEN:
            job = self.by_id[job_id]
            self.assertFalse(job.enabled, job_id)
            self.assertIn("RAN 2026-08-02", job.note, job_id)
            self.assertIn("2026-08-02-maiden-runs.md", job.note, job_id)

    def test_nothing_is_queued(self):
        self.assertEqual(nl.enabled_jobs(), [])

    def test_every_comparison_job_has_at_least_two_arms(self):
        # A comparison needs something to compare against. The
        # reproducibility job is the deliberate exception: it asks whether
        # ONE configuration reproduces itself, which is the question every
        # comparison assumes and none of them asks.
        for j in self.jobs:
            if j.mode == "reproducibility":
                self.assertEqual(len(j.arms), 1, j.id)
            else:
                self.assertGreaterEqual(len(j.arms), 2, j.id)

    def test_every_job_declares_a_criterion_measures_and_a_date(self):
        for j in self.jobs:
            self.assertTrue(j.criterion.strip(), j.id)
            self.assertTrue(j.measures, j.id)
            self.assertTrue(j.registered, j.id)

    def test_every_job_declares_its_mode_and_its_sample_size(self):
        # The sample size belongs in the pre-registration: "five pairs"
        # decided in advance is a criterion, and "kept going until it
        # looked clean" is not. A mode with no declared size can be
        # extended after the numbers come out.
        for j in self.jobs:
            self.assertIn(j.mode, nl.MODES, j.id)
            if j.mode == "paired":
                self.assertGreater(j.pairs, 0, j.id)
            elif j.mode in ("battery", "reproducibility"):
                self.assertGreater(j.runs, 0, j.id)

    def test_every_paired_job_records_load_per_run(self):
        # The whole reason these jobs exist. A paired job that recorded
        # load once per battery would reproduce the defect it replaces.
        for j in self.jobs:
            if j.mode != "paired":
                continue
            measures = " ".join(j.measures).lower()
            self.assertIn("per-pair delta", measures, j.id)
            self.assertRegex(measures, r"load sampled during each individual run",
                             j.id)

    def test_the_registry_records_no_expected_outcome(self):
        # "Never judge the result" -- a pre-registration that predicts a
        # winner invites reading the numbers to match. Criteria say how
        # to decide, never what the answer will be.
        #
        # Whole words, not substrings: the substring form of this test
        # failed on the word "window", which is how the lane's own gate
        # is described.
        banned = ("faster", "slower", "better", "worse", "improvement",
                  "win", "wins", "beats", "outperform", "outperforms",
                  "speedup", "regression", "regressions")
        pattern = re.compile(r"\b(" + "|".join(banned) + r")\b")
        for j in self.jobs:
            haystack = " ".join(
                [j.title, j.question, j.criterion, j.note]
                + [a.note for a in j.arms]).lower()
            # The 122B RPC pair's criterion legitimately says a token
            # mismatch "must not be reported as a regression" -- a rule
            # about how to read a result, not a prediction of one.
            haystack = haystack.replace(
                "must not be reported as a regression", "")
            found = pattern.findall(haystack)
            self.assertEqual(found, [], f"{j.id}: {found}")


class TestTheRpcPairsAreUntouched(unittest.TestCase):
    """The maiden registration was required to append, not to rewrite."""

    def test_neither_rpc_pair_carries_a_post_hoc_schema_field(self):
        raw = json.loads(nl.REGISTRY_PATH.read_text())
        for entry in raw["jobs"]:
            if entry["id"] not in RPC_PAIRS:
                continue
            for added in ("mode", "pairs", "runs", "metric"):
                self.assertNotIn(added, entry, f"{entry['id']}.{added}")

    def test_a_default_valued_field_does_not_serialise(self):
        # What keeps the above true as the schema grows: to_dict() emits a
        # new field only when it carries something, so re-saving the
        # registry cannot make an untouched job look edited.
        job = nl.job_by_id(RPC_PAIRS[0])
        self.assertEqual(job.mode, "block")
        self.assertNotIn("mode", job.to_dict())

    def test_each_rpc_pair_still_has_exactly_one_arm_needing_the_node(self):
        # A pair where both arms need the node has no baseline, and one
        # where neither does is not testing the fleet.
        for job_id in RPC_PAIRS:
            job = nl.job_by_id(job_id)
            needing = [a for a in job.arms if a.requires_nodes]
            self.assertEqual(len(needing), 1, job_id)
            self.assertEqual(list(needing[0].requires_nodes), ["ph16-71-cuda0"])


class TestBlocking(unittest.TestCase):
    def test_a_disabled_job_is_blocked_even_with_the_node_present(self):
        job = nl.job_by_id("ornith-rpc-criterion-2026-08-02")
        reasons = nl.blocking_reasons(job, usable_node_names=["ph16-71-cuda0"])
        self.assertIn("pre-registered but not enabled", reasons)

    def test_a_missing_node_blocks_independently_of_enablement(self):
        job = nl.job_by_id("ornith-rpc-criterion-2026-08-02")
        reasons = nl.blocking_reasons(job, usable_node_names=[])
        self.assertTrue(any("ph16-71-cuda0" in r for r in reasons))

    def test_required_nodes_are_collected_across_arms(self):
        job = nl.job_by_id("qwen122b-remote-experts-hypothesis-2026-08-02")
        self.assertEqual(job.required_nodes, {"ph16-71-cuda0"})

    def test_the_maiden_jobs_need_no_fleet_node(self):
        # They measure the rig. A maiden job that quietly required ph16-71
        # would have been blocked on a node instead of running.
        for job_id in MAIDEN:
            self.assertEqual(nl.job_by_id(job_id).required_nodes, set())

    def test_nothing_in_the_shipped_registry_is_due(self):
        runnable, skipped = nl.due_jobs()
        self.assertEqual(runnable, [])
        self.assertEqual(sorted(j.id for j, _ in skipped),
                         sorted(RPC_PAIRS + MAIDEN))
        for _job, reasons in skipped:
            self.assertTrue(reasons)

    def test_due_jobs_reports_reasons_rather_than_filtering(self):
        # A lane that quietly ran three of five pre-registrations and
        # reported three results would look like it ran everything.
        enabled = _job("live")
        blocked = _job("held", enabled=False)
        runnable, skipped = nl.due_jobs([enabled, blocked])
        self.assertEqual([j.id for j in runnable], ["live"])
        self.assertEqual(skipped[0][1], ["pre-registered but not enabled"])


class _Swap:
    def __init__(self, running=(), raises=None):
        self._running, self._raises = set(running), raises

    def running_model_ids(self):
        if self._raises:
            raise self._raises
        return self._running


class _Load:
    def __init__(self, loadavg_1m):
        self.loadavg_1m = loadavg_1m


class TestWindow(unittest.TestCase):
    def test_idle_swap_and_low_load_opens_the_window(self):
        state = nl.window_state(_Swap(), _Load(0.4))
        self.assertTrue(state.open, state.reasons)
        self.assertEqual(state.reasons, ())
        self.assertEqual(state.observed["loadavg_1m"], 0.4)

    def test_a_resident_model_shuts_the_window(self):
        # A benchmark taken beside a live model measures the two of them.
        state = nl.window_state(_Swap(["laguna-s2.1"]), _Load(0.1))
        self.assertFalse(state.open)
        self.assertTrue(any("laguna-s2.1" in r for r in state.reasons))
        self.assertEqual(state.observed["llama_swap_running"], ["laguna-s2.1"])

    def test_load_above_the_ceiling_shuts_the_window(self):
        state = nl.window_state(_Swap(), _Load(8.99))
        self.assertFalse(state.open)
        self.assertTrue(any("8.99" in r for r in state.reasons))

    def test_the_ceiling_is_inclusive_at_its_own_value(self):
        self.assertTrue(nl.window_state(
            _Swap(), _Load(nl.DEFAULT_LOADAVG_CEILING)).open)

    def test_an_unreachable_swap_fails_closed(self):
        # Not idle -- unknown. It could be mid-restart with a model about
        # to land, and the run would be contaminated from its first token.
        state = nl.window_state(_Swap(raises=OSError("refused")), _Load(0.1))
        self.assertFalse(state.open)
        self.assertTrue(any("could not be reached" in r for r in state.reasons))
        self.assertIn("llama_swap_error", state.observed)

    def test_an_unreadable_loadavg_fails_closed(self):
        state = nl.window_state(_Swap(), _Load(None))
        self.assertFalse(state.open)
        self.assertTrue(any("could not be read" in r for r in state.reasons))

    def test_both_halves_are_reported_not_just_the_first(self):
        state = nl.window_state(_Swap(["ornith-397b"]), _Load(12.0))
        self.assertEqual(len(state.reasons), 2, state.reasons)


class _Manager:
    """Records submissions instead of running them."""

    def __init__(self):
        self.submitted = []

    def submit(self, job_type, title, fn, payload=None, lane="mutation"):
        self.submitted.append({"type": job_type, "title": title, "fn": fn,
                               "payload": payload, "lane": lane})
        return f"store{len(self.submitted)}"


class TestDispatch(unittest.TestCase):
    def _jobs(self):
        return [_job("a"), _job("b"), _job("c"), _job("d"),
                _job("held", enabled=False)]

    def test_a_shut_window_dispatches_nothing(self):
        mgr = _Manager()
        result = nl.dispatch_due(mgr, runner=lambda job, ctx: {},
                                 jobs=self._jobs(),
                                 window=nl.WindowState(False, ("busy",)))
        self.assertEqual(mgr.submitted, [])
        self.assertEqual(result.submitted, [])
        # Every runnable job appears as skipped-with-a-reason, so a night
        # that dispatched nothing can say why rather than looking idle.
        self.assertEqual({j for j, _ in result.skipped},
                         {"a", "b", "c", "d", "held"})
        self.assertTrue(any("window is shut" in r
                            for _j, rs in result.skipped for r in rs))

    def test_an_open_window_dispatches_only_the_enabled_jobs(self):
        mgr = _Manager()
        result = nl.dispatch_due(mgr, runner=lambda job, ctx: {},
                                 jobs=self._jobs(),
                                 window=nl.WindowState(True))
        self.assertEqual(sorted(j for j, _ in result.submitted),
                         ["a", "b", "c", "d"])
        self.assertEqual([j for j, _ in result.skipped], ["held"])

    def test_the_shipped_registry_dispatches_nothing_today(self):
        mgr = _Manager()
        result = nl.dispatch_due(mgr, runner=lambda job, ctx: {},
                                 jobs=nl.load_jobs(),
                                 window=nl.WindowState(True))
        self.assertEqual(result.submitted, [])
        self.assertEqual(sorted(j for j, _ in result.skipped),
                         sorted(RPC_PAIRS + MAIDEN))

    def test_dispatch_uses_the_benchmark_lane(self):
        # One worker, so two night jobs can never share the GPUs -- and
        # the console's jobs page renders that lane already.
        mgr = _Manager()
        nl.dispatch_due(mgr, runner=lambda job, ctx: {}, jobs=self._jobs(),
                        window=nl.WindowState(True))
        self.assertEqual({s["lane"] for s in mgr.submitted}, {nl.LANE})

    def test_the_payload_carries_the_window_it_was_dispatched_under(self):
        mgr = _Manager()
        nl.dispatch_due(mgr, runner=lambda job, ctx: {}, jobs=self._jobs(),
                        window=nl.WindowState(True, (), {"loadavg_1m": 0.3}))
        payload = mgr.submitted[0]["payload"]
        self.assertEqual(payload["window"]["observed"]["loadavg_1m"], 0.3)
        self.assertEqual(payload["night_lane_job"], "a")

    def test_each_submission_closes_over_its_own_job(self):
        # The late-binding trap: a bare `lambda ctx: runner(job, ctx)` in
        # the loop makes every submission run the LAST job, and all four
        # results get filed under four different names.
        mgr = _Manager()
        seen = []
        nl.dispatch_due(mgr, runner=lambda job, ctx: seen.append(job.id),
                        jobs=self._jobs(), window=nl.WindowState(True))
        for submission in mgr.submitted:
            submission["fn"](object())
        self.assertEqual(sorted(seen), ["a", "b", "c", "d"])

    def test_a_limit_reports_what_it_did_not_dispatch(self):
        mgr = _Manager()
        result = nl.dispatch_due(mgr, runner=lambda job, ctx: {},
                                 jobs=self._jobs(),
                                 window=nl.WindowState(True), limit=1)
        self.assertEqual(len(result.submitted), 1)
        deferred = [r for j, r in result.skipped
                    if any("limit of 1" in x for x in r)]
        self.assertEqual(len(deferred), 3)


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.job = nl.job_by_id("determinism-cost-c1-static-2026-08-02")

    def test_evidence_is_filed_with_a_one_line_summary(self):
        with tempfile.TemporaryDirectory() as d:
            record = {"status": "recorded",
                      "load_summary": {"loadavg_1m": {"min": 0.2, "max": 0.9,
                                                      "mean": 0.5}}}
            path = nl.file_evidence(self.job, record, "2026-08-02",
                                    directory=d)
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text())
            self.assertEqual(payload["night_lane_job"]["id"], self.job.id)
            self.assertEqual(payload["record"], record)

            log = (Path(d) / nl.SUMMARY_PATH.name).read_text()
            self.assertIn(self.job.id, log)
            self.assertIn("loadavg(1m) 0.20-0.90 mean 0.50", log)
            self.assertEqual(len([ln for ln in log.splitlines()
                                  if ln.startswith("- ")]), 1)

    def test_a_run_without_a_load_trace_says_so(self):
        # Never "loadavg 0.00": an unrecorded load is unknown, and the
        # summary is the only thing anyone reads in the morning.
        line = nl.summary_line(self.job, {"status": "recorded"}, "2026-08-02")
        self.assertIn("load not recorded", line)
        self.assertNotIn("0.00", line)

    def test_the_summary_line_carries_no_reading_of_the_result(self):
        record = {"status": "recorded", "median_delta": -0.9,
                  "sign_test": {"p_value": 0.0625}}
        line = nl.summary_line(self.job, record, "2026-08-02").lower()
        # The job's own id is quoted verbatim and is a name, not a
        # reading; what this checks is that the module adds no words of
        # its own about how the numbers came out.
        generated = line.replace(self.job.id, "")
        for word in ("faster", "slower", "better", "worse", "cost",
                     "significant", "confirms", "shows", "0.0625", "-0.9"):
            self.assertNotIn(word, generated)

    def test_summaries_append_rather_than_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            for day in ("2026-08-02", "2026-08-03"):
                nl.file_evidence(self.job, {"status": "recorded"}, day,
                                 directory=d)
            log = (Path(d) / nl.SUMMARY_PATH.name).read_text()
            self.assertEqual(len([ln for ln in log.splitlines()
                                  if ln.startswith("- ")]), 2)


class TestArmProfiles(unittest.TestCase):
    """An arm's overrides, applied to a profile that is never saved."""

    BASE = {"name": "qwen122b-a10b", "model_path": "/m.gguf",
            "config": {"ctx": 4096, "fit": "off"},
            "env": ["LD_LIBRARY_PATH=/opt/oneapi/lib"]}

    def test_env_overrides_merge_with_the_profiles_own_env(self):
        arm = nl.Arm(name="a", profile="p",
                     overrides={"env": {"GGML_SYCL_DETERMINISTIC": "0"}})
        out = nl.arm_profile(arm, self.BASE)
        self.assertIn("LD_LIBRARY_PATH=/opt/oneapi/lib", out["env"])
        self.assertIn("GGML_SYCL_DETERMINISTIC=0", out["env"])

    def test_an_env_override_replaces_rather_than_duplicates(self):
        base = dict(self.BASE, env=["GGML_SYCL_DETERMINISTIC=1"])
        arm = nl.Arm(name="a", profile="p",
                     overrides={"env": {"GGML_SYCL_DETERMINISTIC": "0"}})
        self.assertEqual(nl.arm_profile(arm, base)["env"],
                         ["GGML_SYCL_DETERMINISTIC=0"])

    def test_config_overrides_land_in_config(self):
        arm = nl.Arm(name="a", profile="p", overrides={"extra": "-ngl 999"})
        out = nl.arm_profile(arm, self.BASE)
        self.assertEqual(out["config"]["extra"], "-ngl 999")
        self.assertEqual(out["config"]["ctx"], 4096)

    def test_underscored_keys_are_runner_directives_not_config(self):
        arm = nl.Arm(name="a", profile="p",
                     overrides={"_plan_source": "fleet-rpc"})
        self.assertNotIn("_plan_source",
                         nl.arm_profile(arm, self.BASE)["config"])

    def test_the_base_profile_is_not_mutated(self):
        # Two arms of a comparison must differ by exactly what the
        # registry says; a shared dict mutated by the first arm makes the
        # second arm measure something nobody declared.
        arm = nl.Arm(name="a", profile="p",
                     overrides={"extra": "-x", "env": {"K": "V"}})
        nl.arm_profile(arm, self.BASE)
        self.assertNotIn("extra", self.BASE["config"])
        self.assertEqual(self.BASE["env"], ["LD_LIBRARY_PATH=/opt/oneapi/lib"])

    def test_every_shipped_arm_builds(self):
        for job in nl.load_jobs():
            for arm in job.arms:
                nl.arm_profile(arm, self.BASE)


class TestTheOffloadFloor(unittest.TestCase):
    """GGML_OP_OFFLOAD_MIN_BATCH never goes below 32: known correctness
    bug on this hardware. Enforced at the lane rather than left to
    review, because the lane runs with nobody watching and a floor
    violation at 03:00 still produces numbers that look fine."""

    def _arm(self, value):
        return nl.Arm(name="a", profile="p",
                      overrides={"env": {"GGML_OP_OFFLOAD_MIN_BATCH": value}})

    def test_below_the_floor_is_refused(self):
        self.assertTrue(nl.arm_violations(self._arm("16")))
        with self.assertRaises(ValueError):
            nl.arm_profile(self._arm("1"), {"config": {}})

    def test_at_the_floor_is_allowed(self):
        self.assertEqual(nl.arm_violations(self._arm("32")), [])

    def test_above_the_floor_is_allowed(self):
        self.assertEqual(nl.arm_violations(self._arm("64")), [])

    def test_an_unparseable_value_is_refused_not_ignored(self):
        self.assertTrue(nl.arm_violations(self._arm("thirty-two")))

    def test_the_moe_specific_threshold_is_a_different_knob(self):
        # GGML_OP_OFFLOAD_MOE_MIN_BATCH=1 is what every cache condition
        # uses; the global floor does not apply to it.
        arm = nl.Arm(name="a", profile="p",
                     overrides={"env": {"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "1"}})
        self.assertEqual(nl.arm_violations(arm), [])

    def test_no_shipped_arm_violates_the_floor(self):
        for job in nl.load_jobs():
            for arm in job.arms:
                self.assertEqual(nl.arm_violations(arm), [],
                                 f"{job.id}/{arm.name}")

    def test_a_violating_job_is_refused_before_anything_runs(self):
        job = nl.NightLaneJob(
            id="bad", title="t", question="q", criterion="c",
            measures=("m",), mode="paired", pairs=2,
            arms=(self._arm("1"), nl.Arm(name="b", profile="p")))
        calls = []
        record = nl.run_job(job, lambda *a: calls.append(a), clock=lambda: 0.0)
        self.assertEqual(record["status"], "refused")
        self.assertEqual(calls, [])
        self.assertTrue(any("below the 32 floor" in v
                            for v in record["violations"]))


class TestGpuLockIsHeld(unittest.TestCase):
    """The night lane and a session's lane bench are separate schedulers
    that know nothing about each other. The lock is the only thing that
    stops them measuring each other."""

    def _job(self):
        return nl.job_by_id("re-anchor-c1-c2-c3-2026-08-02")

    def test_every_measurement_happens_inside_the_lock(self):
        events = []

        class FakeLock:
            def __init__(self, timeout=None, note=None):
                self.note = note

            def __enter__(self):
                events.append(("acquired", self.note))
                return self

            def __exit__(self, *exc):
                events.append(("released", None))
                return False

        nl.run_job(self._job(),
                   lambda arm, i, slot: events.append(("measure", arm.name))
                   or {"generation_tps": 5.0},
                   clock=lambda: 0.0, gpu_lock=FakeLock)

        self.assertEqual(events[0][0], "acquired")
        self.assertEqual(events[-1][0], "released")
        self.assertIn("night lane", events[0][1])
        self.assertTrue(any(e[0] == "measure" for e in events))

    def test_it_takes_the_same_lock_modelctl_lane_gpu_lock_takes(self):
        import modelctl_lanes
        with mock.patch.object(modelctl_lanes, "gpu_lock") as fake:
            nl.run_job(self._job(), lambda *a: {"generation_tps": 5.0},
                       clock=lambda: 0.0)
        fake.assert_called_once()
        self.assertEqual(fake.call_args.kwargs["timeout"],
                         nl.GPU_LOCK_WAIT_SECONDS)

    def test_a_held_lock_is_a_refusal_and_nothing_is_measured(self):
        import modelctl_lanes
        calls = []
        with modelctl_lanes.gpu_lock(note="a lane bench"):
            record = nl.run_job(self._job(),
                                lambda *a: calls.append(a) or {},
                                clock=lambda: 0.0, gpu_lock_wait=0.2)
        self.assertEqual(record["status"], "refused")
        self.assertEqual(calls, [])
        self.assertTrue(any("GPU lock is held" in v
                            for v in record["violations"]))

    def test_a_job_that_cannot_run_refuses_without_taking_the_lock(self):
        job = nl.NightLaneJob(id="x", title="t", question="q", criterion="c",
                              measures=("m",), mode="paired", pairs=2,
                              arms=(nl.Arm(name="only", profile="p"),))
        import modelctl_lanes
        with mock.patch.object(modelctl_lanes, "gpu_lock") as fake:
            record = nl.run_job(job, lambda *a: {}, clock=lambda: 0.0)
        self.assertEqual(record["status"], "refused")
        fake.assert_not_called()


class TestRunJob(unittest.TestCase):
    def test_a_paired_job_produces_per_pair_deltas(self):
        job = nl.job_by_id("determinism-cost-c1-static-2026-08-02")
        values = {"deterministic-on": 6.0, "deterministic-off": 7.0}
        record = nl.run_job(
            job, lambda arm, i, slot: {"generation_tps": values[arm.name]},
            clock=lambda: 0.0)
        self.assertEqual(record["status"], "recorded")
        self.assertEqual(record["comparison"]["pairs_complete"], 5)
        self.assertEqual(record["comparison"]["per_pair"][0]["delta"], 1.0)
        self.assertEqual(record["comparison"]["sign_test"]["n"], 5)

    def test_a_paired_job_needs_exactly_two_arms(self):
        job = nl.NightLaneJob(id="x", title="t", question="q", criterion="c",
                              measures=("m",), mode="paired", pairs=2,
                              arms=(nl.Arm(name="only", profile="p"),))
        self.assertEqual(nl.run_job(job, lambda *a: {},
                                    clock=lambda: 0.0)["status"], "refused")

    def test_a_battery_runs_every_arm_the_declared_number_of_times(self):
        job = nl.job_by_id("re-anchor-c1-c2-c3-2026-08-02")
        record = nl.run_job(job, lambda arm, i, slot: {"generation_tps": 5.0},
                            clock=lambda: 0.0)
        self.assertEqual(sorted(record["arms"]),
                         ["C1-static", "C2-cache", "C3-cache-hybrid"])
        for arm in record["arms"].values():
            self.assertEqual(len(arm["runs"]), 5)

    def test_a_failed_run_is_recorded_not_fatal(self):
        job = nl.job_by_id("re-anchor-c1-c2-c3-2026-08-02")

        def measure(arm, i, slot):
            if arm.name == "C2-cache" and i == 2:
                raise RuntimeError("server exited rc=134")
            return {"generation_tps": 5.0}

        record = nl.run_job(job, measure, clock=lambda: 0.0)
        self.assertIn("rc=134", record["arms"]["C2-cache"]["runs"][2]["error"])
        # The arm's other runs, and the other two arms, still happen.
        self.assertEqual(len(record["arms"]["C2-cache"]["runs"]), 5)
        self.assertEqual(len(record["arms"]["C3-cache-hybrid"]["runs"]), 5)

    def test_every_run_carries_its_own_load_trace(self):
        job = nl.job_by_id("re-anchor-c1-c2-c3-2026-08-02")
        record = nl.run_job(job, lambda *a: {"generation_tps": 5.0},
                            clock=lambda: 0.0)
        for arm in record["arms"].values():
            for run in arm["runs"]:
                self.assertIn("samples", run["load"])

    def test_every_record_carries_the_criterion_it_was_judged_by(self):
        job = nl.job_by_id("sdpa-reproducibility-2026-08-02")
        record = nl.run_job(job, lambda *a: {"max_abs_dlogprob": 0.0},
                            clock=lambda: 0.0)
        self.assertEqual(record["criterion"], job.criterion)

    def test_the_load_summary_says_it_was_folded_from_per_run_traces(self):
        job = nl.job_by_id("sdpa-reproducibility-2026-08-02")
        record = nl.run_job(job, lambda *a: {"max_abs_dlogprob": 0.0},
                            clock=lambda: 0.0)
        self.assertIn("note", record["load_summary"])

    def test_cancellation_stops_a_battery_partway(self):
        class Ctx:
            def __init__(self):
                self.n = 0

            def is_cancelled(self):
                self.n += 1
                return self.n > 3

        job = nl.job_by_id("re-anchor-c1-c2-c3-2026-08-02")
        record = nl.run_job(job, lambda *a: {"generation_tps": 1.0},
                            ctx=Ctx(), clock=lambda: 0.0)
        total = sum(len(a["runs"]) for a in record["arms"].values())
        self.assertLess(total, 15)
        self.assertTrue(record["notes"])


class TestRoundTrip(unittest.TestCase):
    def test_save_then_load_is_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.json"
            original = nl.load_jobs()
            nl.save_jobs(original, p)
            self.assertEqual(nl.load_jobs(p), original)

    def test_a_missing_registry_is_empty_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(nl.load_jobs(Path(d) / "absent.json"), [])

    def test_the_shipped_file_is_valid_json_with_a_version(self):
        raw = json.loads(nl.REGISTRY_PATH.read_text())
        self.assertEqual(raw["version"], 1)


if __name__ == "__main__":
    unittest.main()
