"""The anchor registry: when a stored number may still be used.

Every test here is about a reason to re-measure. Reusing an anchor is the
cheap path and the registry's whole job is to make the expensive path
happen when it should -- so the cases that matter are the ones where a
naive check would say "still fine": an unrecorded field, a voided value,
and the canary whose fingerprint matches precisely when you want it run.
"""
import json
import tempfile
import unittest
from pathlib import Path

import modelctl_anchors as ma

FP = ma.Fingerprint(build_commit="85b7e6556", profile_hash="p1",
                    env_hash="e1", driver="intel-compute-runtime-26.22")


def _anchor(**kw):
    base = dict(id="a", condition="C1", value=6.0, runs=(6.0, 6.1),
                fingerprint=FP, recorded="2026-08-02")
    base.update(kw)
    return ma.Anchor(**base)


class TestFingerprint(unittest.TestCase):
    def test_an_identical_fingerprint_has_no_differences(self):
        self.assertEqual(FP.differences(FP), [])

    def test_a_changed_field_is_named_with_both_values(self):
        other = ma.Fingerprint(build_commit="deadbeef", profile_hash="p1",
                               env_hash="e1", driver=FP.driver)
        diffs = FP.differences(other)
        self.assertEqual(len(diffs), 1)
        self.assertIn("build_commit", diffs[0])
        self.assertIn("85b7e6556", diffs[0])
        self.assertIn("deadbeef", diffs[0])

    def test_fields_are_named_separately_not_rolled_into_one_digest(self):
        # "the driver moved" and "the binary moved" call for different
        # work; a combined hash cannot tell them apart.
        other = ma.Fingerprint(build_commit="x", profile_hash="y",
                               env_hash="z", driver="w")
        self.assertEqual(len(FP.differences(other)), 4)

    def test_an_unrecorded_field_does_not_match_anything(self):
        partial = ma.Fingerprint(build_commit="85b7e6556")
        diffs = partial.differences(FP)
        self.assertTrue(any("profile_hash was not recorded" in d for d in diffs))
        self.assertTrue(any("env_hash was not recorded" in d for d in diffs))
        self.assertTrue(any("driver was not recorded" in d for d in diffs))

    def test_an_unknown_current_value_does_not_match_either(self):
        diffs = FP.differences(ma.Fingerprint(build_commit="85b7e6556"))
        self.assertTrue(any("is unknown on this machine" in d for d in diffs))

    def test_no_current_fingerprint_at_all(self):
        self.assertEqual(FP.differences(None),
                         ["no current fingerprint to compare against"])


class TestHashes(unittest.TestCase):
    def test_profile_hash_ignores_cosmetic_fields(self):
        a = {"name": "one", "model_path": "/m.gguf",
             "config": {"ctx": 4096, "description": "old words"}}
        b = {"name": "two", "model_path": "/m.gguf",
             "config": {"ctx": 4096, "description": "new words"}}
        self.assertEqual(ma.profile_hash(a), ma.profile_hash(b))

    def test_profile_hash_moves_on_context(self):
        a = {"model_path": "/m.gguf", "config": {"ctx": 4096}}
        b = {"model_path": "/m.gguf", "config": {"ctx": 8192}}
        self.assertNotEqual(ma.profile_hash(a), ma.profile_hash(b))

    def test_profile_hash_moves_on_the_cache_budget(self):
        a = {"model_path": "/m.gguf", "moe_cache": {"mode": "off"}}
        b = {"model_path": "/m.gguf",
             "moe_cache": {"mode": "manual",
                           "gpu": {"budgets_bytes": {"SYCL0": 4294967296}}}}
        self.assertNotEqual(ma.profile_hash(a), ma.profile_hash(b))

    def test_profile_hash_is_key_order_independent(self):
        a = {"model_path": "/m.gguf", "config": {"ctx": 4096, "fit": "off"}}
        b = {"config": {"fit": "off", "ctx": 4096}, "model_path": "/m.gguf"}
        self.assertEqual(ma.profile_hash(a), ma.profile_hash(b))

    def test_no_profile_hashes_to_nothing(self):
        self.assertEqual(ma.profile_hash(None), "")

    def test_env_hash_tracks_the_determinism_knob(self):
        on = ma.env_hash({"GGML_SYCL_DETERMINISTIC": "1"})
        off = ma.env_hash({"GGML_SYCL_DETERMINISTIC": "0"})
        self.assertNotEqual(on, off)

    def test_env_hash_tracks_the_moe_offload_threshold(self):
        # An ambient threshold changes what the runtime does; an anchor
        # taken under one must not be offered for the other.
        a = ma.env_hash({"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "1"})
        b = ma.env_hash({"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "32"})
        self.assertNotEqual(a, b)

    def test_env_hash_ignores_unrelated_shell_noise(self):
        a = ma.env_hash({"GGML_SYCL_DETERMINISTIC": "1", "SSH_TTY": "/dev/x"})
        b = ma.env_hash({"GGML_SYCL_DETERMINISTIC": "1", "LANG": "C"})
        self.assertEqual(a, b)

    def test_an_unset_variable_is_not_its_default(self):
        # The default is a property of the binary, which build_commit
        # already covers; conflating them hides a build whose default moved.
        self.assertNotEqual(ma.env_hash({}),
                            ma.env_hash({"GGML_SYCL_DETERMINISTIC": "1"}))

    def test_driver_identity_is_empty_when_it_cannot_be_read(self):
        ma._DRIVER_CACHE = None
        try:
            def failing(*a, **kw):
                raise FileNotFoundError("no rpm")
            self.assertEqual(ma.driver_identity(runner=failing), "")
        finally:
            ma._DRIVER_CACHE = None

    def test_driver_identity_reads_the_package_version(self):
        ma._DRIVER_CACHE = None
        try:
            class R:
                returncode = 0
                stdout = "26.22.38646.6-3.fc44"
            self.assertEqual(ma.driver_identity(runner=lambda *a, **k: R()),
                             "intel-compute-runtime-26.22.38646.6-3.fc44")
        finally:
            ma._DRIVER_CACHE = None


class TestStaleness(unittest.TestCase):
    def test_a_matching_anchor_is_reusable(self):
        self.assertEqual(ma.staleness(_anchor(), FP), [])
        self.assertFalse(ma.needs_run(_anchor(), FP))

    def test_a_moved_build_stales_it(self):
        other = ma.Fingerprint(build_commit="new", profile_hash="p1",
                               env_hash="e1", driver=FP.driver)
        self.assertTrue(ma.needs_run(_anchor(), other))

    def test_a_voided_anchor_reruns_even_when_everything_matches(self):
        # "The conditions changed" and "that number should not be trusted"
        # are different facts, and only one of them is a fingerprint.
        a = _anchor(void=True, void_reason="load-contaminated")
        reasons = ma.staleness(a, FP)
        self.assertTrue(reasons[0].startswith("voided: load-contaminated"))

    def test_a_void_with_no_reason_still_says_it_is_void(self):
        reasons = ma.staleness(_anchor(void=True), FP)
        self.assertIn("no reason recorded", reasons[0])

    def test_the_void_reason_comes_before_the_fingerprint_verdict(self):
        # Reporting "conditions unchanged" about a bad number is true and
        # useless; the reason someone must act on goes first.
        a = _anchor(void=True, void_reason="bad")
        self.assertIn("voided", ma.staleness(a, FP)[0])

    def test_an_always_run_anchor_is_never_reusable(self):
        a = _anchor(always_run=True)
        reasons = ma.staleness(a, FP)
        self.assertTrue(any("always runs" in r for r in reasons))
        self.assertTrue(ma.needs_run(a, FP))

    def test_an_anchor_that_was_never_measured_is_stale(self):
        a = _anchor(value=None, runs=())
        self.assertTrue(any("no value has ever been recorded" in r
                            for r in ma.staleness(a, FP)))


class TestBatteryPlan(unittest.TestCase):
    def test_it_splits_reusable_from_to_run(self):
        anchors = [_anchor(id="fresh"),
                   _anchor(id="stale", fingerprint=ma.Fingerprint(
                       build_commit="old", profile_hash="p1", env_hash="e1",
                       driver=FP.driver))]
        plan = ma.plan_battery(anchors, FP)
        self.assertEqual([a.id for a, _ in plan.to_run], ["stale"])
        self.assertEqual([a.id for a in plan.reusable], ["fresh"])

    def test_the_plan_carries_the_reason_for_every_re_run(self):
        plan = ma.plan_battery([_anchor(void=True, void_reason="why")], FP)
        self.assertTrue(plan.to_dict()["to_run"][0]["reasons"])

    def test_an_empty_battery_plans_nothing(self):
        plan = ma.plan_battery([], FP)
        self.assertEqual(plan.to_dict(), {"to_run": [], "reusable": [],
                                          "runs": 0, "reused": 0})


class TestRecord(unittest.TestCase):
    def test_recording_clears_the_void_flag(self):
        a = _anchor(void=True, void_reason="load-contaminated")
        fresh = ma.record(a, 6.5, runs=(6.4, 6.6), fingerprint=FP,
                          recorded="2026-08-03")
        self.assertFalse(fresh.void)
        self.assertEqual(fresh.void_reason, "")
        self.assertEqual(fresh.value, 6.5)
        self.assertEqual(ma.staleness(fresh, FP), [])

    def test_recording_does_not_clear_always_run(self):
        # always_run is a property of the anchor's role, not of any one
        # measurement -- a canary that stopped running after its first
        # good result is not a canary.
        fresh = ma.record(_anchor(always_run=True), 14.0, fingerprint=FP)
        self.assertTrue(fresh.always_run)
        self.assertTrue(ma.needs_run(fresh, FP))

    def test_void_anchor_keeps_the_old_value_readable(self):
        voided = ma.void_anchor(_anchor(), "load-contaminated")
        self.assertTrue(voided.void)
        self.assertEqual(voided.value, 6.0)
        self.assertEqual(voided.runs, (6.0, 6.1))


class TestRegistryIO(unittest.TestCase):
    def test_round_trip_is_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            original = ma.load_anchors()
            ma.save_anchors(original, p)
            self.assertEqual(ma.load_anchors(p), original)

    def test_a_missing_registry_is_empty_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ma.load_anchors(Path(d) / "absent.json"), [])

    def test_the_shipped_file_is_valid_json_with_a_version(self):
        raw = json.loads(ma.REGISTRY_PATH.read_text())
        self.assertEqual(raw["version"], 1)


class TestShippedRegistry(unittest.TestCase):
    def setUp(self):
        self.anchors = ma.load_anchors()
        self.by_id = {a.id: a for a in self.anchors}

    def test_it_holds_the_three_conditions_and_the_canary(self):
        self.assertEqual(sorted(self.by_id),
                         ["c1-static-122b", "c2-cache-122b",
                          "c3-cache-hybrid-122b", "laguna-s2.1-canary"])

    def test_the_2026_08_01_battery_is_void_for_load_contamination(self):
        for anchor_id in ("c1-static-122b", "c2-cache-122b",
                          "c3-cache-hybrid-122b"):
            a = self.by_id[anchor_id]
            self.assertTrue(a.void, anchor_id)
            self.assertIn("load-contaminated", a.void_reason)
            self.assertIn("2.63-17.15", a.void_reason)

    def test_the_voided_values_are_kept_not_deleted(self):
        # The record of a measurement that should not be trusted is still
        # evidence, and the re-run has to be comparable against it.
        c1 = self.by_id["c1-static-122b"]
        self.assertAlmostEqual(c1.value, 6.2365, places=3)
        self.assertEqual(len(c1.runs), 5)
        self.assertEqual(len(self.by_id["c2-cache-122b"].runs), 10)

    def test_every_anchor_cites_its_source(self):
        for a in self.anchors:
            self.assertTrue(a.source.endswith(".md"), a.id)

    def test_the_laguna_canary_is_exempt_and_not_void(self):
        canary = self.by_id["laguna-s2.1-canary"]
        self.assertTrue(canary.always_run)
        self.assertFalse(canary.void)
        self.assertEqual(canary.value, 14.20)

    def test_every_battery_anchor_needs_a_run_today(self):
        # Whatever this machine's current fingerprint is: three are void
        # and the fourth is exempt, so a battery run now measures all four.
        # driver is injected rather than probed -- the suite must not
        # shell out to rpm, and the answer must not depend on the box.
        fp = ma.current_fingerprint(build_commit="85b7e6556", driver="d")
        plan = ma.plan_battery(self.anchors, fp)
        self.assertEqual(plan.reusable, [])
        self.assertEqual(len(plan.to_run), 4)

    def test_the_cache_conditions_carry_their_recorded_hit_ratios(self):
        for anchor_id in ("c2-cache-122b", "c3-cache-hybrid-122b"):
            ratios = self.by_id[anchor_id].extra["hit_ratio_per_run"]
            self.assertEqual(len(ratios), 10, anchor_id)

    def test_c1_records_no_hit_ratio_because_it_has_no_cache(self):
        self.assertIsNone(self.by_id["c1-static-122b"]
                          .extra["hit_ratio_per_run"])

    def test_the_effective_cache_budget_is_recorded_as_never_measured(self):
        # The gap the re-anchor job exists to close: the 2026-08-01
        # battery has only the REQUESTED --moe-cache-bytes in its argv.
        for a in self.anchors:
            if a.id == "laguna-s2.1-canary":
                continue
            self.assertIsNone(a.extra["effective_cache_budget_bytes_per_run"])
            self.assertIn("not recorded", a.extra["effective_cache_budget_note"])

    def test_the_battery_load_is_recorded_as_battery_wide(self):
        c1 = self.by_id["c1-static-122b"]
        self.assertEqual(c1.load["loadavg_1m"]["max"], 17.15)
        self.assertIn("not per run", c1.load["note"])


if __name__ == "__main__":
    unittest.main()
