"""Measurement integrity (Phase 7 remediation).

The failure these guard: the managed worker recorded every serving
session as a successful, throughput-less observation, and the
newest-per-plan rule let that row REPLACE the real benchmark. Ranking
then scored a measured plan as unmeasured, and autotune skipped it with
"already validated (None tok/s)".
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import modelctl_runtime
import modelctl_storage
import modelctl_tune


def _run(plan_id, *, kind="benchmark", tps=None, success=True, started,
         state="cold"):
    return {
        "profile_name": "m1", "plan_id": plan_id,
        "hardware_fingerprint": "hw1", "backend_fingerprint": "be1",
        "started_at": started, "finished_at": started + 10,
        "success": success, "generation_tps": tps,
        "run_kind": kind, "cache_state": state,
    }


class ObservationBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = modelctl_runtime.RuntimeDB(Path(self.tmp.name) / "rt.db")


class TestServingRunsDoNotShadowBenchmarks(ObservationBase):
    def test_later_serving_row_does_not_replace_the_measurement(self):
        self.db.record_plan_run(_run("planA", kind="benchmark", tps=42.0,
                                     started=100))
        # A serving session afterwards: newer, successful, no throughput.
        self.db.record_plan_run(_run("planA", kind="serving", tps=None,
                                     started=200, state=""))
        obs = self.db.observations_for_profile("m1", "hw1", "be1")
        self.assertIn("planA", obs)
        self.assertEqual(obs["planA"]["generation_tps"], 42.0,
                         "a serving row shadowed the real measurement")

    def test_serving_only_plan_is_not_an_observation(self):
        self.db.record_plan_run(_run("planB", kind="serving", tps=None,
                                     started=100, state=""))
        obs = self.db.observations_for_profile("m1", "hw1", "be1")
        self.assertNotIn("planB", obs,
                         "a plan with no benchmark must read as unmeasured, "
                         "not as validated with no throughput")

    def test_default_run_kind_is_benchmark(self):
        # Rows written before run_kind existed are measurements.
        run = _run("planC", tps=7.0, started=100)
        del run["run_kind"]
        self.db.record_plan_run(run)
        obs = self.db.observations_for_profile("m1", "hw1", "be1")
        self.assertEqual(obs["planC"]["generation_tps"], 7.0)


class TestUnknownCacheStateIsNotCold(ObservationBase):
    def test_unknown_state_does_not_win_over_a_real_cold_bucket(self):
        # Two real cold measurements vs one state-less legacy row.
        self.db.record_plan_run(_run("p1", tps=10.0, started=100, state="cold"))
        self.db.record_plan_run(_run("p2", tps=11.0, started=101, state="cold"))
        self.db.record_plan_run(_run("p3", tps=99.0, started=102, state=""))
        obs = self.db.observations_for_profile("m1", "hw1", "be1")
        self.assertEqual(set(obs), {"p1", "p2"},
                         "state-less rows were bucketed as cold and could "
                         "outvote genuine cold measurements")


class TestBottleneckAttribution(unittest.TestCase):
    """classify_bottleneck read a flat run['cache_metrics'] with numeric
    hits/lookups -- a shape nothing produced -- so every transfer-bound
    run was reported as GPU-bound."""

    def _row(self, hits, misses, tps=12.0):
        return {"read_bytes_generation": 0, "major_faults": 0,
                "generation_tps": tps,
                "details_json": json.dumps({"cache_metrics": {
                    "SYCL0": {"hits_total": str(hits),
                              "misses_total": str(misses)}}})}

    def test_low_hit_rate_is_h2d(self):
        label, why = modelctl_tune.classify_bottleneck(self._row(10, 90))
        self.assertEqual(label, "h2d")
        self.assertIn("10%", why)

    def test_high_hit_rate_is_not_h2d(self):
        label, _ = modelctl_tune.classify_bottleneck(self._row(95, 5))
        self.assertEqual(label, "gpu")

    def test_no_metrics_still_classifies_gpu(self):
        label, _ = modelctl_tune.classify_bottleneck(
            {"read_bytes_generation": 0, "major_faults": 0,
             "generation_tps": 12.0})
        self.assertEqual(label, "gpu")


class TestFailureClassification(unittest.TestCase):
    """The suppression classes exist only if something produces them."""

    def test_log_tail_drives_the_class(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text("llama_model_load: error: unknown model architecture\n")
            cls = modelctl_tune._classify_failure_from_log(
                {"backend": "llama-cpp"}, 1, str(log))
        self.assertEqual(cls, "unsupported_architecture",
                         "a permanently-broken plan must get a suppressible "
                         "class, not the generic crash that gets retried")

    def test_missing_log_falls_back_safely(self):
        cls = modelctl_tune._classify_failure_from_log(
            {"backend": "llama-cpp"}, None, "/nonexistent/run.log")
        self.assertEqual(cls, "health_timeout")

    def test_oom_is_classified(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            log.write_text("ggml_backend_alloc: out of memory\n")
            cls = modelctl_tune._classify_failure_from_log(
                {"backend": "llama-cpp"}, 1, str(log))
        self.assertEqual(cls, "out_of_ram")


class TestMountPointMatching(unittest.TestCase):
    def test_sibling_prefix_is_not_the_same_mount(self):
        # /mnt/models is a mount; /mnt/models2 is a directory on /.
        import unittest.mock as mock
        mounts = "dev /mnt/models ext4 rw 0 0\n/dev/root / ext4 rw 0 0\n"
        with mock.patch("builtins.open",
                        mock.mock_open(read_data=mounts)):
            got = modelctl_storage._find_mount_point("/mnt/models2/x.gguf")
        self.assertEqual(got, "/",
                         "a sibling with a shared prefix was attributed to "
                         "the wrong mount (wrong device probed)")

    def test_real_child_matches(self):
        import unittest.mock as mock
        mounts = "dev /mnt/models ext4 rw 0 0\n/dev/root / ext4 rw 0 0\n"
        with mock.patch("builtins.open",
                        mock.mock_open(read_data=mounts)):
            got = modelctl_storage._find_mount_point("/mnt/models/x.gguf")
        self.assertEqual(got, "/mnt/models")


if __name__ == "__main__":
    unittest.main()
