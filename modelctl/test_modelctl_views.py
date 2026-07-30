"""Task D5: the history view answers the RAM/SSD questions.

The bottleneck classifier is the part with teeth: a confident wrong
answer sends someone off to optimise the wrong thing, so every branch has
to be backed by a counter the run actually recorded, and "unknown" has to
be a real outcome.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_tune
from modelctl_web.app import create_app
from modelctl_web.jobs import JobStore, JobRunner

MB = 1 << 20
GB = 1 << 30
TOKEN = "test-token"


def run(**kw):
    base = {"read_bytes_generation": 0, "major_faults": 0,
            "generation_tps": 20.0, "claim_json": json.dumps({"ram_bytes": 0})}
    base.update(kw)
    return base


class TestBottleneckClassification(unittest.TestCase):
    def test_reads_during_generation_mean_storage_bound(self):
        label, why = modelctl_tune.classify_bottleneck(
            run(read_bytes_generation=2 * GB))
        self.assertEqual(label, "storage")
        self.assertIn("during generation", why)

    def test_heavy_faults_with_reads_mean_storage_bound(self):
        label, why = modelctl_tune.classify_bottleneck(
            run(major_faults=50_000, read_bytes_generation=128 * MB))
        self.assertEqual(label, "storage")
        self.assertIn("faulting in", why)

    def test_host_weights_without_rereads_mean_cpu_bound(self):
        # Weights sit in RAM and are not being re-read, so the host is
        # computing over them every token.
        label, why = modelctl_tune.classify_bottleneck(
            run(claim_json=json.dumps({"ram_bytes": 40 * GB})))
        self.assertEqual(label, "cpu")
        self.assertIn("CPU is", why)

    def test_low_cache_hit_rate_means_h2d_bound(self):
        label, why = modelctl_tune.classify_bottleneck(
            run(cache_metrics={"lookups": 1000, "hits": 100}))
        self.assertEqual(label, "h2d")
        self.assertIn("PCIe", why)

    def test_high_cache_hit_rate_is_not_h2d_bound(self):
        label, _ = modelctl_tune.classify_bottleneck(
            run(cache_metrics={"lookups": 1000, "hits": 950}))
        self.assertEqual(label, "gpu")

    def test_clean_run_is_gpu_bound(self):
        label, why = modelctl_tune.classify_bottleneck(run())
        self.assertEqual(label, "gpu")
        self.assertIn("GPU-bound", why)

    def test_no_evidence_yields_unknown_not_a_guess(self):
        label, why = modelctl_tune.classify_bottleneck(
            {"generation_tps": None})
        self.assertEqual(label, "unknown")
        self.assertIn("not enough evidence", why)

    def test_storage_beats_cpu_when_both_look_true(self):
        # Offloaded weights AND active paging: the paging is what is
        # actually costing time.
        label, _ = modelctl_tune.classify_bottleneck(
            run(claim_json=json.dumps({"ram_bytes": 40 * GB}),
                read_bytes_generation=2 * GB))
        self.assertEqual(label, "storage")

    def test_missing_counters_do_not_raise(self):
        for payload in ({}, {"major_faults": None}, {"claim_json": "not json"}):
            with self.subTest(payload=payload):
                label, _ = modelctl_tune.classify_bottleneck(payload)
                self.assertIn(label, ("storage", "cpu", "h2d", "gpu", "unknown"))


class TestHistoryPage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        profiles = self.root / "profiles"
        profiles.mkdir()
        p = mock.patch.object(modelctl, "PROFILES_DIR", profiles)
        p.start()
        self.addCleanup(p.stop)

        import modelctl_runtime
        self.db = modelctl_runtime.RuntimeDB(self.root / "rt.db")
        q = mock.patch.object(modelctl_runtime, "RuntimeDB",
                              return_value=self.db)
        q.start()
        self.addCleanup(q.stop)

        store = JobStore(self.root / "jobs.db")
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        self.client = TestClient(create_app(token=TOKEN, store=store, runner=runner))
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    def record(self, **kw):
        payload = {
            "profile_name": "m1", "plan_id": "planabcdef123", "started_at": 1.0,
            "success": True, "generation_tps": 18.0, "prompt_tps": 200.0,
            "load_seconds": 12.0, "peak_ram_bytes": 8 * GB,
            "peak_pss_bytes": 6 * GB,
            "peak_vram_bytes": {"SYCL0": 12 * GB},
            "final_vram_bytes": {"SYCL0": 0},
            "read_bytes": 30 * GB, "read_bytes_warmup": 28 * GB,
            "read_bytes_generation": 2 * GB,
            "read_syscalls": 4242, "disk_read_bytes": 29 * GB,
            "storage_device": "nvme0n1", "major_faults": 55_000,
            "minor_faults": 900, "cache_state": "warm",
            "storage_activity": "storage-backed",
            "storage_activity_detail": "weights were paged in during the run",
            "rates": {"read_bytes_per_second": 1234.0},
        }
        payload.update(kw)
        self.db.record_plan_run(payload)

    def page(self):
        resp = self.client.get("/profiles/m1/history", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        return resp.text

    def test_page_answers_every_required_question(self):
        self.record()
        page = self.page()
        # how much is in VRAM / how much RAM is resident
        self.assertIn("peak", page)
        self.assertIn("PSS", page)
        # is the process taking major faults
        self.assertIn("55000", page)
        self.assertIn("major", page)
        # bytes read during load and during generation
        self.assertIn("during generation", page)
        self.assertIn("load/warmup", page)
        # which cache state this result is
        self.assertIn("warm cache", page)
        # what limited it
        self.assertIn("limited by storage", page)

    def test_unmeasured_fields_say_so_rather_than_showing_zero(self):
        # "not measured" and "zero" are different claims.
        self.record(peak_ram_bytes=None, peak_pss_bytes=None,
                    read_bytes=None, major_faults=None,
                    peak_vram_bytes={}, storage_activity="",
                    storage_activity_detail="")
        page = self.page()
        self.assertIn("not measured", page)

    def test_unreleased_vram_is_flagged(self):
        self.record(final_vram_bytes={"SYCL0": 4 * GB})
        self.assertIn("not fully released", self.page())

    def test_empty_history_is_explained(self):
        self.assertIn("No measurements recorded yet", self.page())

    def test_storage_device_is_named(self):
        self.record()
        self.assertIn("nvme0n1", self.page())


if __name__ == "__main__":
    unittest.main()
