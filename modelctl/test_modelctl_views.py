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


def run(cache_metrics=None, **kw):
    """A plan_run row as the DB really stores one.

    cache_metrics is given as {hits, lookups} for readability and is
    rendered into the shape scrape_moe_cache_metrics actually produces:
    details_json -> {"cache_metrics": {device: {metric: value}}}. The
    old helper put a flat numeric dict at run["cache_metrics"], a shape
    nothing in production ever wrote -- which is how the H2D branch
    stayed "tested" while being unreachable for real runs.
    """
    base = {"read_bytes_generation": 0, "major_faults": 0,
            "generation_tps": 20.0, "claim_json": json.dumps({"ram_bytes": 0})}
    if cache_metrics is not None:
        hits = cache_metrics["hits"]
        misses = cache_metrics["lookups"] - hits
        base["details_json"] = json.dumps({"cache_metrics": {
            "SYCL0": {"hits_total": str(hits), "misses_total": str(misses)}}})
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


class TestHistoryAPI(unittest.TestCase):
    """The history questions, now asked of the API instead of the page.

    The server-rendered /profiles/{name}/history page was demolished with
    the rest of the old console (console phase 3); the same run rows are
    served by GET /api/profiles/{name}/history (every recorded column) and
    GET /api/v2/models/{name}/history (the console's view, which adds the
    bottleneck verdict). The page's job was to answer the RAM/SSD
    questions, so the payloads have to carry the facts it stated -- a
    number the API drops is a question nothing can answer any more.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        profiles = self.root / "profiles"
        profiles.mkdir()
        # /api/v2/models/{name}/history gates on load_profile so an unknown
        # model 404s instead of answering []; the profile only has to exist.
        (profiles / "m1.json").write_text(json.dumps(
            {"name": "m1", "config": {}}))
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

    def rows(self):
        """Every recorded column, from the surviving legacy JSON API."""
        resp = self.client.get("/api/profiles/m1/history", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def console_rows(self):
        """What the /v2 console reads: the same runs plus the verdict."""
        resp = self.client.get("/api/v2/models/m1/history", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def test_api_answers_every_required_question(self):
        self.record()
        row = self.rows()[0]
        # how much is in VRAM / how much RAM is resident
        self.assertEqual(json.loads(row["peak_vram_json"]),
                         {"SYCL0": 12 * GB})
        self.assertEqual(row["peak_ram_bytes"], 8 * GB)
        self.assertEqual(row["peak_pss_bytes"], 6 * GB)
        # is the process taking major faults
        self.assertEqual(row["major_faults"], 55_000)
        self.assertEqual(row["minor_faults"], 900)
        # bytes read during load and during generation, kept apart: the
        # split is the whole point, a total cannot answer either question
        self.assertEqual(row["read_bytes_warmup"], 28 * GB)
        self.assertEqual(row["read_bytes_generation"], 2 * GB)
        self.assertEqual(row["read_bytes"], 30 * GB)
        # which cache state this result is
        self.assertEqual(row["cache_state"], "warm")
        # what limited it -- the judgement itself, not the raw counters
        verdict = self.console_rows()[0]
        self.assertEqual(verdict["cache_state"], "warm")
        self.assertEqual(verdict["bottleneck"], "storage")
        self.assertIn("during generation", verdict["bottleneck_why"])

    def test_unmeasured_fields_are_null_rather_than_zero(self):
        # "not measured" and "zero" are different claims, and JSON has to
        # keep them apart: null is the absent counter, 0 is a real reading.
        self.record(peak_ram_bytes=None, peak_pss_bytes=None,
                    read_bytes=None, major_faults=None,
                    peak_vram_bytes={}, storage_activity="",
                    storage_activity_detail="")
        row = self.rows()[0]
        for field in ("peak_ram_bytes", "peak_pss_bytes", "read_bytes",
                      "major_faults"):
            with self.subTest(field=field):
                self.assertIsNone(row[field])
        self.assertEqual(json.loads(row["peak_vram_json"]), {})
        self.assertEqual(row["storage_activity"], "")
        self.assertEqual(row["storage_activity_detail"], "")

    def test_unreleased_vram_is_visible_in_the_row(self):
        # The page turned a non-zero final VRAM reading into a "not fully
        # released" flag. Nothing judges it server-side now, so the rows
        # have to carry the leftover bytes distinctly from a clean
        # release -- collapse the two and a leak becomes invisible.
        self.record(started_at=1.0)                       # released cleanly
        self.record(started_at=2.0, final_vram_bytes={"SYCL0": 4 * GB})
        leaked, clean = self.rows()  # newest first
        self.assertEqual(json.loads(leaked["final_vram_json"]),
                         {"SYCL0": 4 * GB})
        self.assertEqual(json.loads(clean["final_vram_json"]), {"SYCL0": 0})

    def test_empty_history_is_an_empty_list_not_an_error(self):
        # The page said "No measurements recorded yet"; the API has to say
        # the same thing as data -- 200 with [], never a 404 or a 500 the
        # console would have to render as "history unavailable".
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.console_rows(), [])

    def test_storage_device_is_named(self):
        self.record()
        self.assertEqual(self.rows()[0]["storage_device"], "nvme0n1")


if __name__ == "__main__":
    unittest.main()
