"""Tests for the /v2 console backend: the telemetry snapshot/SSE stream
and the typed JSON endpoints the SPA consumes. Fully hermetic -- every
probe (xpu-smi, llama-swap, /proc, worker /metrics) is injected."""
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from modelctl_web import telemetry
from modelctl_web.app import create_app
from modelctl_web.jobs import JobRunner, JobStore

from test_modelctl_web import TOKEN, WebTestBase

METRICS_TEXT = """\
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 1000
llamacpp:predicted_tokens_seconds 14.2
llamacpp:prompt_tokens_seconds 412
llamacpp:moe_cache_hits_total{device="SYCL0"} 600
llamacpp:moe_cache_misses_total{device="SYCL0"} 400
llamacpp:moe_cache_hit_ratio{device="SYCL0"} 0.6
llamacpp:moe_cache_learning{device="SYCL0"} 0
llamacpp:moe_cache_hits_total{device="SYCL1"} 30
llamacpp:moe_cache_misses_total{device="SYCL1"} 70
llamacpp:moe_cache_hit_ratio{device="SYCL1"} 0.3
llamacpp:moe_cache_learning{device="SYCL1"} 1
"""


def make_collector(store=None, metrics_text=METRICS_TEXT, runtime=None,
                   profiles=None):
    runtime = runtime if runtime is not None else {
        "m1": {"model_id": "m1", "state": "ready", "state_class": "ok",
               "registered": True, "running": True, "pid": 42, "port": 8999,
               "started": time.time()}}
    profiles = profiles if profiles is not None else [{
        "name": "m1", "file": "m1-Q4_K_M", "backend": "llama-cpp",
        "model_path": "", "enabled": True,
        "config": {"split_mode": "layer", "tensor_split": "22,10",
                   "extra": "-ot exps=CPU"},
        "moe_cache": {"mode": "auto"},
    }]
    return telemetry.TelemetryCollector(
        store=store,
        inventory_fn=lambda: [{"device": "SYCL0", "name": "Arc Pro B70",
                               "total_bytes": 32 << 30, "free_bytes": 4 << 30}],
        meminfo_fn=lambda: {"total_bytes": 31 << 30,
                            "available_bytes": 10 << 30,
                            "used_bytes": 21 << 30},
        runtime_fn=lambda: runtime,
        profiles_fn=lambda: profiles,
        metrics_fn=lambda port: metrics_text,
        swap_probe_fn=lambda: {
            "swap": {"ok": True, "latency_ms": 3, "detail": ":9292"},
            "api": {"ok": True, "latency_ms": 5, "detail": ""}})


class TestParsing(unittest.TestCase):
    def test_parse_worker_metrics_splits_plain_and_moe(self):
        parsed = telemetry.parse_worker_metrics(METRICS_TEXT)
        self.assertEqual(parsed["plain"]["llamacpp:tokens_predicted_total"], 1000)
        self.assertEqual(parsed["plain"]["llamacpp:predicted_tokens_seconds"], 14.2)
        # per-device labels survive (collapsing them was the historical bug)
        self.assertEqual(parsed["moe"]["SYCL0"]["hits_total"], "600")
        self.assertEqual(parsed["moe"]["SYCL1"]["misses_total"], "70")

    def test_summarize_cache_aggregates_devices(self):
        parsed = telemetry.parse_worker_metrics(METRICS_TEXT)
        s = telemetry.summarize_cache(parsed["moe"])
        # 630 hits / 1100 lookups across both devices
        self.assertAlmostEqual(s["hit_ratio"], 630 / 1100)
        self.assertTrue(s["learning"])  # SYCL1 still learning -> any=True
        self.assertEqual(s["devices"], ["SYCL0", "SYCL1"])

    def test_summarize_cache_empty_is_none(self):
        self.assertIsNone(telemetry.summarize_cache({}))

    def test_read_meminfo(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "meminfo"
            p.write_text("MemTotal: 100 kB\nMemFree: 10 kB\n"
                         "MemAvailable: 40 kB\n")
            r = telemetry.read_meminfo(str(p))
        self.assertEqual(r["total_bytes"], 100 * 1024)
        self.assertEqual(r["available_bytes"], 40 * 1024)
        self.assertEqual(r["used_bytes"], 60 * 1024)

    def test_read_meminfo_unreadable_is_zeros(self):
        r = telemetry.read_meminfo("/nonexistent/meminfo")
        self.assertEqual(r, {"total_bytes": 0, "available_bytes": 0,
                             "used_bytes": 0})


class TestCollector(unittest.TestCase):
    def test_snapshot_shape(self):
        snap = make_collector().snapshot()
        self.assertEqual(
            set(snap),
            {"ts", "services", "gpus", "ram", "models", "jobs", "errors"})
        # A healthy tick names no failed section: `errors` is how a page
        # tells an empty section from a section whose probe died.
        self.assertEqual(snap["errors"], {})
        self.assertTrue(snap["services"]["swap"]["ok"])
        self.assertIn("console_started", snap["services"])
        g = snap["gpus"][0]
        self.assertEqual(g["used_bytes"], 28 << 30)
        m = snap["models"][0]
        self.assertEqual(m["name"], "m1")
        self.assertEqual(m["state"], "ready")
        self.assertEqual(m["placement"], "split 22,10 (layer) + CPU experts, "
                                         "tensor overrides, MoE cache auto")
        self.assertIsNotNone(m["cache"])
        json.dumps(snap)  # the SSE stream serializes this verbatim

    def test_tok_s_is_a_rate_between_snapshots(self):
        texts = iter([METRICS_TEXT,
                      METRICS_TEXT.replace("tokens_predicted_total 1000",
                                           "tokens_predicted_total 1050")])
        c = make_collector()
        c._metrics_fn = lambda port: next(texts)
        first = c.snapshot()["models"][0]
        self.assertIsNone(first["tok_s"])  # no previous sample yet
        self.assertEqual(first["tok_s_avg"], 14.2)
        time.sleep(0.05)
        second = c.snapshot()["models"][0]
        self.assertIsNotNone(second["tok_s"])
        self.assertGreater(second["tok_s"], 0)

    def test_stopped_model_resets_rate_state(self):
        c = make_collector()
        c.snapshot()
        self.assertIn("m1", c._prev)
        c._runtime_fn = lambda: {"m1": {"model_id": "m1", "state": "stopped",
                                        "state_class": "stopped",
                                        "registered": True, "running": False,
                                        "pid": None, "port": None,
                                        "started": None}}
        snap = c.snapshot()
        self.assertNotIn("m1", c._prev)
        m = snap["models"][0]
        self.assertIsNone(m["tok_s"])
        self.assertIsNone(m["cache"])
        self.assertEqual(m["state"], "stopped")

    def test_unmanaged_runtime_model_is_shown(self):
        c = make_collector(profiles=[])
        rows = c.snapshot()["models"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["backend"], "unmanaged")
        self.assertEqual(rows[0]["placement"], "(no modelctl profile)")

    def test_every_probe_failing_degrades_not_raises(self):
        def boom(*a, **k):
            raise RuntimeError("probe down")
        c = telemetry.TelemetryCollector(
            store=None, inventory_fn=boom, meminfo_fn=boom, runtime_fn=boom,
            profiles_fn=boom, metrics_fn=boom, swap_probe_fn=boom)
        snap = c.snapshot()
        self.assertEqual(snap["gpus"], [])
        self.assertEqual(snap["models"], [])
        self.assertEqual(snap["jobs"], [])
        self.assertFalse(snap["services"]["swap"]["ok"])
        json.dumps(snap)

    def test_sse_stream_emits_tick_events(self):
        c = make_collector()
        gen = telemetry.sse_stream(c, interval=0.01, max_seconds=1)
        frame = next(gen)
        gen.close()
        self.assertTrue(frame.startswith("event: tick\ndata: "))
        payload = json.loads(frame.split("data: ", 1)[1])
        self.assertIn("models", payload)


class ConsoleApiBase(WebTestBase):
    """WebTestBase plus a client whose app has an injected collector."""

    def make_client(self, collector, **app_kwargs):
        from fastapi.testclient import TestClient
        runner = JobRunner(self.store)
        app = create_app(token=TOKEN, store=self.store, runner=runner,
                         collector=collector, **app_kwargs)
        return TestClient(app), runner


class TestV2Api(ConsoleApiBase):
    def test_auth_required(self):
        for path in ("/api/v2/models", "/api/v2/jobs"):
            self.assertEqual(self.client.get(path).status_code, 401, path)
        self.assertEqual(
            self.client.post("/api/v2/jobs/x/cancel").status_code, 401)

    def test_models_endpoint_typed(self):
        client, _ = self.make_client(make_collector(store=self.store))
        rows = client.get("/api/v2/models", headers=self.auth).json()
        self.assertEqual(rows[0]["name"], "m1")
        for key in ("state", "state_class", "placement", "tok_s", "cache",
                    "registered", "running", "port"):
            self.assertIn(key, rows[0])

    def test_jobs_endpoint_typed(self):
        job_id = self.store.create("bench", "bench m1", lane="benchmark")
        self.store.update(job_id, status="running", progress=0.5,
                          detail="run 2/3")
        self.store.append_result_line(job_id, "run 1/3: 14.31 tok/s")
        client, _ = self.make_client(make_collector(store=self.store))
        rows = client.get("/api/v2/jobs", headers=self.auth).json()
        row = next(r for r in rows if r["id"] == job_id)
        self.assertEqual(row["lane"], "benchmark")
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["progress"], 0.5)
        self.assertIn("14.31 tok/s", row["result_tail"])
        self.assertTrue(row["cancellable"])

    def test_cancel_queued_job_succeeds(self):
        job_id = self.store.create("bench", "bench m1", lane="benchmark")
        client, _runner = self.make_client(make_collector(store=self.store))
        res = client.post(f"/api/v2/jobs/{job_id}/cancel",
                          headers=self.auth).json()
        self.assertTrue(res["cancelled"])
        self.assertEqual(res["status"], "cancelled")
        self.assertIsNone(res["reason"])
        self.assertEqual(self.store.get(job_id)["status"], "cancelled")

    def test_cancel_finished_job_refused(self):
        # The reconcile path: the job finished before the optimistic cancel
        # reached the server. The answer must be an explicit refusal, not a
        # rewrite of history to 'cancelled'.
        job_id = self.store.create("bench", "bench m1", lane="benchmark")
        self.store.update(job_id, status="done", finished=time.time())
        client, _ = self.make_client(make_collector(store=self.store))
        res = client.post(f"/api/v2/jobs/{job_id}/cancel",
                          headers=self.auth).json()
        self.assertFalse(res["cancelled"])
        self.assertEqual(res["status"], "done")
        self.assertIn("already done", res["reason"])
        self.assertEqual(self.store.get(job_id)["status"], "done")

    def test_cancel_uncancellable_job_refused(self):
        job_id = self.store.create("mutation", "apply config")
        self.store.update(job_id, status="running", cancellable=0)
        client, _ = self.make_client(make_collector(store=self.store))
        res = client.post(f"/api/v2/jobs/{job_id}/cancel",
                          headers=self.auth).json()
        self.assertFalse(res["cancelled"])
        self.assertIn("not cancellable", res["reason"])
        self.assertEqual(self.store.get(job_id)["status"], "running")

    def test_cancel_unknown_job_404(self):
        client, _ = self.make_client(make_collector(store=self.store))
        self.assertEqual(
            client.post("/api/v2/jobs/nope/cancel", headers=self.auth)
            .status_code, 404)

    def test_events_stream_yields_tick(self):
        # Tiny interval + bounded lifetime so the stream ends by itself:
        # the TestClient portal cannot cancel a sync generator mid-sleep,
        # and a 1h-lifetime stream would hang the suite on teardown.
        client, _ = self.make_client(make_collector(store=self.store),
                                     tick_interval=0.01,
                                     tick_max_seconds=0.2)
        with client.stream("GET", "/api/v2/events", headers=self.auth) as r:
            self.assertEqual(r.status_code, 200)
            self.assertIn("text/event-stream", r.headers["content-type"])
            body = "".join(r.iter_text())
        self.assertIn("event: tick", body)
        payload = json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])
        self.assertIn("models", payload)
        self.assertIn("jobs", payload)

    def test_events_stream_requires_auth(self):
        self.assertEqual(self.client.get("/api/v2/events").status_code, 401)


class TestV2Spa(WebTestBase):
    """The committed dist/ is served at /v2/ with SPA fallback."""

    def test_spa_served(self):
        resp = self.client.get("/v2/", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("<div id=\"app\">", resp.text)

    def test_spa_fallback_for_routes(self):
        resp = self.client.get("/v2/jobs", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("<div id=\"app\">", resp.text)

    def test_spa_asset_served(self):
        index = self.client.get("/v2/", headers=self.auth).text
        import re
        m = re.search(r'src="/v2/(assets/[^"]+\.js)"', index)
        self.assertIsNotNone(m, "index.html references no /v2/assets script")
        resp = self.client.get(f"/v2/{m.group(1)}", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("javascript", resp.headers["content-type"])

    def test_spa_requires_auth(self):
        resp = self.client.get("/v2/", follow_redirects=False)
        self.assertEqual(resp.status_code, 303)
        self.assertIn("/login", resp.headers["location"])

    def test_no_traversal_out_of_dist(self):
        resp = self.client.get("/v2/%2e%2e/%2e%2e/app.py", headers=self.auth)
        # Either the SPA fallback (200 index.html) or a 404 -- never file
        # contents from outside dist.
        self.assertNotIn("create_app", resp.text)


if __name__ == "__main__":
    unittest.main()
