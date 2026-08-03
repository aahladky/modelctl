"""Console phase 2: scratch-safe mode, SSE shutdown, model hub API, and
the /v2 add-wizard API.

Carries its own isolation (tmp state, patched dirs/globs) so single-file
runs never touch real state -- the full suite additionally rides
test__hermeticity's bootstrap.
"""
import json
import os
import struct
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_runtime
import modelctl_tiers
from modelctl_web import telemetry
from modelctl_web import wizard as wiz
from modelctl_web.app import create_app, scratch_missing_redirections
from modelctl_web.jobs import JobRunner, JobStore, sse_job_stream


# Env names the wizard carve-out requires; tests clear them to simulate a
# scratch instance pointed at the live install.
REDIRECTION_VARS = ("MODELCTL_HOME", "MODELCTL_STATE_DIR",
                    "MODELCTL_MODELS_DIR", "MODELCTL_LLAMA_SWAP_CONFIG",
                    "MODELCTL_LLAMA_SWAP_SERVICE",
                    "MODELCTL_LLAMA_SWAP_BASE_URL")


# ---- a real, tiny GGUF ---------------------------------------------------

def _gguf_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _gguf_kv_u32(key, value):
    return _gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _gguf_kv_str(key, value):
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _gguf_tensor(name, dims, type_id=1):
    out = _gguf_string(name)
    out += struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", type_id)
    out += struct.pack("<Q", 0)  # in-shard offset, unread by the parser
    return out


def write_tiny_gguf(path, arch="llama", ctx_max=2048, layers=2,
                    embed=64, heads=4, kv_heads=2):
    """A structurally valid GGUF v3 file small enough to commit nowhere:
    real magic, real KV metadata (architecture, context_length, the KV
    sizing fields), and a few real tensor infos so gguf_model_layout and
    verify_local_gguf both parse it. F16 tensors, no tensor data beyond
    padding -- nothing reads it."""
    kvs = [
        _gguf_kv_str("general.architecture", arch),
        _gguf_kv_str("general.name", "tiny-fixture"),
        _gguf_kv_u32(f"{arch}.context_length", ctx_max),
        _gguf_kv_u32(f"{arch}.block_count", layers),
        _gguf_kv_u32(f"{arch}.embedding_length", embed),
        _gguf_kv_u32(f"{arch}.attention.head_count", heads),
        _gguf_kv_u32(f"{arch}.attention.head_count_kv", kv_heads),
    ]
    tensors = [_gguf_tensor("token_embd.weight", (embed, 128))]
    for i in range(layers):
        tensors.append(_gguf_tensor(f"blk.{i}.attn_q.weight", (embed, embed)))
        tensors.append(_gguf_tensor(f"blk.{i}.ffn_gate.weight", (embed, embed)))
    tensors.append(_gguf_tensor("output.weight", (embed, 128)))
    blob = b"GGUF" + struct.pack("<I", 3)
    blob += struct.pack("<QQ", len(tensors), len(kvs))
    blob += b"".join(kvs) + b"".join(tensors)
    blob += b"\0" * max(0, 4096 - len(blob))  # verify wants >= 1 KiB
    Path(path).write_bytes(blob)
    return path


class TestTinyGgufFixture(unittest.TestCase):
    def test_header_and_layout_parse(self):
        import modelctl_vram
        with TemporaryDirectory() as tmp:
            p = write_tiny_gguf(Path(tmp) / "tiny.gguf")
            meta = modelctl_vram.read_gguf_kv_metadata(p)
            self.assertEqual(meta.get("general.architecture"), "llama")
            self.assertEqual(meta.get("llama.context_length"), 2048)
            layout = modelctl_vram.gguf_model_layout(str(p))
            self.assertIsNotNone(layout)
            self.assertGreater(layout["weight_bytes"], 0)
            self.assertFalse(layout["is_moe"])


# ---- rider a: scratch-safe mode -----------------------------------------

class TestScratchSafeJobStore(unittest.TestCase):
    def _store_with_running_job(self, tmp, **kwargs):
        db = Path(tmp) / "jobs.db"
        seed = JobStore(db, scratch_safe=True)  # seeding must not recover
        job_id = seed.create("bench", "b", lane="benchmark")
        seed.update(job_id, status="running")
        return db, job_id, kwargs

    def test_default_marks_orphaned_jobs_interrupted(self):
        with TemporaryDirectory() as tmp:
            db, job_id, _ = self._store_with_running_job(tmp)
            store = JobStore(db)
            self.assertEqual(store.get(job_id)["status"], "interrupted")

    def test_scratch_safe_leaves_live_jobs_alone(self):
        with TemporaryDirectory() as tmp:
            db, job_id, _ = self._store_with_running_job(tmp)
            store = JobStore(db, scratch_safe=True)
            self.assertEqual(store.get(job_id)["status"], "running")

    def test_env_flag_is_honoured(self):
        with TemporaryDirectory() as tmp:
            db, job_id, _ = self._store_with_running_job(tmp)
            with mock.patch.dict(os.environ, {"MODELCTL_WEB_SCRATCH": "1"}):
                store = JobStore(db)
            self.assertEqual(store.get(job_id)["status"], "running")


class ScratchGateBase(unittest.TestCase):
    """App factory under a controlled scratch env."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        (self.profiles_dir / "m1.json").write_text(json.dumps({
            "name": "m1", "repo_id": "r/m", "file": "f",
            "model_path": str(self.root / "m.gguf"),
            "config": {"device": "", "split_mode": "", "tensor_split": "",
                       "ctx": 4096, "cache_type_k": "q8_0",
                       "cache_type_v": "q8_0", "flash_attn": "auto",
                       "ttl": 60, "mtp": "off", "fit": "off", "extra": ""},
            "env": [], "enabled": True}))
        self.wizards_dir = self.root / "wizards"
        self.wizards_dir.mkdir()
        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
                  mock.patch.object(wiz, "WIZARD_DIR", self.wizards_dir),
                  mock.patch.object(
                      wiz.WizardStore, "__init__",
                      lambda s, base_dir=self.wizards_dir: (
                          setattr(s, "base_dir", Path(base_dir)),
                          Path(base_dir).mkdir(parents=True, exist_ok=True))[0])):
            p.start()
            self.addCleanup(p.stop)

    def make_client(self, env, keep_redirections=False):
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        if not keep_redirections:
            # Simulate a scratch instance sharing the live install: no
            # redirection env at all (the suite bootstrap sets some).
            saved = {}
            for k in REDIRECTION_VARS:
                if k in os.environ:
                    saved[k] = os.environ.pop(k)
            self.addCleanup(lambda: os.environ.update(saved))
        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        app = create_app(store=store, runner=runner)
        self.store = store
        return TestClient(app)

    auth = {}  # auth removed 2026-08-03: console is LAN-open


class TestScratchGateAgainstLiveState(ScratchGateBase):
    def setUp(self):
        super().setUp()
        self.client = self.make_client({"MODELCTL_WEB_SCRATCH": "1"})

    def test_reads_still_work(self):
        r = self.client.get("/api/profiles", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()[0]["name"], "m1")
        self.assertEqual(
            self.client.get("/api/jobs", headers=self.auth).status_code, 200)

    def test_api_mutation_refused_with_reason(self):
        r = self.client.post("/api/v2/jobs/nope/cancel", headers=self.auth)
        self.assertEqual(r.status_code, 405)
        body = r.json()
        self.assertEqual(body["error"], "scratch-safe mode")
        self.assertIn("MODELCTL_WEB_SCRATCH=1", body["reason"])

    def test_html_mutation_refused_with_reason(self):
        r = self.client.post("/profiles/m1", headers=self.auth,
                             data={"ctx": "8192"}, follow_redirects=False)
        self.assertEqual(r.status_code, 405)
        self.assertIn("scratch-safe mode", r.text)

    def test_every_flavor_of_mutation_is_refused(self):
        for method, path in [
                ("post", "/api/tiers/m1/apply"),
                ("post", "/api/profiles/m1"),
                ("post", "/api/pull/org/repo"),
                ("post", "/models/m1/load"),
                ("post", "/profiles/m1/delete"),
                ("post", "/jobs/x/cancel"),
                ("post", "/settings/save"),
                ("post", "/hardware/save"),
                ("post", "/setup/reprobe"),
                ("post", "/runtime/unload-all")]:
            r = getattr(self.client, method)(path, headers=self.auth,
                                             follow_redirects=False)
            self.assertEqual(r.status_code, 405, f"{path} not refused")

    def test_mutating_get_shims_are_refused(self):
        # GET /pull/{repo} and /add/start/{repo} write wizard state.
        for path in ("/pull/org/repo", "/add/start/org/repo"):
            r = self.client.get(path, headers=self.auth,
                                follow_redirects=False)
            self.assertEqual(r.status_code, 405, f"{path} not refused")

    def test_wizard_chain_refused_when_not_redirected(self):
        r = self.client.post("/api/v2/wizards", headers=self.auth)
        self.assertEqual(r.status_code, 405)
        self.assertIn("not redirected", r.json()["reason"])
        r = self.client.post("/add", headers=self.auth,
                             follow_redirects=False)
        self.assertEqual(r.status_code, 405)

    def test_wizard_get_never_rewrites_live_state(self):
        # Several GETs refresh wizard state opportunistically; against
        # the live install a scratch instance must not write those
        # refreshes (they race the live console and bump updated_at).
        state = wiz.WizardState()
        state.source_type = "local_file"
        state.local_path = "/nope/m.gguf"
        state.download_job_id = "job-x"
        path = self.wizards_dir / f"{state.wizard_id}.json"
        payload = json.dumps(state.to_dict(), indent=2)
        path.write_text(payload)
        r = self.client.get(f"/api/v2/wizard/{state.wizard_id}",
                            headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(path.read_text(), payload, "wizard file rewritten")
        # The store-level guard covers every handler, old and new.
        wiz.WizardStore().save(state)
        self.assertEqual(path.read_text(), payload, "save not blocked")
        wiz.WizardStore().delete(state.wizard_id)
        self.assertTrue(path.exists(), "delete not blocked")

    def test_sse_still_streams(self):
        collector = telemetry.TelemetryCollector(
            store=self.store,
            inventory_fn=lambda: [], meminfo_fn=lambda: {
                "total_bytes": 1, "available_bytes": 1, "used_bytes": 0},
            runtime_fn=lambda: {}, profiles_fn=lambda: [],
            metrics_fn=lambda port: None,
            swap_probe_fn=lambda: {"swap": {"ok": False, "latency_ms": None,
                                            "detail": ""},
                                   "api": {"ok": False, "latency_ms": None,
                                           "detail": ""}})
        runner = JobRunner(self.store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        app = create_app(store=self.store, runner=runner,
                         collector=collector, tick_interval=0.01,
                         tick_max_seconds=0.05)
        client = TestClient(app)
        with client.stream("GET", "/api/v2/events", headers=self.auth) as r:
            self.assertEqual(r.status_code, 200)
            body = "".join(r.iter_text())
        self.assertIn("event: tick", body)


class TestScratchGateHermetic(ScratchGateBase):
    def setUp(self):
        super().setUp()
        self.client = self.make_client({
            "MODELCTL_WEB_SCRATCH": "1",
            "MODELCTL_HOME": str(self.root),
            "MODELCTL_MODELS_DIR": str(self.root / "models"),
            "MODELCTL_LLAMA_SWAP_CONFIG": str(self.root / "swap.yaml"),
            "MODELCTL_LLAMA_SWAP_SERVICE": "modelctl-scratch-none.service",
            "MODELCTL_LLAMA_SWAP_BASE_URL": "http://127.0.0.1:9/v1/",
        }, keep_redirections=True)

    def test_redirections_report_complete(self):
        self.assertEqual(scratch_missing_redirections(), [])

    def test_wizard_chain_allowed(self):
        r = self.client.post("/api/v2/wizards", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["wizard_id"])

    def test_everything_else_still_refused(self):
        for path in ("/api/tiers/m1/apply", "/models/m1/load",
                     "/api/profiles/m1", "/settings/save"):
            r = self.client.post(path, headers=self.auth,
                                 follow_redirects=False)
            self.assertEqual(r.status_code, 405, f"{path} not refused")


class TestScratchOffIsUntouched(ScratchGateBase):
    def test_mutations_pass_the_gate_without_the_flag(self):
        client = self.make_client({}, keep_redirections=True)
        with mock.patch("modelctl_web.mutate.submit_edit",
                        return_value="job-1") as sub:
            r = client.post("/api/profiles/m1", headers=self.auth,
                            json={"ctx": "8192"})
        self.assertEqual(r.status_code, 200)
        sub.assert_called_once()


# ---- rider b: SSE shutdown ----------------------------------------------

class TestSseShutdown(unittest.TestCase):
    def _collector(self):
        return telemetry.TelemetryCollector(
            inventory_fn=lambda: [], meminfo_fn=lambda: {
                "total_bytes": 0, "available_bytes": 0, "used_bytes": 0},
            runtime_fn=lambda: {}, profiles_fn=lambda: [],
            metrics_fn=lambda port: None,
            swap_probe_fn=lambda: {"swap": {"ok": True, "latency_ms": 1,
                                            "detail": ""},
                                   "api": {"ok": True, "latency_ms": 1,
                                           "detail": ""}})

    def test_stream_ends_between_ticks_on_shutdown(self):
        stop = threading.Event()
        gen = telemetry.sse_stream(self._collector(), interval=30,
                                   max_seconds=3600, stop=stop)
        t0 = time.monotonic()
        first = next(gen)  # first tick arrives immediately
        self.assertTrue(first.startswith("event: tick"))
        stop.set()  # shutdown mid-pause: the 30s interval must not run out
        self.assertRaises(StopIteration, next, gen)
        self.assertLess(time.monotonic() - t0, 5)

    def test_stream_never_starts_after_shutdown(self):
        stop = threading.Event()
        stop.set()
        gen = telemetry.sse_stream(self._collector(), stop=stop)
        self.assertRaises(StopIteration, next, gen)

    def test_job_stream_ends_between_polls_on_shutdown(self):
        with TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs.db", scratch_safe=True)
            job_id = store.create("bench", "b", lane="benchmark")
            store.update(job_id, status="running")
            stop = threading.Event()
            gen = sse_job_stream(store, job_id, stop=stop)
            t0 = time.monotonic()
            first = next(gen)
            self.assertIn("running", first)
            stop.set()
            self.assertRaises(StopIteration, next, gen)
            self.assertLess(time.monotonic() - t0, 5)

    def test_module_event_is_the_default(self):
        # request_shutdown() must reach generators that took the default.
        self.addCleanup(telemetry.SHUTDOWN.clear)
        telemetry.request_shutdown()
        gen = telemetry.sse_stream(self._collector())
        self.assertRaises(StopIteration, next, gen)


# ---- model hub API -------------------------------------------------------

class HubBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        self.model_path = write_tiny_gguf(self.root / "tiny.gguf")
        self.artifacts = self.profiles_dir / "m1"
        self.artifacts.mkdir()
        profile = {
            "name": "m1", "repo_id": "r/m", "file": "tiny.gguf",
            "model_path": str(self.model_path),
            "mmproj_path": None, "mtp_path": None,
            "artifacts_dir": str(self.artifacts),
            "config": {"device": "", "split_mode": "layer",
                       "tensor_split": "3,1", "ctx": 1024,
                       "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                       "flash_attn": "auto", "ttl": 60, "mtp": "off",
                       "fit": "off", "extra": ""},
            "env": [], "enabled": True,
            "moe_cache": {"mode": "off", "gpu": {"budgets_bytes": {}}},
            # Stored planning inputs make the admission preview
            # deterministic: no live probes, a fixed two-GPU inventory.
            "planning": {
                "inputs": modelctl_tiers.make_planning_inputs(
                    inventory=[
                        {"device": "SYCL0", "name": "Fixture GPU 0",
                         "total_bytes": 8 << 30},
                        {"device": "SYCL1", "name": "Fixture GPU 1",
                         "total_bytes": 4 << 30}],
                    limit_pct=90, primary="SYCL0",
                    ram_available_bytes=16 << 30),
                "recorded_at": "2026-08-01T00:00:00"},
        }
        (self.profiles_dir / "m1.json").write_text(json.dumps(profile))
        no_binaries = str(self.root / "no-binaries" / "*")
        self.runtime_db_path = self.root / "runtime.db"
        real_runtime_db = modelctl_runtime.RuntimeDB

        def _redirected_runtime_db(db_path=None):
            return real_runtime_db(db_path=db_path or self.runtime_db_path)

        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
                  mock.patch.object(modelctl_runtime, "RuntimeDB",
                                    _redirected_runtime_db),
                  mock.patch.object(modelctl, "COMMON_LLAMA_SERVER_GLOBS",
                                    [no_binaries]),
                  mock.patch.object(modelctl, "COMMON_ENV_SCRIPT_GLOBS",
                                    [no_binaries]),
                  mock.patch.object(wiz, "WIZARD_DIR", self.root / "wizards")):
            p.start()
            self.addCleanup(p.stop)
        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.store = store
        self.runner = runner
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        self.client = TestClient(create_app(store=store, runner=runner))
        self.auth = {}

    def wait_job(self, job_id, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.client.get(f"/api/jobs/{job_id}",
                                headers=self.auth).json()
            if j["status"] not in ("queued", "running"):
                return j
            time.sleep(0.1)
        self.fail(f"job {job_id} never finished")


class TestModelDetail(HubBase):
    def test_detail_shape(self):
        r = self.client.get("/api/v2/models/m1", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["name"], "m1")
        self.assertEqual(d["config"]["tensor_split"], "3,1")
        self.assertEqual(d["moe_cache"]["mode"], "off")
        self.assertIn("state", d["runtime"])
        self.assertTrue(d["planning"]["has_stored_inputs"])
        self.assertGreater(d["size_bytes"], 0)
        json.dumps(d)  # SPA contract: serializable verbatim

    def test_unknown_model_is_404(self):
        r = self.client.get("/api/v2/models/nope", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_history_empty_but_typed(self):
        r = self.client.get("/api/v2/models/m1/history", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), [])

    def test_history_rows_carry_bottleneck_judgement(self):
        rdb = modelctl_runtime.RuntimeDB()
        rdb.record_plan_run({
            "profile_name": "m1", "plan_id": "p1", "plan_label": "test plan",
            "run_kind": "benchmark", "success": True,
            "generation_tps": 12.5, "prompt_tps": 80.0,
            "load_seconds": 3.5, "cache_state": "cold"})
        rows = self.client.get("/api/v2/models/m1/history",
                               headers=self.auth).json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plan_id"], "p1")
        self.assertTrue(rows[0]["success"])
        self.assertIn("bottleneck", rows[0])

    def test_logtail_degrades_when_no_log_exists(self):
        r = self.client.get("/api/v2/models/m1/logtail", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["tail"], "")
        self.assertIn("no log yet", body["error"])

    def test_logtail_reads_artifact_logs(self):
        (self.artifacts / "test.log").write_text(
            "\n".join(f"line {i}" for i in range(200)))
        body = self.client.get("/api/v2/models/m1/logtail?lines=50",
                               headers=self.auth).json()
        self.assertEqual(body["source"], "test.log")
        self.assertEqual(len(body["tail"].splitlines()), 50)
        self.assertIn("line 199", body["tail"])


class TestAdmissionPreview(HubBase):
    def test_preview_uses_stored_inputs(self):
        r = self.client.get("/api/v2/models/m1/admission", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["planning_inputs_source"], "stored")
        self.assertIsNotNone(body["plan"])
        adm = body["plan"]["admission"]
        self.assertIn("requested", adm)
        self.assertIn("assumed", adm)
        self.assertIn("chosen", adm)
        self.assertIn("SYCL0", adm["devices"])

    def test_draft_ctx_changes_the_preview_not_the_profile(self):
        before = json.loads((self.profiles_dir / "m1.json").read_text())
        r = self.client.get("/api/v2/models/m1/admission?ctx=2048",
                            headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json()["plan"]["admission"]["requested"]["ctx"], 2048)
        after = json.loads((self.profiles_dir / "m1.json").read_text())
        self.assertEqual(before, after)

    def test_gate_compares_against_the_saved_profile(self):
        body = self.client.get("/api/v2/models/m1/admission",
                               headers=self.auth).json()
        self.assertIn(body["gate"]["kind"], ("none", "pins", "structural"))

    def test_bad_ctx_is_422(self):
        r = self.client.get("/api/v2/models/m1/admission?ctx=lots",
                            headers=self.auth)
        self.assertEqual(r.status_code, 422)

    def test_draft_moe_mode_makes_budgets_count(self):
        # The stored profile has moe_cache off; without the draft mode
        # the planner would drop the budgets and predict a cache-free
        # fit the real (post-save) plan won't have.
        r = self.client.get(
            "/api/v2/models/m1/admission"
            "?moe_mode=auto&budget_bytes.SYCL0=1073741824",
            headers=self.auth)
        self.assertEqual(r.status_code, 200)
        adm = r.json()["plan"]["admission"]
        self.assertEqual(adm["requested"]["cache_budgets_bytes"],
                         {"SYCL0": 1 << 30})


class TestConfigureSave(HubBase):
    def test_unknown_field_rejected(self):
        r = self.client.post("/api/v2/models/m1/config", headers=self.auth,
                             json={"updates": {"rm_rf": "yes"}})
        self.assertEqual(r.status_code, 422)

    def test_structural_change_needs_the_explicit_confirm(self):
        r = self.client.post("/api/v2/models/m1/config", headers=self.auth,
                             json={"updates": {"tensor_split": "1,3"}})
        self.assertEqual(r.status_code, 409)
        gate = r.json()["gate"]
        self.assertEqual(gate["kind"], "structural")
        self.assertTrue(gate["requires_accept"])
        self.assertTrue(any("tensor_split" in c for c in gate["changes"]))

    def test_cache_budget_change_is_structural(self):
        r = self.client.post("/api/v2/models/m1/config", headers=self.auth,
                             json={"budgets_bytes": {"SYCL0": 1 << 30},
                                   "moe_mode": "auto"})
        self.assertEqual(r.status_code, 409)
        self.assertTrue(any("cache budgets" in c
                            for c in r.json()["gate"]["changes"]))

    def test_moe_mode_flip_alone_is_structural(self):
        r = self.client.post("/api/v2/models/m1/config", headers=self.auth,
                             json={"moe_mode": "auto"})
        self.assertEqual(r.status_code, 409)
        self.assertTrue(any("moe_cache mode" in c
                            for c in r.json()["gate"]["changes"]))

    def test_malformed_json_body_is_422_not_500(self):
        r = self.client.post("/api/v2/models/m1/config", headers=self.auth,
                             content=b"not json")
        self.assertEqual(r.status_code, 422)

    def test_non_structural_save_submits_the_edit_job(self):
        with mock.patch("modelctl_web.mutate.submit_edit",
                        return_value="job-edit") as sub:
            r = self.client.post("/api/v2/models/m1/config",
                                 headers=self.auth,
                                 json={"updates": {"ctx": "2048"}})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["jobs"]["config"], "job-edit")
        self.assertIsNone(body["jobs"]["moe_cache"])
        sub.assert_called_once_with(self.runner, "m1", {"ctx": "2048"})

    def test_accepted_structural_save_submits_both_jobs(self):
        with mock.patch("modelctl_web.mutate.submit_edit",
                        return_value="job-edit"), \
             mock.patch("modelctl_web.mutate.submit_moe_cache",
                        return_value="job-moe") as moe:
            r = self.client.post(
                "/api/v2/models/m1/config", headers=self.auth,
                json={"updates": {"tensor_split": "1,3"},
                      "budgets_bytes": {"SYCL0": 1 << 30},
                      "moe_mode": "auto",
                      "accept_structural": True})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["jobs"]["config"], "job-edit")
        self.assertEqual(body["jobs"]["moe_cache"], "job-moe")
        mc = moe.call_args.args[2]
        self.assertEqual(mc["mode"], "auto")
        self.assertEqual(mc["gpu"]["budgets_bytes"], {"SYCL0": 1 << 30})

    def test_plans_endpoint_returns_tagged_rows(self):
        r = self.client.get("/api/v2/models/m1/plans", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        rows = r.json()
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn(row["category"], (
                "baseline", "tested", "untested", "experimental",
                "unavailable"))
            self.assertIn("measured", row)
            self.assertIn("estimated", row)
            self.assertIsInstance(row["warnings"], list)


# ---- add wizard API ------------------------------------------------------

class WizardApiBase(HubBase):
    def new_wizard(self):
        r = self.client.post("/api/v2/wizards", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        return r.json()["wizard_id"]


class TestWizardApiFlow(WizardApiBase):
    def test_create_and_list(self):
        wid = self.new_wizard()
        listed = self.client.get("/api/v2/wizards", headers=self.auth).json()
        self.assertIn(wid, [w["wizard_id"] for w in listed])

    def test_detail_carries_steps_and_gates(self):
        wid = self.new_wizard()
        d = self.client.get(f"/api/v2/wizard/{wid}", headers=self.auth).json()
        self.assertEqual(d["step"], "source")
        self.assertEqual(d["steps"][0], "source")
        self.assertIn("download", d["step_gates"])

    def test_unknown_wizard_404(self):
        r = self.client.get("/api/v2/wizard/nope", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_bad_repo_id_is_a_422_with_the_reason(self):
        wid = self.new_wizard()
        r = self.client.post(f"/api/v2/wizard/{wid}/source",
                             headers=self.auth,
                             json={"source_type": "hf_repo",
                                   "repo_id": "not-a-repo"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("not a repo id", r.json()["error"])

    def test_malformed_wizard_body_is_422_not_500(self):
        wid = self.new_wizard()
        r = self.client.post(f"/api/v2/wizard/{wid}/source",
                             headers=self.auth, content=b"{broken")
        self.assertEqual(r.status_code, 422)
        r = self.client.post(f"/api/v2/wizard/{wid}/plans",
                             headers=self.auth, json=["a", "list"])
        self.assertEqual(r.status_code, 422)

    def test_bad_local_file_rejected_before_any_profile(self):
        wid = self.new_wizard()
        bad = self.root / "notes.txt"
        bad.write_bytes(b"not a model" * 200)
        r = self.client.post(f"/api/v2/wizard/{wid}/source",
                             headers=self.auth,
                             json={"source_type": "local_file",
                                   "local_path": str(bad)})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(len(list(self.profiles_dir.glob("m2*.json"))), 0)

    def test_valid_local_gguf_advances_and_submits_import_once(self):
        wid = self.new_wizard()
        with mock.patch("modelctl_web.mutate.submit_import_local",
                        return_value="job-import") as sub:
            r = self.client.post(f"/api/v2/wizard/{wid}/source",
                                 headers=self.auth,
                                 json={"source_type": "local_file",
                                       "local_path": str(self.model_path)})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["step"], "download")
        self.assertEqual(d["download_job_id"], "job-import")
        sub.assert_called_once()

    def _wizard_at_download(self, job_status, outcome=None, error=""):
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        job_id = self.store.create("import", "import tiny", lane="mutation")
        self.store.update(job_id, status=job_status, error=error,
                          outcome=json.dumps(outcome) if outcome else "")
        state.source_type = "local_file"
        state.local_path = str(self.model_path)
        state.download_job_id = job_id
        state.advance("download")
        wiz.WizardStore().save(state)
        return wid, job_id

    def test_blocked_advance_names_the_reason(self):
        wid, _ = self._wizard_at_download("running")
        r = self.client.post(f"/api/v2/wizard/{wid}/download/next",
                             headers=self.auth)
        self.assertEqual(r.status_code, 409)
        self.assertIn("still running", r.json()["error"])

    def test_failed_download_blocks_with_the_error(self):
        wid, _ = self._wizard_at_download("failed", error="disk full")
        r = self.client.post(f"/api/v2/wizard/{wid}/download/next",
                             headers=self.auth)
        self.assertEqual(r.status_code, 409)
        self.assertIn("disk full", r.json()["error"])

    def test_done_download_advances_and_analyze_reads_the_header(self):
        wid, _ = self._wizard_at_download(
            "done", outcome={"profile": "m1"})
        r = self.client.post(f"/api/v2/wizard/{wid}/download/next",
                             headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["step"], "analyze")
        a = self.client.get(f"/api/v2/wizard/{wid}/analyze",
                            headers=self.auth).json()
        self.assertEqual(a["profile"]["name"], "m1")
        self.assertEqual(a["analysis"]["model_max_ctx"], 2048)
        self.assertEqual(a["analysis"]["arch"], "llama")
        # The header facts persist into wizard state for register.
        d = self.client.get(f"/api/v2/wizard/{wid}", headers=self.auth).json()
        self.assertEqual(d["analysis"]["model_max_ctx"], 2048)

    def test_retry_download_reissues_the_same_job_params(self):
        wid, _ = self._wizard_at_download("failed", error="disk full")
        with mock.patch("modelctl_web.mutate.submit_import_local",
                        return_value="job-retry") as sub:
            r = self.client.post(f"/api/v2/wizard/{wid}/retry/download",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["step"], "download")
        self.assertEqual(d["download_job_id"], "job-retry")
        # Identical params: the same local path the wizard already holds.
        self.assertEqual(sub.call_args.args[1], str(self.model_path))
        self.assertEqual(d["step_gates"]["download"]["outcome"], {})

    def _wizard_at_register(self, with_test="none"):
        wid, _ = self._wizard_at_download("done", outcome={"profile": "m1"})
        state = wiz.WizardStore().load(wid)
        state.profile_name = "m1"
        state.analysis = {"model_max_ctx": 2048, "arch": "llama"}
        if with_test == "running":
            job_id = self.store.create("benchmark", "plan test", lane="benchmark")
            self.store.update(job_id, status="running")
            state.test_job_id = job_id
        elif with_test == "failed":
            job_id = self.store.create("benchmark", "plan test", lane="benchmark")
            self.store.update(job_id, status="failed", error="OOM")
            state.test_job_id = job_id
            state.record_outcome("test", ok=False, job_id=job_id,
                                 status="failed", error="OOM")
        state.advance("register")
        wiz.WizardStore().save(state)
        return wid

    def test_register_form_data_binds_to_real_analysis_and_admission(self):
        wid = self._wizard_at_register()
        r = self.client.get(f"/api/v2/wizard/{wid}/register",
                            headers=self.auth)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["analysis"]["model_max_ctx"], 2048)
        self.assertEqual(d["profile"]["name"], "m1")
        self.assertIsNotNone(d["admission"]["plan"])
        self.assertIn("devices", d["admission"]["plan"]["admission"])

    def test_register_refuses_ctx_above_the_header_maximum(self):
        wid = self._wizard_at_register()
        r = self.client.post(f"/api/v2/wizard/{wid}/register",
                             headers=self.auth, json={"ctx": 999999})
        self.assertEqual(r.status_code, 422)
        self.assertIn("trained maximum", r.json()["error"])
        self.assertIn("2048", r.json()["error"])

    def test_register_ctx_check_survives_missing_persisted_analysis(self):
        # A wizard that never rendered the analyze page has no stored
        # analysis; the register POST must read the header itself rather
        # than skipping the ceiling.
        wid = self._wizard_at_register()
        state = wiz.WizardStore().load(wid)
        state.analysis = {}
        wiz.WizardStore().save(state)
        r = self.client.post(f"/api/v2/wizard/{wid}/register",
                             headers=self.auth, json={"ctx": 999999})
        self.assertEqual(r.status_code, 422)
        self.assertIn("trained maximum", r.json()["error"])

    def test_register_refuses_while_the_test_runs(self):
        wid = self._wizard_at_register(with_test="running")
        r = self.client.post(f"/api/v2/wizard/{wid}/register",
                             headers=self.auth, json={})
        self.assertEqual(r.status_code, 409)
        self.assertIn("still running", r.json()["error"])

    def test_register_refuses_failed_test_without_the_optout(self):
        wid = self._wizard_at_register(with_test="failed")
        r = self.client.post(f"/api/v2/wizard/{wid}/register",
                             headers=self.auth, json={})
        self.assertEqual(r.status_code, 409)
        self.assertTrue(r.json()["requires_register_untested"])

    def test_register_completes_with_the_optout(self):
        wid = self._wizard_at_register(with_test="failed")
        with mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"), \
             mock.patch("modelctl_web.mutate.submit_load",
                        return_value="job-load") as load:
            r = self.client.post(
                f"/api/v2/wizard/{wid}/register", headers=self.auth,
                json={"ctx": 1024, "register_untested": True})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["step"], "done")
        self.assertTrue(d["registration_complete"])
        self.assertTrue(d["endpoint"].endswith("/chat/completions"))
        load.assert_called_once()
        # The ctx override actually landed in the profile.
        saved = json.loads((self.profiles_dir / "m1.json").read_text())
        self.assertEqual(int(saved["config"]["ctx"]), 1024)


class TestSpaServesNewRoutes(HubBase):
    def test_new_routes_fall_back_to_the_spa(self):
        for route in ("/v2/models", "/v2/models/m1", "/v2/add",
                      "/v2/add/abc123"):
            r = self.client.get(route, headers=self.auth)
            self.assertEqual(r.status_code, 200, route)
            self.assertIn('<div id="app">', r.text)


if __name__ == "__main__":
    unittest.main()
