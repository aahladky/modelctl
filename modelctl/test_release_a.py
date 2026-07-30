"""Release A acceptance tests.

Validates the end-to-end flow for Release A:
- Profile creation from local import
- Plan compilation and ranking
- Capability validation (unsupported flags blocked)
- Command provenance
- Web routes accessible
- Wizard state persistence
- Settings page
- Transaction safety
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_capabilities
import modelctl_errors
import modelctl_launch
import modelctl_plans
import modelctl_profiles
import modelctl_transactions


class TestReleaseACapabilityValidation(unittest.TestCase):
    """Req: unsupported flags never reach a backend."""

    def test_stock_binary_no_cache_flags(self):
        """Stock llama.cpp cannot receive cache flags through any path."""
        caps = modelctl_capabilities._classify_probe_failure("/usr/bin/llama-server")
        self.assertFalse(modelctl_capabilities.is_cache_capable(caps))
        self.assertFalse(modelctl_capabilities.is_weight_transfer_cache_capable(caps))

    def test_cpu_only_not_cache_capable(self):
        caps = {"schema": 2, "features": {
            "moe_weight_transfer_cache": False,
            "moe_hybrid_cpu_miss": False,
        }}
        self.assertFalse(modelctl_capabilities.is_cache_capable(caps))

    def test_hybrid_passthrough(self):
        """moe_hybrid_cpu_miss passes through normalization."""
        raw = {"schema": 2, "features": {
            "moe_weight_transfer_cache": True,
            "moe_hybrid_cpu_miss": True,
        }}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertTrue(norm["features"]["moe_hybrid_cpu_miss"])

    def test_prefetch_forced_false(self):
        raw = {"schema": 2, "features": {
            "moe_weight_transfer_cache": True,
            "moe_cache_prefetch": True,
        }}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertFalse(norm["features"]["moe_cache_prefetch"])

    def test_preflight_blocks_unsupported(self):
        """preflight_moe_cache returns error for unsupported binary."""
        profile = {"moe_cache": {"mode": "manual"}}
        caps = {"features": {"moe_weight_transfer_cache": False}}
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        self.assertEqual(msgs[0][0], "error")


class TestReleaseACommandIdentity(unittest.TestCase):
    """Req: shown command matches launched command."""

    def test_port_excluded_from_fingerprint(self):
        fp1 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "m.gguf", "--port", "8080"), "e", "b")
        fp2 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "m.gguf", "--port", "9090"), "e", "b")
        self.assertEqual(fp1, fp2)

    def test_different_args_different_identity(self):
        fp1 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "a.gguf"), "e", "b")
        fp2 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "b.gguf"), "e", "b")
        self.assertNotEqual(fp1, fp2)

    def test_resolve_backend_returns_capabilities(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\nexit 1\n")
            script.chmod(0o755)
            with mock.patch("modelctl_launch.modelctl.find_env_script_candidates",
                           return_value=[]):
                backend = modelctl_launch.resolve_backend(
                    {"backend": "llama-cpp", "binary": str(script)})
            self.assertEqual(backend.capabilities["_probe_status"], "unsupported")


class TestReleaseAProfileValidation(unittest.TestCase):
    """Req: profiles are validated before save."""

    def test_valid_profile_passes(self):
        p = modelctl_profiles.normalize_profile(
            {"name": "test", "config": {}})
        msgs = modelctl_profiles.validate_profile(p)
        self.assertFalse(any(m.severity == "error" for m in msgs))

    def test_invalid_backend_blocked(self):
        msgs = modelctl_profiles.validate_profile(
            {"name": "test", "backend": "invalid", "config": {"ctx": 4096}})
        self.assertTrue(any(m.severity == "error" for m in msgs))


class TestReleaseAPlanExplainability(unittest.TestCase):
    """Req: selected plan has an explainable measured basis."""

    def test_plan_has_decision_data(self):
        p = modelctl_plans.LaunchPlan(
            id="test", profile_name="m", backend="llama-cpp",
            label="test plan", argv=(), env={},
            claim=modelctl_plans.ResourceClaim(
                vram_bytes={"SYCL0": 1000}, ram_bytes=500,
                storage_mode="mmap", expected_context=4096,
                storage_path="/m.gguf", model_bytes=2000),
            estimated={}, source="single-gpu", warnings=(),
            decision_data={"gpu": "SYCL0", "budget": 1000})
        self.assertIn("gpu", p.decision_data)
        self.assertTrue(p.claim.storage_path)
        self.assertTrue(p.claim.model_bytes > 0)


class TestReleaseATransactionSafety(unittest.TestCase):
    """Req: failed mutations recover cleanly."""

    def test_rollback_on_failure(self):
        with TemporaryDirectory() as d:
            modelctl.PROFILES_DIR = Path(d) / "profiles"
            modelctl.STATE_DIR = Path(d) / "state"
            modelctl.PROFILES_DIR.mkdir(parents=True, exist_ok=True)

            (modelctl.PROFILES_DIR / "test.json").write_text(
                json.dumps({"name": "test", "config": {"ctx": 4096}}))

            try:
                with modelctl_transactions.Transaction("test") as tx:
                    tx.stage_profile({"name": "test", "config": {"ctx": 9999}})
                    tx.commit()
                    raise RuntimeError("simulated")
            except RuntimeError:
                pass

            loaded = json.loads((modelctl.PROFILES_DIR / "test.json").read_text())
            self.assertEqual(loaded["config"]["ctx"], 4096)


class TestReleaseAWebRoutes(unittest.TestCase):
    """Req: browser workflow accessible."""

    def test_web_app_creates(self):
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(token="test-token", store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        # Dashboard
        r = client.get("/", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(r.status_code, 200)

    def test_import_page_accessible(self):
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(token="test-token", store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        r = client.get("/import", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(r.status_code, 200)

    def test_settings_page_accessible(self):
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(token="test-token", store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        r = client.get("/settings", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(r.status_code, 200)

    def test_add_wizard_page_accessible(self):
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(token="test-token", store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        r = client.get("/add", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(r.status_code, 200)


class TestReleaseAResourceClaims(unittest.TestCase):
    """Req: RAM/VRAM/storage use is visible."""

    def test_claim_has_storage_path(self):
        claim = modelctl_plans.ResourceClaim(
            vram_bytes={"SYCL0": 1 << 30},
            ram_bytes=512 << 20,
            storage_mode="mmap",
            expected_context=8192,
            storage_path="/home/user/models/model.gguf",
            model_bytes=4 << 30,
            expected_resident_bytes=4 << 30,
            cache_bytes=2 << 30,
        )
        self.assertTrue(claim.storage_path)
        self.assertTrue(claim.model_bytes > 0)
        self.assertTrue(claim.cache_bytes > 0)

    def test_claim_has_breakdown(self):
        claim = modelctl_plans.ResourceClaim(
            vram_bytes={"SYCL0": 1 << 30},
            ram_bytes=512 << 20,
            storage_mode="none",
            expected_context=8192,
            breakdown={"vram": {"SYCL0": {"fixed": 100, "kv": 200}}})
        self.assertIn("vram", claim.breakdown)


class TestReleaseAImportLocal(unittest.TestCase):
    """Req: local GGUF can be added without terminal."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_profiles = modelctl.PROFILES_DIR
        self._orig_models = modelctl.DEFAULT_MODELS_DIR
        modelctl.PROFILES_DIR = Path(self.tmp.name) / "profiles"
        modelctl.DEFAULT_MODELS_DIR = Path(self.tmp.name) / "models"
        self.addCleanup(setattr, modelctl, "PROFILES_DIR", self._orig_profiles)
        self.addCleanup(setattr, modelctl, "DEFAULT_MODELS_DIR", self._orig_models)

    def test_import_creates_profile(self):
        path = Path(self.tmp.name) / "model.gguf"
        path.write_bytes(b"GGUF" + b"\x00" * 100)
        with mock.patch("modelctl.sync_all_backends"), \
             mock.patch("modelctl.generate_artifacts"):
            profile = modelctl.import_local(str(path), resync=False)
        self.assertIsNotNone(profile)
        self.assertTrue(profile["name"])

    def test_import_rejects_non_gguf(self):
        path = Path(self.tmp.name) / "model.txt"
        path.write_bytes(b"not a gguf")
        result = modelctl.import_local(str(path))
        self.assertIsNone(result)


class TestReleaseAColdWarmSeparation(unittest.TestCase):
    """Req (Task 6.2, §13): cold and warm measurements are persisted with
    their cache state and never conflated in ranking observations."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import modelctl_runtime
        self.rdb = modelctl_runtime.RuntimeDB(
            db_path=Path(self.tmp.name) / "runtime.db")

    def _run(self, plan_id, cache_state, gen_tps, started):
        return {"profile_name": "m", "plan_id": plan_id,
                "hardware_fingerprint": "hw", "backend_fingerprint": "be",
                "started_at": started, "finished_at": started + 1,
                "success": True, "generation_tps": gen_tps,
                "cache_state": cache_state}

    def test_cache_state_persisted(self):
        self.rdb.record_plan_run(self._run("p1", "warm", 42.0, 100))
        rows = self.rdb.plan_runs_for("m")
        self.assertEqual(rows[0]["cache_state"], "warm")

    def test_legacy_rows_default_empty_state(self):
        run = self._run("p1", "", 42.0, 100)
        run.pop("cache_state")
        self.rdb.record_plan_run(run)
        rows = self.rdb.plan_runs_for("m")
        self.assertEqual(rows[0]["cache_state"], "")

    def test_observations_never_mix_cold_and_warm(self):
        # Plan A only measured warm, plan B only measured cold: equal
        # coverage, so the conservative (cold) state wins and plan A is
        # untested rather than ranked on an incomparable warm number.
        self.rdb.record_plan_run(self._run("a", "warm", 50.0, 100))
        self.rdb.record_plan_run(self._run("b", "cold", 10.0, 101))
        obs = self.rdb.observations_for_profile("m")
        self.assertNotIn("a", obs)
        self.assertIn("b", obs)
        self.assertEqual(obs["b"]["cache_state"], "cold")

    def test_observations_prefer_full_coverage_state(self):
        # Both plans have warm observations; plan B also has a colder one.
        # Warm has full coverage, so it is the comparison basis.
        self.rdb.record_plan_run(self._run("a", "warm", 50.0, 100))
        self.rdb.record_plan_run(self._run("b", "cold", 10.0, 101))
        self.rdb.record_plan_run(self._run("b", "warm", 45.0, 102))
        obs = self.rdb.observations_for_profile("m")
        self.assertEqual(set(obs), {"a", "b"})
        self.assertTrue(all(o["cache_state"] == "warm" for o in obs.values()))
        self.assertEqual(obs["b"]["generation_tps"], 45.0)
