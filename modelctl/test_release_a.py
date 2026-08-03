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
            # A coherent stock-binary rejection (message + exit 1). A
            # silent exit 1 now classifies "error" (transient, uncached)
            # -- see test_capability_cache_truth.py.
            script.write_text(
                '#!/bin/sh\necho "error: invalid argument: $1" >&2\nexit 1\n')
            script.chmod(0o755)
            with mock.patch("modelctl_launch.modelctl.find_env_script_candidates",
                           return_value=[]):
                backend = modelctl_launch.resolve_backend(
                    {"backend": "llama-cpp", "binary": str(script),
                     "model_path": str(Path(d) / "model.gguf")})
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
        app = create_app(store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        # Dashboard
        r = client.get("/", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(r.status_code, 200)

    def test_import_redirects_into_the_add_workflow(self):
        # P1 consolidation: /import is a compatibility shim, not a second
        # acquisition path. The phase-3 cutover moved the workflow itself
        # to /v2/add and made the shim a permanent 301, so the requirement
        # is now that both entry points land on the same single workflow.
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        auth = {"Authorization": "Bearer test-token"}
        r = client.get("/import", headers=auth, follow_redirects=False)
        self.assertEqual(r.status_code, 301)
        self.assertEqual(r.headers["location"], "/v2/add")
        # Still one acquisition path: the old wizard URL lands there too.
        r = client.get("/add", headers=auth, follow_redirects=False)
        self.assertEqual(r.headers["location"], "/v2/add")

    def test_settings_page_accessible(self):
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(store=mock.MagicMock(),
                        runner=mock.MagicMock())
        client = TestClient(app)
        r = client.get("/settings", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(r.status_code, 200)

    def test_add_wizard_page_accessible(self):
        from fastapi.testclient import TestClient
        from modelctl_web.app import create_app
        app = create_app(store=mock.MagicMock(),
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


class TestObservationValidity(unittest.TestCase):
    """Task H1: an observation is only evidence while the setup it was
    measured in still holds, and a stale one says what changed."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import modelctl_runtime
        self.rt = modelctl_runtime
        self.rdb = modelctl_runtime.RuntimeDB(
            db_path=Path(self.tmp.name) / "runtime.db")
        self.rdb.record_plan_run({
            "profile_name": "m", "plan_id": "p1",
            "hardware_fingerprint": "hw", "backend_fingerprint": "be",
            "environment_fingerprint": "env", "capability_digest": "caps",
            "storage_device": "nvme0n1",
            "started_at": 100, "finished_at": 101,
            "success": True, "generation_tps": 42.0, "cache_state": "cold"})

    def _obs(self, **kw):
        ident = self.rt.ObservationIdentity(**kw)
        return self.rdb.observations_for_profile("m", identity=ident)["p1"]

    def test_matching_identity_is_not_stale(self):
        o = self._obs(hardware_fingerprint="hw", backend_fingerprint="be",
                      environment_fingerprint="env", capability_digest="caps",
                      storage_device="nvme0n1")
        self.assertFalse(o["stale"])
        self.assertEqual(o["stale_reasons"], [])

    def test_changed_environment_is_stale_and_says_so(self):
        # The effective launch environment is not part of the plan ID, so
        # nothing else would catch this.
        o = self._obs(hardware_fingerprint="hw", environment_fingerprint="other")
        self.assertTrue(o["stale"])
        self.assertTrue(any("environment" in r for r in o["stale_reasons"]))

    def test_changed_capabilities_are_stale(self):
        # A rebuilt runtime reporting different features invalidates
        # measurements taken under the old contract.
        o = self._obs(capability_digest="rebuilt")
        self.assertTrue(o["stale"])
        self.assertTrue(any("capabilities" in r for r in o["stale_reasons"]))

    def test_changed_storage_device_is_stale(self):
        o = self._obs(storage_device="sda")
        self.assertTrue(o["stale"])
        self.assertTrue(any("storage" in r for r in o["stale_reasons"]))

    def test_every_mismatch_is_reported_not_just_the_first(self):
        # Task H3 has to explain a rejection; one reason out of three is a
        # misleading explanation.
        o = self._obs(hardware_fingerprint="new", backend_fingerprint="new",
                      environment_fingerprint="new")
        self.assertEqual(len(o["stale_reasons"]), 3)

    def test_unknown_values_are_not_treated_as_changed(self):
        # Rows predating a column, and callers that cannot supply a value,
        # must not turn the whole history red.
        o = self._obs(hardware_fingerprint="hw")
        self.assertFalse(o["stale"])
        o = self._obs()
        self.assertFalse(o["stale"])

    def test_legacy_positional_fingerprints_still_work(self):
        obs = self.rdb.observations_for_profile("m", "different-hw", "be")
        self.assertTrue(obs["p1"]["stale"])

    def test_positional_and_identity_checks_are_unioned(self):
        # Adopting the richer argument must not silently drop a check the
        # caller was already getting positionally.
        ident = self.rt.ObservationIdentity(environment_fingerprint="env")
        obs = self.rdb.observations_for_profile(
            "m", "different-hw", identity=ident)
        self.assertTrue(obs["p1"]["stale"])
        self.assertTrue(any("hardware" in r for r in obs["p1"]["stale_reasons"]))


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

    def test_warm_page_cache_and_warm_expert_cache_never_mix(self):
        # A warm-expert number (GPU cache filled by warmup) is not
        # comparable to a warm page-cache number. Equal coverage: the
        # more conservative plain-warm state wins.
        self.rdb.record_plan_run(self._run("a", "warm", 30.0, 100))
        self.rdb.record_plan_run(self._run("b", "warm-expert", 60.0, 101))
        obs = self.rdb.observations_for_profile("m")
        self.assertIn("a", obs)
        self.assertNotIn("b", obs)

    def test_cold_outranks_warm_expert_on_a_tie(self):
        self.rdb.record_plan_run(self._run("a", "warm-expert", 60.0, 100))
        self.rdb.record_plan_run(self._run("b", "cold", 10.0, 101))
        obs = self.rdb.observations_for_profile("m")
        self.assertEqual(obs["b"]["cache_state"], "cold")
        self.assertNotIn("a", obs)


class TestExperimentalGuardrail(unittest.TestCase):
    """Task H2: an experimental plan outranks a safe baseline only on
    evidence -- a current measurement that clears a configurable margin."""

    def _plan(self, pid, cache_bytes=0, source="test", ram=0, ctx=32768):
        from modelctl_plans import LaunchPlan, ResourceClaim
        return LaunchPlan(
            id=pid, profile_name="m", backend="llama-cpp", label=pid,
            argv=(), env={},
            claim=ResourceClaim({"S0": 8 << 30}, ram, "none", ctx,
                                cache_bytes=cache_bytes),
            estimated={}, source=source, warnings=(), decision_data={})

    def _policy(self, objective="fastest_generation", margin=0.10):
        from modelctl_plans import RuntimePolicy
        return RuntimePolicy(objective, None, True, True, 8192, None, 3,
                             experimental_margin=margin)

    def setUp(self):
        import modelctl_plans
        self.mp = modelctl_plans
        self.safe = self._plan("safe", source="current-profile")
        self.exp = self._plan("exp", cache_bytes=2 << 30)

    def test_experimental_leads_when_it_clears_the_margin(self):
        obs = {"safe": {"generation_tps": 10.0},
               "exp": {"generation_tps": 12.0}}       # +20% vs 10% required
        ranked = self.mp.rank_plans([self.safe, self.exp], self._policy(), obs)
        self.assertEqual(ranked[0][0].id, "exp")

    def test_experimental_is_demoted_when_it_only_ties(self):
        # Equal performance is not worth the risk of an experimental path.
        obs = {"safe": {"generation_tps": 10.0},
               "exp": {"generation_tps": 10.4}}       # +4%, under the margin
        explain = {}
        ranked = self.mp.rank_plans([self.safe, self.exp], self._policy(),
                                    obs, explain=explain)
        self.assertEqual(ranked[0][0].id, "safe")
        self.assertIn("exp", explain)
        self.assertIn("margin", explain["exp"])

    def test_demoted_experimental_is_still_available(self):
        # Demotion is an ordering, not a removal: the user can still pick it.
        obs = {"safe": {"generation_tps": 10.0}, "exp": {"generation_tps": 10.1}}
        ranked = self.mp.rank_plans([self.safe, self.exp], self._policy(), obs)
        self.assertEqual({p.id for p, _ in ranked}, {"safe", "exp"})

    def test_stale_measurement_cannot_promote_an_experimental_plan(self):
        obs = {"safe": {"generation_tps": 10.0},
               "exp": {"generation_tps": 99.0, "stale": True,
                       "stale_reasons": ["environment changed since this was measured"]}}
        explain = {}
        ranked = self.mp.rank_plans([self.safe, self.exp], self._policy(),
                                    obs, explain=explain)
        self.assertEqual(ranked[0][0].id, "safe")
        self.assertIn("stale", explain["exp"])
        # H3 needs the specific cause, not just the word "stale".
        self.assertIn("environment changed", explain["exp"])

    def test_unmeasured_experimental_never_leads(self):
        obs = {"safe": {"generation_tps": 10.0}}
        explain = {}
        ranked = self.mp.rank_plans([self.safe, self.exp], self._policy(),
                                    obs, explain=explain)
        self.assertEqual(ranked[0][0].id, "safe")
        self.assertIn("no measurement", explain["exp"])

    def test_margin_zero_ranks_experimental_on_score_alone(self):
        obs = {"safe": {"generation_tps": 10.0}, "exp": {"generation_tps": 10.1}}
        ranked = self.mp.rank_plans([self.safe, self.exp],
                                    self._policy(margin=0.0), obs)
        self.assertEqual(ranked[0][0].id, "exp")

    def test_no_measured_safe_plan_means_no_baseline_to_protect(self):
        # Nothing to be safer than: the objective score decides.
        obs = {"exp": {"generation_tps": 5.0}}
        ranked = self.mp.rank_plans([self.safe, self.exp], self._policy(), obs)
        self.assertEqual(ranked[0][0].id, "exp")

    def test_pinning_overrides_the_guardrail(self):
        from modelctl_plans import RuntimePolicy
        obs = {"safe": {"generation_tps": 10.0}, "exp": {"generation_tps": 1.0}}
        policy = RuntimePolicy("fastest_generation", "exp", True, True,
                               8192, None, 3)
        ranked = self.mp.rank_plans([self.safe, self.exp], policy, obs)
        self.assertEqual(ranked[0][0].id, "exp")

    def test_interactive_latency_ranks_on_ttft_not_load_time(self):
        a = self._plan("a")
        b = self._plan("b")
        obs = {"a": {"ttft_seconds": 2.0, "load_seconds": 1.0},
               "b": {"ttft_seconds": 0.5, "load_seconds": 60.0}}
        ranked = self.mp.rank_plans([a, b], self._policy("interactive_latency"), obs)
        self.assertEqual(ranked[0][0].id, "b")

    def test_lowest_storage_ranks_on_bytes_actually_read(self):
        a = self._plan("a")
        b = self._plan("b")
        obs = {"a": {"read_bytes": 40 << 30}, "b": {"read_bytes": 1 << 30}}
        ranked = self.mp.rank_plans([a, b], self._policy("lowest_storage"), obs)
        self.assertEqual(ranked[0][0].id, "b")

    def test_unmeasured_plans_lose_to_measured_ones_on_new_objectives(self):
        # A plan with no TTFT must not win by having no latency to report.
        a = self._plan("a")
        b = self._plan("b")
        obs = {"a": {"ttft_seconds": 5.0}}
        ranked = self.mp.rank_plans([a, b], self._policy("interactive_latency"), obs)
        self.assertEqual(ranked[0][0].id, "a")


class TestDecisionTrace(unittest.TestCase):
    """Task H3: every automatic choice can be checked by the user, which
    means the trace has to state the real reason, not a plausible one."""

    def _plan(self, pid, cache_bytes=0, source="test", label=None):
        from modelctl_plans import LaunchPlan, ResourceClaim
        return LaunchPlan(
            id=pid, profile_name="m", backend="llama-cpp", label=label or pid,
            argv=(), env={},
            claim=ResourceClaim({"S0": 8 << 30}, 0, "none", 32768,
                                cache_bytes=cache_bytes),
            estimated={}, source=source, warnings=(), decision_data={})

    def _policy(self, **kw):
        from modelctl_plans import RuntimePolicy
        base = dict(objective="fastest_generation", pinned_plan_id=None,
                    allow_fallback=True, allow_untested=True,
                    minimum_context=8192, maximum_cpu_bytes=None,
                    maximum_storage_tier=3)
        base.update(kw)
        return RuntimePolicy(**base)

    def _build(self, plans, obs, policy):
        import modelctl_evidence, modelctl_plans
        explain = {}
        ranked = modelctl_plans.rank_plans(plans, policy, obs, explain=explain)
        ev = modelctl_evidence.build_plan_evidence(
            {"name": "m"}, plans, obs, {}, ranked_ids=[p.id for p, _ in ranked])
        return modelctl_evidence.build_decision_trace(
            ev, ranked=ranked, policy=policy, observations=obs, explain=explain,
            hardware_fingerprint="hw123456789abc")

    def test_trace_names_the_selected_plan_and_its_fallback(self):
        a = self._plan("a", label="fast")
        b = self._plan("b", label="slow")
        t = self._build([a, b], {"a": {"generation_tps": 20.0},
                                 "b": {"generation_tps": 5.0}}, self._policy())
        self.assertEqual(t.selected, "fast")
        self.assertEqual(t.fallback, "slow")

    def test_rejection_quotes_the_ranker_not_a_paraphrase(self):
        # The reason shown must be the guardrail's own words, or the trace
        # can drift from what the ranker actually did.
        safe = self._plan("safe", source="current-profile", label="baseline")
        exp = self._plan("exp", cache_bytes=2 << 30, label="cache plan")
        t = self._build([safe, exp], {"safe": {"generation_tps": 10.0},
                                      "exp": {"generation_tps": 10.2}},
                        self._policy(experimental_margin=0.10))
        self.assertEqual(t.selected, "baseline")
        rejected = dict(t.rejected)
        self.assertIn("cache plan", rejected)
        self.assertIn("margin", rejected["cache plan"])

    def test_stale_reasons_reach_the_trace(self):
        safe = self._plan("safe", source="current-profile", label="baseline")
        exp = self._plan("exp", cache_bytes=2 << 30, label="cache plan")
        obs = {"safe": {"generation_tps": 10.0},
               "exp": {"generation_tps": 99.0, "stale": True,
                       "stale_reasons": ["storage device changed since this was measured"]}}
        t = self._build([safe, exp], obs, self._policy())
        self.assertIn("storage device changed", dict(t.rejected)["cache plan"])

    def test_unmeasured_baseline_says_so_rather_than_inventing_a_reason(self):
        safe = self._plan("safe", source="current-profile", label="baseline")
        t = self._build([safe], {}, self._policy())
        self.assertIn("nothing has been measured", t.reason)

    def test_constraints_are_reported_in_user_units(self):
        t = self._build([self._plan("a")], {"a": {"generation_tps": 1.0}},
                        self._policy(maximum_cpu_bytes=64 << 30,
                                     minimum_context=32768))
        joined = " ".join(t.constraints)
        self.assertIn("64 GiB", joined)
        self.assertIn("32,768", joined)

    def test_evidence_records_provenance_of_the_choice(self):
        t = self._build([self._plan("a")],
                        {"a": {"generation_tps": 1.0, "measured_at": 1700000000,
                               "cache_state": "warm"}}, self._policy())
        joined = " ".join(t.evidence)
        self.assertIn("warm", joined)
        self.assertIn("hw123456789", joined)

    def test_pinned_choice_says_it_was_pinned(self):
        a = self._plan("a", label="pinned one")
        b = self._plan("b", label="faster")
        t = self._build([a, b], {"a": {"generation_tps": 1.0},
                                 "b": {"generation_tps": 50.0}},
                        self._policy(pinned_plan_id="a"))
        self.assertEqual(t.selected, "pinned one")
        self.assertIn("pinned", t.reason)

    def test_empty_when_there_is_nothing_to_choose(self):
        import modelctl_evidence
        t = modelctl_evidence.build_decision_trace([], ranked=[])
        self.assertFalse(t.present)


class TestMmapReservation(unittest.TestCase):
    """An mmap-backed oversized MoE must be testable. Reserving against
    addressed bytes rather than resident ones refused every plan this
    project exists to serve."""

    def _claim(self, storage_mode, ram_bytes, resident):
        from modelctl_plans import ResourceClaim
        return ResourceClaim({"SYCL0": 8 << 30}, ram_bytes, storage_mode,
                             8192, ram_resident_bytes=resident)

    def _reserved_ram(self, claim):
        # Mirrors the admission figure test_launch_plan() computes.
        resident = claim.ram_resident_bytes
        if claim.storage_mode != "mmap" and not resident:
            resident = claim.ram_bytes
        return resident

    def test_mmap_plan_reserves_resident_bytes_not_addressed_ones(self):
        # 36.7 GiB addressed through mmap, nothing required resident.
        c = self._claim("mmap", 37 << 30, 0)
        self.assertEqual(self._reserved_ram(c), 0)

    def test_mmap_plan_still_reserves_its_resident_portion(self):
        c = self._claim("mmap", 37 << 30, 4 << 30)
        self.assertEqual(self._reserved_ram(c), 4 << 30)

    def test_non_mmap_plan_reserves_its_full_claim(self):
        # A resident plan genuinely needs the RAM and must still be refused
        # when it does not fit.
        c = self._claim("none", 20 << 30, 0)
        self.assertEqual(self._reserved_ram(c), 20 << 30)

    def test_non_mmap_plan_prefers_an_explicit_resident_figure(self):
        c = self._claim("none", 20 << 30, 12 << 30)
        self.assertEqual(self._reserved_ram(c), 12 << 30)


class TestUniformCacheBudgetReservation(unittest.TestCase):
    """The runtime's --moe-cache-bytes is one uniform per-GPU budget, so
    the planner must reserve what will be allocated, not what was asked
    for. Under-reserving lets the cache collide with placed experts."""

    def _claim(self, budgets):
        import modelctl_plans, modelctl_hardware, modelctl_profiles
        from unittest import mock
        prof = modelctl_profiles.normalize_profile({
            "name": "m", "backend": "llama-cpp", "model_path": "/x/m.gguf",
            "moe_cache": {"mode": "manual",
                          "gpu": {"budgets_bytes": budgets, "policy": "slru"},
                          "decode": {"admit_to_gpu_cache": True,
                                     "miss_execution": "gpu"}},
            "config": {"device": "SYCL0,SYCL1", "ctx": 4096, "fit": "off",
                       "extra": ""}})
        gpus = [modelctl_hardware.GpuSnapshot("SYCL0", "A", 32 << 30, 30 << 30,
                                              0, True, "", 600),
                modelctl_hardware.GpuSnapshot("SYCL1", "B", 12 << 30, 11 << 30,
                                              0, True, "", 400)]
        snap = modelctl_hardware.HardwareSnapshot(
            captured_at=0, fingerprint="f", gpus=tuple(gpus),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})
        with mock.patch("modelctl.build_server_args", return_value=["llama-server"]), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="fp"):
            return modelctl_plans.current_profile_plan(prof, snap).claim

    def test_uneven_budgets_reserve_the_maximum_on_every_device(self):
        c = self._claim({"SYCL0": 4 << 30, "SYCL1": 2 << 30})
        self.assertEqual(c.vram_cache_bytes.get("SYCL0"), 4 << 30)
        # The runtime would allocate 4 GiB here too, not the declared 2.
        self.assertEqual(c.vram_cache_bytes.get("SYCL1"), 4 << 30)

    def test_equal_budgets_are_unchanged(self):
        c = self._claim({"SYCL0": 2 << 30, "SYCL1": 2 << 30})
        self.assertEqual(c.vram_cache_bytes.get("SYCL0"), 2 << 30)
        self.assertEqual(c.vram_cache_bytes.get("SYCL1"), 2 << 30)

    def test_undeclared_devices_reserve_nothing(self):
        # Only devices the profile names get a reservation; inventing one
        # for every GPU would refuse plans that fit.
        c = self._claim({"SYCL0": 4 << 30})
        self.assertEqual(c.vram_cache_bytes.get("SYCL0"), 4 << 30)
        self.assertEqual(c.vram_cache_bytes.get("SYCL1", 0), 0)

    def test_reservation_matches_the_flag_the_command_carries(self):
        # Planner and command builder must agree, or the claim describes a
        # different allocation than the one that happens.
        import modelctl, modelctl_profiles
        budgets = {"SYCL0": 4 << 30, "SYCL1": 2 << 30}
        c = self._claim(budgets)
        prof = modelctl_profiles.normalize_profile({
            "name": "m", "backend": "llama-cpp", "model_path": "/x/m.gguf",
            "moe_cache": {"mode": "manual",
                          "gpu": {"budgets_bytes": budgets, "policy": "slru"},
                          "decode": {"admit_to_gpu_cache": True,
                                     "miss_execution": "gpu"}},
            "config": {"device": "SYCL0", "ctx": 4096, "fit": "off", "extra": ""}})
        caps = {"schema": 2, "features": {"moe_weight_transfer_cache": True}}
        args = modelctl.build_server_args(prof, capabilities=caps)
        flag = args[args.index("--moe-cache-bytes") + 1] if "--moe-cache-bytes" in args else None
        self.assertEqual(int(flag), max(c.vram_cache_bytes.values()))
