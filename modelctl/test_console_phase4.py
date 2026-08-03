"""Console phase 4: the operations the phase-3 cutover left behind.

Every case here asks one of two questions. "Does this control submit the
same thing the demolished route submitted?" -- because phase 4 is a
re-home, and a control that quietly does something slightly different is
worse than a missing one. And "does a scratch instance refuse it, with a
reason?" -- because the refusal transcript is how a walk of this surface
is verified without driving the live stack.

Carries its own isolation (tmp state, patched dirs) so single-file runs
never touch real state; the full suite additionally rides
test__hermeticity's bootstrap.
"""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_hardware
from modelctl_web import hub
from modelctl_web import wizard as wiz
from modelctl_web.app import create_app
from modelctl_web.jobs import JobRunner, JobStore


PROFILE = {
    "name": "m1", "repo_id": "r/m", "file": "m-Q4_K_M",
    "model_path": "/x/m.gguf", "mmproj_path": None, "mtp_path": None,
    "backend": "llama-cpp",
    "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
               "ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q4_0",
               "flash_attn": "auto", "ttl": 3600, "mtp": "off", "fit": "on",
               "extra": ""},
    "env": [], "enabled": True,
}

# Every mutating endpoint phase 4 adds, with a body that would be valid
# if the instance were live. One list, used by the action tests and by
# the scratch-safe transcript, so an endpoint cannot be added to the
# surface and left out of the refusal check.
PHASE4_WRITES = [
    ("/api/v2/models/m1/restart", {}),
    ("/api/v2/models/m1/cache/reset", {}),
    ("/api/v2/models/m1/plans/p1/select", {}),
    ("/api/v2/models/m1/plans/p1/disable", {}),
    ("/api/v2/models/m1/plans/p1/enable", {}),
    ("/api/v2/models/m1/plans/p1/test", {}),
    ("/api/v2/models/m1/tier/apply", {"accept_tier_change": True}),
    ("/api/v2/models/m1/runtime-policy", {"mode": "fixed"}),
    ("/api/v2/models/m1/delete", {}),
    ("/api/v2/models/m1/bench", {"max_tokens": 64, "runs": 1}),
    ("/api/v2/models/m1/smoke", {}),
    ("/api/v2/models/m1/autotune", {"objective": "balanced"}),
    ("/api/v2/runtime/unload-all", {}),
    ("/api/v2/settings/routing/apply", {}),
]


class Phase4Base(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        (self.profiles_dir / "m1.json").write_text(json.dumps(PROFILE))
        self.swap_config = self.root / "config.yaml"
        self.swap_config.write_text("models: {}\n")
        no_binaries = str(self.root / "no-binaries" / "*")

        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
                  mock.patch.object(modelctl, "LLAMA_SWAP_CONFIG_PATH",
                                    self.swap_config),
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
        # Short, on purpose. The lane workers block forever on their own
        # queues, so this join can only ever time out -- its job is to let
        # an in-flight worker finish before the tmpdir goes away, and
        # nothing here submits a real one. The inherited 1 s spends a
        # second per test waiting for a thread that is never going to
        # exit, which across this file is most of its wall time.
        self.addCleanup(lambda: runner._thread.join(timeout=0.05) or None)
        self.client = TestClient(create_app(store=store, runner=runner))
        # Auth removed 2026-08-03 (owner decision: LAN-open like :9292).
        self.auth = {}

    def running(self, port=9101):
        """Patch llama-swap into reporting m1 resident on `port`."""
        return mock.patch(
            "modelctl_web.swap.LlamaSwapClient.runtime_state",
            return_value={"m1": {"model_id": "m1", "state": "running",
                                 "running": True, "registered": True,
                                 "port": port, "pid": 4242, "started": 0,
                                 "state_class": "active"}})


class TestRuntimeActions(Phase4Base):
    def test_restart_submits_the_same_helper_the_old_route_did(self):
        with mock.patch("modelctl_web.mutate.submit_restart",
                        return_value="job-restart") as sub:
            r = self.client.post("/api/v2/models/m1/restart", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"job_id": "job-restart"})
        self.assertEqual(sub.call_args[0][1], "m1")

    def test_unload_all_submits_the_same_helper_the_old_route_did(self):
        with mock.patch("modelctl_web.mutate.submit_unload_all",
                        return_value="job-ua") as sub:
            r = self.client.post("/api/v2/runtime/unload-all", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"job_id": "job-ua"})
        self.assertTrue(sub.called)

    def test_a_slash_in_the_model_id_survives_the_route(self):
        """llama-swap model ids can contain a slash, which is why load and
        unload use {name:path}. Restart is the same kind of id."""
        with mock.patch("modelctl_web.mutate.submit_restart",
                        return_value="job-r") as sub:
            r = self.client.post("/api/v2/models/org/model/restart",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sub.call_args[0][1], "org/model")


class TestCacheReset(Phase4Base):
    """The one phase-4 action that is not a job: the answer has to be
    synchronous, because a measurement taken right after a reset that
    silently did nothing is a wrong measurement."""

    def test_a_model_that_is_not_running_is_refused_not_faked(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.runtime_state",
                        return_value={}):
            r = self.client.post("/api/v2/models/m1/cache/reset",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 409)
        self.assertIn("not running", r.json()["error"])

    def test_a_failed_reset_is_a_502_naming_the_failure(self):
        with self.running(), mock.patch("urllib.request.urlopen",
                                        side_effect=OSError("connection refused")):
            r = self.client.post("/api/v2/models/m1/cache/reset",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 502)
        self.assertIn("connection refused", r.json()["error"])

    def test_a_successful_reset_reports_the_port_it_reset(self):
        with self.running(port=9191), mock.patch("urllib.request.urlopen") as u:
            r = self.client.post("/api/v2/models/m1/cache/reset",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "name": "m1", "port": 9191})
        self.assertEqual(u.call_args[0][0].full_url,
                         "http://127.0.0.1:9191/cache/reset")
        self.assertEqual(u.call_args[0][0].method, "POST")

    def test_an_unreadable_runtime_is_a_502_not_a_traceback(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.runtime_state",
                        side_effect=RuntimeError("swap is down")):
            r = self.client.post("/api/v2/models/m1/cache/reset",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 502)
        self.assertIn("swap is down", r.json()["error"])


class TestPlanActions(Phase4Base):
    def test_select_and_disable_ride_the_one_helper_with_its_flag(self):
        for action, disable in (("select", False), ("disable", True)):
            with self.subTest(action=action):
                with mock.patch("modelctl_web.mutate.submit_plan_select",
                                return_value="job-p") as sub:
                    r = self.client.post(
                        f"/api/v2/models/m1/plans/plan-7/{action}",
                        headers=self.auth)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json(), {"job_id": "job-p"})
                self.assertEqual(sub.call_args[0][1:], ("m1", "plan-7"))
                self.assertEqual(sub.call_args[1].get("disable", False), disable)

    def test_enable_and_test_submit_their_own_helpers(self):
        for action, target in (
                ("enable", "modelctl_web.mutate.submit_plan_enable"),
                ("test", "modelctl_web.mutate.submit_plan_test")):
            with self.subTest(action=action):
                with mock.patch(target, return_value=f"job-{action}") as sub:
                    r = self.client.post(
                        f"/api/v2/models/m1/plans/plan-7/{action}",
                        headers=self.auth)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json(), {"job_id": f"job-{action}"})
                self.assertEqual(sub.call_args[0][1:], ("m1", "plan-7"))

    def test_enable_goes_through_the_plan_service_the_old_route_used(self):
        """submit_plan_enable is new code for an old submission: the old
        console built this job inline. It must still call the same
        service function, on the same lane."""
        from modelctl_web import mutate
        result = mock.Mock(ok=True, messages=["enabled"])
        runner = mock.Mock()
        runner.submit.return_value = "job-e"
        with mock.patch("modelctl_services.plan_service.enable_plan",
                        return_value=result) as svc:
            job_id = mutate.submit_plan_enable(runner, "m1", "plan-7")
            fn = runner.submit.call_args[0][2]
            fn(mock.Mock())
        self.assertEqual(job_id, "job-e")
        self.assertEqual(runner.submit.call_args[0][0], "mutation")
        svc.assert_called_once_with("m1", "plan-7")

    def test_a_failing_enable_fails_the_job(self):
        from modelctl_web import mutate
        result = mock.Mock(ok=False, messages=["no such plan"])
        runner = mock.Mock()
        with mock.patch("modelctl_services.plan_service.enable_plan",
                        return_value=result):
            mutate.submit_plan_enable(runner, "m1", "plan-7")
            fn = runner.submit.call_args[0][2]
            with self.assertRaises(RuntimeError) as caught:
                fn(mock.Mock())
        self.assertIn("no such plan", str(caught.exception))


class TestTierApply(Phase4Base):
    PLAN = {"tier": 2, "config": {}, "layout": [], "warnings": [],
            "admission": {}}

    _DEFAULT = object()

    def _plan(self, plan=_DEFAULT):
        return mock.patch.object(
            modelctl, "plan_tiers_for_profile",
            return_value=(self.PLAN if plan is self._DEFAULT else plan,
                          {}, "stored"))

    def _gate(self, requires):
        import modelctl_tiers
        return mock.patch.object(
            modelctl_tiers, "tier_change_gate",
            return_value={"kind": "structural" if requires else "none",
                          "changes": ["tier 3 -> 2"] if requires else [],
                          "requires_accept": requires})

    def test_a_gated_replan_is_refused_with_its_changes_until_confirmed(self):
        with self._plan(), self._gate(True), \
                mock.patch("modelctl_web.mutate.submit_tier_apply") as sub:
            r = self.client.post("/api/v2/models/m1/tier/apply",
                                 headers=self.auth, json={})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["gate"]["changes"], ["tier 3 -> 2"])
        self.assertFalse(sub.called, "the gate must stop the submission, not "
                                     "just decorate it")

    def test_the_confirm_carries_accept_tier_change_into_the_job(self):
        with self._plan(), self._gate(True), \
                mock.patch("modelctl_web.mutate.submit_tier_apply",
                           return_value="job-t") as sub:
            r = self.client.post("/api/v2/models/m1/tier/apply",
                                 headers=self.auth,
                                 json={"accept_tier_change": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], "job-t")
        self.assertTrue(sub.call_args[1]["accept_tier_change"])

    def test_an_ungated_replan_applies_without_a_confirm(self):
        with self._plan(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_tier_apply",
                           return_value="job-t") as sub:
            r = self.client.post("/api/v2/models/m1/tier/apply",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(sub.call_args[1]["accept_tier_change"])

    def test_an_unplannable_model_is_a_409_not_a_job_that_will_fail(self):
        with self._plan(plan=None):
            r = self.client.post("/api/v2/models/m1/tier/apply",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 409)
        self.assertIn("couldn't analyze", r.json()["error"])

    def test_the_preview_the_page_shows_is_the_plan_apply_computes(self):
        """The button sits under the admission preview. If the two used
        different plan paths the operator would confirm one thing and get
        another -- so both go through plan_tiers_for_profile on the same
        profile."""
        with self._plan(), self._gate(False):
            preview = self.client.get("/api/v2/models/m1/admission",
                                      headers=self.auth).json()
            with mock.patch("modelctl_web.mutate.submit_tier_apply",
                            return_value="job-t"):
                applied = self.client.post("/api/v2/models/m1/tier/apply",
                                           headers=self.auth).json()
        self.assertEqual(preview["plan"]["tier"], self.PLAN["tier"])
        self.assertEqual(applied["gate"]["requires_accept"], False)


class TestRuntimePolicy(Phase4Base):
    def test_fixed_drops_the_policy_exactly_as_the_old_form_did(self):
        with mock.patch("modelctl_web.mutate.submit_runtime_policy",
                        return_value="job-rp") as sub:
            r = self.client.post("/api/v2/models/m1/runtime-policy",
                                 headers=self.auth, json={"mode": "fixed"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["mode"], "fixed")
        self.assertIsNone(sub.call_args[0][2],
                          "fixed mode passes runtime=None, which is what "
                          "update_runtime_policy reads as 'drop the section'")

    def test_managed_builds_the_whole_policy_dict(self):
        body = {"mode": "managed", "objective": "fastest_load",
                "pinned_plan_id": "p9", "allow_fallback": False,
                "allow_untested": True, "minimum_context": 16384,
                "maximum_cpu_bytes": 8 << 30, "maximum_storage_tier": 2,
                "disabled_plan_ids": ["p1", "p2"]}
        with mock.patch("modelctl_web.mutate.submit_runtime_policy",
                        return_value="job-rp") as sub:
            r = self.client.post("/api/v2/models/m1/runtime-policy",
                                 headers=self.auth, json=body)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sub.call_args[0][2], {
            "mode": "managed", "objective": "fastest_load",
            "pinned_plan_id": "p9", "allow_fallback": False,
            "allow_untested": True, "minimum_context": 16384,
            "maximum_cpu_bytes": 8 << 30, "maximum_storage_tier": 2,
            "disabled_plan_ids": ["p1", "p2"]})

    def test_the_form_offers_exactly_the_objectives_the_write_accepts(self):
        """The phase-3 settings rule, applied to an enum: a select that
        offers what the endpoint rejects is a dead control, and one that
        hides what it accepts makes the JSON file the only way in. The old
        template's list was hand-maintained and had drifted -- it omitted
        interactive_latency and lowest_storage."""
        with mock.patch.object(hub, "plan_rows", return_value=[]):
            offered = self.client.get("/api/v2/models/m1/runtime-policy",
                                      headers=self.auth).json()["objectives"]
        self.assertEqual(set(offered), set(hub.RUNTIME_OBJECTIVES))
        for objective in offered:
            with self.subTest(objective=objective):
                with mock.patch("modelctl_web.mutate.submit_runtime_policy",
                                return_value="j"):
                    r = self.client.post(
                        "/api/v2/models/m1/runtime-policy", headers=self.auth,
                        json={"mode": "managed", "objective": objective})
                self.assertEqual(r.status_code, 200, objective)

    def test_every_offered_objective_is_one_the_ranker_scores(self):
        """Guards the other direction: an objective in the list that the
        ranker does not branch on would silently rank as balanced."""
        import inspect
        import modelctl_plans
        source = inspect.getsource(modelctl_plans)
        for objective in hub.RUNTIME_OBJECTIVES:
            if objective == "balanced":
                continue  # the fall-through, never compared by name
            with self.subTest(objective=objective):
                self.assertIn(f'objective == "{objective}"', source)

    def test_rejections_name_what_was_wrong(self):
        for body, expect in (
                ({"mode": "sideways"}, "mode must be"),
                ({"mode": "managed", "objective": "go_faster"},
                 "unknown objective"),
                ({"mode": "managed", "maximum_storage_tier": 9},
                 "maximum_storage_tier is 1"),
                ({"mode": "managed", "minimum_context": "lots"},
                 "must be integers"),
                ({"mode": "managed", "disabled_plan_ids": "p1"},
                 "list of plan ids"),
        ):
            with self.subTest(body=body):
                r = self.client.post("/api/v2/models/m1/runtime-policy",
                                     headers=self.auth, json=body)
                self.assertEqual(r.status_code, 422)
                self.assertIn(expect, r.json()["error"])

    def test_managed_on_an_unresolvable_backend_is_refused(self):
        """Managed placement IS the backend picking a plan at launch, so a
        backend that cannot be resolved cannot be managed. The old handler
        refused this for the same reason."""
        import modelctl_backends
        with mock.patch.object(modelctl_backends, "get_backend",
                               side_effect=modelctl_backends.BackendError(
                                   "no such backend 'nope'")):
            r = self.client.post("/api/v2/models/m1/runtime-policy",
                                 headers=self.auth, json={"mode": "managed"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no such backend", r.json()["error"])

    def test_the_get_degrades_when_plans_cannot_compile(self):
        with mock.patch.object(hub, "plan_rows",
                               side_effect=RuntimeError("no hardware")):
            r = self.client.get("/api/v2/models/m1/runtime-policy",
                                headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["plans"], [])
        self.assertTrue(r.json()["objectives"])


class TestRunCommandPreview(Phase4Base):
    def test_it_publishes_the_resolved_binary_not_the_pinned_one(self):
        cmd = mock.Mock(argv=("/opt/resolved/llama-server", "-m", "/x/m.gguf"),
                        command_fingerprint="abc123def456", warnings=())
        with mock.patch.object(modelctl, "canonical_launch_command",
                               return_value=(cmd, True, ["ok"])):
            r = self.client.get("/api/v2/models/m1/run-command",
                                headers=self.auth)
        body = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["resolved_binary"], "/opt/resolved/llama-server")
        self.assertEqual(body["argv"][0], "/opt/resolved/llama-server")
        self.assertIn(" \\\n  ", body["run_sh"])
        self.assertEqual(body["command_fingerprint"], "abc123def456")

    def test_it_degrades_instead_of_500ing(self):
        with mock.patch.object(modelctl, "canonical_launch_command",
                               side_effect=RuntimeError("no binary anywhere")):
            r = self.client.get("/api/v2/models/m1/run-command",
                                headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertIn("no binary anywhere", r.json()["error"])
        self.assertEqual(r.json()["argv"], [])

    def test_it_is_a_read(self):
        """A GET that writes is the bug the wizard's download step had.
        Nothing about the profile may move when this is fetched."""
        before = (self.profiles_dir / "m1.json").read_text()
        cmd = mock.Mock(argv=("/b", "-m", "/x"), command_fingerprint="f",
                        warnings=())
        with mock.patch.object(modelctl, "canonical_launch_command",
                               return_value=(cmd, True, [])):
            self.client.get("/api/v2/models/m1/run-command", headers=self.auth)
        self.assertEqual((self.profiles_dir / "m1.json").read_text(), before)


class TestDeleteBenchSmokeAutotune(Phase4Base):
    def test_delete_404s_for_an_unknown_profile_before_queueing_anything(self):
        with mock.patch("modelctl_web.mutate.submit_remove") as sub:
            r = self.client.post("/api/v2/models/nope/delete", headers=self.auth)
        self.assertEqual(r.status_code, 404)
        self.assertFalse(sub.called)

    def test_delete_submits_the_remove_job(self):
        with mock.patch("modelctl_web.mutate.submit_remove",
                        return_value="job-rm") as sub:
            r = self.client.post("/api/v2/models/m1/delete", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"job_id": "job-rm"})
        self.assertEqual(sub.call_args[0][1], "m1")

    def test_remove_runs_cmd_remove_the_way_the_old_route_did(self):
        from modelctl_web import mutate
        runner = mock.Mock()
        runner.submit.return_value = "job-rm"
        with mock.patch.object(modelctl, "cmd_remove") as removed:
            mutate.submit_remove(runner, "m1")
            fn = runner.submit.call_args[0][2]
            fn(mock.Mock())
        args = removed.call_args[0][0]
        self.assertEqual(args.name, "m1")
        self.assertTrue(args.no_hermes)
        self.assertFalse(args.no_router_restart)
        self.assertEqual(runner.submit.call_args[0][0], "remove")

    def test_bench_clamps_the_overrides_to_the_bounds_the_old_form_used(self):
        for sent, expect in (({"max_tokens": 99999, "runs": 99}, (4096, 10)),
                             ({"max_tokens": 0, "runs": 0}, (1, 1)),
                             ({"max_tokens": "sixty", "runs": None}, (256, 3)),
                             ({}, (256, 3))):
            with self.subTest(sent=sent):
                with mock.patch("modelctl_web.mutate.submit_bench",
                                return_value="job-b") as sub:
                    r = self.client.post("/api/v2/models/m1/bench",
                                         headers=self.auth, json=sent)
                self.assertEqual(r.status_code, 200)
                self.assertEqual((sub.call_args[1]["max_tokens"],
                                  sub.call_args[1]["runs"]), expect)
                self.assertEqual((r.json()["max_tokens"], r.json()["runs"]),
                                 expect)

    def test_smoke_submits_onto_the_benchmark_lane(self):
        with mock.patch("modelctl_web.mutate.submit_smoke_test",
                        return_value="job-s") as sub:
            r = self.client.post("/api/v2/models/m1/smoke", headers=self.auth)
        self.assertEqual(r.json(), {"job_id": "job-s"})
        self.assertEqual(sub.call_args[0][1], "m1")

    def test_autotune_passes_objective_and_candidates(self):
        with mock.patch("modelctl_web.mutate.submit_autotune",
                        return_value="job-a") as sub:
            r = self.client.post(
                "/api/v2/models/m1/autotune", headers=self.auth,
                json={"objective": "fastest_load", "plan_ids": ["p1", "p2"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sub.call_args[1]["objective"], "fastest_load")
        self.assertEqual(sub.call_args[1]["candidate_ids"], ["p1", "p2"])

    def test_autotune_rejects_an_objective_the_ranker_does_not_know(self):
        r = self.client.post("/api/v2/models/m1/autotune", headers=self.auth,
                             json={"objective": "vibes"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("unknown objective", r.json()["error"])


class TestJobDeepLinks(Phase4Base):
    def test_one_job_is_readable_on_its_own(self):
        job_id = self.store.create("bench", "benchmark m1", lane="benchmark")
        r = self.client.get(f"/api/v2/jobs/{job_id}", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["id"], job_id)
        self.assertEqual(body["title"], "benchmark m1")
        self.assertEqual(body["lane"], "benchmark")

    def test_an_unknown_job_is_a_404(self):
        r = self.client.get("/api/v2/jobs/nope", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_the_per_job_row_is_the_same_shape_the_list_serves(self):
        """The detail page and the list render the same component tree;
        two shapes would mean two renderers."""
        job_id = self.store.create("bench", "benchmark m1", lane="benchmark")
        one = self.client.get(f"/api/v2/jobs/{job_id}", headers=self.auth).json()
        listed = self.client.get("/api/v2/jobs", headers=self.auth).json()
        match = [j for j in listed if j["id"] == job_id]
        self.assertEqual(len(match), 1)
        self.assertEqual(set(one), set(match[0]))

    def test_an_old_job_url_lands_on_that_job_again(self):
        """Phase 3 sent every /jobs/{id} to the list because the SPA had
        no per-job URL. It has one again, so the id survives the
        redirect."""
        r = self.client.get("/jobs/abc123", headers=self.auth,
                            follow_redirects=False)
        self.assertEqual(r.status_code, 301)
        self.assertEqual(r.headers["location"], "/v2/jobs/abc123")

    def test_an_old_job_sub_path_keeps_only_the_id(self):
        r = self.client.get("/jobs/abc123/log", headers=self.auth,
                            follow_redirects=False)
        self.assertEqual(r.headers["location"], "/v2/jobs/abc123")


class TestRoutingMatrix(Phase4Base):
    GENERATED = {
        "vars": {"mc_m1": "m1"},
        "evict_costs": {"mc_m1": 1.5},
        "sets": {"mc_m1": "m1", "mc_m1_plus_helper": "m1 & helper"},
        "claims": {"m1": {"SYCL0": 1 << 30}},
        "excluded": [{"name": "m2", "reason": "claim unknown -- not guessed"}],
    }

    def _generated(self):
        import modelctl_matrix
        return mock.patch.object(modelctl_matrix, "generate_matrix",
                                 return_value=self.GENERATED)

    def test_the_grid_says_what_each_set_would_become(self):
        self.swap_config.write_text(modelctl.yaml.safe_dump({
            "matrix": {"sets": {"mc_m1": "m1-old", "handmade": "a & b"}}}))
        with self._generated():
            r = self.client.get("/api/v2/settings/routing", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        rows = {row["key"]: row for row in r.json()["rows"]}
        self.assertEqual(rows["mc_m1"]["change"], "changed")
        self.assertEqual(rows["mc_m1"]["before"], "m1-old")
        self.assertEqual(rows["mc_m1"]["after"], "m1")
        self.assertEqual(rows["mc_m1_plus_helper"]["change"], "added")
        self.assertEqual(rows["handmade"]["change"], "unchanged")

    def test_hand_authored_sets_are_marked_and_survive_the_merge(self):
        """The question the old YAML blob could not answer: does applying
        this touch anything I wrote by hand?"""
        self.swap_config.write_text(modelctl.yaml.safe_dump({
            "matrix": {"sets": {"handmade": "a & b"}}}))
        with self._generated():
            body = self.client.get("/api/v2/settings/routing",
                                   headers=self.auth).json()
        rows = {row["key"]: row for row in body["rows"]}
        self.assertFalse(rows["handmade"]["managed"])
        self.assertTrue(rows["mc_m1"]["managed"])
        self.assertEqual(body["merged"]["sets"]["handmade"], "a & b")

    def test_an_unreadable_config_is_a_state_not_a_500(self):
        self.swap_config.write_text("{{{ not yaml")
        with self._generated():
            r = self.client.get("/api/v2/settings/routing", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertIn("existing", r.json()["errors"])

    def test_a_generator_that_throws_leaves_nothing_to_apply(self):
        import modelctl_matrix
        with mock.patch.object(modelctl_matrix, "generate_matrix",
                               side_effect=RuntimeError("no inventory")):
            body = self.client.get("/api/v2/settings/routing",
                                   headers=self.auth).json()
        self.assertIsNone(body["generated"])
        self.assertEqual(body["rows"], [])
        self.assertIn("no inventory", body["errors"]["generated"])

    def test_apply_submits_the_service_that_owns_the_rollback(self):
        with mock.patch("modelctl_web.mutate.submit_matrix_apply",
                        return_value="job-mx") as sub:
            r = self.client.post("/api/v2/settings/routing/apply",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"job_id": "job-mx"})
        self.assertTrue(sub.called)

    def test_the_read_does_not_write_the_config(self):
        before = self.swap_config.read_text()
        with self._generated():
            self.client.get("/api/v2/settings/routing", headers=self.auth)
        self.assertEqual(self.swap_config.read_text(), before)


class TestScratchSafeCoversPhase4(unittest.TestCase):
    """The scratch walk of this surface, as an assertion.

    A scratch instance exists to be walked and must never drive the
    stack, so the walk of a page full of new buttons is the transcript of
    every one of them being refused with the reason. Anything reachable
    from a phase-4 control that is NOT in PHASE4_WRITES would walk
    straight through to the live install.
    """

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        (self.profiles_dir / "m1.json").write_text(json.dumps(PROFILE))
        patcher = mock.patch.dict(os.environ, {"MODELCTL_WEB_SCRATCH": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        p = mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir)
        p.start()
        self.addCleanup(p.stop)
        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=0.05) or None)
        self.client = TestClient(create_app(store=store, runner=runner))
        # Auth removed 2026-08-03 (owner decision: LAN-open like :9292).
        self.auth = {}

    def test_every_phase4_write_is_refused_with_a_reason(self):
        for path, body in PHASE4_WRITES:
            with self.subTest(path=path):
                r = self.client.post(path, headers=self.auth, json=body)
                self.assertEqual(r.status_code, 405, path)
                reason = r.json()["reason"]
                self.assertIn("scratch-safe mode", reason)
                # the reason names the request, so a transcript of these
                # refusals says which control was refused
                self.assertIn(path, reason)

    def test_no_phase4_write_reaches_its_job_runner(self):
        """The refusal is in the middleware, before the handler -- so a
        refused control cannot have queued anything on the way out."""
        with mock.patch("modelctl_web.jobs.JobRunner.submit") as submit:
            for path, body in PHASE4_WRITES:
                self.client.post(path, headers=self.auth, json=body)
        self.assertFalse(submit.called)

    def test_the_phase4_reads_behave_exactly_as_live(self):
        import modelctl_matrix
        # The generator walks the real GPU inventory; the point here is
        # that scratch mode does not gate reads, not that this machine
        # has devices.
        with mock.patch.object(modelctl_matrix, "generate_matrix",
                               return_value={"vars": {}, "evict_costs": {},
                                             "sets": {}, "claims": {},
                                             "excluded": []}), \
                mock.patch.object(hub, "plan_rows", return_value=[]):
            for path in ("/api/v2/models/m1/runtime-policy",
                         "/api/v2/settings/routing",
                         "/api/v2/jobs"):
                with self.subTest(path=path):
                    r = self.client.get(path, headers=self.auth)
                    self.assertEqual(r.status_code, 200, path)

    def test_the_write_list_covers_every_mutating_phase4_route(self):
        """The list above is what the transcript walks. A route added to
        the app and not to the list would be walked past, not refused --
        so the app's own route table is the authority here."""
        store = JobStore(self.root / "check.db", scratch_safe=True)
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        app = create_app(store=store, runner=runner)
        phase4_prefixes = ("/api/v2/models/", "/api/v2/runtime/",
                           "/api/v2/settings/routing")
        mutating = set()
        for route in app.routes:
            methods = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            if not (methods & {"POST", "PUT", "PATCH", "DELETE"}):
                continue
            if not any(path.startswith(p) for p in phase4_prefixes):
                continue
            mutating.add(path)
        # phases 1-3 already owned these three under the same prefixes
        inherited = {"/api/v2/models/{name:path}/load",
                     "/api/v2/models/{name:path}/unload",
                     "/api/v2/models/{name}/config"}
        covered = {
            "/api/v2/models/{name:path}/restart",
            "/api/v2/models/{name:path}/cache/reset",
            "/api/v2/models/{name}/plans/{plan_id}/select",
            "/api/v2/models/{name}/plans/{plan_id}/disable",
            "/api/v2/models/{name}/plans/{plan_id}/enable",
            "/api/v2/models/{name}/plans/{plan_id}/test",
            "/api/v2/models/{name}/tier/apply",
            "/api/v2/models/{name}/runtime-policy",
            "/api/v2/models/{name}/delete",
            "/api/v2/models/{name}/bench",
            "/api/v2/models/{name}/smoke",
            "/api/v2/models/{name}/autotune",
            "/api/v2/runtime/unload-all",
            "/api/v2/settings/routing/apply",
        }
        self.assertEqual(mutating - inherited, covered)
        self.assertEqual(len(covered), len(PHASE4_WRITES))


if __name__ == "__main__":
    unittest.main()
