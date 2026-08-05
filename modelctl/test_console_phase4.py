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
import contextlib
import json
import os
import time
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
    ("/api/v2/models/m1/placement", {"selection": {}}),
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


GIB = 1 << 30


@contextlib.contextmanager
def _both(*patches):
    """Enter several patches together, yielding the FIRST one's mock.

    The placement reads now touch two seams -- the planner and the machine
    snapshot -- and every call site here wants to inspect the planner."""
    mocks = [p.start() for p in patches]
    try:
        yield mocks[0]
    finally:
        for p in reversed(patches):
            p.stop()


# What the planner answers for a selection that spills. Shaped like a real
# plan_tiers result -- the derived view is only trustworthy if it is read
# off the same keys production produces.
PLAN_SPILL = {
    "tier": 4,
    "config": {"device": "", "split_mode": "layer", "tensor_split": "24,12",
               "extra": "--fit off -ngl 99 --device SYCL0,SYCL1"},
    "layout": [("all GPUs (fixed)", 6.0, "attention/embeddings/KV"),
               ("SYCL0", 12.0, "experts layers 0-10 (11)"),
               ("CPU", 30.0, "experts layers 40-59 (20), SSD via mmap")],
    "warnings": ["the CPU share is bigger than the RAM budget"],
    "analysis": {"weights_gib": 60.0, "ram_budget_gib": 19.6},
    "cache_budgets": None,
    "admission": {
        "fits": True,
        "devices": {
            "SYCL0": {"demand_bytes": 19 * GIB, "usable_bytes": 20 * GIB,
                      "capacity_bytes": 24 * GIB, "fits": True},
            "SYCL1": {"demand_bytes": 10 * GIB, "usable_bytes": 11 * GIB,
                      "capacity_bytes": 12 * GIB, "fits": True}}},
}


class _StubProbe:
    """The attributes presence_state actually reads."""

    def __init__(self, reachable=True, pin_agrees=True, detail="", age=0.0):
        self.reachable = reachable
        self.pin_agrees = pin_agrees
        self.detail = detail
        self.protocol = "hello"
        self.probed_at = time.time() - age


class _StubDevice:
    def __init__(self, name, total_bytes=0):
        self.name = name
        self.total_bytes = total_bytes


class _StubNode:
    def __init__(self, name, devices):
        self.name = name
        self.devices = [_StubDevice(*d) if isinstance(d, tuple)
                        else _StubDevice(d) for d in devices]


# The fleet these tests plan against. Stubbed rather than read: Phase4Base
# does not redirect FLEET_PATH/PRESENCE_PATH, so a real read would make a
# single-file run depend on whether the laptop happens to be awake.
FLEET_NODE = "ph16-71-cuda0"
FLEET_KEY = f"RPC:{FLEET_NODE}:CUDA0"


def _plan_variant(**over):
    plan = json.loads(json.dumps(PLAN_SPILL))
    plan["layout"] = [tuple(r) for r in plan["layout"]]
    plan.update(over)
    return plan


class TestPlacementPreview(Phase4Base):
    """GET /api/v2/models/{name}/placement -- where the planner puts the
    weights for a chosen set of devices.

    The screen this feeds must never work the split out for itself: the
    moment it disagrees with the planner the console lies about where the
    weights went, which is the same failure as silent SSD streaming
    relocated into the browser. So every number here is read off a plan.
    """

    INPUTS = {"inventory": [{"device": "SYCL0", "name": "Arc A",
                             "total_bytes": 24 * GIB},
                            {"device": "SYCL1", "name": "Arc B",
                             "total_bytes": 12 * GIB}],
              "vram_limit_pct": 90, "primary": "SYCL0",
              "ram_available_bytes": 31 * GIB,
              # Declared remote ceilings, recorded in the snapshot and
              # deliberately presence-independent (modelctl_fleet.
              # budget_input): a closed laptop must not collapse the bar
              # its device draws.
              "fleet_budgets": {"RPC:ph16-71-cuda0:CUDA0": 10 * GIB}}

    # Installed memory, distinct from the 31 GiB the planner spent, so a
    # test cannot pass by confusing capacity with usable.
    RAM_TOTAL = 64 * GIB

    def _planner(self, plan=None, source="stored", probe=None):
        planner = mock.patch("modelctl_plans.plan_for_selection",
                             return_value=plan or _plan_variant())
        inputs = mock.patch.object(modelctl, "resolve_planning_inputs",
                                   return_value=(self.INPUTS, source))
        ram = mock.patch("modelctl_vram.system_ram_total",
                         return_value=self.RAM_TOTAL)
        nodes = mock.patch(
            "modelctl_fleet.load_fleet",
            return_value=[_StubNode(FLEET_NODE, [("CUDA0", 12 * GIB)])])
        presence = mock.patch(
            "modelctl_fleet.load_presence",
            return_value={FLEET_NODE: probe if probe is not None
                          else _StubProbe()})
        return _both(planner, inputs, ram, nodes, presence)

    def _selection_seen(self, query=""):
        with self._planner() as p:
            r = self.client.get(f"/api/v2/models/m1/placement{query}",
                                headers=self.auth)
        self.assertEqual(r.status_code, 200, r.text)
        return p.call_args[0][1]

    # --- the query is the operator's choice -----------------------------

    def test_no_query_is_the_automatic_placement(self):
        """What the machine would do on its own -- what the operator sees
        before touching anything."""
        self.assertEqual(self._selection_seen(), {})

    def test_a_device_switched_off_reaches_the_planner_as_off(self):
        self.assertEqual(self._selection_seen("?on.RAM=0"),
                         {"RAM": {"on": False}})

    def test_a_ceiling_reaches_the_planner_in_bytes(self):
        self.assertEqual(self._selection_seen(f"?ceiling.SYCL0={8 * GIB}"),
                         {"SYCL0": {"ceiling_bytes": 8 * GIB}})

    def test_a_remote_device_keeps_its_admission_key(self):
        """RPC keys carry colons; they must survive the query intact or the
        selection silently addresses a device that does not exist."""
        key = "RPC:ph16-71-cuda0:CUDA0"
        self.assertEqual(self._selection_seen(f"?on.{key}=1"),
                         {key: {"on": True}})

    def test_a_switch_and_a_ceiling_for_one_device_are_one_entry(self):
        self.assertEqual(
            self._selection_seen(f"?on.SYCL0=1&ceiling.SYCL0={4 * GIB}"),
            {"SYCL0": {"on": True, "ceiling_bytes": 4 * GIB}})

    def test_a_ceiling_that_is_not_a_number_is_refused(self):
        """Dropping it would answer for a layout the operator did not ask
        for, which is worse than saying no."""
        with self._planner() as p:
            r = self.client.get(
                "/api/v2/models/m1/placement?ceiling.SYCL0=lots",
                headers=self.auth)
        self.assertEqual(r.status_code, 422)
        self.assertFalse(p.called)

    def test_a_negative_ceiling_is_refused(self):
        r = self.client.get("/api/v2/models/m1/placement?ceiling.SYCL0=-1",
                            headers=self.auth)
        self.assertEqual(r.status_code, 422)

    def test_an_unreadable_switch_is_refused(self):
        r = self.client.get("/api/v2/models/m1/placement?on.SYCL0=maybe",
                            headers=self.auth)
        self.assertEqual(r.status_code, 422)

    # --- the answer the screen renders ----------------------------------

    def test_every_gpu_reports_the_bytes_the_planner_charged_it(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"]["SYCL0"]["bytes"], 19 * GIB)
        self.assertEqual(body["devices"]["SYCL1"]["bytes"], 10 * GIB)
        self.assertEqual(body["devices"]["SYCL0"]["backing"], "VRAM")

    def test_the_host_share_says_ssd_when_the_plan_streams_it(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"]["RAM"]["backing"], "SSD via mmap")
        self.assertEqual(body["devices"]["RAM"]["bytes"], 30 * GIB)
        self.assertEqual(body["spill_bytes"], 30 * GIB,
                         "the bytes with nowhere to go are the headline "
                         "number on this screen")

    def test_the_host_share_says_ram_when_the_plan_holds_it_resident(self):
        """--no-mmap in the emitted config is the whole difference between
        30 GiB living in memory and 30 GiB streaming off the SSD."""
        resident = _plan_variant(config={
            **PLAN_SPILL["config"],
            "extra": "--fit off -ngl 99 --device SYCL0,SYCL1 --no-mmap"})
        with self._planner(resident):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"]["RAM"]["backing"], "RAM")
        self.assertEqual(body["spill_bytes"], 0)

    def test_a_plan_with_no_host_share_reports_no_spill(self):
        whole = _plan_variant(layout=[("SYCL0 (whole model)", 18.0,
                                       "weights + KV + overhead")])
        with self._planner(whole):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        # The row is present and empty rather than absent: memory is a
        # device you can tick, so it has to be on screen to be ticked.
        self.assertEqual(body["devices"]["RAM"]["bytes"], 0)
        self.assertEqual(body["spill_bytes"], 0)

    def test_remote_devices_report_under_their_admission_key(self):
        key = "RPC:ph16-71-cuda0:CUDA0"
        remote = _plan_variant(config={
            **PLAN_SPILL["config"],
            "rpc": {"endpoints": ["10.0.0.9:50052"], "placements": [],
                    "admission": {key: 5 * GIB}}})
        with self._planner(remote):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"][key]["bytes"], 5 * GIB)
        self.assertEqual(body["devices"][key]["backing"], "over RPC")

    # --- every row is a bar that can be drawn ---------------------------

    def test_the_host_row_carries_the_memory_it_is_measured_against(self):
        """A bar with no bound cannot be drawn.

        The host row shipped bytes against capacity 0 and usable 0, so the
        one control this screen exists for -- committed against a ceiling
        you can move -- had nothing to render for the device the spill
        actually lands on.
        """
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        ram = body["devices"]["RAM"]
        self.assertEqual(ram["usable_bytes"], 31 * GIB,
                         "usable is the memory the planner spent, so it "
                         "agrees with planned_against instead of with free "
                         "memory re-read at render time")
        self.assertEqual(ram["capacity_bytes"], self.RAM_TOTAL,
                         "capacity is installed memory -- a physical "
                         "constant, so reading it live adds no second clock")

    def test_a_remote_rows_capacity_is_the_hardware_not_the_ceiling(self):
        """capacity == usable would say a card is exactly as big as the
        ceiling someone set on it, so a remote bar could never show the
        headroom a ceiling is currently withholding."""
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        row = body["devices"][FLEET_KEY]
        self.assertEqual(row["capacity_bytes"], 12 * GIB, "the device total")
        self.assertEqual(row["usable_bytes"], 10 * GIB, "the declared budget")

    def test_a_remote_row_falls_back_to_its_ceiling_when_the_node_is_gone(self):
        """Understating headroom beats inventing it."""
        with self._planner(), mock.patch("modelctl_fleet.load_fleet",
                                         return_value=[]):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        row = body["devices"][FLEET_KEY]
        self.assertEqual(row["capacity_bytes"], 10 * GIB)
        self.assertEqual(row["state"], "STALE")

    def test_a_remote_row_carries_the_ceiling_it_was_planned_against(self):
        """The recorded fleet budget, not a live probe: presence belongs to
        the device's state, never to the bound its bar is drawn against."""
        key = "RPC:ph16-71-cuda0:CUDA0"
        remote = _plan_variant(config={
            **PLAN_SPILL["config"],
            "rpc": {"endpoints": ["10.0.0.9:50052"], "placements": [],
                    "admission": {key: 5 * GIB}}})
        with self._planner(remote):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"][key]["usable_bytes"], 10 * GIB)

    def test_no_device_row_reports_bytes_against_a_zero_capacity(self):
        """Asserted over every row rather than over the host row by name, so
        a backing type added later cannot reintroduce the gap quietly."""
        key = "RPC:ph16-71-cuda0:CUDA0"
        everything = _plan_variant(config={
            **PLAN_SPILL["config"],
            "rpc": {"endpoints": ["10.0.0.9:50052"], "placements": [],
                    "admission": {key: 5 * GIB}}})
        with self._planner(everything):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        for name, row in body["devices"].items():
            if row["bytes"] <= 0:
                continue
            self.assertGreater(row["usable_bytes"], 0,
                               f"{name} holds bytes with no bound to draw "
                               f"them against")

    # --- fits answers the question a person reads it as -----------------

    def test_a_host_share_that_streams_is_not_reported_as_fitting(self):
        """fits: True beside 30 GiB streaming off the SSD is exactly the
        reassurance this screen exists to remove -- the backing field told
        the truth while the boolean next to it said the opposite."""
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"]["RAM"]["backing"], "SSD via mmap")
        self.assertFalse(body["devices"]["RAM"]["fits"])

    def test_a_host_share_held_in_memory_does_fit(self):
        resident = _plan_variant(config={
            **PLAN_SPILL["config"],
            "extra": "--fit off -ngl 99 --device SYCL0,SYCL1 --no-mmap"})
        with self._planner(resident):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertTrue(body["devices"]["RAM"]["fits"])

    def test_a_remote_device_over_its_declared_ceiling_does_not_fit(self):
        key = "RPC:ph16-71-cuda0:CUDA0"
        over = _plan_variant(config={
            **PLAN_SPILL["config"],
            "rpc": {"endpoints": ["10.0.0.9:50052"], "placements": [],
                    "admission": {key: 12 * GIB}}})
        with self._planner(over):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertFalse(body["devices"][key]["fits"],
                         "12 GiB against a declared 10 GiB ceiling")

    # --- presence is a state, not a filter -------------------------------

    def test_a_device_the_layout_does_not_use_still_gets_a_row(self):
        """You cannot tick on a device that is not on screen. A remote key
        the current layout ignores was absent from the answer entirely,
        which is why ticking one looked inert -- there was nothing to
        render a tick against."""
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertIn(FLEET_KEY, body["devices"])
        self.assertEqual(body["devices"][FLEET_KEY]["bytes"], 0)
        self.assertEqual(body["devices"][FLEET_KEY]["usable_bytes"], 10 * GIB,
                         "an unused device still shows the room it offers")

    def test_local_devices_are_present_by_construction(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        for key in ("SYCL0", "RAM"):
            self.assertEqual(body["devices"][key]["state"], "PRESENT")

    def test_a_remote_device_carries_its_nodes_presence(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"][FLEET_KEY]["state"], "PRESENT")

    def test_an_unreachable_node_makes_its_device_stale_not_absent(self):
        """The device keeps its row and its ceiling; only the state moves.
        Deleting it would make the tick list flicker with the network."""
        with self._planner(probe=_StubProbe(reachable=False,
                                            detail="connection refused")):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        row = body["devices"][FLEET_KEY]
        self.assertEqual(row["state"], "STALE")
        self.assertIn("connection refused", row["detail"])
        self.assertEqual(row["usable_bytes"], 10 * GIB)

    def test_a_node_on_another_commit_is_never_merely_present(self):
        """PIN_MISMATCH is up but unusable -- placing a graph across two
        ggml builds gives wrong numbers rather than an error."""
        with self._planner(probe=_StubProbe(pin_agrees=False)):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["devices"][FLEET_KEY]["state"], "PIN_MISMATCH")

    def test_ticking_on_a_device_that_is_not_present_says_so(self):
        """The probe that started this: ?on.<remote>=1 answered
        byte-identically to the baseline, so the screen could not tell a
        request that was honoured from one that was quietly dropped."""
        with self._planner(probe=_StubProbe(reachable=False,
                                            detail="connection refused")):
            body = self.client.get(
                f"/api/v2/models/m1/placement?on.{FLEET_KEY}=1",
                headers=self.auth).json()
        self.assertTrue(
            any(FLEET_KEY in w for w in body["warnings"]),
            "turning on a device nothing can reach must not read as success")

    def test_ticking_on_a_present_device_warns_about_nothing(self):
        with self._planner():
            body = self.client.get(
                f"/api/v2/models/m1/placement?on.{FLEET_KEY}=1",
                headers=self.auth).json()
        self.assertFalse([w for w in body["warnings"] if FLEET_KEY in w])

    # --- a key the machine does not have ---------------------------------

    def test_a_device_the_machine_does_not_have_is_refused(self):
        """select_inputs only ever looks up keys it already knows, so an
        unknown one was accepted and then never read: the answer described
        a selection nobody made. Same silent-no-op class as a dropped
        ceiling, which this endpoint already refuses."""
        with self._planner():
            r = self.client.get("/api/v2/models/m1/placement?on.NOPE0=1",
                                headers=self.auth)
        self.assertEqual(r.status_code, 422)
        self.assertIn("NOPE0", r.json()["error"])

    def test_the_refusal_names_the_devices_that_do_exist(self):
        """A refusal that does not say what the valid keys are leaves the
        caller guessing at an enum the machine already knows."""
        with self._planner():
            r = self.client.get("/api/v2/models/m1/placement?ceiling.GPU9=1",
                                headers=self.auth)
        self.assertEqual(r.status_code, 422)
        error = r.json()["error"]
        for known in ("SYCL0", "SYCL1", "RAM", "RPC:ph16-71-cuda0:CUDA0"):
            self.assertIn(known, error)

    def test_the_keys_the_machine_does_have_are_accepted(self):
        for key in ("SYCL0", "SYCL1", "RAM", "RPC:ph16-71-cuda0:CUDA0"):
            with self.subTest(key=key), self._planner():
                r = self.client.get(
                    f"/api/v2/models/m1/placement?on.{key}=1",
                    headers=self.auth)
            self.assertEqual(r.status_code, 200, r.text)

    def test_a_remote_key_stays_valid_while_its_node_is_unreachable(self):
        """fleet_budgets is presence-independent by design. Refusing a key
        because a laptop is closed would make the valid device set flicker
        with the network."""
        with self._planner():
            r = self.client.get(
                "/api/v2/models/m1/placement?ceiling.RPC:ph16-71-cuda0:CUDA0="
                f"{4 * GIB}", headers=self.auth)
        self.assertEqual(r.status_code, 200, r.text)

    # --- the storage floor reaches this path too -------------------------

    def _with_runtime(self, **runtime):
        p = json.loads((self.profiles_dir / "m1.json").read_text())
        p["runtime"] = {**(p.get("runtime") or {}), **runtime}
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))

    def test_a_layout_that_streams_names_the_floor_it_crosses(self):
        """maximum_storage_tier is a real filter on the ranked path
        (plans.py:1697 drops mmap plans below tier 3) and was invisible
        here, so the screen offered layouts the model's own policy
        forbids -- live, the 122B is set to tier 2 and previewed 39.52 GiB
        of SSD streaming anyway."""
        self._with_runtime(maximum_storage_tier=2)
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        floor = body["storage_floor"]
        self.assertEqual(floor["maximum_storage_tier"], 2)
        self.assertTrue(floor["crossed"])
        self.assertIn("30.0 GiB", floor["detail"],
                      "the shortfall is named in the units the row shows")
        self.assertTrue(any("storage" in w for w in body["warnings"]),
                        "the crossing rides the warnings channel the apply "
                        "job already logs, so it cannot be applied silently")

    def test_a_layout_held_in_memory_clears_the_floor(self):
        self._with_runtime(maximum_storage_tier=2)
        resident = _plan_variant(config={
            **PLAN_SPILL["config"],
            "extra": "--fit off -ngl 99 --device SYCL0,SYCL1 --no-mmap"})
        with self._planner(resident):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertFalse(body["storage_floor"]["crossed"])

    def test_a_profile_that_permits_storage_reports_no_crossing(self):
        """Tier 3 is the default and means SSD streaming is allowed. A
        spilling layout is then a choice, not a violation."""
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["storage_floor"]["maximum_storage_tier"], 3)
        self.assertFalse(body["storage_floor"]["crossed"])

    def test_the_layout_rows_survive_as_named_fields(self):
        """The planner's rows are tuples; JSON would hand the screen
        positional arrays to index into."""
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["layout"][0]["label"], "all GPUs (fixed)")
        self.assertEqual(body["layout"][2]["label"], "CPU")
        self.assertEqual(body["layout"][2]["gib"], 30.0)

    def test_the_plan_arrives_whole(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["tier"], 4)
        self.assertEqual(body["config"], PLAN_SPILL["config"])
        self.assertEqual(body["warnings"], PLAN_SPILL["warnings"])
        self.assertEqual(body["analysis"], PLAN_SPILL["analysis"])
        self.assertTrue(body["admission"]["fits"])

    def test_the_answer_carries_what_is_applied_right_now(self):
        """The screen opens on what is set to run and needs to know when it
        has drifted from it. Reconstructing that from the emitted -ot rules
        would be a second reader of placement."""
        applied = {"RAM": {"on": False}}
        p = json.loads((self.profiles_dir / "m1.json").read_text())
        p["planning"] = {"selection": applied}
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement?on.SYCL0=1",
                                   headers=self.auth).json()
        self.assertEqual(body["applied_selection"], applied)
        self.assertEqual(body["selection"], {"SYCL0": {"on": True}},
                         "what was asked for stays distinct from what runs")

    def test_a_model_that_was_never_placed_reports_no_applied_selection(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["applied_selection"], {})

    def test_the_answer_says_which_snapshot_it_planned_against(self):
        """The row shows free memory as it is now; the planner spends a
        recorded snapshot. Printing both without saying so is what made a
        three-day-old picture read as a broken layout."""
        p = json.loads((self.profiles_dir / "m1.json").read_text())
        p["planning"] = {"recorded_at": "2026-08-01T17:57:49"}
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["planned_against"]["source"], "stored")
        self.assertEqual(body["planned_against"]["recorded_at"],
                         "2026-08-01T17:57:49")
        self.assertEqual(body["planned_against"]["ram_available_bytes"],
                         31 * GIB)

    def test_a_machine_never_planned_against_says_live(self):
        with self._planner(source="live"):
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["planned_against"]["source"], "live")
        self.assertIsNone(body["planned_against"]["recorded_at"])

    def test_fresh_reads_the_machine_instead_of_the_record(self):
        with self._planner() as _p:
            self.client.get("/api/v2/models/m1/placement?fresh=1",
                            headers=self.auth)
            refreshed = modelctl.resolve_planning_inputs.call_args[1]["refresh"]
        self.assertTrue(refreshed, "?fresh=1 is the re-read: without it the "
                                   "screen can only ever show the snapshot")

    def test_the_default_read_does_not_disturb_the_record(self):
        with self._planner():
            self.client.get("/api/v2/models/m1/placement", headers=self.auth)
            refreshed = modelctl.resolve_planning_inputs.call_args[1]["refresh"]
        self.assertFalse(refreshed)

    def test_the_planner_is_handed_the_resolved_inputs(self):
        """Preview and the annotation must describe the same snapshot. If
        plan_for_selection resolved its own, the screen could name one
        machine and plan against another."""
        with self._planner() as p:
            self.client.get("/api/v2/models/m1/placement", headers=self.auth)
        self.assertEqual(p.call_args[1]["inputs"], self.INPUTS)

    def test_an_unplannable_model_is_a_409_not_an_empty_screen(self):
        with mock.patch("modelctl_plans.plan_for_selection",
                        return_value=None):
            r = self.client.get("/api/v2/models/m1/placement",
                                headers=self.auth)
        self.assertEqual(r.status_code, 409)
        self.assertIn("couldn't analyze", r.json()["error"])


class TestPlacementApply(Phase4Base):
    """POST /api/v2/models/{name}/placement -- run the model this way."""

    INPUTS = {"inventory": [{"device": "SYCL0", "name": "Arc A",
                             "total_bytes": 24 * GIB},
                            {"device": "SYCL1", "name": "Arc B",
                             "total_bytes": 12 * GIB}],
              "vram_limit_pct": 90, "primary": "SYCL0",
              "ram_available_bytes": 31 * GIB,
              "fleet_budgets": {"RPC:ph16-71-cuda0:CUDA0": 10 * GIB}}

    def _planner(self, plan=None):
        return _both(
            mock.patch("modelctl_plans.plan_for_selection",
                       return_value=plan or _plan_variant()),
            mock.patch.object(modelctl, "resolve_planning_inputs",
                              return_value=(self.INPUTS, "stored")))

    def _gate(self, requires):
        import modelctl_tiers
        return mock.patch.object(
            modelctl_tiers, "tier_change_gate",
            return_value={"kind": "structural" if requires else "none",
                          "changes": ["tier 3 -> 4"] if requires else [],
                          "requires_accept": requires})

    def test_applying_a_device_the_machine_does_not_have_is_refused(self):
        """Worse here than on the read: submit_placement_apply RECORDS the
        selection at profile.planning.selection, so an unknown key would
        persist as a stored intent naming a device that does not exist."""
        with self._planner(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply") as sub:
            r = self.client.post("/api/v2/models/m1/placement",
                                 headers=self.auth,
                                 json={"selection": {"NOPE0": {"on": True}}})
        self.assertEqual(r.status_code, 422)
        self.assertIn("NOPE0", r.json()["error"])
        self.assertFalse(sub.called, "nothing may be recorded for a device "
                                     "the machine does not have")

    def test_applying_a_key_the_machine_does_have_is_not_refused(self):
        with self._planner(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job1"):
            r = self.client.post(
                "/api/v2/models/m1/placement", headers=self.auth,
                json={"selection": {"RPC:ph16-71-cuda0:CUDA0": {"on": True}}})
        self.assertEqual(r.status_code, 200, r.text)

    def test_a_gated_placement_is_refused_until_confirmed(self):
        with self._planner(), self._gate(True), \
                mock.patch("modelctl_web.mutate.submit_placement_apply") as sub:
            r = self.client.post("/api/v2/models/m1/placement",
                                 headers=self.auth, json={"selection": {}})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["gate"]["changes"], ["tier 3 -> 4"])
        self.assertFalse(sub.called, "the gate must stop the submission, not "
                                     "just decorate it")

    def test_the_confirm_carries_the_selection_into_the_job(self):
        selection = {"RAM": {"on": False}, "SYCL0": {"ceiling_bytes": 4 * GIB}}
        with self._planner(), self._gate(True), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p") as sub:
            r = self.client.post(
                "/api/v2/models/m1/placement", headers=self.auth,
                json={"selection": selection, "accept_tier_change": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], "job-p")
        self.assertEqual(sub.call_args[0][2], selection)
        self.assertTrue(sub.call_args[1]["accept_tier_change"])

    def test_an_ungated_placement_applies_without_a_confirm(self):
        with self._planner(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p") as sub:
            r = self.client.post("/api/v2/models/m1/placement",
                                 headers=self.auth, json={"selection": {}})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(sub.call_args[1]["accept_tier_change"])

    def test_the_gate_is_read_against_the_selected_plan(self):
        """Not against the automatic one. A confirm shown for a placement
        the operator did not choose is a confirm for the wrong change."""
        plan = _plan_variant()
        import modelctl_tiers
        with self._planner(plan), \
                mock.patch.object(
                    modelctl_tiers, "tier_change_gate",
                    return_value={"kind": "none", "changes": [],
                                  "requires_accept": False}) as g, \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p"):
            self.client.post("/api/v2/models/m1/placement", headers=self.auth,
                             json={"selection": {"RAM": {"on": False}}})
        self.assertEqual(g.call_args[0][1], plan)

    def test_a_selection_that_is_not_an_object_is_refused(self):
        """Including an empty one. A list is a malformed selection whether
        or not it happens to be empty, and reading [] as "automatic" would
        place the model somewhere the caller never asked for."""
        for bad in (["SYCL0"], [], "SYCL0", 0):
            with self.subTest(selection=bad), \
                    mock.patch("modelctl_web.mutate."
                               "submit_placement_apply") as sub:
                r = self.client.post("/api/v2/models/m1/placement",
                                     headers=self.auth,
                                     json={"selection": bad})
                self.assertEqual(r.status_code, 422)
                self.assertFalse(sub.called)

    def test_a_missing_selection_is_the_automatic_placement(self):
        with self._planner(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p") as sub:
            r = self.client.post("/api/v2/models/m1/placement",
                                 headers=self.auth, json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sub.call_args[0][2], {})

    def test_an_unplannable_model_is_a_409_not_a_job_that_will_fail(self):
        with mock.patch("modelctl_plans.plan_for_selection",
                        return_value=None):
            r = self.client.post("/api/v2/models/m1/placement",
                                 headers=self.auth, json={"selection": {}})
        self.assertEqual(r.status_code, 409)
        self.assertIn("couldn't analyze", r.json()["error"])

    def test_a_fresh_apply_carries_the_re_read_into_the_job(self):
        """Applying what a re-read showed must record THAT machine. Without
        this the screen would show today's layout and save yesterday's
        snapshot underneath it."""
        with self._planner(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p") as sub:
            r = self.client.post("/api/v2/models/m1/placement",
                                 headers=self.auth,
                                 json={"selection": {}, "fresh": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(sub.call_args[1]["refresh"])

    def test_an_ordinary_apply_leaves_the_snapshot_alone(self):
        with self._planner(), self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p") as sub:
            self.client.post("/api/v2/models/m1/placement",
                             headers=self.auth, json={"selection": {}})
        self.assertFalse(sub.call_args[1]["refresh"])

    def test_the_preview_and_the_apply_plan_the_same_selection(self):
        """The button sits under the preview; if the two planned different
        selections the operator would confirm one layout and get another."""
        selection = {"RAM": {"on": False}}
        with self._planner() as p, self._gate(False), \
                mock.patch("modelctl_web.mutate.submit_placement_apply",
                           return_value="job-p"):
            self.client.get("/api/v2/models/m1/placement?on.RAM=0",
                            headers=self.auth)
            previewed = p.call_args[0][1]
            self.client.post("/api/v2/models/m1/placement", headers=self.auth,
                             json={"selection": selection})
            applied = p.call_args[0][1]
        self.assertEqual(previewed, selection)
        self.assertEqual(applied, selection)


class TestPlacementApplyJob(Phase4Base):
    """The work submit_placement_apply does inside the lane."""

    def _run(self, selection, accept=True, refresh=False):
        from modelctl_web import mutate
        runner = mock.Mock()
        inputs = {"inventory": [{"device": "SYCL0", "total_bytes": 24 * GIB}],
                  "ram_available_bytes": 20 * GIB, "vram_limit_pct": 90,
                  "primary": "SYCL0"}
        self.resolved = mock.Mock(return_value=(inputs, "stored"))
        with mock.patch("modelctl_plans.plan_for_selection",
                        return_value=_plan_variant()), \
                mock.patch.object(modelctl, "resolve_planning_inputs",
                                  self.resolved), \
                mock.patch.object(modelctl, "generate_artifacts"), \
                mock.patch("modelctl_web.mutate._sync_backends",
                           return_value=True):
            mutate.submit_placement_apply(runner, "m1", selection,
                                          accept_tier_change=accept,
                                          refresh=refresh)
            fn = runner.submit.call_args[0][2]
            result = fn(mock.Mock())
        saved = json.loads((self.profiles_dir / "m1.json").read_text())
        return result, saved, inputs

    def test_the_applied_selection_is_recorded_on_the_profile(self):
        """Reopening the screen has to show what is set to run, and the
        only honest source for that is what was applied."""
        selection = {"RAM": {"on": False}}
        _result, saved, _inputs = self._run(selection)
        self.assertEqual(saved["planning"]["selection"], selection)

    def test_the_recorded_inputs_are_the_machine_not_the_selection(self):
        """Planning inputs are the machine snapshot. Recording the FILTERED
        inputs would make the operator's ceiling look like a smaller
        computer, and every later automatic replan would inherit it."""
        _result, saved, inputs = self._run({"RAM": {"on": False}})
        self.assertEqual(saved["planning"]["inputs"], inputs)

    def test_the_plan_config_lands_on_the_profile(self):
        result, saved, _inputs = self._run({})
        self.assertTrue(result["applied"])
        self.assertEqual(saved["config"]["tensor_split"], "24,12")

    def test_a_gated_change_without_the_confirm_leaves_the_profile_alone(self):
        import modelctl_tiers
        before = (self.profiles_dir / "m1.json").read_text()
        with mock.patch.object(
                modelctl_tiers, "tier_change_gate",
                return_value={"kind": "structural", "changes": ["tier 3 -> 4"],
                              "requires_accept": True}):
            result, _saved, _inputs = self._run({}, accept=False)
        self.assertFalse(result["applied"])
        self.assertEqual((self.profiles_dir / "m1.json").read_text(), before)

    def test_a_fresh_apply_re_reads_the_machine_before_planning(self):
        """The job resolves inputs itself, so the refresh has to reach it --
        a screen that showed a re-read and then saved the old snapshot
        would be the same lie in the other direction."""
        self._run({}, refresh=True)
        self.assertTrue(self.resolved.call_args[1]["refresh"])

    def test_an_ordinary_apply_keeps_the_recorded_snapshot(self):
        self._run({})
        self.assertFalse(self.resolved.call_args[1]["refresh"])

    def test_an_unplannable_model_fails_the_job_loudly(self):
        from modelctl_web import mutate
        runner = mock.Mock()
        with mock.patch("modelctl_plans.plan_for_selection",
                        return_value=None):
            mutate.submit_placement_apply(runner, "m1", {})
            fn = runner.submit.call_args[0][2]
            with self.assertRaises(RuntimeError) as caught:
                fn(mock.Mock())
        self.assertIn("couldn't analyze", str(caught.exception))


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
            "/api/v2/models/{name}/placement",
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


class TestLegacyPin(Phase4Base):
    """Phase 3.1: a profile still launched by a pinned plan id says so.

    The placement surface owns intent; a pin names a compiled artifact.
    The two cannot be reconciled by this screen, so it reports the pin
    rather than rendering an automatic placement that is not what runs --
    which is exactly how qwen3-5-122b-a10b-ud came to read as "on
    automatic" while a four-device ladder plan was launching.
    """

    INPUTS = TestPlacementPreview.INPUTS
    RAM_TOTAL = TestPlacementPreview.RAM_TOTAL
    _planner = TestPlacementPreview._planner

    def _profile(self, **over):
        p = json.loads((self.profiles_dir / "m1.json").read_text())
        p.update(over)
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))
        return p

    def test_a_profile_on_a_pinned_plan_says_so(self):
        self._profile(runtime={"mode": "managed", "pinned_plan_id": "abc123"})
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertEqual(body["legacy_pin"]["plan_id"], "abc123")

    def test_a_selection_supersedes_the_pin(self):
        """The launcher ignores the pin once a selection exists, so the
        screen must not go on reporting it as what decides the launch."""
        self._profile(runtime={"mode": "managed", "pinned_plan_id": "abc123"},
                      planning={"selection": {"SYCL0": {"on": True}}})
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertIsNone(body["legacy_pin"])

    def test_a_profile_with_no_pin_reports_none(self):
        with self._planner():
            body = self.client.get("/api/v2/models/m1/placement",
                                   headers=self.auth).json()
        self.assertIsNone(body["legacy_pin"])


class TestAdoptingAPin(Phase4Base):
    """Phase 3.1: the equivalent selection is offered, never installed."""

    INPUTS = TestPlacementPreview.INPUTS
    RAM_TOTAL = TestPlacementPreview.RAM_TOTAL
    _planner = TestPlacementPreview._planner

    def _pinned(self, devices=("SYCL0",), ram=0):
        claim = mock.Mock()
        claim.vram_admission_bytes.return_value = {d: 4 * GIB
                                                   for d in devices}
        claim.ram_admission_bytes.return_value = ram
        return mock.Mock(id="abc123", label="tier 4 plan", claim=claim)

    def _with_pin(self):
        p = json.loads((self.profiles_dir / "m1.json").read_text())
        p["runtime"] = {"mode": "managed", "pinned_plan_id": "abc123"}
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))

    def test_the_offered_selection_names_the_devices_the_pin_uses(self):
        self._with_pin()
        with self._planner(), mock.patch(
                "modelctl_plans.compile_launch_plans",
                return_value=[self._pinned(devices=("SYCL0",))]):
            body = self.client.get("/api/v2/models/m1/placement/adopt-pin",
                                   headers=self.auth).json()
        self.assertTrue(body["selection"]["SYCL0"]["on"])
        self.assertFalse(body["selection"]["SYCL1"]["on"],
                         "a device the pinned plan does not use is off, not "
                         "absent -- absent means 'as the machine offers'")

    def test_the_offer_comes_with_the_layout_it_would_produce(self):
        """One click to preview, not one click to apply. The operator sees
        what adopting costs before anything is written."""
        self._with_pin()
        with self._planner(), mock.patch(
                "modelctl_plans.compile_launch_plans",
                return_value=[self._pinned()]):
            body = self.client.get("/api/v2/models/m1/placement/adopt-pin",
                                   headers=self.auth).json()
        self.assertIn("devices", body["placement"])
        self.assertIn("caveat", body)

    def test_nothing_is_written_by_the_offer(self):
        """A derivation that installed itself would be a fifth writer
        winning quietly, which is what this whole alignment removes."""
        self._with_pin()
        before = (self.profiles_dir / "m1.json").read_text()
        with self._planner(), mock.patch(
                "modelctl_plans.compile_launch_plans",
                return_value=[self._pinned()]):
            self.client.get("/api/v2/models/m1/placement/adopt-pin",
                            headers=self.auth)
        self.assertEqual((self.profiles_dir / "m1.json").read_text(), before)

    def test_a_profile_with_no_pin_has_nothing_to_adopt(self):
        with self._planner():
            r = self.client.get("/api/v2/models/m1/placement/adopt-pin",
                                headers=self.auth)
        self.assertEqual(r.status_code, 409)

    def test_a_pin_that_no_longer_compiles_says_so(self):
        """Plan ids hash the compiled config, so a planner improvement can
        strand a pin -- the 122B's moved cf4274bf -> 2b426a8f. That is the
        case adoption most needs to handle honestly."""
        self._with_pin()
        with self._planner(), mock.patch(
                "modelctl_plans.compile_launch_plans", return_value=[]):
            r = self.client.get("/api/v2/models/m1/placement/adopt-pin",
                                headers=self.auth)
        self.assertEqual(r.status_code, 409)
        self.assertIn("abc123", r.json()["error"])
