"""The console fleet surface: one read model, one writer, three states.

Two questions, as in phase 4. "Does this control submit the same thing
the primitive does?" -- a budget editor that writes the registry itself
would be a second writer with a second idea of the ceiling. And "does a
scratch instance refuse it, with a reason?" -- the refusal transcript is
how this surface is walked without moving a live node's budget.

The state machine gets its own class. PRESENT / STALE / PIN_MISMATCH is
not a nicety: a pin-mismatched node answers a handshake in milliseconds
and is unusable, so the one rendering that must be impossible is the one
where it looks available.

Carries its own isolation (tmp state, patched dirs) so single-file runs
never touch real state.
"""
import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_fleet as fleet
import modelctl_fsutil
import modelctl_nightlane
import modelctl_tiers
from modelctl_web import fleet as fleetview
from modelctl_web.app import create_app
from modelctl_web.jobs import JobRunner, JobStore

TOKEN = "test-token"
GIB = 1 << 30
PIN = "85b7e6556b6b83026d1a17df2635bc1173db1f97"
OTHER_PIN = "0000000000000000000000000000000000000000"

# Every mutating fleet endpoint, with a body that would be valid on a
# live instance. One list, used by the refusal transcript and by the
# route-table check, so an endpoint cannot join the surface and skip the
# scratch-safe assertion.
FLEET_WRITES = [
    ("/api/v2/fleet/probe", {}),
    ("/api/v2/fleet/nodes/ph16-71-cpu0/probe", {}),
    ("/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
     {"budget_bytes": 18 * GIB}),
]


def cpu_node(name="ph16-71-cpu0", pin=PIN, budget=16 * GIB, cap=20 * GIB,
             enabled=True):
    return fleet.FleetNode(
        name=name, host="192.168.0.76", port=50053, variant="cpu", pin=pin,
        enabled=enabled, note="rpc-cpu0.service (user unit)",
        devices=(fleet.FleetDevice(name="CPU", kind="cpu",
                                   total_bytes=30 * GIB, budget_bytes=budget,
                                   cap_bytes=cap),))


def gpu_node(name="ph16-71-cuda0", pin=PIN):
    return fleet.FleetNode(
        name=name, host="192.168.0.76", port=50052, variant="cuda", pin=pin,
        devices=(fleet.FleetDevice(name="CUDA0", kind="gpu",
                                   total_bytes=12 * GIB,
                                   budget_bytes=10 * GIB),))


def probe(node, reachable=True, agrees=True, age=0.0, detail=""):
    return fleet.NodeProbe(
        node=node, endpoint="192.168.0.76:50053", reachable=reachable,
        protocol="5.0.0" if reachable else "", pin=PIN, pin_agrees=agrees,
        detail=detail, probed_at=time.time() - age)


class FleetConsoleBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        self.fleet_path = self.root / "fleet.json"
        self.presence_path = self.root / "presence.json"
        self.night_path = self.root / "night-lane.json"
        for target, attr, val in (
                (fleet, "FLEET_PATH", self.fleet_path),
                (fleet, "PRESENCE_PATH", self.presence_path),
                (modelctl_nightlane, "REGISTRY_PATH", self.night_path),
                (modelctl_fsutil, "STATE_DIR", self.root),
                (modelctl, "PROFILES_DIR", self.profiles_dir)):
            p = mock.patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"MODELCTL_FLEET_PIN": PIN})
        p.start()
        self.addCleanup(p.stop)
        # The local node walks the real GPU inventory; these cases are
        # about node shape, not about this machine having a card. Kept
        # under its own name so the one case that IS about the real
        # assembler can still reach it.
        self._real_local_node = fleetview._local_node
        p = mock.patch.object(fleetview, "_local_node",
                              return_value=self.local_stub())
        p.start()
        self.addCleanup(p.stop)

    def local_stub(self):
        return {"name": "rig (this machine)", "location": "local",
                "host": "127.0.0.1", "port": None, "endpoint": "local",
                "variant": "local", "enabled": True, "note": "",
                "pin": {"node": PIN, "expected": PIN, "agrees": True},
                "presence": {"state": "PRESENT", "detail": "",
                             "reachable": True, "protocol": "",
                             "probed_at": None, "ttl_seconds": 900},
                "devices": [{"name": "SYCL0", "kind": "gpu", "label": "B70",
                             "budget_bytes": 28 * GIB,
                             "total_bytes": 32 * GIB, "cap_bytes": 0,
                             "ceiling_bytes": 32 * GIB,
                             "ceiling_basis": "reported device total",
                             "admission_key": "SYCL0", "editable": False,
                             "edit_note": "90% VRAM limit minus reserve"}]}


class ClientBase(FleetConsoleBase):
    def setUp(self):
        super().setUp()
        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.store, self.runner = store, runner
        self.addCleanup(lambda: runner._thread.join(timeout=0.05) or None)
        self.client = TestClient(create_app(token=TOKEN, store=store,
                                            runner=runner))
        self.auth = {"Authorization": f"Bearer {TOKEN}"}


class TestPresenceTriState(FleetConsoleBase):
    """The three states, and the fact that they are three."""

    def test_a_fresh_agreeing_probe_is_present(self):
        state, _ = fleetview.presence_state(probe("n", age=10))
        self.assertEqual(state, fleetview.PRESENT)

    def test_a_node_reachable_on_the_wrong_commit_is_pin_mismatch(self):
        """Up, fast, healthy protocol -- and unusable. This is the state a
        boolean cannot carry."""
        state, detail = fleetview.presence_state(
            probe("n", agrees=False, detail="node built at abc, checkout pins def"))
        self.assertEqual(state, fleetview.PIN_MISMATCH)
        self.assertIn("built at", detail)

    def test_pin_mismatch_outranks_freshness(self):
        """A probe from one second ago on the wrong commit is still a
        mismatch, not a present node with a caveat."""
        state, _ = fleetview.presence_state(probe("n", agrees=False, age=1))
        self.assertEqual(state, fleetview.PIN_MISMATCH)

    def test_an_expired_probe_is_stale_and_says_how_old(self):
        state, detail = fleetview.presence_state(probe("n", age=5000))
        self.assertEqual(state, fleetview.STALE)
        self.assertIn("presence TTL", detail)

    def test_a_node_never_probed_is_stale_for_that_reason(self):
        state, detail = fleetview.presence_state(None)
        self.assertEqual(state, fleetview.STALE)
        self.assertEqual(detail, "never probed")

    def test_an_unreachable_node_is_stale_with_the_error(self):
        state, detail = fleetview.presence_state(
            probe("n", reachable=False, detail="ConnectionRefusedError"))
        self.assertEqual(state, fleetview.STALE)
        self.assertIn("ConnectionRefused", detail)

    def test_the_three_states_are_distinct_strings(self):
        self.assertEqual(
            len({fleetview.PRESENT, fleetview.STALE, fleetview.PIN_MISMATCH}), 3)

    def test_presence_matches_what_the_planner_will_actually_use(self):
        """The view and usable_nodes must not disagree: a card saying
        PRESENT for a node the planner skips is a lie with a green dot."""
        fleet.save_fleet([cpu_node()])
        for pr, expect_usable in ((probe("ph16-71-cpu0"), True),
                                  (probe("ph16-71-cpu0", agrees=False), False),
                                  (probe("ph16-71-cpu0", age=5000), False),
                                  (probe("ph16-71-cpu0", reachable=False), False)):
            with self.subTest(detail=pr.detail, reachable=pr.reachable):
                fleet.save_presence([pr])
                state, _ = fleetview.presence_state(pr)
                usable = [n.name for n in fleet.usable_nodes()]
                self.assertEqual(state == fleetview.PRESENT,
                                 usable == ["ph16-71-cpu0"])


class TestReadModel(FleetConsoleBase):
    def test_the_rig_is_the_first_node_in_the_same_shape(self):
        fleet.save_fleet([cpu_node()])
        view = fleetview.fleet_view()
        local, remote = view["nodes"][0], view["nodes"][1]
        self.assertEqual(local["location"], "local")
        self.assertEqual(remote["location"], "remote")
        # same keys: one renderer, not two
        self.assertEqual(set(local), set(remote))
        self.assertEqual(set(local["devices"][0]) - {"label"},
                         set(remote["devices"][0]) - {"label"})

    def test_each_device_carries_budget_total_and_ceiling(self):
        fleet.save_fleet([cpu_node()])
        (device,) = fleetview.fleet_view()["nodes"][1]["devices"]
        self.assertEqual(device["budget_bytes"], 16 * GIB)
        self.assertEqual(device["total_bytes"], 30 * GIB)
        self.assertEqual(device["cap_bytes"], 20 * GIB)
        self.assertEqual(device["ceiling_bytes"],
                         20 * GIB - fleet.runtime_headroom(20 * GIB))
        self.assertEqual(device["ceiling_basis"], "systemd MemoryMax")

    def test_a_remote_device_is_editable_here(self):
        fleet.save_fleet([cpu_node()])
        self.assertTrue(
            fleetview.fleet_view()["nodes"][1]["devices"][0]["editable"])

    def test_the_real_local_node_is_read_only_and_says_where_it_is_edited(self):
        """One editor per number: the local budget is limit_pct of the
        card minus its reserve, and both knobs live on the settings page.
        Runs the real assembler, not the stub the other cases use."""
        import modelctl_hardware
        card = mock.Mock(device="SYCL0", name="Arc Pro B70",
                         total_bytes=32 * GIB, reserve_bytes=2 * GIB)
        with mock.patch.object(modelctl_hardware, "capture_hardware_snapshot",
                               return_value=mock.Mock()), \
             mock.patch.object(modelctl_hardware, "enabled_gpus",
                               return_value=[card]), \
             mock.patch.object(modelctl, "load_defaults",
                               return_value={"vram_limit_pct": 90}):
            local = self._real_local_node({})
        (device,) = local["devices"]
        self.assertFalse(device["editable"])
        self.assertIn("settings page", device["edit_note"])
        # the number it shows is the one admission actually charges
        self.assertEqual(device["budget_bytes"],
                         int(32 * GIB * 0.90) - 2 * GIB)
        self.assertEqual(local["presence"]["state"], fleetview.PRESENT)

    def test_pin_agreement_is_a_field_not_a_footnote(self):
        fleet.save_fleet([cpu_node(pin=OTHER_PIN)])
        node = fleetview.fleet_view()["nodes"][1]
        self.assertEqual(node["pin"]["node"], OTHER_PIN)
        self.assertEqual(node["pin"]["expected"], PIN)
        self.assertFalse(node["pin"]["agrees"])

    def test_the_view_opens_no_socket(self):
        """Rendering is not a network event -- probing is the explicit
        POST. A view that probed would make every page open a LAN sweep."""
        fleet.save_fleet([cpu_node(), gpu_node()])
        with mock.patch.object(fleet, "probe_node",
                               side_effect=AssertionError("probed on a read")), \
             mock.patch("socket.create_connection",
                        side_effect=AssertionError("opened a socket")):
            view = fleetview.fleet_view()
        self.assertEqual(len(view["nodes"]), 3)

    def test_a_corrupt_registry_degrades_to_the_local_node(self):
        self.fleet_path.write_text("{not json")
        view = fleetview.fleet_view()
        self.assertEqual([n["location"] for n in view["nodes"]], ["local"])

    def test_night_lane_rows_are_read_only_and_carry_enabled(self):
        self.night_path.write_text(json.dumps({"version": 1, "jobs": [
            {"id": "rpc-pair", "title": "a pre-registered pair",
             "question": "q", "criterion": "c", "measures": [],
             "mode": "paired", "pairs": 3, "enabled": False,
             "registered": "2026-08-02",
             "arms": [{"name": "local", "profile": "m1", "overrides": {},
                       "requires_nodes": []},
                      {"name": "rpc", "profile": "m1", "overrides": {},
                       "requires_nodes": ["ph16-71-cpu0"]}]},
            {"id": "local-only", "title": "needs no node", "question": "q",
             "criterion": "c", "measures": [], "enabled": True,
             "arms": [{"name": "a", "profile": "m1", "overrides": {},
                       "requires_nodes": []}]}]}))
        rows = fleetview.night_lane_rows()
        self.assertEqual([r["id"] for r in rows], ["rpc-pair"])
        self.assertFalse(rows[0]["enabled"])
        self.assertEqual(rows[0]["requires_nodes"], ["ph16-71-cpu0"])
        self.assertEqual(rows[0]["mode"], "paired")

    def test_stale_profiles_ride_along_with_the_view(self):
        fleet.save_fleet([cpu_node()])
        inputs = modelctl_tiers.make_planning_inputs(
            [], 90, "SYCL0", 31 * GIB,
            fleet_budgets={"RPC:ph16-71-cpu0:CPU": 8 * GIB})
        (self.profiles_dir / "m1.json").write_text(json.dumps(
            {"name": "m1", "config": {}, "planning": {"inputs": inputs}}))
        view = fleetview.fleet_view()
        self.assertEqual([p["name"] for p in view["stale_profiles"]], ["m1"])


class TestFleetRoutes(ClientBase):
    def test_the_read_endpoint_serves_the_view(self):
        fleet.save_fleet([cpu_node()])
        r = self.client.get("/api/v2/fleet", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual([n["name"] for n in body["nodes"]][1], "ph16-71-cpu0")
        json.dumps(body)  # SPA contract: serializable verbatim

    def test_budget_submits_through_the_one_mutation_entry(self):
        """Phase 4's entry is modelctl_web.mutate.submit_*; a budget edit
        that wrote the registry from the route would be a second path
        with a second idea of the rules."""
        fleet.save_fleet([cpu_node()])
        with mock.patch("modelctl_web.mutate.submit_fleet_budget",
                        return_value="job-b") as sub:
            r = self.client.post(
                "/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
                headers=self.auth, json={"budget_bytes": 18 * GIB})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], "job-b")
        self.assertEqual(sub.call_args[0][1:], ("ph16-71-cpu0", "CPU", 18 * GIB))

    def test_the_submitted_job_calls_the_step_one_primitive(self):
        """...and the primitive is where the ceiling, the lock and the
        staling live, so there is exactly one implementation of each."""
        from modelctl_web import mutate
        runner = mock.Mock()
        runner.submit.return_value = "job-b"
        with mock.patch.object(fleet, "set_device_budget",
                               return_value={"changed": True,
                                             "ceiling_bytes": 19 * GIB,
                                             "ceiling_basis": "systemd MemoryMax",
                                             "staled_profiles": []}) as prim:
            mutate.submit_fleet_budget(runner, "ph16-71-cpu0", "CPU", 18 * GIB)
            fn = runner.submit.call_args[0][2]
            fn(mock.Mock())
        prim.assert_called_once_with("ph16-71-cpu0", "CPU", 18 * GIB)
        self.assertEqual(runner.submit.call_args[1]["lane"], "mutation")

    def test_an_over_ceiling_budget_is_refused_before_it_is_queued(self):
        fleet.save_fleet([cpu_node()])
        with mock.patch("modelctl_web.mutate.submit_fleet_budget") as sub:
            r = self.client.post(
                "/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
                headers=self.auth, json={"budget_bytes": 25 * GIB})
        self.assertEqual(r.status_code, 422)
        self.assertIn("19.00 GiB", r.json()["error"])
        self.assertEqual(r.json()["ceiling_bytes"],
                         20 * GIB - fleet.runtime_headroom(20 * GIB))
        self.assertFalse(sub.called, "the refusal must stop the submission")

    def test_the_refusal_names_the_number_to_ask_for(self):
        """A refusal that says only "too big" sends the operator to a
        JSON file to find the ceiling."""
        fleet.save_fleet([cpu_node()])
        r = self.client.post(
            "/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
            headers=self.auth, json={"budget_bytes": 99 * GIB})
        self.assertIn("systemd MemoryMax", r.json()["error"])
        self.assertIn("ceiling", r.json()["error"])

    def test_unknown_node_and_device_are_404s(self):
        fleet.save_fleet([cpu_node()])
        for path in ("/api/v2/fleet/nodes/nope/devices/CPU/budget",
                     "/api/v2/fleet/nodes/ph16-71-cpu0/devices/GPU9/budget"):
            with self.subTest(path=path):
                r = self.client.post(path, headers=self.auth,
                                     json={"budget_bytes": GIB})
                self.assertEqual(r.status_code, 404)

    def test_a_non_integer_budget_is_a_422_naming_the_field(self):
        fleet.save_fleet([cpu_node()])
        for body in ({"budget_bytes": "lots"}, {}, {"budget_bytes": -1}):
            with self.subTest(body=body):
                r = self.client.post(
                    "/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
                    headers=self.auth, json=body)
                self.assertEqual(r.status_code, 422)
                self.assertIn("budget_bytes", r.json()["error"])

    def test_probe_records_presence_for_every_enabled_node(self):
        fleet.save_fleet([cpu_node(), cpu_node(name="off", enabled=False)])
        with mock.patch.object(fleet, "probe_node",
                               side_effect=lambda n, **kw: probe(n.name)) as pr:
            r = self.client.post("/api/v2/fleet/probe", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual([p["node"] for p in r.json()["probed"]],
                         ["ph16-71-cpu0"])
        self.assertEqual([c[0][0].name for c in pr.call_args_list],
                         ["ph16-71-cpu0"])
        self.assertIn("ph16-71-cpu0", fleet.load_presence())

    def test_probing_one_node_does_not_erase_the_others(self):
        """save_presence writes the whole file; a per-node probe that
        replaced it would make the other node read as never probed."""
        fleet.save_fleet([cpu_node(), gpu_node()])
        fleet.save_presence([probe("ph16-71-cpu0"), probe("ph16-71-cuda0")])
        with mock.patch.object(fleet, "probe_node",
                               side_effect=lambda n, **kw: probe(n.name)):
            r = self.client.post("/api/v2/fleet/nodes/ph16-71-cuda0/probe",
                                 headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(fleet.load_presence()),
                         ["ph16-71-cpu0", "ph16-71-cuda0"])

    def test_probing_an_unknown_node_is_a_404(self):
        fleet.save_fleet([cpu_node()])
        r = self.client.post("/api/v2/fleet/nodes/nope/probe", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_the_read_is_a_read(self):
        fleet.save_fleet([cpu_node()])
        before = self.fleet_path.read_text()
        self.client.get("/api/v2/fleet", headers=self.auth)
        self.assertEqual(self.fleet_path.read_text(), before)
        self.assertFalse(self.presence_path.exists())


class TestScratchSafeCoversTheFleet(FleetConsoleBase):
    """The scratch walk of this surface, as an assertion.

    A scratch console must be walkable without moving a live node's
    budget or reaching across the LAN. Every write here answers 405 with
    a reason naming the request, which is what the walk transcript is
    made of.
    """

    def setUp(self):
        super().setUp()
        p = mock.patch.dict(os.environ, {"MODELCTL_WEB_SCRATCH": "1"})
        p.start()
        self.addCleanup(p.stop)
        fleet.save_fleet([cpu_node()])
        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=0.05) or None)
        self.app = create_app(token=TOKEN, store=store, runner=runner)
        self.client = TestClient(self.app)
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    def test_every_fleet_write_is_refused_with_a_reason(self):
        for path, body in FLEET_WRITES:
            with self.subTest(path=path):
                r = self.client.post(path, headers=self.auth, json=body)
                self.assertEqual(r.status_code, 405, path)
                reason = r.json()["reason"]
                self.assertIn("scratch-safe mode", reason)
                self.assertIn(path, reason)

    def test_no_fleet_write_reaches_the_registry_or_the_wire(self):
        with mock.patch.object(fleet, "set_device_budget") as setter, \
             mock.patch.object(fleet, "probe_node") as prober, \
             mock.patch("modelctl_web.jobs.JobRunner.submit") as submit:
            for path, body in FLEET_WRITES:
                self.client.post(path, headers=self.auth, json=body)
        self.assertFalse(setter.called)
        self.assertFalse(prober.called)
        self.assertFalse(submit.called)

    def test_the_fleet_read_behaves_exactly_as_live(self):
        r = self.client.get("/api/v2/fleet", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["nodes"][1]["name"], "ph16-71-cpu0")

    def test_the_write_list_covers_every_mutating_fleet_route(self):
        """The app's own route table is the authority: a fleet write
        added and not listed would be walked past, not refused."""
        mutating = set()
        for route in self.app.routes:
            methods = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            if not (methods & {"POST", "PUT", "PATCH", "DELETE"}):
                continue
            if path.startswith("/api/v2/fleet"):
                mutating.add(path)
        self.assertEqual(mutating, {
            "/api/v2/fleet/probe",
            "/api/v2/fleet/nodes/{node}/probe",
            "/api/v2/fleet/nodes/{node}/devices/{device}/budget",
        })
        self.assertEqual(len(mutating), len(FLEET_WRITES))


if __name__ == "__main__":
    unittest.main()
