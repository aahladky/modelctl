"""The one writer of a fleet device's budget, and what a budget change
invalidates.

Before this, `budget_bytes` was a number only enrollment ever wrote: the
registry had a reader (`admission_budgets`) and no setter, so raising a
node's budget meant hand-editing fleet.json -- with no ceiling, no lock,
and nothing telling the profiles planned against the old number that it
had moved. Three properties are pinned here:

* **A budget cannot exceed what the node can actually give.** The
  ceiling is the node's own OS-enforced limit (systemd MemoryMax on a
  cpu unit) or the device total, minus runtime headroom. Over-ceiling is
  refused with the ceiling named -- the failure it prevents is an OOM
  kill on the far side of a 2.5GbE link, an hour into a load.
* **Two writers cannot lose each other's edit.** The registry is one
  JSON document; a read-modify-write without the state lock drops the
  concurrent edit silently.
* **A budget is a planning input.** Admission spends it exactly as it
  spends local VRAM, so moving it stales every stored-input plan built
  against the old number, and this says which ones.

Hermetic: tmp registry, tmp profile dir, tmp state lock, no sockets.
"""
import os
import shutil
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import modelctl
import modelctl_fleet as fleet
import modelctl_fsutil
import modelctl_tiers

GIB = 1 << 30
PIN = "85b7e6556b6b83026d1a17df2635bc1173db1f97"


def cpu_node(name="ph16-71-cpu0", budget=16 * GIB, cap=20 * GIB,
             total=30 * GIB, enabled=True):
    return fleet.FleetNode(
        name=name, host="192.168.0.76", port=50053, variant="cpu", pin=PIN,
        enabled=enabled,
        devices=(fleet.FleetDevice(name="CPU", kind="cpu", total_bytes=total,
                                   budget_bytes=budget, cap_bytes=cap),))


def gpu_node(name="ph16-71-cuda0", budget=10 * GIB, total=12 * GIB):
    return fleet.FleetNode(
        name=name, host="192.168.0.76", port=50052, variant="cuda", pin=PIN,
        devices=(fleet.FleetDevice(name="CUDA0", kind="gpu",
                                   total_bytes=total, budget_bytes=budget),))


class BudgetBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fleet-budget-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fleet_path = self.tmp / "fleet.json"
        self.profiles = self.tmp / "profiles"
        self.profiles.mkdir()
        for target, attr, val in (
                (fleet, "FLEET_PATH", self.fleet_path),
                (fleet, "PRESENCE_PATH", self.tmp / "presence.json"),
                # the flock must not land in the real state dir, where the
                # live console is holding the same file
                (modelctl_fsutil, "STATE_DIR", self.tmp),
                (modelctl, "PROFILES_DIR", self.profiles)):
            p = mock.patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"MODELCTL_FLEET_PIN": PIN})
        p.start()
        self.addCleanup(p.stop)

    def write_profile(self, name, fleet_budgets=None, no_fleet_block=False):
        """A profile with recorded planning inputs, optionally predating
        the fleet-budget field."""
        inputs = modelctl_tiers.make_planning_inputs(
            [], 90, "SYCL0", 31 * GIB, fleet_budgets=fleet_budgets or {})
        if no_fleet_block:
            inputs.pop("fleet")
        profile = {"name": name, "model_path": "/fake/m.gguf",
                   "config": {"ctx": 8192}, "planning": {"inputs": inputs}}
        (self.profiles / f"{name}.json").write_text(
            modelctl.json.dumps(profile))
        return profile


class TestCeiling(BudgetBase):
    def test_a_cpu_device_is_limited_by_its_memorymax_not_the_machine_ram(self):
        """The cgroup limit is what kills the process; the box having
        30 GiB is irrelevant next to a MemoryMax of 20G."""
        device = cpu_node().devices[0]
        ceiling, basis = fleet.device_ceiling(device)
        self.assertEqual(basis, "systemd MemoryMax")
        self.assertEqual(ceiling, 20 * GIB - fleet.runtime_headroom(20 * GIB))
        self.assertLess(ceiling, 30 * GIB)

    def test_a_cpu_device_with_no_recorded_cap_says_it_is_guessing(self):
        device = cpu_node(cap=0).devices[0]
        ceiling, basis = fleet.device_ceiling(device)
        self.assertIn("no MemoryMax recorded", basis)
        self.assertEqual(ceiling, 30 * GIB - fleet.runtime_headroom(30 * GIB))

    def test_a_gpu_device_is_limited_by_what_the_node_reports(self):
        device = gpu_node().devices[0]
        ceiling, basis = fleet.device_ceiling(device)
        self.assertEqual(basis, "reported device total")
        self.assertEqual(ceiling, 12 * GIB - fleet.runtime_headroom(12 * GIB))

    def test_headroom_has_a_floor_so_small_devices_keep_some(self):
        """5% of a 4 GiB card is 200 MiB, which the rpc server's own
        resident set alone exceeds."""
        self.assertEqual(fleet.runtime_headroom(4 * GIB), GIB)
        self.assertEqual(fleet.runtime_headroom(40 * GIB), 2 * GIB)

    def test_every_budget_in_the_shipped_registry_is_under_its_ceiling(self):
        """The ceiling is retroactive: a rule that condemns the budgets
        already in force would refuse the operator's next edit for a
        reason that has nothing to do with the edit."""
        for node in (cpu_node(), gpu_node()):
            for device in node.devices:
                ceiling, basis = fleet.device_ceiling(device)
                self.assertLessEqual(device.budget_bytes, ceiling,
                                     f"{node.name}/{device.name} ({basis})")


class TestSetBudget(BudgetBase):
    def test_a_budget_under_the_ceiling_is_written(self):
        fleet.save_fleet([cpu_node()])
        r = fleet.set_device_budget("ph16-71-cpu0", "CPU", 18 * GIB)
        self.assertTrue(r["changed"])
        self.assertEqual(r["previous_bytes"], 16 * GIB)
        self.assertEqual(r["budget_bytes"], 18 * GIB)
        (back,) = fleet.load_fleet()
        self.assertEqual(back.devices[0].budget_bytes, 18 * GIB)
        # everything else about the node survives the edit
        self.assertEqual(back.devices[0].cap_bytes, 20 * GIB)
        self.assertEqual(back.devices[0].total_bytes, 30 * GIB)
        self.assertEqual(back.pin, PIN)

    def test_an_over_ceiling_budget_is_refused_with_the_ceiling_named(self):
        fleet.save_fleet([cpu_node()])
        with self.assertRaises(ValueError) as caught:
            fleet.set_device_budget("ph16-71-cpu0", "CPU", 25 * GIB)
        msg = str(caught.exception)
        self.assertIn("19.00 GiB", msg)          # the ceiling itself
        self.assertIn("systemd MemoryMax", msg)  # and where it comes from
        # and nothing was written
        (back,) = fleet.load_fleet()
        self.assertEqual(back.devices[0].budget_bytes, 16 * GIB)

    def test_the_ceiling_moves_when_the_cap_does(self):
        """The point of step 2: raising MemoryMax on the node is what
        makes a bigger budget legal here."""
        fleet.save_fleet([cpu_node()])
        with self.assertRaises(ValueError):
            fleet.set_device_budget("ph16-71-cpu0", "CPU", 24 * GIB)
        fleet.set_device_cap("ph16-71-cpu0", "CPU", 26 * GIB)
        r = fleet.set_device_budget("ph16-71-cpu0", "CPU", 24 * GIB)
        self.assertEqual(r["budget_bytes"], 24 * GIB)
        self.assertEqual(r["ceiling_bytes"],
                         26 * GIB - fleet.runtime_headroom(26 * GIB))

    def test_a_negative_budget_is_refused(self):
        fleet.save_fleet([cpu_node()])
        with self.assertRaises(ValueError):
            fleet.set_device_budget("ph16-71-cpu0", "CPU", -1)

    def test_an_unknown_node_or_device_names_what_exists(self):
        fleet.save_fleet([cpu_node()])
        with self.assertRaises(KeyError) as caught:
            fleet.set_device_budget("nope", "CPU", GIB)
        self.assertIn("nope", str(caught.exception))
        with self.assertRaises(KeyError) as caught:
            fleet.set_device_budget("ph16-71-cpu0", "CUDA9", GIB)
        self.assertIn("CPU", str(caught.exception))

    def test_setting_the_same_number_is_not_a_write(self):
        fleet.save_fleet([cpu_node()])
        before = self.fleet_path.read_text()
        r = fleet.set_device_budget("ph16-71-cpu0", "CPU", 16 * GIB)
        self.assertFalse(r["changed"])
        self.assertEqual(self.fleet_path.read_text(), before)

    def test_one_node_edit_leaves_the_others_alone(self):
        fleet.save_fleet([cpu_node(), gpu_node()])
        fleet.set_device_budget("ph16-71-cpu0", "CPU", 18 * GIB)
        by_name = {n.name: n for n in fleet.load_fleet()}
        self.assertEqual(by_name["ph16-71-cuda0"].devices[0].budget_bytes,
                         10 * GIB)


class TestSetCap(BudgetBase):
    def test_recording_a_cap_does_not_move_the_budget(self):
        fleet.save_fleet([cpu_node()])
        r = fleet.set_device_cap("ph16-71-cpu0", "CPU", 26 * GIB)
        self.assertTrue(r["changed"])
        (back,) = fleet.load_fleet()
        self.assertEqual(back.devices[0].cap_bytes, 26 * GIB)
        self.assertEqual(back.devices[0].budget_bytes, 16 * GIB)

    def test_a_cap_below_the_budget_in_force_is_refused(self):
        """Otherwise the registry would hold a state its own setter could
        not reproduce."""
        fleet.save_fleet([cpu_node()])
        with self.assertRaises(ValueError) as caught:
            fleet.set_device_cap("ph16-71-cpu0", "CPU", 8 * GIB)
        self.assertIn("below", str(caught.exception))
        (back,) = fleet.load_fleet()
        self.assertEqual(back.devices[0].cap_bytes, 20 * GIB)


class TestConcurrentWritesSerialize(BudgetBase):
    def test_two_concurrent_budget_writes_both_survive(self):
        """Both threads read the whole registry, change one device and
        write the whole registry back. Without the state lock the second
        write is computed from a pre-first-write read and silently drops
        it -- the classic lost update, on a file two processes (CLI and
        console job worker) really do write."""
        fleet.save_fleet([cpu_node(), gpu_node()])
        real_load = fleet.load_fleet
        start = threading.Barrier(2)

        def slow_load(*a, **kw):
            nodes = real_load(*a, **kw)
            # widen the read-modify-write window so an unlocked
            # implementation loses the race deterministically
            try:
                start.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass
            return nodes

        errors = []

        def run(node, device, value):
            try:
                fleet.set_device_budget(node, device, value)
            except Exception as e:  # surfaced below, not swallowed
                errors.append(e)

        with mock.patch.object(fleet, "load_fleet", slow_load):
            threads = [
                threading.Thread(target=run,
                                 args=("ph16-71-cpu0", "CPU", 18 * GIB)),
                threading.Thread(target=run,
                                 args=("ph16-71-cuda0", "CUDA0", 9 * GIB)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        self.assertEqual(errors, [])
        by_name = {n.name: n for n in real_load()}
        self.assertEqual(by_name["ph16-71-cpu0"].devices[0].budget_bytes,
                         18 * GIB)
        self.assertEqual(by_name["ph16-71-cuda0"].devices[0].budget_bytes,
                         9 * GIB)


class TestBudgetIsAPlanningInput(BudgetBase):
    def test_budget_input_reads_the_registry_not_presence(self):
        """admission_budgets drops an unreachable node -- correct for
        "what may this spend now", wrong for "what did the operator
        declare". A closed laptop must not read as an input change."""
        fleet.save_fleet([cpu_node(), gpu_node()])
        self.assertEqual(fleet.admission_budgets(), {})  # never probed
        self.assertEqual(fleet.budget_input(), {
            "RPC:ph16-71-cpu0:CPU": 16 * GIB,
            "RPC:ph16-71-cuda0:CUDA0": 10 * GIB,
        })

    def test_a_disabled_node_is_not_an_input(self):
        fleet.save_fleet([cpu_node(enabled=False)])
        self.assertEqual(fleet.budget_input(), {})

    def test_a_replan_records_the_budget_in_force(self):
        fleet.save_fleet([cpu_node()])
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]), \
             mock.patch.object(modelctl, "load_defaults",
                               return_value={"vram_limit_pct": 90,
                                             "primary_gpu": "SYCL0"}), \
             mock.patch.object(modelctl, "resolve_primary_gpu",
                               return_value="SYCL0"), \
             mock.patch("modelctl_vram.system_ram_available",
                        return_value=31 * GIB):
            inputs, source = modelctl.resolve_planning_inputs(
                {"name": "m1"}, refresh=True)
        self.assertEqual(source, "live")
        self.assertEqual(
            modelctl_tiers.planning_input_fleet_budgets(inputs),
            {"RPC:ph16-71-cpu0:CPU": 16 * GIB})

    def test_staling_fires_for_the_profiles_planned_against_the_old_number(self):
        fleet.save_fleet([cpu_node()])
        self.write_profile("depends",
                           fleet_budgets={"RPC:ph16-71-cpu0:CPU": 16 * GIB})
        self.write_profile("already-current",
                           fleet_budgets={"RPC:ph16-71-cpu0:CPU": 18 * GIB})
        r = fleet.set_device_budget("ph16-71-cpu0", "CPU", 18 * GIB)
        staled = [p["name"] for p in r["staled_profiles"]]
        self.assertEqual(staled, ["depends"])
        self.assertEqual(
            r["staled_profiles"][0]["changed"]["RPC:ph16-71-cpu0:CPU"],
            {"recorded": 16 * GIB, "live": 18 * GIB})

    def test_a_record_predating_the_field_makes_no_claim(self):
        """A profile planned before budgets were an input cannot be said
        to disagree with one -- that is silence, not drift."""
        fleet.save_fleet([cpu_node()])
        self.write_profile("legacy", no_fleet_block=True)
        r = fleet.set_device_budget("ph16-71-cpu0", "CPU", 18 * GIB)
        self.assertEqual(r["staled_profiles"], [])

    def test_a_profile_with_no_recorded_inputs_has_nothing_to_stale(self):
        fleet.save_fleet([cpu_node()])
        (self.profiles / "fresh.json").write_text(
            modelctl.json.dumps({"name": "fresh", "config": {}}))
        r = fleet.set_device_budget("ph16-71-cpu0", "CPU", 18 * GIB)
        self.assertEqual(r["staled_profiles"], [])

    def test_enrolling_a_node_stales_a_profile_that_recorded_none(self):
        """{} and None are different claims: a profile that recorded "no
        fleet budgets" IS contradicted by a node enrolled afterwards."""
        fleet.save_fleet([cpu_node()])
        self.write_profile("planned-fleet-free", fleet_budgets={})
        staled = fleet.stale_input_profiles()
        self.assertEqual([p["name"] for p in staled], ["planned-fleet-free"])

    def test_the_mismatch_line_names_both_numbers(self):
        inputs = modelctl_tiers.make_planning_inputs(
            [], 90, "SYCL0", 31 * GIB,
            fleet_budgets={"RPC:ph16-71-cpu0:CPU": 16 * GIB})
        line = modelctl_tiers.fleet_budget_mismatch(
            inputs, {"RPC:ph16-71-cpu0:CPU": 24 * GIB})
        self.assertIn("16.00 GiB -> 24.00 GiB", line)
        self.assertIn("--refresh-inputs", line)
        self.assertIsNone(modelctl_tiers.fleet_budget_mismatch(
            inputs, {"RPC:ph16-71-cpu0:CPU": 16 * GIB}))

    def test_a_legacy_record_never_produces_a_mismatch_line(self):
        inputs = modelctl_tiers.make_planning_inputs([], 90, "SYCL0", 31 * GIB)
        inputs.pop("fleet")
        self.assertIsNone(modelctl_tiers.fleet_budget_mismatch(
            inputs, {"RPC:ph16-71-cpu0:CPU": 24 * GIB}))


class TestCli(BudgetBase):
    def _run(self, **kw):
        import io
        import contextlib
        args = types.SimpleNamespace(**kw)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = modelctl.cmd_fleet(args)
        return code, out.getvalue(), err.getvalue()

    def test_set_budget_exercises_the_primitive(self):
        fleet.save_fleet([cpu_node()])
        code, out, _ = self._run(fleet_command="set-budget",
                                 node="ph16-71-cpu0", device="CPU",
                                 bytes=18 * GIB)
        self.assertEqual(code, 0)
        self.assertIn("16.00 GiB -> 18.00 GiB", out)
        self.assertIn("ceiling 19.00 GiB", out)
        (back,) = fleet.load_fleet()
        self.assertEqual(back.devices[0].budget_bytes, 18 * GIB)

    def test_an_over_ceiling_request_exits_nonzero_with_the_reason(self):
        fleet.save_fleet([cpu_node()])
        code, _, err = self._run(fleet_command="set-budget",
                                 node="ph16-71-cpu0", device="CPU",
                                 bytes=25 * GIB)
        self.assertEqual(code, 2)
        self.assertIn("refused", err)
        self.assertIn("19.00 GiB", err)

    def test_the_cli_reports_which_profiles_went_stale(self):
        fleet.save_fleet([cpu_node()])
        self.write_profile("depends",
                           fleet_budgets={"RPC:ph16-71-cpu0:CPU": 16 * GIB})
        _, out, _ = self._run(fleet_command="set-budget",
                              node="ph16-71-cpu0", device="CPU",
                              bytes=18 * GIB)
        self.assertIn("stale planning inputs: depends", out)

    def test_set_cap_is_reachable_from_the_shell(self):
        fleet.save_fleet([cpu_node()])
        code, out, _ = self._run(fleet_command="set-cap",
                                 node="ph16-71-cpu0", device="CPU",
                                 bytes=26 * GIB)
        self.assertEqual(code, 0)
        self.assertIn("20.00 GiB -> 26.00 GiB", out)
        (back,) = fleet.load_fleet()
        self.assertEqual(back.devices[0].cap_bytes, 26 * GIB)


if __name__ == "__main__":
    unittest.main()
