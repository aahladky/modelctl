"""Who holds the memory: the display fold behind every RAM row.

The admission fold (`charged_bytes`) deliberately charges an ACTIVE
row's `RPC:` keys and nothing else -- local free bytes already have a
running model subtracted, so counting it again would refuse a launch
that fits. That is correct for the gate and useless for a screen: free
memory shrinks with no holder named, and "no room" renders exactly like
"no room because the model you are looking at already holds it"
(open-items 2026-08-04, the entry that blocks every memory view).

`live_holdings` is the other answer: every live reservation contributes
everything it holds, local keys and RAM included, attributed to the
profile holding it. The fleet view writes it onto device rows as
`held`/`held_bytes`; the placement answer annotates its rows with
`held_bytes` and `held_by_this_model_bytes`.

Hermetic: every RuntimeDB here is a throwaway file, `_pid_alive` is
mocked where a live PID would decide, and the fleet/placement cases
stub `RuntimeDB` so no test reads the rig's real runtime.db.
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modelctl_runtime


GIB = 1 << 30


class HoldingsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ram-holdings-")
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.db_path = Path(self.tmp) / "rt.db"
        self.rdb = modelctl_runtime.RuntimeDB(self.db_path)
        self.pid = os.getpid()
        self.budgets = {"SYCL0": 30 * GIB, "RAM": 30 * GIB,
                        "RPC:ph16-71-cpu0:CPU": 25 * GIB}

    def _hold(self, vram, ram=0, state="active", pid=None, name="holder"):
        with mock.patch.object(modelctl_runtime, "_pid_alive",
                               return_value=True):
            res = self.rdb.acquire_reservation(
                name, "hp", {"vram_bytes": vram, "ram_bytes": ram},
                self.pid + 1 if pid is None else pid, budgets=self.budgets)
        self.assertIsNotNone(res)
        if state != "pending":
            self.rdb.update_reservation(res["id"], state=state)
        return res

    def _holdings(self, alive=True, exclude_pid=None):
        with mock.patch.object(modelctl_runtime, "_pid_alive",
                               return_value=alive):
            return self.rdb.live_holdings(exclude_pid=exclude_pid)


class TestLiveHoldingsNameTheHolder(HoldingsBase):
    """The fold itself: attribution, hygiene, and the split from the gate."""

    def test_an_active_row_holds_its_local_ram_and_devices(self):
        self._hold({"SYCL0": 6 * GIB}, ram=12 * GIB, name="laguna")
        holdings = self._holdings()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["profile"], "laguna")
        self.assertEqual(holdings[0]["state"], "active")
        self.assertEqual(holdings[0]["devices"],
                         {"SYCL0": 6 * GIB, "RAM": 12 * GIB})

    def test_the_gate_view_still_omits_what_the_display_view_names(self):
        # The reason this fold exists at all: same row, two answers,
        # both correct for their question.
        self._hold({"SYCL0": 6 * GIB}, ram=12 * GIB)
        with mock.patch.object(modelctl_runtime, "_pid_alive",
                               return_value=True):
            charged = self.rdb.charged_bytes()
        self.assertNotIn("RAM", charged)
        self.assertNotIn("SYCL0", charged)
        held = self._holdings()[0]["devices"]
        self.assertEqual(held["RAM"], 12 * GIB)
        self.assertEqual(held["SYCL0"], 6 * GIB)

    def test_a_pending_row_is_a_holder_too(self):
        self._hold({"SYCL0": 2 * GIB}, ram=1 * GIB, state="pending",
                   name="loading")
        holdings = self._holdings()
        self.assertEqual(holdings[0]["state"], "pending")
        self.assertEqual(holdings[0]["devices"]["RAM"], 1 * GIB)

    def test_remote_keys_ride_along(self):
        self._hold({"RPC:ph16-71-cpu0:CPU": 24 * GIB}, name="q122b")
        held = self._holdings()[0]["devices"]
        self.assertEqual(held["RPC:ph16-71-cpu0:CPU"], 24 * GIB)

    def test_a_dead_owner_holds_nothing(self):
        self._hold({"SYCL0": 6 * GIB}, ram=12 * GIB)
        self.assertEqual(self._holdings(alive=False), [])

    def test_the_excluded_pid_holds_nothing(self):
        self._hold({"SYCL0": 6 * GIB}, pid=4242)
        self.assertEqual(self._holdings(exclude_pid=4242), [])

    def test_zero_ram_yields_no_ram_key(self):
        self._hold({"SYCL0": 6 * GIB}, ram=0)
        self.assertNotIn("RAM", self._holdings()[0]["devices"])

    def test_a_claim_that_does_not_decode_is_skipped_not_raised(self):
        self._hold({"SYCL0": 6 * GIB}, name="good")
        self._hold({"SYCL0": 1 * GIB}, name="bad")
        with sqlite3.connect(self.db_path) as c:
            c.execute("UPDATE reservations SET claim_json='not json' "
                      "WHERE profile_name='bad'")
        holdings = self._holdings()
        self.assertEqual([h["profile"] for h in holdings], ["good"])

    def test_unpriceable_byte_counts_are_dropped(self):
        self._hold({"SYCL0": 1 * GIB}, name="messy")
        with sqlite3.connect(self.db_path) as c:
            c.execute("UPDATE reservations SET claim_json=? "
                      "WHERE profile_name='messy'",
                      ('{"vram_bytes": {"SYCL0": "lots", "SYCL1": -5},'
                       ' "ram_bytes": -1}',))
        holdings = self._holdings()
        # Nothing priceable left: the row is not a holder at all.
        self.assertEqual(holdings, [])


class TestFleetRowsCarryHeld(unittest.TestCase):
    """The fleet view writes holdings onto the device rows that hold them."""

    def _node(self, keys):
        return {"name": "rig", "location": "local", "devices": [
            {"name": k.split(":")[-1], "admission_key": k} for k in keys]}

    def _attach(self, holdings, nodes):
        from modelctl_web import fleet as fleetview
        rdb = mock.Mock()
        rdb.live_holdings.return_value = holdings
        with mock.patch.object(modelctl_runtime, "RuntimeDB",
                               return_value=rdb):
            fleetview._attach_holdings(nodes)
        return nodes

    def test_held_lands_on_the_matching_admission_key(self):
        nodes = [self._node(["SYCL0", "RAM"])]
        holdings = [{"profile": "laguna", "state": "active",
                     "devices": {"SYCL0": 6 * GIB, "RAM": 12 * GIB}}]
        nodes = self._attach(holdings, nodes)
        ram_row = nodes[0]["devices"][1]
        self.assertEqual(ram_row["held_bytes"], 12 * GIB)
        self.assertEqual(ram_row["held"],
                         [{"profile": "laguna", "bytes": 12 * GIB}])

    def test_two_holders_sum_and_both_are_named(self):
        nodes = [self._node(["RAM"])]
        holdings = [
            {"profile": "a", "state": "active", "devices": {"RAM": 4 * GIB}},
            {"profile": "b", "state": "pending", "devices": {"RAM": 2 * GIB}},
        ]
        nodes = self._attach(holdings, nodes)
        row = nodes[0]["devices"][0]
        self.assertEqual(row["held_bytes"], 6 * GIB)
        self.assertEqual({h["profile"] for h in row["held"]}, {"a", "b"})

    def test_a_device_nothing_holds_says_zero_not_nothing(self):
        nodes = [self._node(["SYCL1"])]
        nodes = self._attach([], nodes)
        row = nodes[0]["devices"][0]
        self.assertEqual(row["held_bytes"], 0)
        self.assertEqual(row["held"], [])

    def test_remote_rows_get_their_rpc_holdings(self):
        nodes = [self._node(["RPC:ph16-71-cpu0:CPU"])]
        holdings = [{"profile": "q122b", "state": "active",
                     "devices": {"RPC:ph16-71-cpu0:CPU": 24 * GIB}}]
        nodes = self._attach(holdings, nodes)
        row = nodes[0]["devices"][0]
        self.assertEqual(row["held_bytes"], 24 * GIB)

    def test_an_unreadable_runtime_db_degrades_to_an_error_not_a_crash(self):
        from modelctl_web import fleet as fleetview
        errors = {}
        nodes = [self._node(["RAM"])]
        with mock.patch.object(modelctl_runtime, "RuntimeDB",
                               side_effect=OSError("locked")):
            # fleet_view guards the attach with exactly this shape.
            try:
                fleetview._attach_holdings(nodes)
            except Exception as e:
                errors["holdings"] = str(e)
        self.assertIn("holdings", errors)
        # The row survives without the fields rather than lying with 0.
        self.assertNotIn("held_bytes", nodes[0]["devices"][0])


class TestPlacementRowsNameTheirOwnHolding(unittest.TestCase):
    """The placement answer's rows can finally tell the two apart:
    "no room" and "no room because this model already holds it"."""

    def _devices(self):
        return {"RAM": {"bytes": 5 * GIB, "backing": "RAM", "fits": True,
                        "capacity_bytes": 31 * GIB,
                        "usable_bytes": 20 * GIB},
                "SYCL0": {"bytes": 27 * GIB, "backing": "VRAM", "fits": True,
                          "capacity_bytes": 31 * GIB,
                          "usable_bytes": 28 * GIB}}

    def _annotate(self, devices, profile, holdings):
        from modelctl_web import hub
        rdb = mock.Mock()
        rdb.live_holdings.return_value = holdings
        with mock.patch.object(modelctl_runtime, "RuntimeDB",
                               return_value=rdb):
            return hub._annotate_holdings(devices, profile)

    def test_the_models_own_holding_is_split_from_the_total(self):
        devices = self._devices()
        holdings = [
            {"profile": "q122b", "state": "active",
             "devices": {"RAM": 5 * GIB, "SYCL0": 27 * GIB}},
            {"profile": "other", "state": "active",
             "devices": {"RAM": 2 * GIB}},
        ]
        warn = self._annotate(devices, "q122b", holdings)
        self.assertIsNone(warn)
        self.assertEqual(devices["RAM"]["held_bytes"], 7 * GIB)
        self.assertEqual(devices["RAM"]["held_by_this_model_bytes"], 5 * GIB)
        self.assertEqual(devices["SYCL0"]["held_by_this_model_bytes"],
                         27 * GIB)

    def test_nothing_running_reads_as_zero_held(self):
        devices = self._devices()
        warn = self._annotate(devices, "q122b", [])
        self.assertIsNone(warn)
        self.assertEqual(devices["RAM"]["held_bytes"], 0)
        self.assertEqual(devices["RAM"]["held_by_this_model_bytes"], 0)

    def test_a_holding_on_a_device_the_answer_does_not_draw_is_ignored(self):
        devices = self._devices()
        holdings = [{"profile": "q122b", "state": "active",
                     "devices": {"RPC:gone:CPU": 3 * GIB}}]
        warn = self._annotate(devices, "q122b", holdings)
        self.assertIsNone(warn)
        self.assertEqual(devices["RAM"]["held_bytes"], 0)

    def test_an_unreadable_runtime_db_returns_a_warning_and_no_fields(self):
        from modelctl_web import hub
        devices = self._devices()
        with mock.patch.object(modelctl_runtime, "RuntimeDB",
                               side_effect=OSError("locked")):
            warn = hub._annotate_holdings(devices, "q122b")
        self.assertIsInstance(warn, str)
        self.assertIn("held", warn)
        # Absent, not zero: a zero here would claim "nothing is held"
        # on the strength of not having been able to look.
        self.assertNotIn("held_bytes", devices["RAM"])


if __name__ == "__main__":
    unittest.main()
