"""Reservation denials say WHY: capacity shortfall vs live-worker conflict.

Pins the fix for the 2026-08-03 baseline finding: a plan whose claim can
never fit the budgets was reported as "reservation conflict -- another
worker holds the resources" with zero other workers alive
(2026-08-03-time-to-serve-baseline.md). Capacity denials must carry the
shortfall (insufficient_vram / insufficient_ram, well-known codes from
modelctl_errors.CODES); reservation_conflict stays reserved for genuine
collisions with live holders.
"""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import modelctl_runtime

GIB = 1 << 30


class TestAcquireVerdict(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rdb = modelctl_runtime.RuntimeDB(Path(self.tmp.name) / "rt.db")
        self.pid = os.getpid()

    def test_success_returns_reservation_and_no_denial(self):
        res, denial = self.rdb.acquire_reservation_verdict(
            "a", "p1", {"vram_bytes": {"SYCL0": 6 * GIB}, "ram_bytes": 0},
            self.pid, budgets={"SYCL0": 10 * GIB, "RAM": 0})
        self.assertIsNotNone(res)
        self.assertIsNone(denial)

    def test_capacity_denial_names_the_shortfall(self):
        res, denial = self.rdb.acquire_reservation_verdict(
            "a", "p1", {"vram_bytes": {"SYCL0": 12 * GIB}, "ram_bytes": 0},
            self.pid, budgets={"SYCL0": 10 * GIB, "RAM": 0})
        self.assertIsNone(res)
        self.assertEqual(denial["code"], "insufficient_vram")
        self.assertEqual(denial["resource"], "SYCL0")
        self.assertEqual(denial["need_bytes"], 12 * GIB)
        self.assertEqual(denial["budget_bytes"], 10 * GIB)
        self.assertEqual(denial["pending_bytes"], 0)
        self.assertEqual(denial["holders"], [])

    def test_ram_capacity_denial(self):
        res, denial = self.rdb.acquire_reservation_verdict(
            "a", "p1", {"vram_bytes": {}, "ram_bytes": 9 * GIB},
            self.pid, budgets={"SYCL0": 10 * GIB, "RAM": 8 * GIB})
        self.assertIsNone(res)
        self.assertEqual(denial["code"], "insufficient_ram")
        self.assertEqual(denial["resource"], "RAM")
        self.assertEqual(denial["need_bytes"], 9 * GIB)
        self.assertEqual(denial["budget_bytes"], 8 * GIB)

    def test_capacity_denial_lists_live_holders(self):
        with mock.patch.object(modelctl_runtime, "_pid_alive",
                               return_value=True):
            first = self.rdb.acquire_reservation(
                "holder-profile", "p1",
                {"vram_bytes": {"SYCL0": 6 * GIB}, "ram_bytes": 0},
                self.pid + 1, budgets={"SYCL0": 10 * GIB, "RAM": 0})
            self.assertIsNotNone(first)
            res, denial = self.rdb.acquire_reservation_verdict(
                "b", "p2", {"vram_bytes": {"SYCL0": 5 * GIB}, "ram_bytes": 0},
                self.pid, budgets={"SYCL0": 10 * GIB, "RAM": 0})
        self.assertIsNone(res)
        self.assertEqual(denial["code"], "insufficient_vram")
        self.assertEqual(denial["pending_bytes"], 6 * GIB)
        self.assertEqual(denial["holders"], ["holder-profile"])

    def test_legacy_overlap_is_a_conflict(self):
        with mock.patch.object(modelctl_runtime, "_pid_alive",
                               return_value=True):
            first = self.rdb.acquire_reservation(
                "holder-profile", "p1",
                {"vram_bytes": {"SYCL0": 1 * GIB}}, self.pid + 1)
            self.assertIsNotNone(first)
            res, denial = self.rdb.acquire_reservation_verdict(
                "b", "p2", {"vram_bytes": {"SYCL0": 1 * GIB}}, self.pid)
        self.assertIsNone(res)
        self.assertEqual(denial["code"], "reservation_conflict")
        self.assertEqual(denial["holders"], ["holder-profile"])

    def test_wrapper_backcompat(self):
        ok = self.rdb.acquire_reservation(
            "a", "p1", {"vram_bytes": {"SYCL0": 6 * GIB}, "ram_bytes": 0},
            self.pid, budgets={"SYCL0": 10 * GIB, "RAM": 0})
        self.assertIsInstance(ok, dict)
        denied = self.rdb.acquire_reservation(
            "b", "p2", {"vram_bytes": {"SYCL0": 12 * GIB}, "ram_bytes": 0},
            self.pid, budgets={"SYCL0": 10 * GIB, "RAM": 0})
        self.assertIsNone(denied)


class TestTuneDenialMessage(unittest.TestCase):
    """test_launch_plan surfaces the denial, not a blanket conflict."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import modelctl_tune
        self.tune = modelctl_tune
        self.rdb = modelctl_runtime.RuntimeDB(Path(self.tmp.name) / "rt.db")

    def _run(self, vram_need_gib=12, budget_gib=10):
        plan = SimpleNamespace(
            id="p1", label="fake plan",
            claim=SimpleNamespace(
                vram_admission_bytes=lambda: {"SYCL0": vram_need_gib * GIB},
                ram_admission_bytes=lambda: 0,
                storage_mode="mmap", vram_cache_bytes={},
                ram_resident_bytes=0, mmap_bytes=0))
        snap = SimpleNamespace(
            fingerprint="hw", backend_fingerprints={},
            ram_available_bytes=8 * GIB, ram_reserve_bytes=0)
        gpu = SimpleNamespace(device="SYCL0", free_bytes=budget_gib * GIB,
                              reserve_bytes=0)
        profile = {"name": "prof", "backend": "llama-cpp"}
        with mock.patch.object(self.tune.modelctl_hardware,
                               "capture_hardware_snapshot",
                               return_value=snap), \
             mock.patch.object(self.tune.modelctl_hardware, "enabled_gpus",
                               return_value=[gpu]), \
             mock.patch.object(self.tune.modelctl_plans,
                               "compile_launch_plans", return_value=[plan]), \
             mock.patch.object(self.tune.modelctl_runtime, "RuntimeDB",
                               return_value=self.rdb):
            self.tune.test_launch_plan("prof", "p1",
                                       profile_override=profile)

    def test_capacity_denial_message_and_class(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(vram_need_gib=12, budget_gib=10)
        msg = str(ctx.exception)
        self.assertIn("12.0 GiB", msg)
        self.assertIn("SYCL0", msg)
        self.assertIn("10.0 GiB", msg)
        self.assertNotIn("another worker", msg)
        with self.rdb._conn() as c:
            rows = c.execute(
                "SELECT failure_class FROM plan_runs").fetchall()
        self.assertEqual([r["failure_class"] for r in rows],
                         ["insufficient_vram"])


if __name__ == "__main__":
    unittest.main()
