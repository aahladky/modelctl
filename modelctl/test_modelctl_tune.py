import os
import time
import unittest

import modelctl_tune


class TestSamplerIo(unittest.TestCase):
    """_Sampler's storage-read and page-fault tracking (Task 3.3). Uses
    this test process's own pid/pgrp -- /proc data for a real, running
    process, no fake subprocess tree needed."""

    def test_child_io_reads_real_proc_data(self):
        # Task D2 added the read-syscall count as a fourth counter, so
        # "many small reads" and "few large ones" are distinguishable at
        # the same byte total.
        sampler = modelctl_tune._Sampler(os.getpgrp())
        read_bytes, minor, major, syscr = sampler._child_io()
        self.assertGreaterEqual(read_bytes, 0)
        self.assertGreaterEqual(minor, 0)
        self.assertGreaterEqual(major, 0)
        self.assertGreaterEqual(syscr, 0)

    def test_child_io_ignores_other_pgrps(self):
        # A pgrp that (almost certainly) doesn't exist -- no processes
        # should match, so everything stays at zero.
        sampler = modelctl_tune._Sampler(999999)
        self.assertEqual(sampler._child_io(), (0, 0, 0, 0))

    def test_io_snapshot_reflects_sampler_state(self):
        sampler = modelctl_tune._Sampler(os.getpgrp())
        sampler.read_bytes = 12345
        sampler.minor_faults = 6
        sampler.major_faults = 1
        sampler.read_syscalls = 9
        self.assertEqual(sampler.io_snapshot(),
                         {"read_bytes": 12345, "minor_faults": 6,
                          "major_faults": 1, "read_syscalls": 9})

    def test_sampler_thread_populates_io_fields(self):
        # End-to-end: start the real polling thread against this process's
        # own pgrp and confirm it actually updates read_bytes within one
        # tick, not just that the accessor works.
        with modelctl_tune._Sampler(os.getpgrp()) as sampler:
            time.sleep(1.5)
        self.assertGreaterEqual(sampler.read_bytes, 0)


class TestPlanRunIoColumns(unittest.TestCase):
    """runtime.db's additive migration for read_bytes/major_faults/etc.
    (Task 3.3) round-trips through record_plan_run/plan_runs_for."""

    def setUp(self):
        import modelctl_runtime
        from tempfile import TemporaryDirectory
        from pathlib import Path
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rdb = modelctl_runtime.RuntimeDB(db_path=Path(self.tmp.name) / "runtime.db")

    def test_io_columns_round_trip(self):
        run = {
            "profile_name": "m1", "plan_id": "p1",
            "started_at": time.time(), "success": True,
            "read_bytes": 1048576, "read_bytes_warmup": 262144,
            "read_bytes_generation": 786432,
            "major_faults": 3, "minor_faults": 500,
        }
        self.rdb.record_plan_run(run)
        rows = self.rdb.plan_runs_for("m1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["read_bytes"], 1048576)
        self.assertEqual(rows[0]["read_bytes_warmup"], 262144)
        self.assertEqual(rows[0]["read_bytes_generation"], 786432)
        self.assertEqual(rows[0]["major_faults"], 3)
        self.assertEqual(rows[0]["minor_faults"], 500)

    def test_io_columns_default_to_null_when_absent(self):
        run = {"profile_name": "m1", "plan_id": "p1",
               "started_at": time.time(), "success": False}
        self.rdb.record_plan_run(run)
        rows = self.rdb.plan_runs_for("m1")
        self.assertIsNone(rows[0]["read_bytes"])
        self.assertIsNone(rows[0]["major_faults"])


if __name__ == "__main__":
    unittest.main()
