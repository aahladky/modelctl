"""Task D2: process memory and I/O sampling, and honest storage claims.

The rule these enforce: what the storage did is read off counters, never
inferred from `mmap=true`. A fully page-cached mmap model reads nothing
from disk, and calling that "SSD streaming" is how a plan gets rejected
for a cost it never paid.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modelctl_tune

MB = 1 << 20
GB = 1 << 30


class TestStorageActivityClassification(unittest.TestCase):
    def test_mmap_with_no_reads_is_page_cache_served(self):
        label, why = modelctl_tune.classify_storage_activity(
            read_bytes=1 * MB, major_faults=12, storage_mode="mmap")
        self.assertEqual(label, "page-cache-served")
        self.assertIn("page cache", why)

    def test_mmap_alone_does_not_imply_storage_backed(self):
        # The specific error being prevented: storage_mode says the model
        # *can* be paged from disk, not that it was.
        label, _ = modelctl_tune.classify_storage_activity(
            read_bytes=0, major_faults=0, storage_mode="mmap")
        self.assertNotEqual(label, "storage-backed")

    def test_heavy_faults_and_reads_are_storage_backed(self):
        label, why = modelctl_tune.classify_storage_activity(
            read_bytes=8 * GB, major_faults=250_000, storage_mode="mmap",
            elapsed_seconds=100)
        self.assertEqual(label, "storage-backed")
        self.assertIn("paged in", why)
        self.assertIn("/s", why)

    def test_bulk_read_without_faults_reads_as_model_load(self):
        # Loading 20 GiB through read() is not the same as paging during
        # generation, and must not be reported as if it were.
        label, why = modelctl_tune.classify_storage_activity(
            read_bytes=20 * GB, major_faults=10, storage_mode="none")
        self.assertEqual(label, "bulk-read")
        self.assertIn("model load", why)

    def test_no_activity_without_mmap_is_reported_plainly(self):
        label, _ = modelctl_tune.classify_storage_activity(
            read_bytes=0, major_faults=0, storage_mode="none")
        self.assertEqual(label, "no-storage-activity")

    def test_faults_without_reads_is_flagged_as_unusual(self):
        label, why = modelctl_tune.classify_storage_activity(
            read_bytes=1 * MB, major_faults=200_000, storage_mode="none")
        self.assertEqual(label, "fault-heavy")
        self.assertIn("memory pressure", why)


class TestDeviceHelpers(unittest.TestCase):
    def test_short_device_name_strips_dev_prefix(self):
        self.assertEqual(modelctl_tune._short_device_name("/dev/nvme1n1p1"),
                         "nvme1n1p1")
        self.assertEqual(modelctl_tune._short_device_name("nvme1n1p1"),
                         "nvme1n1p1")
        self.assertEqual(modelctl_tune._short_device_name(""), "")

    def test_disk_read_bytes_reads_diskstats(self):
        # field 6 (index 5) is sectors read; the interface's sector is
        # always 512 bytes regardless of physical sector size.
        stats = ("   8       0 sda 100 0 2048 0 0 0 0 0 0 0 0\n"
                 " 259       0 nvme0n1 5 0 4096 0 0 0 0 0 0 0 0\n")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "diskstats"
            path.write_text(stats)
            real_open = open

            def fake_open(name, *a, **kw):
                if name == "/proc/diskstats":
                    return real_open(path, *a, **kw)
                return real_open(name, *a, **kw)

            with mock.patch("builtins.open", fake_open):
                self.assertEqual(modelctl_tune._disk_read_bytes("nvme0n1"),
                                 4096 * 512)
                self.assertEqual(modelctl_tune._disk_read_bytes("sda"),
                                 2048 * 512)

    def test_unknown_device_returns_none_not_zero(self):
        # None means "we could not measure"; zero would claim the disk was
        # idle, which is a different and possibly false statement.
        self.assertIsNone(modelctl_tune._disk_read_bytes(""))
        self.assertIsNone(modelctl_tune._disk_read_bytes("no-such-device"))


class TestSamplerCounters(unittest.TestCase):
    def sampler(self):
        return modelctl_tune._Sampler(os.getpid(), storage_device="/dev/fake0")

    def test_sampler_exposes_every_required_counter(self):
        s = self.sampler()
        for field in ("peak_vram", "baseline_vram", "final_vram", "peak_ram",
                      "peak_pss", "read_bytes", "read_syscalls",
                      "minor_faults", "major_faults", "disk_read_bytes"):
            self.assertTrue(hasattr(s, field), field)

    def test_io_snapshot_includes_syscalls(self):
        snap = self.sampler().io_snapshot()
        for key in ("read_bytes", "read_syscalls", "minor_faults",
                    "major_faults"):
            self.assertIn(key, snap)

    def test_rates_are_derived_from_raw_counters(self):
        s = self.sampler()
        s.started_at = 100.0
        s.finished_at = 110.0
        s.read_bytes = 100 * MB
        s.disk_read_bytes = 50 * MB
        s.major_faults = 500
        r = s.rates()
        self.assertAlmostEqual(r["elapsed_seconds"], 10.0)
        self.assertAlmostEqual(r["read_bytes_per_second"], 10 * MB)
        self.assertAlmostEqual(r["disk_read_bytes_per_second"], 5 * MB)
        self.assertAlmostEqual(r["major_faults_per_second"], 50.0)

    def test_rates_never_divide_by_zero(self):
        s = self.sampler()
        s.started_at = s.finished_at = 100.0
        self.assertGreater(s.rates()["elapsed_seconds"], 0)

    def test_child_memory_reports_rss_and_pss_for_this_process(self):
        # Runs against the real /proc for the test process itself.
        s = modelctl_tune._Sampler(os.getpgid(os.getpid()))
        rss, pss = s._child_memory()
        self.assertGreater(rss, 0)
        # PSS comes from smaps_rollup, which may be unreadable; 0 means
        # unknown, and that is allowed.
        self.assertGreaterEqual(pss, 0)

    def test_child_io_returns_four_counters(self):
        s = modelctl_tune._Sampler(os.getpgid(os.getpid()))
        read_bytes, minor, major, syscr = s._child_io()
        self.assertGreaterEqual(minor, 0)
        self.assertGreaterEqual(syscr, 0)

    def test_sampler_captures_final_vram_on_exit(self):
        inventory = [{"device": "SYCL0", "total_bytes": 16 * GB,
                      "free_bytes": 16 * GB}]
        with mock.patch.object(modelctl_tune.modelctl, "get_gpu_inventory",
                               return_value=inventory):
            with modelctl_tune._Sampler(os.getpid()) as s:
                pass
        # Baseline and final both observed, so a device that never released
        # its memory is visible rather than being read as peak usage.
        self.assertIn("SYCL0", s.final_vram)


class TestRuntimeRecording(unittest.TestCase):
    def test_new_counters_round_trip_through_the_database(self):
        import modelctl_runtime
        with tempfile.TemporaryDirectory() as d:
            db = modelctl_runtime.RuntimeDB(Path(d) / "rt.db")
            db.record_plan_run({
                "profile_name": "m1", "plan_id": "p1", "started_at": 1.0,
                "success": True,
                "peak_pss_bytes": 4 * GB, "read_syscalls": 1234,
                "disk_read_bytes": 9 * GB, "storage_device": "nvme0n1",
                "baseline_vram_bytes": {"SYCL0": 1}, "final_vram_bytes": {"SYCL0": 2},
                "storage_activity": "storage-backed",
                "storage_activity_detail": "paged in",
                "rates": {"read_bytes_per_second": 1.5},
            })
            row = dict(db.plan_runs_for("m1")[0])
        self.assertEqual(row["peak_pss_bytes"], 4 * GB)
        self.assertEqual(row["read_syscalls"], 1234)
        self.assertEqual(row["disk_read_bytes"], 9 * GB)
        self.assertEqual(row["storage_device"], "nvme0n1")
        self.assertEqual(row["storage_activity"], "storage-backed")
        self.assertIn("read_bytes_per_second", row["rates_json"])

    def test_legacy_rows_survive_the_migration(self):
        # Rows written before D2 have no such counters; they must read back
        # as unknown rather than breaking the query.
        import modelctl_runtime
        with tempfile.TemporaryDirectory() as d:
            db = modelctl_runtime.RuntimeDB(Path(d) / "rt.db")
            db.record_plan_run({"profile_name": "m1", "plan_id": "p1",
                                "started_at": 1.0, "success": True})
            row = dict(db.plan_runs_for("m1")[0])
        self.assertIsNone(row["peak_pss_bytes"])
        self.assertEqual(row["storage_activity"], "")


if __name__ == "__main__":
    unittest.main()
