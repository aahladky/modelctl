"""Task D4: storage calibration that knows when it measured RAM.

A calibration silently served from the page cache reports RAM throughput
under a disk's name, typically an order of magnitude too fast, and every
storage estimate built on it is then wrong. These tests are mostly about
that detection.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modelctl_benchmark
import modelctl_storage
from modelctl_services import cache_service as cs

MB = 1 << 20


class TestMountOptions(unittest.TestCase):
    def test_probe_reports_mount_options(self):
        # noatime, discard, compress= and ro all change what a storage
        # measurement means.
        info = modelctl_storage.probe_storage(os.path.expanduser("~"))
        self.assertIsInstance(info.mount_options, str)
        self.assertIn("rw", info.mount_options.split(","))

    def test_missing_mount_point_yields_empty_options(self):
        self.assertEqual(
            modelctl_storage._find_mount_options("/no/such/mount"), "")


class CalibrationBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"
        self.path.write_bytes(os.urandom(8 * MB))


class TestContaminationDetection(CalibrationBase):
    def test_cached_file_is_flagged_as_contaminated(self):
        cached = modelctl_benchmark.Residency(8 * MB, 8 * MB)
        with mock.patch.object(modelctl_benchmark, "page_cache_residency",
                               return_value=cached), \
             mock.patch.object(modelctl_benchmark, "evict_from_page_cache",
                               return_value=True):
            r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertTrue(r.ok)
        self.assertTrue(r.page_cache_contaminated)
        self.assertEqual(r.residency_before, 1.0)
        self.assertTrue(any("page cache" in w for w in r.warnings))

    def test_evicted_file_is_not_flagged(self):
        # The temp dir may itself be tmpfs, which is legitimately always
        # contaminated -- pin a real filesystem so this tests eviction.
        cold = modelctl_benchmark.Residency(0, 8 * MB)
        real_fs = mock.Mock(filesystem="btrfs", mount_point="/home",
                            mount_options="rw", block_devices=("/dev/nvme0n1p1",),
                            transport="nvme", rotational=False, raid_level=None)
        with mock.patch.object(modelctl_benchmark, "page_cache_residency",
                               return_value=cold), \
             mock.patch.object(modelctl_benchmark, "evict_from_page_cache",
                               return_value=True), \
             mock.patch.object(cs, "_storage_identity", return_value=real_fs):
            r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertFalse(r.page_cache_contaminated)
        self.assertEqual(r.residency_before, 0.0)

    def test_ram_backed_filesystem_is_always_contaminated(self):
        # There is no device to calibrate on tmpfs.
        cold = modelctl_benchmark.Residency(0, 8 * MB)
        fake = mock.Mock(filesystem="tmpfs", mount_point="/tmp",
                         mount_options="rw", block_devices=(),
                         transport="unknown", rotational=None, raid_level=None)
        with mock.patch.object(modelctl_benchmark, "page_cache_residency",
                               return_value=cold), \
             mock.patch.object(cs, "_storage_identity", return_value=fake):
            r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertTrue(r.page_cache_contaminated)
        self.assertTrue(any("RAM-backed" in w for w in r.warnings))

    def test_small_sample_is_warned_about(self):
        r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertTrue(any("readahead" in w for w in r.warnings))

    def test_implausibly_fast_random_reads_are_called_out(self):
        # No storage device answers a 4 KiB read in single-digit
        # microseconds; that is a memory copy.
        with mock.patch.object(cs, "_measure_random_reads",
                               return_value={"p50_us": 1.0, "p99_us": 140.0,
                                             "iops": 30000, "bps": 1 << 30}):
            r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertTrue(any("faster than any storage device" in w
                            for w in r.warnings))


class TestCalibrationContent(CalibrationBase):
    def test_calibration_records_storage_identity(self):
        r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertTrue(r.ok)
        self.assertTrue(r.filesystem)
        self.assertTrue(r.mount_point)
        self.assertGreater(r.measured_at, 0)

    def test_sequential_and_random_are_both_measured(self):
        # Sequential throughput says nothing about pulling one expert out
        # of the middle of a large file, which is the access pattern an
        # offloaded MoE actually has.
        r = cs.calibrate_storage_sequential(self.path, read_bytes=4 * MB)
        self.assertGreater(r.sequential_read_bps, 0)
        self.assertGreater(r.random_read_iops, 0)
        self.assertGreater(r.random_read_latency_us_p99, 0)

    def test_missing_file_fails_cleanly(self):
        r = cs.calibrate_storage_sequential("/nonexistent/model.gguf")
        self.assertFalse(r.ok)
        self.assertIn("not found", r.messages[0])

    def test_empty_file_fails_cleanly(self):
        empty = Path(self.tmp.name) / "empty.gguf"
        empty.touch()
        r = cs.calibrate_storage_sequential(empty)
        self.assertFalse(r.ok)

    def test_random_measurement_skips_tiny_files(self):
        tiny = Path(self.tmp.name) / "tiny.gguf"
        tiny.write_bytes(b"x" * 100)
        self.assertEqual(cs._measure_random_reads(tiny, 100), {})


class TestCalibrationStaleness(unittest.TestCase):
    def test_fresh_calibration_is_not_stale(self):
        import time
        age, stale = cs.calibration_age(time.time() - 60)
        self.assertLess(age, 120)
        self.assertFalse(stale)

    def test_old_calibration_is_stale(self):
        # Disks get replaced and filesystems fill up; a year-old number
        # must not silently steer a placement decision.
        import time
        age, stale = cs.calibration_age(time.time() - 400 * 86400)
        self.assertTrue(stale)

    def test_never_calibrated_reads_as_stale(self):
        age, stale = cs.calibration_age(None)
        self.assertIsNone(age)
        self.assertTrue(stale)


if __name__ == "__main__":
    unittest.main()
