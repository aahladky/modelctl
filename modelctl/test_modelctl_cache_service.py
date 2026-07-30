import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from modelctl_services.cache_service import (
    CacheMetrics, BenchmarkResult, CalibrationResult,
    calibrate_storage_sequential,
)


class TestCacheMetrics(unittest.TestCase):
    def test_defaults(self):
        m = CacheMetrics()
        self.assertEqual(m.lookups, 0)
        self.assertEqual(m.hit_ratio, 0.0)

    def test_hit_ratio(self):
        m = CacheMetrics(lookups=100, hits=75, misses=25)
        # hit_ratio is set by the caller (scrape_cache_metrics), not auto-computed.
        self.assertEqual(m.lookups, 100)
        self.assertEqual(m.hits, 75)
        self.assertEqual(m.misses, 25)


class TestCalibrationResult(unittest.TestCase):
    def test_defaults(self):
        r = CalibrationResult(ok=False)
        self.assertFalse(r.ok)
        self.assertEqual(r.sequential_read_bps, 0)


class TestCalibrateStorage(unittest.TestCase):
    def test_calibrate_real_file(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "test.bin"
            # Write 2 MiB of data.
            path.write_bytes(b"\x00" * (2 * 1024 * 1024))
            result = calibrate_storage_sequential(str(path), read_bytes=1024 * 1024)
            self.assertTrue(result.ok)
            self.assertTrue(result.sequential_read_bps > 0)
            self.assertEqual(result.file_size_bytes, 2 * 1024 * 1024)
            self.assertTrue(result.elapsed_seconds >= 0)

    def test_calibrate_nonexistent(self):
        result = calibrate_storage_sequential("/nonexistent/file.gguf")
        self.assertFalse(result.ok)

    def test_calibrate_empty_file(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "empty.bin"
            path.write_bytes(b"")
            result = calibrate_storage_sequential(str(path))
            self.assertFalse(result.ok)
