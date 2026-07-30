import unittest

import modelctl_benchmark
from modelctl_benchmark import BenchmarkMode, BenchmarkContext, validate_mode, describe_mode, label_for_result


class TestBenchmarkMode(unittest.TestCase):
    def test_describe_all_modes(self):
        for mode in BenchmarkMode:
            desc = describe_mode(mode)
            self.assertTrue(len(desc) > 10, f"{mode} description too short")

    def test_natural_always_valid(self):
        ctx = BenchmarkContext()
        issues = validate_mode(BenchmarkMode.NATURAL, ctx)
        self.assertEqual(issues, [])

    def test_process_cold_always_valid(self):
        ctx = BenchmarkContext()
        issues = validate_mode(BenchmarkMode.PROCESS_COLD, ctx)
        self.assertEqual(issues, [])

    def test_expert_cache_warm_needs_metrics(self):
        ctx = BenchmarkContext(has_cache_metrics=False)
        issues = validate_mode(BenchmarkMode.EXPERT_CACHE_WARM, ctx)
        self.assertTrue(len(issues) > 0)

    def test_expert_cache_warm_with_metrics(self):
        ctx = BenchmarkContext(has_cache_metrics=True)
        issues = validate_mode(BenchmarkMode.EXPERT_CACHE_WARM, ctx)
        self.assertEqual(issues, [])

    def test_storage_cold_needs_privileges(self):
        ctx = BenchmarkContext(can_drop_caches=False)
        issues = validate_mode(BenchmarkMode.STORAGE_COLD, ctx)
        self.assertTrue(len(issues) > 0)

    def test_storage_cold_with_privileges(self):
        ctx = BenchmarkContext(can_drop_caches=True)
        issues = validate_mode(BenchmarkMode.STORAGE_COLD, ctx)
        self.assertEqual(issues, [])


class TestLabelForResult(unittest.TestCase):
    def test_natural_label(self):
        self.assertEqual(label_for_result(BenchmarkMode.NATURAL, True), "natural")

    def test_cold_unverified_label(self):
        label = label_for_result(BenchmarkMode.STORAGE_COLD, False)
        self.assertIn("cold_unverified", label)

    def test_process_cold_unverified_label(self):
        label = label_for_result(BenchmarkMode.PROCESS_COLD, False)
        self.assertIn("unverified", label)

    def test_validated_label(self):
        label = label_for_result(BenchmarkMode.EXPERT_CACHE_WARM, True)
        self.assertEqual(label, "expert-cache-warm")
