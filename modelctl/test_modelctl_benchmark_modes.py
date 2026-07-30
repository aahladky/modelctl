"""Task D3: benchmark modes as controlled procedures, not labels.

The rule: a mode whose precondition was not met is never recorded under
that mode's name. A cold label on a warm measurement corrupts every later
comparison that trusts it.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modelctl_benchmark as bench
from modelctl_benchmark import BenchmarkContext, BenchmarkMode

MB = 1 << 20


class FakeStorage:
    def __init__(self, filesystem):
        self.filesystem = filesystem


class TestResidencyMeasurement(unittest.TestCase):
    """These run against the real kernel; a file just written is cached."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"
        self.path.write_bytes(os.urandom(4 * MB))

    def test_a_freshly_written_file_reads_as_resident(self):
        r = bench.page_cache_residency(self.path)
        self.assertIsNotNone(r)
        self.assertEqual(r.total_bytes, 4 * MB)
        self.assertGreater(r.fraction, 0.5)

    def test_empty_file_does_not_divide_by_zero(self):
        empty = Path(self.tmp.name) / "empty.gguf"
        empty.touch()
        r = bench.page_cache_residency(empty)
        self.assertEqual(r.fraction, 0.0)

    def test_unmeasurable_path_returns_none_not_zero(self):
        # None means "could not measure"; zero would assert the file is
        # cold, which is a different and possibly false claim.
        self.assertIsNone(bench.page_cache_residency("/nonexistent/model.gguf"))
        self.assertIsNone(bench.page_cache_residency([]))

    def test_residency_sums_across_shards(self):
        second = Path(self.tmp.name) / "shard2.gguf"
        second.write_bytes(os.urandom(2 * MB))
        r = bench.page_cache_residency([self.path, second])
        self.assertEqual(r.total_bytes, 6 * MB)

    def test_eviction_call_reports_whether_it_ran(self):
        self.assertTrue(bench.evict_from_page_cache(self.path))
        self.assertFalse(bench.evict_from_page_cache("/nonexistent/x.gguf"))


class TestStorageColdVerification(unittest.TestCase):
    """Eviction succeeding is not evidence that anything got evicted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"
        self.path.write_bytes(os.urandom(MB))

    def ctx(self, **kw):
        return BenchmarkContext(model_path=str(self.path),
                                consent_storage_cold=True, **kw)

    def test_ineffective_eviction_is_labelled_cold_unverified(self):
        # posix_fadvise returns success while dropping nothing whenever the
        # pages are still referenced -- e.g. the model is still loaded.
        still_cached = bench.Residency(MB, MB)
        with mock.patch.object(bench, "evict_from_page_cache", return_value=True), \
             mock.patch.object(bench, "page_cache_residency",
                               return_value=still_cached):
            prep = bench.prepare_mode(BenchmarkMode.STORAGE_COLD, self.ctx())
        self.assertFalse(prep.achieved)
        self.assertEqual(prep.label, "cold_unverified")
        self.assertIn("still cached", prep.messages[0])

    def test_effective_eviction_earns_the_cold_label(self):
        residencies = [bench.Residency(MB, MB), bench.Residency(0, MB)]
        with mock.patch.object(bench, "evict_from_page_cache", return_value=True), \
             mock.patch.object(bench, "page_cache_residency",
                               side_effect=residencies):
            prep = bench.prepare_mode(BenchmarkMode.STORAGE_COLD, self.ctx())
        self.assertTrue(prep.achieved)
        self.assertEqual(prep.label, "storage-cold")
        self.assertEqual(prep.evidence["residency_fraction_after"], 0.0)

    def test_unmeasurable_residency_is_not_reported_as_cold(self):
        with mock.patch.object(bench, "evict_from_page_cache", return_value=True), \
             mock.patch.object(bench, "page_cache_residency", return_value=None):
            prep = bench.prepare_mode(BenchmarkMode.STORAGE_COLD, self.ctx())
        self.assertFalse(prep.achieved)
        self.assertEqual(prep.label, "cold_unverified")

    def test_storage_cold_without_consent_is_refused(self):
        prep = bench.prepare_mode(
            BenchmarkMode.STORAGE_COLD,
            BenchmarkContext(model_path=str(self.path)))
        self.assertTrue(prep.refused)
        self.assertFalse(prep.achieved)
        self.assertEqual(prep.label, "cold_unverified")

    def test_evidence_records_the_before_and_after(self):
        residencies = [bench.Residency(MB, MB), bench.Residency(0, MB)]
        with mock.patch.object(bench, "evict_from_page_cache", return_value=True), \
             mock.patch.object(bench, "page_cache_residency",
                               side_effect=residencies):
            prep = bench.prepare_mode(BenchmarkMode.STORAGE_COLD, self.ctx())
        self.assertEqual(prep.evidence["residency_fraction_before"], 1.0)
        self.assertEqual(prep.evidence["residency_fraction_after"], 0.0)
        self.assertTrue(prep.evidence["evict_call_succeeded"])


class TestProcessColdIsHonestAboutPageCache(unittest.TestCase):
    def test_a_fresh_process_over_a_cached_model_is_not_cold(self):
        # "process-cold" says nothing about the page cache; measuring it
        # turns that unknown into a number.
        with mock.patch.object(bench, "page_cache_residency",
                               return_value=bench.Residency(MB, MB)):
            prep = bench.prepare_mode(BenchmarkMode.PROCESS_COLD,
                                      BenchmarkContext(model_path="/m.gguf"))
        self.assertFalse(prep.achieved)
        self.assertIn("page cache unverified", prep.label)
        self.assertIn("already in the page cache", prep.messages[0])

    def test_a_fresh_process_over_an_uncached_model_is_cold(self):
        with mock.patch.object(bench, "page_cache_residency",
                               return_value=bench.Residency(0, MB)):
            prep = bench.prepare_mode(BenchmarkMode.PROCESS_COLD,
                                      BenchmarkContext(model_path="/m.gguf"))
        self.assertTrue(prep.achieved)
        self.assertEqual(prep.label, "process-cold")


class TestExpertCacheWarm(unittest.TestCase):
    def test_without_metrics_the_mode_cannot_claim_it_warmed(self):
        prep = bench.prepare_mode(BenchmarkMode.EXPERT_CACHE_WARM,
                                  BenchmarkContext(has_cache_metrics=False))
        self.assertFalse(prep.achieved)
        self.assertIn("unverified", prep.label)

    def test_with_metrics_the_mode_is_attemptable(self):
        prep = bench.prepare_mode(BenchmarkMode.EXPERT_CACHE_WARM,
                                  BenchmarkContext(has_cache_metrics=True))
        self.assertTrue(prep.achieved)
        self.assertEqual(prep.label, "expert-cache-warm")

    def test_stabilisation_needs_a_settled_window(self):
        self.assertTrue(bench.expert_cache_stabilized([0.1, 0.80, 0.805, 0.798]))

    def test_a_still_climbing_hit_rate_is_not_stable(self):
        self.assertFalse(bench.expert_cache_stabilized([0.2, 0.4, 0.6, 0.8]))

    def test_too_few_samples_is_not_stable(self):
        self.assertFalse(bench.expert_cache_stabilized([0.8, 0.8]))
        self.assertFalse(bench.expert_cache_stabilized([]))


class TestNaturalAndPageCacheWarm(unittest.TestCase):
    def test_natural_claims_nothing(self):
        with mock.patch.object(bench, "page_cache_residency",
                               return_value=bench.Residency(MB // 2, MB)):
            prep = bench.prepare_mode(BenchmarkMode.NATURAL,
                                      BenchmarkContext(model_path="/m.gguf"))
        self.assertTrue(prep.achieved)
        self.assertEqual(prep.label, "natural")
        self.assertIn("no cache manipulation", prep.messages[0])
        # It still records what the state was, so the result is auditable.
        self.assertEqual(prep.evidence["residency_fraction"], 0.5)

    def test_page_cache_warm_records_its_starting_point(self):
        with mock.patch.object(bench, "page_cache_residency",
                               return_value=bench.Residency(0, MB)):
            prep = bench.prepare_mode(BenchmarkMode.PAGE_CACHE_WARM,
                                      BenchmarkContext(model_path="/m.gguf"))
        self.assertIn("residency_fraction_before", prep.evidence)


class TestModeValidation(unittest.TestCase):
    def test_tmpfs_makes_storage_cold_impossible(self):
        # On a RAM-backed filesystem the cached pages *are* the file, so
        # there is no storage layer to go cold.
        issues = bench.validate_mode(
            BenchmarkMode.STORAGE_COLD,
            BenchmarkContext(can_drop_caches=True, consent_storage_cold=True,
                             storage_info=FakeStorage("tmpfs")))
        self.assertTrue(any("impossible" in i for i in issues))

    def test_network_filesystem_is_flagged(self):
        issues = bench.validate_mode(
            BenchmarkMode.STORAGE_COLD,
            BenchmarkContext(can_drop_caches=True, consent_storage_cold=True,
                             storage_info=FakeStorage("nfs")))
        self.assertTrue(any("nfs" in i for i in issues))

    def test_missing_consent_is_an_issue(self):
        issues = bench.validate_mode(
            BenchmarkMode.STORAGE_COLD,
            BenchmarkContext(can_drop_caches=True))
        self.assertTrue(any("opt-in" in i for i in issues))

    def test_local_disk_with_consent_has_no_issues(self):
        issues = bench.validate_mode(
            BenchmarkMode.STORAGE_COLD,
            BenchmarkContext(can_drop_caches=True, consent_storage_cold=True,
                             storage_info=FakeStorage("btrfs")))
        self.assertEqual(issues, [])


class TestBenchmarkServiceIntegration(unittest.TestCase):
    def test_prepare_mode_surfaces_refusal_as_not_ok(self):
        from modelctl_services import benchmark_service
        ctx = BenchmarkContext(model_path="/m.gguf")
        with mock.patch.object(benchmark_service, "context_for_profile",
                               return_value=ctx):
            r = benchmark_service.prepare_mode("m1", "storage-cold")
        self.assertFalse(r.ok)
        self.assertTrue(r.data["refused"])
        self.assertEqual(r.data["label"], "cold_unverified")

    def test_unachieved_precondition_warns_and_relabels(self):
        from modelctl_services import benchmark_service
        ctx = BenchmarkContext(model_path="/m.gguf", has_cache_metrics=False)
        with mock.patch.object(benchmark_service, "context_for_profile",
                               return_value=ctx):
            r = benchmark_service.prepare_mode("m1", "expert-cache-warm")
        self.assertTrue(r.ok)
        self.assertFalse(r.data["achieved"])
        self.assertIn("unverified", r.data["label"])
        self.assertTrue(any("precondition was not met" in w for w in r.warnings))

    def test_unknown_mode_is_rejected(self):
        from modelctl_services import benchmark_service
        self.assertFalse(benchmark_service.prepare_mode("m1", "warp-speed").ok)


if __name__ == "__main__":
    unittest.main()
