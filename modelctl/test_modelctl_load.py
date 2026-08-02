"""Load sampling: what it records, and what it refuses to invent.

The one property that matters more than any other here is that an
unreadable field stays unreadable all the way into the summary. A load
recorder that reports 0.0 for a loadavg it could not read makes the worst
run of a battery look like the quietest, which is the exact failure mode
the 2026-08-01 numbers already have.
"""
import threading
import unittest

import modelctl_load as ml


class TestSample(unittest.TestCase):
    def test_a_live_sample_reads_the_machine(self):
        s = ml.sample_load()
        self.assertIsNotNone(s.loadavg_1m)
        self.assertIsNotNone(s.mem_available_bytes)
        self.assertGreater(s.at, 0)

    def test_sampling_never_raises_when_proc_is_unreadable(self):
        original = (ml.LOADAVG, ml.MEMINFO)
        try:
            ml.LOADAVG = ml.Path("/nonexistent/loadavg")
            ml.MEMINFO = ml.Path("/nonexistent/meminfo")
            s = ml.sample_load()
        finally:
            ml.LOADAVG, ml.MEMINFO = original
        # Unreadable, not zero.
        self.assertIsNone(s.loadavg_1m)
        self.assertIsNone(s.mem_available_bytes)

    def test_a_malformed_loadavg_line_yields_nothing_rather_than_garbage(self):
        original = ml.LOADAVG
        import tempfile
        from pathlib import Path
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "loadavg"
                p.write_text("not a load average at all\n")
                ml.LOADAVG = p
                s = ml.sample_load()
        finally:
            ml.LOADAVG = original
        self.assertIsNone(s.loadavg_1m)
        self.assertIsNone(s.runnable)


class TestTrace(unittest.TestCase):
    def _trace(self, values):
        t = ml.LoadTrace(interval=5.0)
        for i, v in enumerate(values):
            t.add(ml.LoadSample(at=float(i), loadavg_1m=v))
        return t

    def test_summary_reports_min_mean_max(self):
        stat = self._trace([1.0, 3.0, 2.0]).summary()["loadavg_1m"]
        self.assertEqual(stat["min"], 1.0)
        self.assertEqual(stat["max"], 3.0)
        self.assertEqual(stat["mean"], 2.0)
        self.assertEqual(stat["n"], 3)

    def test_unreadable_samples_are_excluded_not_counted_as_zero(self):
        stat = self._trace([2.0, None, 4.0]).summary()["loadavg_1m"]
        self.assertEqual(stat["n"], 2)
        self.assertEqual(stat["min"], 2.0)
        self.assertEqual(stat["mean"], 3.0)

    def test_a_field_no_sample_could_read_is_absent_from_the_summary(self):
        summary = self._trace([1.0, 2.0]).summary()
        self.assertIn("loadavg_1m", summary)
        self.assertNotIn("mem_available_bytes", summary)

    def test_an_empty_trace_says_the_load_is_unknown(self):
        summary = ml.LoadTrace().summary()
        self.assertEqual(summary["samples"], 0)
        self.assertIn("unknown, not zero", summary["note"])
        self.assertNotIn("loadavg_1m", summary)


class TestRecorder(unittest.TestCase):
    def test_a_run_shorter_than_the_interval_still_gets_one_sample(self):
        # Otherwise the shortest runs -- the ones most easily perturbed --
        # are exactly the ones with no load record.
        rec = ml.LoadRecorder(interval=3600, sampler=lambda: ml.LoadSample(
            at=1.0, loadavg_1m=0.5))
        with rec:
            pass
        self.assertEqual(rec.trace.summary()["samples"], 1)

    def test_the_trace_survives_an_exception_in_the_measured_run(self):
        rec = ml.LoadRecorder(interval=3600, sampler=lambda: ml.LoadSample(
            at=1.0, loadavg_1m=0.5))
        with self.assertRaises(ValueError):
            with rec:
                raise ValueError("the run failed")
        # A failed run's load is often why it failed.
        self.assertEqual(rec.trace.summary()["samples"], 1)
        self.assertIsNotNone(rec.trace.finished)

    def test_samples_accumulate_over_time(self):
        seen = threading.Event()
        counter = {"n": 0}

        def sampler():
            counter["n"] += 1
            if counter["n"] >= 3:
                seen.set()
            return ml.LoadSample(at=float(counter["n"]), loadavg_1m=1.0)

        rec = ml.LoadRecorder(interval=0.01, sampler=sampler)
        rec.start()
        seen.wait(timeout=5)
        rec.stop()
        self.assertGreaterEqual(rec.trace.summary()["samples"], 3)

    def test_a_sampler_that_raises_is_counted_not_fatal(self):
        calls = {"n": 0}
        done = threading.Event()

        def sampler():
            calls["n"] += 1
            if calls["n"] == 1:
                return ml.LoadSample(at=0.0, loadavg_1m=1.0)
            done.set()
            raise OSError("proc went away")

        rec = ml.LoadRecorder(interval=0.01, sampler=sampler)
        rec.start()
        done.wait(timeout=5)
        rec.stop()
        self.assertGreaterEqual(rec.trace.missed, 1)
        self.assertIn("missed", rec.trace.summary())

    def test_stop_is_idempotent(self):
        rec = ml.LoadRecorder(interval=3600,
                              sampler=lambda: ml.LoadSample(at=1.0))
        rec.start()
        rec.stop()
        rec.stop()


class TestComparability(unittest.TestCase):
    def _summary(self, mean):
        t = ml.LoadTrace()
        t.add(ml.LoadSample(at=0.0, loadavg_1m=mean))
        return t.summary()

    def test_two_quiet_runs_are_within_tolerance(self):
        check = ml.comparable_load([self._summary(0.4), self._summary(0.6)])
        self.assertTrue(check["checked"])
        self.assertTrue(check["within_tolerance"])
        self.assertAlmostEqual(check["loadavg_1m_spread"], 0.2)

    def test_the_void_battery_spread_is_out_of_tolerance(self):
        # 2.63 to 17.15 across the 2026-08-01 battery: the reason those
        # block deltas cannot be read as a property of the attribute.
        check = ml.comparable_load([self._summary(2.63), self._summary(17.15)])
        self.assertFalse(check["within_tolerance"])
        self.assertAlmostEqual(check["loadavg_1m_spread"], 14.52)

    def test_one_run_cannot_be_compared_with_itself(self):
        check = ml.comparable_load([self._summary(0.4)])
        self.assertFalse(check["checked"])
        self.assertIn("fewer than two", check["reason"])

    def test_runs_without_a_readable_load_do_not_count_as_agreeing(self):
        check = ml.comparable_load([ml.LoadTrace().summary(),
                                    ml.LoadTrace().summary()])
        self.assertFalse(check["checked"])
        self.assertEqual(check["runs_with_load"], 0)


if __name__ == "__main__":
    unittest.main()
