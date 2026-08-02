"""Paired benchmarking: the alternation, the deltas, and the sign test.

These tests are written against the specific defect the module replaces.
The 2026-08-01 "cost of determinism" figures were two blocks of runs
differenced across a machine whose loadavg moved by an order of magnitude
between them, so the tests below care about three things: that the arms
actually alternate, that a delta is computed within a pair and never
across the whole run, and that the module reports statistics without
reaching a verdict.
"""
import math
import unittest

import modelctl_paired as mp

A = mp.Condition(name="deterministic-on", config={"env": {"D": "1"}})
B = mp.Condition(name="deterministic-off", config={"env": {"D": "0"}})


class _NullRecorder:
    """A recorder that samples nothing, so tests do not race a thread."""

    def __init__(self):
        self.trace = mp.modelctl_load.LoadTrace()

    def start(self):
        return self

    def stop(self):
        return self.trace


def _fixed(values):
    """measure() returning a scripted value per (condition, pair)."""
    def measure(condition, pair, slot):
        return {"generation_tps": values[condition.name][pair]}
    return measure


def _run(values, pairs=None, **kw):
    n = pairs if pairs is not None else len(values[A.name])
    return mp.run_paired(A, B, _fixed(values), pairs=n,
                         recorder_factory=_NullRecorder,
                         clock=lambda: 0.0, **kw)


class TestSchedule(unittest.TestCase):
    def test_the_order_alternates(self):
        self.assertEqual(mp.schedule(4), [(0, 1), (1, 0), (0, 1), (1, 0)])

    def test_zero_pairs_is_an_empty_schedule_not_an_error(self):
        self.assertEqual(mp.schedule(0), [])
        self.assertEqual(mp.schedule(-3), [])

    def test_each_arm_leads_half_the_time_on_an_even_count(self):
        # The point of alternating: "back to back" is not symmetric, and a
        # fixed order folds the second-slot advantage into every delta
        # with the same sign.
        leads = [order[0] for order in mp.schedule(6)]
        self.assertEqual(leads.count(0), leads.count(1))


class TestSignTest(unittest.TestCase):
    def test_five_agreeing_deltas(self):
        t = mp.sign_test([-1.0, -0.5, -2.0, -0.1, -3.0])
        self.assertEqual((t.n, t.positive, t.negative, t.ties), (5, 0, 5, 0))
        self.assertAlmostEqual(t.p_value, 2 * (1 / 32))

    def test_a_perfect_split(self):
        t = mp.sign_test([1.0, -1.0, 1.0, -1.0])
        self.assertEqual((t.positive, t.negative), (2, 2))
        self.assertAlmostEqual(t.p_value, 1.0)

    def test_ties_leave_the_sample(self):
        # A zero delta is evidence of neither direction; counting it as
        # either would manufacture agreement.
        t = mp.sign_test([1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertEqual((t.n, t.ties, t.positive), (3, 2, 3))
        self.assertAlmostEqual(t.p_value, 0.25)

    def test_dropped_pairs_are_not_counted(self):
        t = mp.sign_test([1.0, None, 1.0])
        self.assertEqual(t.n, 2)

    def test_no_non_tied_pair_has_no_p_value(self):
        # Not 1.0: "the test had nothing to work with" and "the test came
        # out perfectly balanced" are different facts.
        t = mp.sign_test([0.0, 0.0])
        self.assertIsNone(t.p_value)
        self.assertEqual(t.n, 0)

    def test_no_deltas_at_all(self):
        self.assertIsNone(mp.sign_test([]).p_value)

    def test_the_p_value_never_exceeds_one(self):
        for n in range(1, 12):
            for k in range(n + 1):
                deltas = [1.0] * k + [-1.0] * (n - k)
                p = mp.sign_test(deltas).p_value
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_it_matches_the_binomial_definition(self):
        deltas = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
        t = mp.sign_test(deltas)
        n, k = 7, 5
        expected = min(1.0, 2 * sum(math.comb(n, i)
                                    for i in range(k, n + 1)) / 2 ** n)
        self.assertAlmostEqual(t.p_value, expected)

    def test_the_result_carries_no_threshold(self):
        payload = mp.sign_test([-1.0] * 5).to_dict()
        self.assertNotIn("significant", payload)
        self.assertIn("no threshold is applied", payload["note"])


class TestRunPaired(unittest.TestCase):
    VALUES = {A.name: [6.0, 6.2, 6.1, 6.3, 6.0],
              B.name: [7.0, 7.1, 7.3, 7.0, 7.2]}

    def test_both_arms_run_in_every_pair(self):
        c = _run(self.VALUES)
        self.assertEqual(len(c.pairs), 5)
        for p in c.pairs:
            self.assertEqual(set(p.runs), {A.name, B.name})
            self.assertTrue(p.complete)

    def test_the_recorded_order_alternates(self):
        c = _run(self.VALUES)
        self.assertEqual([p.order[0] for p in c.pairs],
                         [A.name, B.name, A.name, B.name, A.name])

    def test_slots_are_recorded_per_run(self):
        c = _run(self.VALUES)
        # Pair 1 runs B first, so B is in slot 0 there.
        self.assertEqual(c.pairs[0].runs[A.name].slot, 0)
        self.assertEqual(c.pairs[1].runs[A.name].slot, 1)

    def test_deltas_are_b_minus_a_within_each_pair(self):
        c = _run(self.VALUES)
        for i, d in enumerate(c.deltas()):
            self.assertAlmostEqual(
                d, self.VALUES[B.name][i] - self.VALUES[A.name][i])

    def test_a_pair_is_never_differenced_against_another_pair(self):
        # The defect being replaced. Arm A drifts upward across pairs and
        # arm B drifts downward; blocked means would report a large
        # difference where every within-pair delta is exactly 1.0.
        values = {A.name: [1.0, 2.0, 3.0, 4.0],
                  B.name: [2.0, 3.0, 4.0, 5.0]}
        c = _run(values)
        self.assertEqual(c.deltas(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(c.to_dict()["median_delta"], 1.0)

    def test_relative_deltas(self):
        c = _run({A.name: [2.0, 4.0], B.name: [3.0, 5.0]})
        self.assertEqual(c.relative_deltas(), [0.5, 0.25])

    def test_a_zero_baseline_yields_no_relative_delta(self):
        c = _run({A.name: [0.0], B.name: [1.0]})
        self.assertEqual(c.relative_deltas(), [None])

    def test_a_failed_arm_leaves_its_pair_incomplete_not_the_run_aborted(self):
        def measure(condition, pair, slot):
            if pair == 2 and condition.name == B.name:
                raise RuntimeError("server exited rc=134")
            return {"generation_tps": 5.0 if condition is A else 6.0}

        c = mp.run_paired(A, B, measure, pairs=4,
                          recorder_factory=_NullRecorder, clock=lambda: 0.0)
        self.assertEqual(len(c.pairs), 4)
        self.assertEqual([p.index for p in c.incomplete_pairs()], [2])
        self.assertEqual(c.deltas()[2], None)
        self.assertEqual(c.sign_test().n, 3)
        self.assertIn("rc=134", c.pairs[2].runs[B.name].error)

    def test_a_dropped_pair_stays_in_the_raw_record(self):
        def measure(condition, pair, slot):
            if pair == 1 and condition is B:
                raise RuntimeError("boom")
            return {"generation_tps": 1.0}

        c = mp.run_paired(A, B, measure, pairs=3,
                          recorder_factory=_NullRecorder, clock=lambda: 0.0)
        payload = c.to_dict()
        self.assertEqual(payload["pairs_run"], 3)
        self.assertEqual(payload["pairs_complete"], 2)
        self.assertEqual(payload["pairs_incomplete"], [1])
        self.assertEqual(len(payload["per_pair"]), 3)
        self.assertEqual(len(payload["runs"]), 6)

    def test_stopping_early_records_how_far_it_got(self):
        state = {"n": 0}

        def should_stop():
            state["n"] += 1
            return state["n"] > 2

        c = _run(self.VALUES, should_stop=should_stop)
        self.assertEqual(len(c.pairs), 2)
        self.assertTrue(any("2 of 5 pairs" in n for n in c.notes))

    def test_between_runs_inside_a_pair_only(self):
        # A cooldown belongs between the two arms of a pair, where it is
        # applied equally to both; a gap between pairs cannot bias a
        # within-pair delta but a gap after the second arm can.
        calls = []
        _run(self.VALUES, between=lambda: calls.append(1))
        self.assertEqual(len(calls), 5)

    def test_every_run_carries_a_load_summary(self):
        c = _run(self.VALUES)
        for p in c.pairs:
            for r in p.runs.values():
                self.assertIn("samples", r.load)


class TestRecord(unittest.TestCase):
    def test_the_delta_convention_is_stated(self):
        payload = _run({A.name: [1.0], B.name: [2.0]}).to_dict()
        self.assertEqual(payload["delta_convention"],
                         "delta = deterministic-off - deterministic-on")

    def test_both_arms_configs_are_recorded_verbatim(self):
        payload = _run({A.name: [1.0], B.name: [2.0]}).to_dict()
        self.assertEqual(payload["a"]["config"], {"env": {"D": "1"}})
        self.assertEqual(payload["b"]["config"], {"env": {"D": "0"}})

    def test_the_record_names_no_winner(self):
        payload = _run({A.name: [1.0, 1.0], B.name: [9.0, 9.0]}).to_dict()
        text = str(payload).lower()
        for word in ("winner", "faster", "slower", "better", "worse",
                     "significant", "verdict"):
            self.assertNotIn(word, text)

    def test_per_pair_rows_carry_both_values_and_the_sign(self):
        row = _run({A.name: [1.0], B.name: [2.0]}).to_dict()["per_pair"][0]
        self.assertEqual(row[A.name], 1.0)
        self.assertEqual(row[B.name], 2.0)
        self.assertEqual(row["delta"], 1.0)
        self.assertEqual(row["sign"], 1)

    def test_load_comparability_is_reported_per_pair(self):
        payload = _run({A.name: [1.0, 1.0], B.name: [2.0, 2.0]}).to_dict()
        self.assertEqual([c["pair"] for c in payload["load_comparability"]],
                         [0, 1])


if __name__ == "__main__":
    unittest.main()
