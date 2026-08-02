"""The pre-registered night-lane registry stays pre-registered.

These are guard rails around a discipline, not around an algorithm: the
registry's value is that nothing here runs by accident and that the
criteria were written before any numbers existed.
"""
import json
import tempfile
import unittest
from pathlib import Path

import modelctl_nightlane as nl


class TestShippedRegistry(unittest.TestCase):
    def setUp(self):
        self.jobs = nl.load_jobs()

    def test_both_pre_registered_pairs_are_present(self):
        self.assertEqual(
            sorted(j.id for j in self.jobs),
            ["ornith-rpc-criterion-2026-08-02",
             "qwen122b-remote-experts-hypothesis-2026-08-02"])

    def test_every_job_is_disabled(self):
        # The whole point. A pre-registration that runs itself is a
        # schedule, and nothing here was asked to be scheduled.
        for j in self.jobs:
            self.assertFalse(j.enabled, j.id)
        self.assertEqual(nl.enabled_jobs(), [])

    def test_every_job_is_a_pair(self):
        for j in self.jobs:
            self.assertEqual(len(j.arms), 2, j.id)

    def test_every_job_has_a_criterion_and_measures(self):
        for j in self.jobs:
            self.assertTrue(j.criterion.strip(), j.id)
            self.assertTrue(j.measures, j.id)
            self.assertTrue(j.registered, j.id)

    def test_exactly_one_arm_per_pair_needs_the_node(self):
        # A pair where both arms need the node has no baseline, and one
        # where neither does is not testing the fleet.
        for j in self.jobs:
            needing = [a for a in j.arms if a.requires_nodes]
            self.assertEqual(len(needing), 1, j.id)
            self.assertEqual(list(needing[0].requires_nodes), ["ph16-71-cuda0"])

    def test_the_registry_records_no_expected_outcome(self):
        # "Never judge the result" -- a pre-registration that predicts a
        # winner invites reading the numbers to match. Criteria say how
        # to decide, never what the answer will be.
        banned = ("faster", "slower", "better", "worse", "improvement",
                  "win", "beats", "should outperform", "expect a speedup")
        for j in self.jobs:
            haystack = " ".join(
                [j.title, j.question, j.criterion, j.note]
                + [a.note for a in j.arms]).lower()
            for word in banned:
                self.assertNotIn(word, haystack, f"{j.id}: '{word}'")


class TestBlocking(unittest.TestCase):
    def test_a_disabled_job_is_blocked_even_with_the_node_present(self):
        job = nl.job_by_id("ornith-rpc-criterion-2026-08-02")
        reasons = nl.blocking_reasons(job, usable_node_names=["ph16-71-cuda0"])
        self.assertIn("pre-registered but not enabled", reasons)

    def test_a_missing_node_blocks_independently_of_enablement(self):
        job = nl.job_by_id("ornith-rpc-criterion-2026-08-02")
        reasons = nl.blocking_reasons(job, usable_node_names=[])
        self.assertTrue(any("ph16-71-cuda0" in r for r in reasons))

    def test_required_nodes_are_collected_across_arms(self):
        job = nl.job_by_id("qwen122b-remote-experts-hypothesis-2026-08-02")
        self.assertEqual(job.required_nodes, {"ph16-71-cuda0"})


class TestRoundTrip(unittest.TestCase):
    def test_save_then_load_is_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.json"
            original = nl.load_jobs()
            nl.save_jobs(original, p)
            self.assertEqual(nl.load_jobs(p), original)

    def test_a_missing_registry_is_empty_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(nl.load_jobs(Path(d) / "absent.json"), [])

    def test_the_shipped_file_is_valid_json_with_a_version(self):
        raw = json.loads(nl.REGISTRY_PATH.read_text())
        self.assertEqual(raw["version"], 1)


if __name__ == "__main__":
    unittest.main()
