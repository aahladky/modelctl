import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest import mock

import modelctl
import modelctl_transactions


class TestTransaction(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_profiles = modelctl.PROFILES_DIR
        self._orig_state = modelctl.STATE_DIR
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.state_dir = Path(self.tmp.name) / "state"
        modelctl.PROFILES_DIR = self.profiles_dir
        modelctl.STATE_DIR = self.state_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(setattr, modelctl, "PROFILES_DIR", self._orig_profiles)
        self.addCleanup(setattr, modelctl, "STATE_DIR", self._orig_state)

    def test_commit_writes_profile(self):
        profile = {"name": "test", "config": {"ctx": 4096}}
        with modelctl_transactions.Transaction("test") as tx:
            tx.stage_profile(profile)
            tx.commit()
        path = self.profiles_dir / "test.json"
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text())
        self.assertEqual(loaded["name"], "test")

    def test_rollback_on_exception(self):
        profile = {"name": "test", "config": {"ctx": 4096}}
        # Save initial profile.
        (self.profiles_dir / "test.json").write_text(json.dumps(profile))

        try:
            with modelctl_transactions.Transaction("test") as tx:
                tx.stage_profile({"name": "test", "config": {"ctx": 8192}})
                tx.commit()
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass

        # Should be rolled back to original.
        loaded = json.loads((self.profiles_dir / "test.json").read_text())
        self.assertEqual(loaded["config"]["ctx"], 4096)

    def test_stage_artifact(self):
        artifact_path = Path(self.tmp.name) / "test.sh"
        with modelctl_transactions.Transaction("test") as tx:
            tx.stage_artifact(artifact_path, "#!/bin/sh\necho hello\n")
            tx.commit()
        self.assertTrue(artifact_path.exists())
        self.assertIn("echo hello", artifact_path.read_text())

    def test_invalid_profile_name_raises(self):
        with self.assertRaises(modelctl_transactions.TransactionError):
            with modelctl_transactions.Transaction("test") as tx:
                tx.stage_profile({"config": {}})

    def test_multiple_profiles(self):
        with modelctl_transactions.Transaction("test") as tx:
            tx.stage_profile({"name": "alpha", "config": {"ctx": 1024}})
            tx.stage_profile({"name": "beta", "config": {"ctx": 2048}})
            tx.commit()
        self.assertTrue((self.profiles_dir / "alpha.json").exists())
        self.assertTrue((self.profiles_dir / "beta.json").exists())


class TestRollbackRestoresEverything(unittest.TestCase):
    """Regression: rollback used to restore profiles only.

    Found by the Task E2 real-process suite. _rollback() handled profiles
    and left a comment saying artifacts and configs were preserved in the
    backup directory "for manual recovery" -- but __exit__ deletes that
    directory, so a failed multi-file mutation left artifacts and the
    llama-swap config modified with no way back at all.
    """

    def setUp(self):
        from tempfile import TemporaryDirectory
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles),
                  mock.patch.object(modelctl, "STATE_DIR", self.root)):
            p.start()
            self.addCleanup(p.stop)

    def test_artifacts_are_restored_on_failure(self):
        artifact = self.root / "run.sh"
        artifact.write_text("original")
        with self.assertRaises(RuntimeError):
            with modelctl_transactions.Transaction("t") as tx:
                tx.stage_artifact(artifact, "replaced")
                tx.commit()
                raise RuntimeError("boom")
        self.assertEqual(artifact.read_text(), "original")

    def test_configs_are_restored_on_failure(self):
        config = self.root / "config.yaml"
        config.write_text("models: {}\n")
        with self.assertRaises(RuntimeError):
            with modelctl_transactions.Transaction("t") as tx:
                tx.stage_config(config, "models: {broken}\n")
                tx.commit()
                raise RuntimeError("boom")
        self.assertEqual(config.read_text(), "models: {}\n")

    def test_files_created_by_the_transaction_are_removed(self):
        # Undoing a creation means the file should not be left behind.
        created = self.root / "new-artifact.sh"
        with self.assertRaises(RuntimeError):
            with modelctl_transactions.Transaction("t") as tx:
                tx.stage_artifact(created, "content")
                tx.commit()
                raise RuntimeError("boom")
        self.assertFalse(created.exists())

    def test_same_basename_in_different_dirs_does_not_collide(self):
        # Backups keyed by basename alone overwrote each other, so one of
        # the two run.sh files was restored with the other's content.
        a = self.root / "a" / "run.sh"
        b = self.root / "b" / "run.sh"
        a.parent.mkdir()
        b.parent.mkdir()
        a.write_text("A original")
        b.write_text("B original")
        with self.assertRaises(RuntimeError):
            with modelctl_transactions.Transaction("t") as tx:
                tx.stage_artifact(a, "A replaced")
                tx.stage_artifact(b, "B replaced")
                tx.commit()
                raise RuntimeError("boom")
        self.assertEqual(a.read_text(), "A original")
        self.assertEqual(b.read_text(), "B original")

    def test_successful_transaction_keeps_its_writes(self):
        artifact = self.root / "run.sh"
        artifact.write_text("original")
        with modelctl_transactions.Transaction("t") as tx:
            tx.stage_artifact(artifact, "replaced")
            tx.commit()
        self.assertEqual(artifact.read_text(), "replaced")
