import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
