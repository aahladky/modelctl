"""The profile read-modify-write cores hold state_lock for the whole
load -> mutate -> save span.

save_profile's internal lock only makes the final write atomic, not the
update: two concurrent editors (CLI vs web) could each load the same
profile, then serially save, and the second save would silently drop the
first one's fields. These tests pin that load and save both happen
inside one held lock span (state_lock is reentrant per thread, so the
inner save_profile lock nests instead of deadlocking).
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_fsutil

PROFILE = {
    "name": "m1", "repo_id": "r/m", "file": "m-Q4_K_M",
    "model_path": "/x/m.gguf", "mmproj_path": None, "mtp_path": None,
    "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
               "ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q4_0",
               "flash_attn": "auto", "ttl": 3600, "mtp": "off", "fit": "on",
               "extra": ""},
    "env": [], "enabled": True,
}


def _lock_depth():
    return getattr(modelctl_fsutil._tls, "depth", 0)


class LockSpanBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        (self.profiles_dir / "m1.json").write_text(json.dumps(PROFILE))

        self.depths = {}
        real_load, real_save = modelctl.load_profile, modelctl.save_profile

        def spying_load(name):
            self.depths["load"] = _lock_depth()
            return real_load(name)

        def spying_save(profile):
            self.depths["save"] = _lock_depth()
            return real_save(profile)

        patches = [
            mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
            mock.patch.object(modelctl_fsutil, "STATE_DIR",
                              Path(self.tmp.name) / "state"),
            mock.patch.object(modelctl, "load_profile", spying_load),
            mock.patch.object(modelctl, "save_profile", spying_save),
            mock.patch.object(modelctl, "generate_artifacts",
                              lambda profile: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def assert_read_modify_write_was_one_lock_span(self):
        self.assertGreaterEqual(
            self.depths.get("load", 0), 1,
            "load_profile ran outside state_lock -- a concurrent editor "
            "can slip in between the read and the write")
        self.assertGreaterEqual(self.depths.get("save", 0), 1)


class TestUpdateProfileConfigLock(LockSpanBase):
    def test_update_holds_lock_across_read_modify_write(self):
        with mock.patch.object(modelctl, "preflight",
                               return_value=(True, None, None, [])):
            profile, _messages = modelctl.update_profile_config(
                "m1", {"ctx": 1024}, resync=False)
        self.assert_read_modify_write_was_one_lock_span()
        self.assertEqual(profile["config"]["ctx"], 1024)
        on_disk = json.loads((self.profiles_dir / "m1.json").read_text())
        self.assertEqual(on_disk["config"]["ctx"], 1024)


class TestUpdateRuntimePolicyLock(LockSpanBase):
    def test_policy_update_holds_lock_across_read_modify_write(self):
        modelctl.update_runtime_policy(
            "m1", {"mode": "managed", "objective": "balanced"}, resync=False)
        self.assert_read_modify_write_was_one_lock_span()
        on_disk = json.loads((self.profiles_dir / "m1.json").read_text())
        self.assertEqual(on_disk["runtime"]["objective"], "balanced")


if __name__ == "__main__":
    unittest.main()
