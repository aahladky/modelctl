import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from modelctl_web.wizard import WizardState, WizardStore, STEPS


class TestWizardState(unittest.TestCase):
    def test_default_state(self):
        s = WizardState()
        self.assertEqual(s.step, "source")
        self.assertTrue(s.wizard_id)
        self.assertFalse(s.download_complete)

    def test_advance(self):
        s = WizardState()
        s.advance("inspect")
        self.assertEqual(s.step, "inspect")
        s.advance("done")
        self.assertEqual(s.step, "done")

    def test_advance_invalid_step_ignored(self):
        s = WizardState()
        s.advance("invalid_step")
        self.assertEqual(s.step, "source")

    def test_set_error(self):
        s = WizardState()
        s.set_error("something broke")
        self.assertEqual(len(s.errors), 1)
        self.assertIn("broke", s.errors[0]["message"])

    def test_roundtrip(self):
        s = WizardState(repo_id="test/repo", step="plans")
        d = s.to_dict()
        s2 = WizardState.from_dict(d)
        self.assertEqual(s2.repo_id, "test/repo")
        self.assertEqual(s2.step, "plans")

    def test_from_dict_ignores_unknown(self):
        d = {"wizard_id": "x", "step": "source", "unknown_field": 42}
        s = WizardState.from_dict(d)
        self.assertEqual(s.wizard_id, "x")


class TestWizardStore(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WizardStore(Path(self.tmp.name) / "wizards")

    def test_save_and_load(self):
        s = WizardState(repo_id="test/repo")
        self.store.save(s)
        loaded = self.store.load(s.wizard_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.repo_id, "test/repo")

    def test_load_nonexistent(self):
        self.assertIsNone(self.store.load("nonexistent"))

    def test_delete(self):
        s = WizardState()
        self.store.save(s)
        self.store.delete(s.wizard_id)
        self.assertIsNone(self.store.load(s.wizard_id))

    def test_list_active(self):
        s1 = WizardState(step="source")
        s2 = WizardState(step="plans")
        s3 = WizardState(step="done")
        s3.updated_at = time.time() - 100000  # old
        self.store.save(s1)
        self.store.save(s2)
        self.store.save(s3)
        active = self.store.list_active()
        ids = [w.wizard_id for w in active]
        self.assertIn(s1.wizard_id, ids)
        self.assertIn(s2.wizard_id, ids)
        self.assertNotIn(s3.wizard_id, ids)

    def test_list_active_sorted(self):
        s1 = WizardState(step="source")
        s1.updated_at = time.time() - 100
        s2 = WizardState(step="plans")
        s2.updated_at = time.time() - 10
        self.store.save(s1)
        self.store.save(s2)
        active = self.store.list_active()
        self.assertEqual(active[0].wizard_id, s2.wizard_id)

    def test_cleanup_removes_old(self):
        s = WizardState()
        self.store.save(s)
        # Manually backdate the file after save.
        path = self.store._path(s.wizard_id)
        data = json.loads(path.read_text())
        data["updated_at"] = time.time() - 700000
        path.write_text(json.dumps(data))
        self.store.cleanup()
        self.assertIsNone(self.store.load(s.wizard_id))
