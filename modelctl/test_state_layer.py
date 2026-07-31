"""Regression tests for the state layer (Phase 3 of the 2026-07 remediation).

Atomic writes, the cross-process state lock, the unified state dir,
resilient bulk readers, defaults preservation, and import hygiene.
"""
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_fsutil
import modelctl_transactions


class TestAtomicWriteText(unittest.TestCase):
    def test_writes_content_and_leaves_no_tmp(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.json"
            modelctl_fsutil.atomic_write_text(target, '{"a": 1}')
            self.assertEqual(target.read_text(), '{"a": 1}')
            leftovers = [p for p in Path(tmp).iterdir() if p != target]
            self.assertEqual(leftovers, [])

    def test_overwrite_is_replace_not_truncate(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.json"
            target.write_text("old")
            # A reader holding the old fd keeps seeing complete old bytes.
            with open(target) as reader:
                modelctl_fsutil.atomic_write_text(target, "new")
                self.assertEqual(reader.read(), "old")
            self.assertEqual(target.read_text(), "new")


class TestStateLock(unittest.TestCase):
    def test_reentrant_within_a_thread(self):
        with TemporaryDirectory() as tmp, \
             mock.patch.object(modelctl_fsutil, "STATE_DIR", Path(tmp)):
            with modelctl_fsutil.state_lock():
                with modelctl_fsutil.state_lock():
                    pass  # must not deadlock

    def test_serializes_across_threads(self):
        order = []
        with TemporaryDirectory() as tmp, \
             mock.patch.object(modelctl_fsutil, "STATE_DIR", Path(tmp)):
            entered = threading.Event()
            release = threading.Event()

            def holder():
                with modelctl_fsutil.state_lock():
                    order.append("A-in")
                    entered.set()
                    release.wait(timeout=5)
                    order.append("A-out")

            def contender():
                entered.wait(timeout=5)
                with modelctl_fsutil.state_lock():
                    order.append("B-in")

            ta = threading.Thread(target=holder)
            tb = threading.Thread(target=contender)
            ta.start(); tb.start()
            entered.wait(timeout=5)
            time.sleep(0.1)   # give B a chance to (wrongly) enter
            release.set()
            ta.join(timeout=5); tb.join(timeout=5)
        self.assertEqual(order, ["A-in", "A-out", "B-in"])


class TestUnifiedStateDir(unittest.TestCase):
    def test_modelctl_home_moves_every_module(self):
        # Import-time resolution: check in a subprocess with the env set.
        code = (
            "import modelctl, modelctl_hardware, modelctl_capabilities, "
            "modelctl_runtime; from modelctl_web import jobs, wizard; "
            "dirs = {str(modelctl.STATE_DIR), str(jobs.STATE_DIR), "
            "str(modelctl_hardware.STATE_DIR), "
            "str(modelctl_capabilities.STATE_DIR), "
            "str(modelctl_runtime.STATE_DIR), str(wizard.STATE_DIR)}; "
            "print(dirs.pop() if len(dirs) == 1 else 'SPLIT: %s' % dirs)"
        )
        with TemporaryDirectory() as tmp:
            env = {**os.environ, "MODELCTL_HOME": tmp}
            env.pop("MODELCTL_STATE_DIR", None)
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, env=env,
                cwd=str(Path(__file__).parent))
            self.assertEqual(out.stdout.strip(), tmp,
                             f"state dirs split: {out.stdout} {out.stderr}")


class TestImportHasNoSideEffects(unittest.TestCase):
    def test_importing_the_web_app_touches_no_state(self):
        # Importing modelctl_web.app used to construct a JobStore against
        # the live web_jobs.db (marking running jobs 'interrupted') and
        # spawn six worker threads.
        code = (
            "import modelctl_web.app as a; "
            "assert not hasattr(a, 'app'), 'module-level app is back'; "
            "print('clean')"
        )
        with TemporaryDirectory() as tmp:
            env = {**os.environ, "MODELCTL_HOME": tmp}
            env.pop("MODELCTL_STATE_DIR", None)
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, env=env,
                cwd=str(Path(__file__).parent))
            self.assertEqual(out.stdout.strip(), "clean", out.stderr)
            self.assertFalse((Path(tmp) / "web_jobs.db").exists(),
                             "import created the jobs DB")


class TestResilientReaders(unittest.TestCase):
    def test_one_corrupt_profile_does_not_brick_the_rest(self):
        with TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles"
            profiles.mkdir()
            (profiles / "good.json").write_text(json.dumps(
                {"name": "good", "enabled": True}))
            (profiles / "torn.json").write_text('{"name": "to')  # truncated
            with mock.patch.object(modelctl, "PROFILES_DIR", profiles):
                out = modelctl._read_all_profiles()
        self.assertEqual([p["name"] for p in out], ["good"])


class TestPromptDefaultsPreservesUnknownKeys(unittest.TestCase):
    def test_web_settings_survive_cli_defaults(self):
        with TemporaryDirectory() as tmp:
            defaults_path = Path(tmp) / "defaults.json"
            defaults_path.write_text(json.dumps(
                {"ttl": 3600, "vram_limit_pct": 70, "primary_gpu": "SYCL1"}))
            with mock.patch.object(modelctl, "DEFAULTS_PATH", defaults_path), \
                 mock.patch.object(modelctl_fsutil, "STATE_DIR", Path(tmp)), \
                 mock.patch("builtins.input", return_value=""), \
                 mock.patch.object(modelctl, "prompt_int",
                                   side_effect=lambda label, cur: cur):
                modelctl.prompt_defaults()
            saved = json.loads(defaults_path.read_text())
        self.assertEqual(saved.get("vram_limit_pct"), 70,
                         "prompt_defaults erased the web-set VRAM budget")
        self.assertEqual(saved.get("primary_gpu"), "SYCL1")


class TestTransactionBackups(unittest.TestCase):
    def test_backup_dirs_never_collide(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.object(modelctl, "STATE_DIR", Path(tmp)), \
                 mock.patch.object(modelctl, "PROFILES_DIR",
                                   Path(tmp) / "profiles"):
                with modelctl_transactions.Transaction("same") as t1, \
                     modelctl_transactions.Transaction("same") as t2:
                    self.assertNotEqual(t1._backup_dir, t2._backup_dir)

    def test_orphaned_backups_are_listed(self):
        with TemporaryDirectory() as tmp:
            orphan = Path(tmp) / ".tx_backups" / "dead_123_1_abcd"
            orphan.mkdir(parents=True)
            with mock.patch.object(modelctl, "STATE_DIR", Path(tmp)):
                found = modelctl_transactions.orphaned_backups()
        self.assertEqual(found, [orphan])


class TestWebInstallEnvPropagation(unittest.TestCase):
    def test_unit_carries_modelctl_env(self):
        import io
        from argparse import Namespace
        from contextlib import redirect_stdout
        with TemporaryDirectory() as tmp:
            unit_path = Path(tmp) / "modelctl-web.service"
            ok = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch.object(modelctl, "WEB_UNIT_PATH", unit_path), \
                 mock.patch.object(modelctl.subprocess, "run",
                                   return_value=ok), \
                 mock.patch.object(modelctl, "web_token",
                                   return_value="tok"), \
                 mock.patch.dict(os.environ,
                                 {"MODELCTL_HOME": "/data/mc"}), \
                 redirect_stdout(io.StringIO()):
                modelctl.cmd_web_install(Namespace(bind=None))
            unit = unit_path.read_text()
        self.assertIn('Environment="MODELCTL_HOME=/data/mc"', unit,
                      "installed unit dropped the CLI's MODELCTL_HOME")
        self.assertIn('Environment="MODELCTL_WEB_BIND=', unit)


if __name__ == "__main__":
    unittest.main()
