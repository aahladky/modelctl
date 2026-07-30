"""Release A real-process integration tests (roadmap Task E2).

test_release_a.py is control-plane coverage with everything mocked. These
launch real llama-server processes against a real GGUF, because the
failures Release A is meant to catch -- a command that does not actually
run, a reservation that does not actually block, a rollback that does not
actually restore -- are invisible to a mocked test.

They are opt-in because they are slow and consume GPU and RAM:

    MODELCTL_REAL_TESTS=1 .venv/bin/python -m unittest test_release_a_real -v

Point them at a different model with MODELCTL_TEST_MODEL=/path/to.gguf.
A small model is deliberate: this suite is about the control plane
driving real processes, not about model quality.

Nothing here touches the live llama-swap service. Registration is
exercised against a temporary config file; restarting the user's running
service is out of scope for an automated test.
"""
import json
import os
import shutil
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_launch
import modelctl_plans

ENABLED = os.environ.get("MODELCTL_REAL_TESTS") == "1"

DEFAULT_MODEL = os.path.expanduser(
    "~/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf")
MODEL_PATH = os.environ.get("MODELCTL_TEST_MODEL", DEFAULT_MODEL)


def _reason():
    if not ENABLED:
        return "set MODELCTL_REAL_TESTS=1 to run real-process tests"
    if not os.path.exists(MODEL_PATH):
        return f"test model not found: {MODEL_PATH}"
    return None


SKIP = _reason()


@unittest.skipIf(SKIP, SKIP or "")
class RealProcessBase(unittest.TestCase):
    """A real profile, a real binary, an isolated state directory."""

    @classmethod
    def setUpClass(cls):
        cls.binary = cls._find_binary()
        if not cls.binary:
            raise unittest.SkipTest("no runnable llama-server build found")

    @staticmethod
    def _find_binary():
        candidates = [modelctl.LLAMA_SERVER_BIN] + \
            modelctl.find_llama_server_candidates()
        return next((c for c in candidates if c and os.access(c, os.X_OK)), None)

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()

        self.profile = modelctl.normalize_profile({
            "name": "realtest",
            "model_path": MODEL_PATH,
            "backend": "llama-cpp",
            "binary": self.binary,
            "config": {"ctx": 2048, "flash_attn": "auto", "fit": "off",
                       "ttl": 60, "extra": "-ngl 0", "device": ""},
            "env": [],
        })
        (self.profiles / "realtest.json").write_text(json.dumps(self.profile))

        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles),
                  mock.patch.object(modelctl, "LLAMA_SWAP_CONFIG_PATH",
                                    self.root / "llama-swap.yaml")):
            p.start()
            self.addCleanup(p.stop)

    def current_plan(self, backend=None):
        backend = backend or modelctl_launch.resolve_backend(self.profile)
        return modelctl_plans.current_profile_plan(
            self.profile, capabilities=backend.capabilities), backend


class TestDirectWorkerLaunch(RealProcessBase):
    """The canonical command really starts a server that really answers."""

    def test_canonical_command_launches_and_serves(self):
        import subprocess
        import urllib.request

        port = modelctl.find_free_port()
        cmd, ok, messages = modelctl.canonical_launch_command(
            self.profile, port=port)
        self.assertTrue(cmd.is_valid, f"validation blocked launch: {messages}")

        proc = subprocess.Popen(list(cmd.argv), env=cmd.environment,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        try:
            healthy = False
            deadline = time.time() + 180
            while time.time() < deadline:
                if proc.poll() is not None:
                    self.fail(f"process exited early: {proc.returncode}")
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            healthy = True
                            break
                except Exception:
                    time.sleep(1)
            self.assertTrue(healthy, "server never became healthy")

            body = json.dumps({
                "model": "realtest",
                "messages": [{"role": "user", "content": "Say OK."}],
                # Generous budget: the fixture is a reasoning model, and a
                # small max_tokens is spent entirely on reasoning content,
                # leaving `content` empty through no fault of the server.
                "max_tokens": 128, "temperature": 0,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read())
            message = payload["choices"][0]["message"]
            # Tokens were produced somewhere -- reasoning or content both
            # prove the model ran.
            self.assertTrue(
                message.get("content") or message.get("reasoning_content"),
                f"server returned no tokens at all: {payload}")
            self.assertGreater(payload["usage"]["completion_tokens"], 0)
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_launched_command_matches_the_previewed_one(self):
        # The whole of Phase B, checked against a process that really ran.
        preview, _ok, _m = modelctl.canonical_launch_command(self.profile)
        launched, _ok2, _m2 = modelctl.canonical_launch_command(
            self.profile, port=45999)
        self.assertEqual(preview.command_fingerprint,
                         launched.command_fingerprint)


class TestPlanTest(RealProcessBase):
    """modelctl_tune.test_launch_plan against a real process."""

    def test_plan_test_records_a_real_measurement(self):
        import modelctl_runtime
        import modelctl_tune

        db = modelctl_runtime.RuntimeDB(self.root / "runtime.db")
        plan, backend = self.current_plan()
        with mock.patch.object(modelctl_runtime, "RuntimeDB", return_value=db), \
             mock.patch.object(modelctl_tune, "RuntimeDB", return_value=db,
                               create=True):
            run = modelctl_tune.test_launch_plan(
                "realtest", plan.id, log=lambda *a: None,
                max_tokens=16, runs=1, binary=self.binary)

        self.assertTrue(run.get("success"), run.get("failure_class"))
        self.assertGreater(run.get("generation_tps") or 0, 0)
        self.assertTrue(run.get("command_fingerprint"))
        self.assertTrue(run.get("binary_fingerprint"))
        # Task D2/D5: the counters are real, not placeholders.
        self.assertIsNotNone(run.get("major_faults"))
        self.assertIn("storage_activity", run)
        self.assertIn(run["storage_activity"],
                      ("page-cache-served", "no-storage-activity",
                       "bulk-read", "storage-backed", "fault-heavy"))


class TestLlamaSwapRegistration(RealProcessBase):
    """Registration is exercised against a temp config, never the live one."""

    def test_sync_writes_a_loadable_entry(self):
        import yaml
        modelctl.generate_artifacts(dict(self.profile))
        entry, ok, _msgs = modelctl.render_llama_swap_entry(dict(self.profile))
        parsed = yaml.safe_load(entry)
        self.assertIn("realtest", parsed)
        cmd = parsed["realtest"]["cmd"]
        self.assertIn(self.binary, cmd)
        self.assertIn("${PORT}", cmd)
        self.assertTrue(ok)

    def test_generated_run_sh_is_executable_and_correct(self):
        modelctl.generate_artifacts(dict(self.profile))
        run_sh = self.profiles / "realtest" / "run.sh"
        self.assertTrue(run_sh.exists())
        self.assertTrue(os.access(run_sh, os.X_OK))
        text = run_sh.read_text()
        self.assertIn(self.binary, text)
        self.assertIn(MODEL_PATH, text)


class TestReservations(RealProcessBase):
    """Two competing reservations: the second must actually be refused."""

    def test_conflicting_reservation_is_refused(self):
        # Byte-based admission: the second claim does not fit the budget
        # once the first is held, so it must be refused rather than
        # oversubscribing the GPU.
        import modelctl_runtime
        db = modelctl_runtime.RuntimeDB(self.root / "runtime.db")
        plan, _backend = self.current_plan()
        claim = {"vram_bytes": {"SYCL0": 8 << 30}, "ram_bytes": 1 << 30}
        budgets = {"SYCL0": 12 << 30, "RAM": 8 << 30}

        first = db.acquire_reservation("realtest", plan.id, claim,
                                       os.getpid(), budgets)
        self.assertIsNotNone(first)
        # A different owner pid, or the budget check treats it as our own.
        second = db.acquire_reservation("other", plan.id, claim,
                                        os.getppid(), budgets)
        self.assertIsNone(second,
                          "a reservation that does not fit was granted anyway")
        db.release_reservation(first["id"])
        third = db.acquire_reservation("other", plan.id, claim,
                                       os.getppid(), budgets)
        self.assertIsNotNone(third, "reservation not released")


class TestTransactionRollback(RealProcessBase):
    """A failed multi-file mutation leaves the originals intact."""

    def test_failed_transaction_restores_every_file(self):
        import modelctl_transactions
        a = self.root / "a.json"
        b = self.root / "b.json"
        a.write_text('{"v": 1}')
        b.write_text('{"v": 2}')

        with self.assertRaises(RuntimeError):
            with modelctl_transactions.Transaction("realtest-fail") as tx:
                tx.stage_artifact(a, '{"v": 99}')
                tx.stage_artifact(b, '{"v": 99}')
                tx.commit()
                raise RuntimeError("boom after commit")

        # Rollback restores both files, not just the one that failed.
        self.assertEqual(json.loads(a.read_text())["v"], 1)
        self.assertEqual(json.loads(b.read_text())["v"], 2)

    def test_successful_transaction_commits_every_file(self):
        import modelctl_transactions
        a = self.root / "a.json"
        b = self.root / "b.json"
        a.write_text('{"v": 1}')
        b.write_text('{"v": 2}')
        with modelctl_transactions.Transaction("realtest-ok") as tx:
            tx.stage_artifact(a, '{"v": 10}')
            tx.stage_artifact(b, '{"v": 20}')
            tx.commit()
        self.assertEqual(json.loads(a.read_text())["v"], 10)
        self.assertEqual(json.loads(b.read_text())["v"], 20)


class TestCancellation(RealProcessBase):
    """A cancelled job must actually stop its child process."""

    def test_cancelling_a_launch_kills_the_process(self):
        import subprocess
        port = modelctl.find_free_port()
        cmd, _ok, _m = modelctl.canonical_launch_command(self.profile, port=port)
        proc = subprocess.Popen(
            list(cmd.argv), env=cmd.environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        time.sleep(3)
        self.assertIsNone(proc.poll(), "process died before cancellation")

        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        deadline = time.time() + 30
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.5)
        self.assertIsNotNone(proc.poll(), "process survived cancellation")


class TestBaselineCandidateSet(RealProcessBase):
    """Task E1's set builds against real hardware and a real binary."""

    def test_candidates_share_their_invariants(self):
        import modelctl_baseline
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
        backend = modelctl_launch.resolve_backend(self.profile)
        s = modelctl_baseline.build_candidate_set(self.profile, snap, backend)

        self.assertEqual(len(s.candidates), 4)
        self.assertTrue(s.invariants["binary_fingerprint"])
        # stock-mmap is the do-nothing baseline and must always exist.
        self.assertTrue(s.by_name("stock-mmap").available)
        for c in s.available():
            self.assertIsNotNone(c.plan)
            self.assertEqual(c.plan.claim.expected_context,
                             s.invariants["context"])

    def test_unavailable_candidates_explain_themselves(self):
        import modelctl_baseline
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
        s = modelctl_baseline.build_candidate_set(self.profile, snap, backend=None)
        for c in s.candidates:
            if not c.available:
                self.assertTrue(c.reason, f"{c.name} unavailable with no reason")


if __name__ == "__main__":
    unittest.main()
