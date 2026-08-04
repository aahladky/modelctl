import json
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_runtime
from modelctl_web.app import create_app
from modelctl_web.jobs import JobStore, JobRunner

PROFILE = {
    "name": "m1", "repo_id": "r/m", "file": "m-Q4_K_M",
    "model_path": "/x/m.gguf", "mmproj_path": None, "mtp_path": None,
    "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
               "ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q4_0",
               "flash_attn": "auto", "ttl": 3600, "mtp": "off", "fit": "on",
               "extra": ""},
    "env": [], "enabled": True,
}


class WebTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        (self.profiles_dir / "m1.json").write_text(json.dumps(PROFILE))
        store = JobStore(Path(self.tmp.name) / "jobs.db")
        runner = JobRunner(store)
        self.store = store
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        app = create_app(store=store, runner=runner)
        self.client = TestClient(app)
        # Auth removed 2026-08-03 (owner decision: LAN-open like :9292).
        # Empty dict keeps the many headers=self.auth call sites valid.
        self.auth = {}
        # Endpoints construct RuntimeDB() with its default path, which is
        # the real ~/.local/share/modelctl/runtime.db -- that is how test
        # plan-run rows (m1/p1) leaked into production plan history.
        # Redirect default constructions into this test's tmp dir while
        # still honoring an explicit db_path.
        self.runtime_db_path = Path(self.tmp.name) / "runtime.db"
        real_runtime_db = modelctl_runtime.RuntimeDB

        def _redirected_runtime_db(db_path=None):
            return real_runtime_db(db_path=db_path or self.runtime_db_path)

        # Setup banners and plan pages sweep these globs and probe what
        # they find; keep single-file runs off the real llama.cpp builds
        # (the full suite already empties them via test__hermeticity).
        no_binaries = str(Path(self.tmp.name) / "no-binaries" / "*")
        import modelctl_web.wizard as wizard_mod
        self.patches = [
            mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
            mock.patch.object(modelctl_runtime, "RuntimeDB",
                              _redirected_runtime_db),
            mock.patch.object(modelctl, "COMMON_LLAMA_SERVER_GLOBS",
                              [no_binaries]),
            mock.patch.object(modelctl, "COMMON_ENV_SCRIPT_GLOBS",
                              [no_binaries]),
            mock.patch.object(wizard_mod, "WIZARD_DIR",
                              Path(self.tmp.name) / "wizards"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def resolvable_backend(self):
        """Make the setup probe see a runnable llama-server, so readiness
        resolves the same way on every machine. Without this the tests that
        reach the setup probe (now folded into /api/v2/settings) only
        behaved consistently on machines that happened to have a real
        llama.cpp build for the candidate sweep to find."""
        binary = Path(self.tmp.name) / "llama-server"
        binary.write_text("#!/bin/sh\nexit 1\n")
        binary.chmod(0o755)
        return mock.patch.multiple(modelctl,
                                   LLAMA_SERVER_BIN=str(binary),
                                   find_llama_server_candidates=lambda: [])

    def wait_job(self, job_id, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.client.get(f"/api/jobs/{job_id}", headers=self.auth).json()
            if j["status"] not in ("queued", "running"):
                return j
            time.sleep(0.1)
        self.fail(f"job {job_id} never finished")


class TestOpenAccess(WebTestBase):
    """No auth layer: the console is deliberately LAN-open (owner decision
    2026-08-03, same posture as llama-swap on :9292). What remains is the
    cross-origin POST rejection -- without it any website open in a LAN
    browser could drive the mutating API blind."""

    def test_api_open_without_credentials(self):
        self.assertEqual(self.client.get("/api/jobs").status_code, 200)

    def test_root_moves_to_console(self):
        resp = self.client.get("/", follow_redirects=False)
        self.assertEqual(resp.headers["location"], "/v2/")

    def test_login_routes_gone(self):
        self.assertEqual(self.client.get("/login").status_code, 404)
        self.assertEqual(
            self.client.post("/login", data={"token_field": "x"}).status_code,
            404)
        self.assertEqual(self.client.get("/logout").status_code, 404)

    def test_token_rotation_gone(self):
        resp = self.client.post("/api/v2/settings/token/rotate",
                                json={"confirm": True})
        self.assertEqual(resp.status_code, 404)

    def test_cross_origin_post_rejected(self):
        resp = self.client.post("/api/profiles/m1", headers={
            "Origin": "http://evil.example"}, json={})
        self.assertEqual(resp.status_code, 403)

    def test_settings_access_reports_no_token(self):
        access = self.client.get("/api/v2/settings").json()["access"]
        self.assertNotIn("token_path", access)
        self.assertIn("bind", access)

    def test_healthz_open(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)


class TestDownloadHonesty(WebTestBase):
    """The console must not lie about downloads: cancel kills the puller,
    deleting a wizard cancels its jobs, and a pull that cannot fit on disk
    is refused before it starts (2026-08-03 review findings)."""

    def test_wizard_delete_cancels_its_live_jobs(self):
        import threading
        from modelctl_web.wizard import WizardState, WizardStore
        release = threading.Event()

        def blocked(ctx):
            release.wait(timeout=10)
            return {}

        job_id = self.store and None
        runner = self.client.app.state.runner
        job_id = runner.submit("pull", "stub download", blocked,
                               lane="download")
        state = WizardState()
        state.download_job_id = job_id
        WizardStore().save(state)
        deadline = time.time() + 5
        while time.time() < deadline:
            if self.store.get(job_id)["status"] == "running":
                break
            time.sleep(0.05)
        resp = self.client.post(f"/api/v2/wizard/{state.wizard_id}/delete")
        self.assertEqual(resp.status_code, 200)
        release.set()
        self.assertEqual(self.store.get(job_id)["status"], "cancelled")

    def test_pull_refused_when_disk_space_short(self):
        from modelctl_web import mutate
        from modelctl_services.result import ServiceResult
        with mock.patch.object(
                mutate.acquisition_service, "check_destination_space",
                return_value=ServiceResult.failure("only 1 GiB free but this needs 50 GiB")):
            job_id = mutate.submit_pull(
                self.client.app.state.runner, "some/repo",
                needed_bytes=50 << 30)
            j = self.wait_job(job_id)
        self.assertEqual(j["status"], "failed")
        self.assertIn("free", j["error"])

    def test_pull_downloads_in_a_killable_subprocess(self):
        from modelctl_web import mutate
        registered = []

        class FakeProc:
            returncode = 0
            pid = 4242

            def __init__(self):
                import io as _io
                self.stdout = _io.StringIO(
                    "  downloading m.gguf -> /x ...\n"
                    "Done. Profile 'm1' created.\n")

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

        class Ctx:
            job_id = "j1"
            store = self.store

            def log(self, m):
                pass

            def set_progress(self, *a, **k):
                pass

            def raise_if_cancelled(self):
                pass

            def register_process(self, proc):
                registered.append(proc)

        shared = {"file": None}
        with mock.patch.object(mutate.subprocess, "Popen",
                               return_value=FakeProc()) as popen:
            result = mutate._run_pull_subprocess(
                Ctx(), "r/m", quant_label="Q4_K_M", want_mtp=True,
                current=shared)
        argv = popen.call_args.args[0]
        self.assertIn("pull", argv)
        self.assertIn("r/m", argv)
        self.assertIn("--quant", argv)
        self.assertEqual(registered, [popen.return_value])
        self.assertEqual(result["profile"], "m1")
        # The poller reads THIS dict; the reader must write into it, or
        # the progress bar sits at 0% for the whole download.
        self.assertEqual(shared["file"], "m.gguf")
        self.assertEqual(
            popen.call_args.kwargs["env"]["HF_HUB_DISABLE_PROGRESS_BARS"], "1")

    def test_sync_warning_reaches_the_job(self):
        from modelctl_web import mutate

        class Ctx:
            logged = []

            def log(self, m):
                self.logged.append(m)

        with mock.patch.object(
                mutate.modelctl, "sync_all_backends",
                return_value={"profiles": 1, "restarted": False,
                              "restart_error": "unit not found"}):
            reloaded = mutate._sync_backends(Ctx())
        self.assertFalse(reloaded)
        self.assertTrue(any("NOT reloaded" in m for m in Ctx.logged),
                        Ctx.logged)

    class _PullCtx:
        """The slice of JobContext _run_pull_subprocess touches."""

        def __init__(self):
            self.logged = []

        def log(self, m):
            self.logged.append(m)

        def set_progress(self, *a, **k):
            pass

        def raise_if_cancelled(self):
            pass

        def register_process(self, proc):
            pass

    @staticmethod
    def _failing_proc(stdout_text):
        import io as _io

        class FailingProc:
            returncode = 1
            pid = 4242
            stdout = _io.StringIO(stdout_text)

            def wait(self, timeout=None):
                return 1

            def poll(self):
                return 1

        return FailingProc()

    def test_failed_pull_removes_partial_files(self):
        """A pull that dies mid-file must not leave the half-written
        target or hf's .incomplete spool behind -- multi-GB junk that
        holds disk until someone finds it by hand."""
        from modelctl_web import mutate
        models_dir = Path(self.tmp.name) / "models"
        repo_dir = models_dir / "r" / "m"
        spool = repo_dir / ".cache" / "huggingface" / "download"
        spool.mkdir(parents=True)
        (repo_dir / "m.gguf").write_bytes(b"x" * 10)
        (spool / "m.gguf.abc123.incomplete").write_bytes(b"x" * 10)
        ctx = self._PullCtx()
        with mock.patch.object(
                mutate.subprocess, "Popen",
                return_value=self._failing_proc(
                    "  downloading m.gguf -> /x ...\n")), \
             mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir), \
             mock.patch.object(modelctl, "_remote_file_size",
                               return_value=100):
            with self.assertRaises(RuntimeError):
                mutate._run_pull_subprocess(ctx, "r/m")
        self.assertFalse((repo_dir / "m.gguf").exists())
        self.assertFalse((spool / "m.gguf.abc123.incomplete").exists())
        self.assertTrue(any("removed" in m for m in ctx.logged), ctx.logged)

    def test_pull_cleanup_keeps_complete_and_unverifiable_files(self):
        """Cleanup only deletes what is provably partial: a file whose
        size matches the repo's copy finished before the failure, and a
        file whose remote size can't be fetched (offline hiccup) might
        be complete -- deleting either would throw away good gigabytes."""
        from modelctl_web import mutate
        models_dir = Path(self.tmp.name) / "models"
        repo_dir = models_dir / "r" / "m"
        repo_dir.mkdir(parents=True)
        (repo_dir / "a.gguf").write_bytes(b"x" * 5)   # matches remote: keep
        (repo_dir / "b.gguf").write_bytes(b"x" * 3)   # remote unknown: keep
        sizes = {"a.gguf": 5, "b.gguf": None}
        ctx = self._PullCtx()
        with mock.patch.object(
                mutate.subprocess, "Popen",
                return_value=self._failing_proc(
                    "  downloading a.gguf -> /x ...\n"
                    "  downloading b.gguf -> /x ...\n")), \
             mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir), \
             mock.patch.object(modelctl, "_remote_file_size",
                               side_effect=lambda r, f: sizes.get(f)):
            with self.assertRaises(RuntimeError):
                mutate._run_pull_subprocess(ctx, "r/m")
        self.assertTrue((repo_dir / "a.gguf").exists())
        self.assertTrue((repo_dir / "b.gguf").exists())


class TestPullProgress(WebTestBase):
    """A pull that is moving has to say it is moving.

    The bar sat at 0 for entire multi-hour downloads. Two causes, both
    in the old poller: it waited on a "downloading X ->" line from a
    child whose stdout block-buffers for the whole transfer, and even
    once it had a filename it stat()'d the destination path -- which
    hf_hub_download does not create until the file is complete. Progress
    is now read from the bytes actually on disk.
    """

    def _repo_dirs(self, subfolder=""):
        models_dir = Path(self.tmp.name) / "models"
        repo_dir = models_dir / "r" / "m"
        spool = repo_dir / ".cache" / "huggingface" / "download" / subfolder
        spool.mkdir(parents=True)
        return models_dir, repo_dir, spool

    def test_bytes_on_disk_counts_finished_files_and_the_live_spool(self):
        """In-flight bytes live in a .incomplete spool whose name is a
        hash of the filename, not the filename -- so counting only the
        destination path reads 0 until the very last moment."""
        from modelctl_web import pull_progress
        models_dir, repo_dir, spool = self._repo_dirs()
        (repo_dir / "a.gguf").write_bytes(b"x" * 300)
        (spool / "9RmZbQ.etag123.ab12cd34.incomplete").write_bytes(b"x" * 200)
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            have = pull_progress.bytes_on_disk("r/m", ["a.gguf", "b.gguf"])
        self.assertEqual(have, 500)

    def test_bytes_on_disk_ignores_files_this_pull_is_not_fetching(self):
        """One repo dir can hold an earlier quant. Counting it would pin
        the bar near 100% from the first tick of an untouched download."""
        from modelctl_web import pull_progress
        models_dir, repo_dir, _spool = self._repo_dirs()
        (repo_dir / "wanted.gguf").write_bytes(b"x" * 100)
        (repo_dir / "other-quant.gguf").write_bytes(b"x" * 9000)
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            have = pull_progress.bytes_on_disk("r/m", ["wanted.gguf"])
        self.assertEqual(have, 100)

    def test_bytes_on_disk_finds_a_subfolder_quants_spool(self):
        """Repos that ship a quant inside a subfolder (BF16/model.gguf)
        spool into the matching subfolder; a flat glob misses it."""
        from modelctl_web import pull_progress
        models_dir, _repo_dir, spool = self._repo_dirs(subfolder="BF16")
        (spool / "hash.etag.incomplete").write_bytes(b"x" * 700)
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            have = pull_progress.bytes_on_disk("r/m", ["BF16/model.gguf"])
        self.assertEqual(have, 700)

    def test_bytes_on_disk_survives_a_missing_repo_dir(self):
        """The first tick fires before the child has created anything."""
        from modelctl_web import pull_progress
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR",
                               Path(self.tmp.name) / "models"):
            self.assertEqual(pull_progress.bytes_on_disk("r/m", ["a.gguf"]), 0)

    def test_poll_once_reports_the_fraction_landed_so_far(self):
        from modelctl_web import pull_progress
        models_dir, _repo_dir, spool = self._repo_dirs()
        (spool / "hash.etag.incomplete").write_bytes(b"x" * 250)
        seen = []
        poller = pull_progress.PullProgressPoller(
            "r/m", ["a.gguf"], 1000,
            on_progress=lambda v, d: seen.append((v, d)))
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            poller.poll_once()
        value, detail = seen[-1]
        self.assertAlmostEqual(value, 0.25)
        self.assertIn("/", detail)

    def test_poll_once_never_claims_done_before_the_child_exits(self):
        """Only the subprocess's exit code says the pull finished. A bar
        that reads 100% while the child is still running is the same lie
        in a different place."""
        from modelctl_web import pull_progress
        models_dir, repo_dir, _spool = self._repo_dirs()
        (repo_dir / "a.gguf").write_bytes(b"x" * 5000)
        seen = []
        poller = pull_progress.PullProgressPoller(
            "r/m", ["a.gguf"], 1000,
            on_progress=lambda v, d: seen.append((v, d)))
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            poller.poll_once()
        self.assertAlmostEqual(seen[-1][0], 0.99)

    def test_progress_detail_names_the_file_being_fetched(self):
        from modelctl_web import pull_progress
        models_dir, repo_dir, _spool = self._repo_dirs()
        (repo_dir / "a.gguf").write_bytes(b"x" * 100)
        seen = []
        poller = pull_progress.PullProgressPoller(
            "r/m", ["a.gguf"], 1000,
            on_progress=lambda v, d: seen.append((v, d)),
            label_fn=lambda: "a.gguf")
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            poller.poll_once()
        self.assertTrue(seen[-1][1].startswith("a.gguf"), seen)

    def test_poller_cannot_measure_without_a_size_or_a_file_list(self):
        """No repo metadata means no denominator. Report nothing rather
        than invent a fraction -- and let the caller say so in the log."""
        from modelctl_web import pull_progress
        seen = []
        blind = pull_progress.PullProgressPoller(
            "r/m", ["a.gguf"], 0, on_progress=lambda v, d: seen.append(v))
        self.assertFalse(blind.can_measure)
        blind.poll_once()
        self.assertEqual(seen, [])

    def test_poller_thread_stops_when_the_job_leaves_running(self):
        """A daemon thread polling a finished job keeps writing progress
        onto a completed row."""
        from modelctl_web import pull_progress
        seen = []
        poller = pull_progress.PullProgressPoller(
            "r/m", ["a.gguf"], 1000, on_progress=lambda v, d: seen.append(v),
            should_continue=lambda: False, interval=0.01)
        poller.start()
        poller.join(timeout=2)
        self.assertFalse(poller.is_alive())
        self.assertEqual(seen, [])

    def test_shutdown_waits_for_a_tick_that_is_already_in_flight(self):
        """Setting the stop flag does not recall a tick that is already
        mid-write. That write lands after the runner marks the job done
        and rewinds a finished pull to 99%."""
        import threading as _threading
        from modelctl_web import pull_progress
        models_dir, repo_dir, _spool = self._repo_dirs()
        (repo_dir / "a.gguf").write_bytes(b"x" * 100)
        entered, release, writes = _threading.Event(), _threading.Event(), []

        def slow_progress(value, detail):
            entered.set()
            release.wait(3)
            writes.append(value)

        poller = pull_progress.PullProgressPoller(
            "r/m", ["a.gguf"], 1000, on_progress=slow_progress, interval=0.01)
        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir):
            poller.start()
            self.assertTrue(entered.wait(3), "poller never ticked")
            release.set()
            poller.shutdown()
            self.assertFalse(poller.is_alive())
        self.assertEqual(writes, [0.1])

    def test_pull_shuts_the_poller_down_before_the_job_finishes(self):
        """Ordering contract: the poller is fully stopped inside the job
        function, so the runner's progress=1.0 is the last word."""
        from modelctl_services.result import ServiceResult
        from modelctl_web import mutate, pull_progress
        events = []

        class Recorder:
            can_measure = True

            def __init__(self, *a, **kw):
                pass

            def start(self):
                events.append("start")

            def shutdown(self):
                events.append("shutdown")

        def fake_pull(ctx, repo_id, **kw):
            events.append("download")
            return {"profile": "m1"}

        with mock.patch.object(pull_progress, "PullProgressPoller", Recorder), \
             mock.patch.object(mutate, "_run_pull_subprocess", fake_pull), \
             mock.patch.object(mutate.acquisition_service,
                               "check_destination_space",
                               return_value=ServiceResult()):
            job_id = mutate.submit_pull(
                self.client.app.state.runner, "r/m", needed_bytes=1000,
                needed_files=["a.gguf"])
            j = self.wait_job(job_id)
        self.assertEqual(j["status"], "done")
        self.assertEqual(events, ["start", "download", "shutdown"])

    def test_pull_job_progress_moves_while_bytes_are_landing(self):
        """The headline defect: bytes were landing on disk and the job's
        progress stayed at 0 for the whole transfer."""
        from modelctl_services.result import ServiceResult
        from modelctl_web import mutate, pull_progress
        models_dir, _repo_dir, spool = self._repo_dirs()
        observed = {}

        def fake_pull(ctx, repo_id, quant_label=None, want_mtp=True,
                      current=None):
            (spool / "hash.etag.ab12cd34.incomplete").write_bytes(b"x" * 400)
            deadline = time.time() + 5
            while time.time() < deadline:
                job = self.store.get(ctx.job_id)
                if job and job["progress"]:
                    observed["progress"] = job["progress"]
                    observed["detail"] = job["detail"]
                    break
                time.sleep(0.02)
            return {"profile": "m1"}

        with mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", models_dir), \
             mock.patch.object(pull_progress, "POLL_INTERVAL_SECONDS", 0.02), \
             mock.patch.object(mutate, "_run_pull_subprocess", fake_pull), \
             mock.patch.object(mutate.acquisition_service,
                               "check_destination_space",
                               return_value=ServiceResult()):
            job_id = mutate.submit_pull(
                self.client.app.state.runner, "r/m", needed_bytes=1000,
                needed_files=["a.gguf"])
            j = self.wait_job(job_id)
        self.assertEqual(j["status"], "done")
        self.assertAlmostEqual(observed.get("progress", 0), 0.4)

    def test_pull_without_a_measurable_size_says_so_in_the_log(self):
        from modelctl_services.result import ServiceResult
        from modelctl_web import mutate
        with mock.patch.object(mutate, "_run_pull_subprocess",
                               return_value={"profile": "m1"}), \
             mock.patch.object(mutate.acquisition_service,
                               "check_destination_space",
                               return_value=ServiceResult()):
            job_id = mutate.submit_pull(self.client.app.state.runner, "r/m")
            j = self.wait_job(job_id)
        self.assertEqual(j["status"], "done")
        self.assertIn("progress bar", j["result"])

    def test_pull_child_runs_unbuffered(self):
        """The child's per-file lines are the operator's ground truth. A
        block-buffered child delivers them after the download they were
        describing, and leaves the poller's filename label empty too."""
        from modelctl_web import mutate
        import io as _io

        class FakeProc:
            returncode = 0
            pid = 4242
            stdout = _io.StringIO("Done. Profile 'm1' created.\n")

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

        with mock.patch.object(mutate.subprocess, "Popen",
                               return_value=FakeProc()) as popen:
            mutate._run_pull_subprocess(TestDownloadHonesty._PullCtx(), "r/m")
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")


class TestSyncRetry(WebTestBase):
    """A job that saved config but couldn't reload the router leaves the
    old config serving; the console offers a one-click retry. The retry
    endpoint submits a sync job that tells the truth: done when the
    router reloaded, failed (with the reason) when it didn't."""

    def test_sync_endpoint_submits_job_that_succeeds(self):
        from modelctl_web import mutate
        with mock.patch.object(
                mutate.modelctl, "sync_all_backends",
                return_value={"profiles": 2, "restarted": True,
                              "restart_error": ""}):
            resp = self.client.post("/api/v2/sync")
            self.assertEqual(resp.status_code, 200)
            job_id = resp.json()["job_id"]
            j = self.wait_job(job_id)
        self.assertEqual(j["status"], "done")
        row = self.client.get(f"/api/v2/jobs/{job_id}").json()
        self.assertIs(row["router_reloaded"], True)

    def test_sync_job_fails_when_router_not_reloaded(self):
        from modelctl_web import mutate
        with mock.patch.object(
                mutate.modelctl, "sync_all_backends",
                return_value={"profiles": 2, "restarted": False,
                              "restart_error": "unit not found"}):
            job_id = self.client.post("/api/v2/sync").json()["job_id"]
            j = self.wait_job(job_id)
        self.assertEqual(j["status"], "failed")
        self.assertIn("unit not found", j["error"])

    def test_job_row_surfaces_router_reloaded_from_outcome(self):
        """tier-apply and moe-cache jobs already record router_reloaded
        in their outcome; the SPA needs it as a typed field, not by
        grepping the log tail."""
        from modelctl_web import telemetry
        job_id = self.store.create("tier-apply", "auto-place m1")
        self.store.update(job_id, outcome=json.dumps(
            {"applied": True, "router_reloaded": False}))
        row = telemetry.job_row(self.store.get(job_id))
        self.assertIs(row["router_reloaded"], False)

    def test_job_row_router_reloaded_absent_or_bad_outcome_is_none(self):
        from modelctl_web import telemetry
        job_id = self.store.create("edit", "edit m1")
        row = telemetry.job_row(self.store.get(job_id))
        self.assertIsNone(row["router_reloaded"])
        self.store.update(job_id, outcome="not json{")
        row = telemetry.job_row(self.store.get(job_id))
        self.assertIsNone(row["router_reloaded"])


class TestProfilesApi(WebTestBase):
    # Phase-3 cutover: profile *deletion* left the console with the old
    # server-rendered pages. POST /profiles/{name}/delete was the only
    # caller of modelctl.cmd_remove from the web, and no /api/v2 endpoint
    # replaced it -- removing a model is a CLI-only operation now. The
    # confirm-flow test that guarded it went with the route.

    def test_list_profiles(self):
        resp = self.client.get("/api/profiles", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([p["name"] for p in resp.json()], ["m1"])

    def test_models_api_lists_profiles(self):
        """The dashboard's model table is now the operate page fed by
        GET /api/v2/models; a saved profile has to appear in it whether or
        not llama-swap knows about the model yet."""
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_web.swap.LlamaSwapClient.runtime_state",
                        return_value={}), \
             self.resolvable_backend():
            resp = self.client.get("/api/v2/models", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        rows = {r["name"]: r for r in resp.json()}
        self.assertIn("m1", rows)
        # Unknown to llama-swap, but still the profile's own facts.
        self.assertEqual(rows["m1"]["file"], "m-Q4_K_M")
        self.assertFalse(rows["m1"]["running"])

    def test_edit_submits_job_and_applies(self):
        with mock.patch.object(modelctl, "update_profile_config",
                               return_value=(PROFILE, ["ok"])) as upd, \
             mock.patch.object(modelctl, "build_server_args", return_value=["x"]):
            resp = self.client.post(
                "/api/profiles/m1", headers=self.auth,
                json={"ctx": 65536, "bogus_field": "x"})
            job_id = resp.json()["job"]
            job = self.wait_job(job_id)
        self.assertEqual(job["status"], "done")
        upd.assert_called_once()
        self.assertEqual(upd.call_args[0][1]["ctx"], 65536)


class TestTiersApi(WebTestBase):
    PLAN = {"tier": 1, "config": {"device": "SYCL0", "split_mode": "",
                                  "tensor_split": "", "extra": ""},
            "layout": [("SYCL0", 10.0, "weights + KV")],
            "warnings": [],
            "analysis": {"weights_gib": 10, "kv_gib": 1, "is_moe": False,
                         "n_layers": 32, "ram_budget_gib": 20}}

    def test_tier_dry_run(self):
        import modelctl_tiers
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[
                {"device": "SYCL0", "name": "big", "total_bytes": 32 << 30,
                 "free_bytes": 30 << 30}]), \
             mock.patch.object(modelctl_tiers, "plan_tiers", return_value=self.PLAN):
            resp = self.client.get("/api/tiers/m1", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["tier"], 1)

    def test_tier_apply_serializes_through_worker(self):
        import modelctl_tiers
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[
                {"device": "SYCL0", "name": "big", "total_bytes": 32 << 30,
                 "free_bytes": 30 << 30}]), \
             mock.patch.object(modelctl_tiers, "plan_tiers", return_value=self.PLAN), \
             mock.patch.object(modelctl, "save_profile"), \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends") as sync:
            resp = self.client.post("/api/tiers/m1/apply", headers=self.auth)
            job = self.wait_job(resp.json()["job"])
        self.assertEqual(job["status"], "done")
        sync.assert_called_once()


class TestJobsApi(WebTestBase):
    def test_smoke_test_job_lifecycle_records(self):
        """The smoke test's measured throughput has to reach the job log.

        Its only remaining trigger is the wizard's test step with no plan
        selected (POST /api/v2/wizard/{id}/test/run -> submit_smoke_test);
        the old POST /profiles/{name}/test went with the console."""
        from modelctl_web.wizard import WizardState, WizardStore
        state = WizardState(step="test", profile_name="m1")
        WizardStore().save(state)
        with mock.patch.object(modelctl, "smoke_test_profile",
                               return_value={"ok": True, "stage": "done",
                                             "messages": ["fine"],
                                             "tok_per_s": 42.0}):
            resp = self.client.post(
                f"/api/v2/wizard/{state.wizard_id}/test/run", headers=self.auth)
            self.assertEqual(resp.status_code, 200)
            job = self.wait_job(resp.json()["test_job_id"])
        self.assertEqual(job["status"], "done")
        self.assertIn("42.0 tok/s", job["result"])

    def test_cancel_marks_job(self):
        with mock.patch("modelctl_web.mutate.submit_pull", return_value="fakejob1"):
            resp = self.client.post("/api/pull/x/y", headers=self.auth, json={})
        self.assertEqual(resp.json()["job"], "fakejob1")


class TestWizardAcquisition(WebTestBase):
    """HTTP-level coverage of the add-model wizard. The inspect step
    500'd in production twice (total_bytes vs total_size, then unguarded
    HF fetches) because nothing served these steps in tests.

    Re-homed onto /api/v2/wizard/* at the phase-3 cutover. Two things the
    old form routes owned have no successor and were dropped here:
    the malformed-repo-id rejection (identical coverage now lives in
    test_console_phase2.py TestWizardApiFlow.
    test_bad_repo_id_is_a_422_with_the_reason), and POST
    /profiles/{name}/bench -- the console has no benchmark trigger at all
    any more, so submit_bench's max_tokens/runs parsing is unreachable
    from the web.
    """

    REPO_CONTENTS = {
        "quant_groups": [
            {"label": "m-Q4_K_M", "files": ["m-Q4_K_M.gguf"],
             "sharded": False, "total_size": 50 << 30},
            {"label": "m-Q8_0", "files": ["m-Q8_0.gguf"],
             "sharded": False, "total_size": None},
        ],
        "mmproj_files": [], "mtp_files": [],
    }

    def _wizard(self, **fields):
        from modelctl_web.wizard import WizardState, WizardStore
        state = WizardState(**fields)
        WizardStore().save(state)
        return state

    def test_inspect_returns_real_producer_shape(self):
        """The endpoint carries what get_repo_contents() actually returns
        -- including a group whose total_size is None (missing HF sizes),
        which must stay None rather than being faked into a number."""
        w = self._wizard(step="inspect", repo_id="org/model",
                         source_type="hf_repo")
        with mock.patch.object(modelctl, "get_repo_contents",
                               return_value=self.REPO_CONTENTS):
            resp = self.client.get(f"/api/v2/wizard/{w.wizard_id}/inspect",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        groups = {g["label"]: g for g in resp.json()["contents"]["quant_groups"]}
        self.assertEqual(groups["m-Q4_K_M"]["total_size"], 50 << 30)
        self.assertIsNone(groups["m-Q8_0"]["total_size"])
        self.assertEqual(resp.json()["error"], "")

    def test_inspect_bad_repo_reports_error_not_500(self):
        w = self._wizard(step="inspect", repo_id="org/nonexistent",
                         source_type="hf_repo")
        with mock.patch.object(modelctl, "get_repo_contents",
                               side_effect=RuntimeError("404 not found")):
            resp = self.client.get(f"/api/v2/wizard/{w.wizard_id}/inspect",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["contents"])
        self.assertIn("org/nonexistent", resp.json()["error"])
        self.assertIn("404 not found", resp.json()["error"])

    def test_source_submit_requires_repo_id(self):
        w = self._wizard()
        resp = self.client.post(f"/api/v2/wizard/{w.wizard_id}/source",
                                headers=self.auth,
                                json={"source_type": "hf_repo", "repo_id": ""})
        self.assertEqual(resp.status_code, 422)
        self.assertIn("repo id", resp.json()["error"])
        self.assertEqual(resp.json()["state"]["step"], "source")
        from modelctl_web.wizard import WizardStore
        state = WizardStore().load(w.wizard_id)
        self.assertEqual(state.step, "source")
        self.assertTrue(state.errors)

    def test_inspect_without_repo_is_refused_not_bounced(self):
        """A repo-less wizard on inspect used to bounce Resume -> /add
        forever with no error shown; the endpoint now names the gap."""
        w = self._wizard(step="inspect")
        resp = self.client.get(f"/api/v2/wizard/{w.wizard_id}/inspect",
                               headers=self.auth)
        self.assertEqual(resp.status_code, 409)
        self.assertIn("no repository selected", resp.json()["error"])

    def test_partial_config_save_does_not_disable(self):
        """A programmatic save setting one field used to read the absent
        checkbox as 'disable', which unregistered the model on sync. The
        typed body has no checkbox: an absent `enabled` stays absent, and
        only an explicit false disables.

        An `extra` edit is a structural change, so both saves carry the
        confirm -- the gate itself is test_console_phase3.py's subject,
        not this one's."""
        with mock.patch("modelctl_web.mutate.submit_edit",
                        return_value="job-edit-1") as sub:
            resp = self.client.post(
                "/api/v2/models/m1/config", headers=self.auth,
                json={"updates": {"extra": "--foo"},
                      "accept_structural": True})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("enabled", sub.call_args[0][2])
        with mock.patch("modelctl_web.mutate.submit_edit",
                        return_value="job-edit-2") as sub:
            resp = self.client.post(
                "/api/v2/models/m1/config", headers=self.auth,
                json={"updates": {"extra": "--foo", "enabled": False},
                      "accept_structural": True})
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(sub.call_args[0][2]["enabled"])

    def test_wizard_delete_removes_it(self):
        w = self._wizard(step="inspect", repo_id="org/model")
        resp = self.client.post(f"/api/v2/wizard/{w.wizard_id}/delete",
                                headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["deleted"], w.wizard_id)
        from modelctl_web.wizard import WizardStore
        self.assertIsNone(WizardStore().load(w.wizard_id))

    def test_wizard_read_is_render_only(self):
        """The download GET used to submit the job as a side effect; two
        tabs raced the guard into duplicate concurrent pulls. Reading the
        wizard must never start one -- only POST /inspect does."""
        w = self._wizard(step="download", repo_id="org/model",
                         source_type="hf_repo")
        with mock.patch("modelctl_web.mutate.submit_pull") as sub:
            resp = self.client.get(f"/api/v2/wizard/{w.wizard_id}",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        sub.assert_not_called()
        # Still parked on download with nothing started: the client can
        # offer "start download" and the state backs that up.
        self.assertEqual(resp.json()["step"], "download")
        self.assertEqual(resp.json()["download_job_id"], "")

    def test_inspect_submit_starts_download_once(self):
        w = self._wizard(step="inspect", repo_id="org/model",
                         source_type="hf_repo")
        with mock.patch("modelctl_web.mutate.submit_pull",
                        return_value="job-pull-1") as sub:
            self.client.post(f"/api/v2/wizard/{w.wizard_id}/inspect",
                             headers=self.auth, json={"quant": "m-Q4_K_M"})
            # A double-click / resubmission must not start a second pull.
            self.client.post(f"/api/v2/wizard/{w.wizard_id}/inspect",
                             headers=self.auth, json={"quant": "m-Q4_K_M"})
        sub.assert_called_once()
        from modelctl_web.wizard import WizardStore
        self.assertEqual(WizardStore().load(w.wizard_id).download_job_id,
                         "job-pull-1")

    def test_api_pull_dedups_active_repo(self):
        job_id = self.store.create("pull", "pull org/model",
                                   payload={"repo_id": "org/model"},
                                   lane="download")
        self.store.update(job_id, status="running")
        with mock.patch("modelctl_web.mutate.submit_pull") as sub:
            resp = self.client.post("/api/pull/org/model", headers=self.auth)
        sub.assert_not_called()
        self.assertEqual(resp.json()["job"], job_id)
        self.assertTrue(resp.json()["deduplicated"])

    def test_register_blocked_while_test_job_live(self):
        """A register POST issued while the plan-test's llama-server is
        still running used to slip past the gate and race it for VRAM.

        The 409 is also asserted in test_console_phase2.py; what is only
        checked here is that apply_plan never fired -- the gate has to
        stop the write, not just report one."""
        job_id = self.store.create("benchmark", "plan test m1/p1",
                                   lane="benchmark")
        self.store.update(job_id, status="running")
        w = self._wizard(step="register", profile_name="m1",
                         selected_plan_id="p1", test_job_id=job_id)
        with mock.patch("modelctl_services.plan_service.apply_plan") as ap:
            resp = self.client.post(f"/api/v2/wizard/{w.wizard_id}/register",
                                    headers=self.auth, json={})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("still running", resp.json()["error"])
        ap.assert_not_called()
        from modelctl_web.wizard import WizardStore
        state = WizardStore().load(w.wizard_id)
        self.assertFalse(state.registration_complete)

    def test_measured_results_reach_the_wizard_state(self):
        """The plan-test job records generation_tps etc., but nothing
        folded them into wizard state -- the done step called every
        tested registration 'not measured'."""
        run = {"success": True, "generation_tps": 12.3, "prompt_tps": 456.7,
               "load_seconds": 41.5, "cache_state": "cold"}
        job_id = self.store.create("benchmark", "plan test m1/p1",
                                   lane="benchmark")
        self.store.update(job_id, status="done", outcome=json.dumps(run))
        w = self._wizard(step="test", profile_name="m1",
                         selected_plan_id="p1", test_job_id=job_id)
        resp = self.client.post(f"/api/v2/wizard/{w.wizard_id}/test/next",
                                headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["step"], "register")
        from modelctl_web.wizard import WizardStore
        state = WizardStore().load(w.wizard_id)
        self.assertEqual(state.measured.get("generation_tps"), 12.3)
        self.assertEqual(state.test_observations["p1"]["cache_state"], "cold")
        # And the numbers come back out on the read the done step uses,
        # rather than living only in the file.
        detail = self.client.get(f"/api/v2/wizard/{w.wizard_id}",
                                 headers=self.auth).json()
        self.assertEqual(detail["measured"]["generation_tps"], 12.3)
        self.assertEqual(detail["measured"]["prompt_tps"], 456.7)

    def test_analyze_tolerates_legacy_scalar_outcome(self):
        """Legacy job rows can hold a JSON scalar in outcome; the analyze
        step 500'd on .get() (seen live Jul 30)."""
        job_id = self.store.create("pull", "pull org/model", lane="download")
        self.store.update(job_id, status="done",
                          outcome=json.dumps("a bare string"))
        w = self._wizard(step="analyze", repo_id="org/model",
                         source_type="hf_repo", download_job_id=job_id)
        resp = self.client.get(f"/api/v2/wizard/{w.wizard_id}/analyze",
                               headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        # A scalar outcome carries no profile name, so the step degrades
        # to "nothing analysed yet" instead of raising.
        self.assertIsNone(resp.json()["profile"])


class TestRuntimeApi(WebTestBase):
    # Phase-3 cutover: POST /models/{name}/restart and POST
    # /runtime/unload-all had no /api/v2 successor, so restart-in-place
    # and unload-everything are gone from the console (mutate.submit_restart
    # / submit_unload_all survive but nothing calls them). The route tests
    # that covered them went with the routes; load and unload are covered
    # by test_console_phase3.py TestModelActionsRehomed.

    def _mock_runtime(self, state=None):
        if state is None:
            state = {
                "m1": {"model_id": "m1", "state": "ready", "registered": True,
                       "running": True, "pid": 1234, "port": 5000,
                       "started": time.time() - 60, "state_class": "ok"},
            }
        return mock.patch("modelctl_web.swap.LlamaSwapClient.runtime_state",
                          return_value=state)

    def test_model_detail_api_carries_runtime_identity(self):
        """The per-model page's header: the worker's pid and port, not
        just its state name."""
        with self._mock_runtime():
            resp = self.client.get("/api/v2/models/m1", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["name"], "m1")
        self.assertEqual(body["runtime"]["state"], "ready")
        self.assertEqual(body["runtime"]["pid"], 1234)
        self.assertEqual(body["runtime"]["port"], 5000)
        self.assertTrue(body["runtime"]["running"])

    def test_runtime_api_returns_json(self):
        with self._mock_runtime():
            resp = self.client.get("/api/runtime", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("m1", data)
        self.assertEqual(data["m1"]["state"], "ready")
        self.assertEqual(data["m1"]["pid"], 1234)

    def test_runtime_api_empty_when_swap_down(self):
        with self._mock_runtime(state={}):
            resp = self.client.get("/api/runtime", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    def test_runtime_model_endpoint(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.model_state",
                        return_value={"state": "stopped", "worker": None}):
            resp = self.client.get("/api/runtime/models/m1", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["state"], "stopped")

    def test_logtail_api_and_its_page_header(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.logs",
                        return_value="line1\nline2\n"), \
             self._mock_runtime():
            resp = self.client.get("/api/v2/models/m1/logtail",
                                   headers=self.auth)
            # Same page, one fetch over: pid and port come from
            # runtime_state(), not model_state()'s {state, worker} shape,
            # which is why they used to be missing from this view.
            header = self.client.get("/api/v2/models/m1",
                                     headers=self.auth).json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["source"], "llama-swap")
        self.assertIn("line1", resp.json()["tail"])
        self.assertEqual(header["runtime"]["pid"], 1234)
        self.assertEqual(header["runtime"]["port"], 5000)

    def test_stopped_model_is_reported_loadable(self):
        """The logs view offered a Load button for a registered but
        stopped model; the SPA derives that button from these two flags,
        so they have to survive the join."""
        stopped = {"m1": {"model_id": "m1", "state": "stopped",
                          "registered": True, "running": False, "pid": None,
                          "port": None, "started": None,
                          "state_class": "stopped"}}
        with mock.patch("modelctl_web.swap.LlamaSwapClient.logs",
                        return_value=""), \
             self._mock_runtime(state=stopped):
            resp = self.client.get("/api/v2/models/m1", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        rt = resp.json()["runtime"]
        self.assertEqual(rt["state"], "stopped")
        self.assertTrue(rt["registered"])
        self.assertFalse(rt["running"])
        with mock.patch("modelctl_web.mutate.submit_load",
                        return_value="job-load") as sub:
            load = self.client.post("/api/v2/models/m1/load", headers=self.auth)
        self.assertEqual(load.json(), {"job_id": "job-load"})
        sub.assert_called_once()

    def test_runtime_logs_api(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.logs",
                        return_value="log output"):
            resp = self.client.get("/api/runtime/logs/m1", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.text, "log output")

    def test_runtime_logs_api_error(self):
        from modelctl_web.swap import ModelctlSwapError
        with mock.patch("modelctl_web.swap.LlamaSwapClient.logs",
                        side_effect=ModelctlSwapError("LLAMA_SWAP_UNAVAILABLE", "down")):
            resp = self.client.get("/api/runtime/logs/m1", headers=self.auth)
        self.assertEqual(resp.status_code, 502)

    def test_models_api_joins_runtime_state(self):
        """The operate table's rows are the profile joined with what
        llama-swap reports, so a running model reads as running there."""
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_web.telemetry.fetch_metrics_text",
                        return_value=None), \
             self._mock_runtime(), self.resolvable_backend():
            resp = self.client.get("/api/v2/models", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        rows = {r["name"]: r for r in resp.json()}
        self.assertIn("m1", rows)
        self.assertEqual(rows["m1"]["state"], "ready")
        self.assertTrue(rows["m1"]["running"])
        self.assertEqual(rows["m1"]["port"], 5000)

    def test_model_id_simple_names_work(self):
        # The load route takes {name:path} so llama-swap ids can hold a
        # slash; a plain name must not pick up one on the way through.
        with mock.patch("modelctl_web.mutate.submit_load", return_value="job5") as sub:
            resp = self.client.post("/api/v2/models/my-model/load",
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"job_id": "job5"})
        sub.assert_called_once()
        self.assertEqual(sub.call_args[0][1], "my-model")


class TestSwapClient(unittest.TestCase):
    def test_runtime_state_merges_registered_and_running(self):
        from modelctl_web.swap import LlamaSwapClient
        client = LlamaSwapClient()
        with mock.patch.object(client, "registered_models",
                               return_value=[{"id": "m1"}, {"id": "m2"}]), \
             mock.patch.object(client, "running_models",
                               return_value=[{"model": "m1", "state": "ready",
                                              "pid": 100, "port": 5000}]):
            state = client.runtime_state()
        self.assertEqual(state["m1"]["state"], "ready")
        self.assertEqual(state["m1"]["pid"], 100)
        self.assertTrue(state["m1"]["running"])
        self.assertTrue(state["m1"]["registered"])
        self.assertEqual(state["m2"]["state"], "stopped")
        self.assertFalse(state["m2"]["running"])
        self.assertTrue(state["m2"]["registered"])

    def test_runtime_state_derives_port_from_proxy(self):
        # Newer llama-swap /running payloads have no "port" key; the
        # upstream address is in "proxy". The cache-metric scrape depends
        # on this port being populated.
        from modelctl_web.swap import LlamaSwapClient
        client = LlamaSwapClient()
        with mock.patch.object(client, "registered_models", return_value=[]), \
             mock.patch.object(client, "running_models",
                               return_value=[{"model": "m1", "state": "ready",
                                              "proxy": "http://localhost:5815"}]):
            state = client.runtime_state()
        self.assertEqual(state["m1"]["port"], 5815)

    def test_runtime_state_empty_when_swap_down(self):
        from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError
        client = LlamaSwapClient()
        with mock.patch.object(client, "registered_models",
                               side_effect=ModelctlSwapError("LLAMA_SWAP_UNAVAILABLE", "down")):
            state = client.runtime_state()
        self.assertEqual(state, {})

    def test_runtime_state_raises_when_requested(self):
        from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError
        client = LlamaSwapClient()
        with mock.patch.object(client, "registered_models",
                               side_effect=ModelctlSwapError("LLAMA_SWAP_UNAVAILABLE", "down")):
            with self.assertRaises(ModelctlSwapError):
                client.runtime_state(raise_on_unavailable=True)

    def test_model_state_unavailable_when_swap_down(self):
        from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError
        client = LlamaSwapClient()
        with mock.patch.object(client, "registered_models",
                               side_effect=ModelctlSwapError("LLAMA_SWAP_UNAVAILABLE", "down")):
            state = client.model_state("m1")
        self.assertEqual(state["state"], "unavailable")

    def test_state_class(self):
        from modelctl_web.swap import LlamaSwapClient
        self.assertEqual(LlamaSwapClient._state_class("ready"), "ok")
        self.assertEqual(LlamaSwapClient._state_class("loading"), "active")
        self.assertEqual(LlamaSwapClient._state_class("failed"), "warn")
        self.assertEqual(LlamaSwapClient._state_class("stopped"), "stopped")
        self.assertEqual(LlamaSwapClient._state_class("bogus"), "")

    def test_url_encoding(self):
        from modelctl_web.swap import LlamaSwapClient
        self.assertEqual(LlamaSwapClient._enc("a/b"), "a%2Fb")
        self.assertEqual(LlamaSwapClient._enc("simple"), "simple")


class TestJobLanes(unittest.TestCase):
    def _make_manager(self):
        from modelctl_web.jobs import JobStore, JobManager
        tmp = TemporaryDirectory()
        store = JobStore(Path(tmp.name) / "jobs.db")
        mgr = JobManager(store)
        return store, mgr, tmp

    def test_lanes_exist(self):
        store, mgr, tmp = self._make_manager()
        self.assertIn("mutation", mgr.lanes)
        self.assertIn("runtime", mgr.lanes)
        self.assertIn("download", mgr.lanes)
        self.assertIn("benchmark", mgr.lanes)

    def test_jobs_route_to_correct_lane(self):
        import threading
        store, mgr, tmp = self._make_manager()
        done = threading.Event()

        def runtime_fn(ctx):
            done.set()
            return {"ok": True}

        job_id = mgr.submit("test", "test job", runtime_fn, lane="runtime")
        done.wait(timeout=5)
        time.sleep(0.5)
        job = store.get(job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["lane"], "runtime")

    def test_download_does_not_block_runtime(self):
        import threading
        store, mgr, tmp = self._make_manager()
        download_blocker = threading.Event()
        runtime_done = threading.Event()

        def download_fn(ctx):
            download_blocker.wait(timeout=10)
            return {"ok": True}

        def runtime_fn(ctx):
            runtime_done.set()
            return {"ok": True}

        mgr.submit("download", "blocking download", download_fn, lane="download")
        mgr.submit("runtime", "quick runtime", runtime_fn, lane="runtime")
        self.assertTrue(runtime_done.wait(timeout=5),
                        "runtime should not be blocked by download")
        download_blocker.set()

    def test_cancel_sets_status(self):
        import threading
        store, mgr, tmp = self._make_manager()
        started = threading.Event()

        def slow_fn(ctx):
            started.set()
            time.sleep(30)
            return {"ok": True}

        job_id = mgr.submit("test", "slow job", slow_fn, lane="benchmark")
        started.wait(timeout=5)
        mgr.cancel(job_id)
        time.sleep(1)
        job = store.get(job_id)
        self.assertEqual(job["status"], "cancelled")

    def test_cancelled_job_does_not_resurrect_as_done(self):
        """A cancel during an uninterruptible call (hf_hub_download, a
        30-min warm load) used to flip back to 'done' when the call
        finally returned -- profile created, router restarted, after the
        UI said cancelled."""
        import threading
        store, mgr, tmp = self._make_manager()
        started = threading.Event()
        release = threading.Event()

        def stubborn_fn(ctx):
            started.set()
            release.wait(timeout=10)  # blocking work that ignores cancel
            return {"ok": True}

        job_id = mgr.submit("test", "stubborn", stubborn_fn, lane="benchmark")
        started.wait(timeout=5)
        mgr.cancel(job_id)
        release.set()  # the blocking work completes after the cancel
        time.sleep(1)
        self.assertEqual(store.get(job_id)["status"], "cancelled")

    def test_cancel_of_finished_job_is_a_noop(self):
        """Cancelling in the 2s polling window after a job finished used
        to rewrite its terminal status, falsifying history."""
        import threading
        store, mgr, tmp = self._make_manager()
        done = threading.Event()
        job_id = mgr.submit("test", "quick",
                            lambda ctx: done.set() or {"ok": True},
                            lane="runtime")
        done.wait(timeout=5)
        time.sleep(0.5)
        mgr.cancel(job_id)
        self.assertEqual(store.get(job_id)["status"], "done")

    def test_lane_survives_bookkeeping_failure(self):
        """If recording a failure itself raises (SQLITE_BUSY past the
        timeout), the lane worker used to die -- and with it, every
        later job on a single-worker lane."""
        import threading
        store, mgr, tmp = self._make_manager()
        boom = threading.Event()
        real_update = store.update

        def flaky_update(job_id, **fields):
            if fields.get("status") == "failed" and not boom.is_set():
                boom.set()
                raise RuntimeError("database is locked")
            return real_update(job_id, **fields)

        with mock.patch.object(store, "update", side_effect=flaky_update):
            mgr.submit("test", "bad", lambda ctx: 1 / 0, lane="mutation")
            for _ in range(50):
                if boom.is_set():
                    break
                time.sleep(0.1)
            self.assertTrue(boom.is_set())
        survived = threading.Event()
        mgr.submit("test", "good",
                   lambda ctx: survived.set() or {"ok": True},
                   lane="mutation")
        self.assertTrue(survived.wait(timeout=5),
                        "mutation lane worker died after a bookkeeping error")


class TestJobContext(unittest.TestCase):
    def test_log_appends(self):
        from modelctl_web.jobs import JobStore, JobContext
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        store = JobStore(Path(tmp.name) / "jobs.db")
        job_id = store.create("test", "test job")
        ctx = JobContext(store, job_id)
        ctx.log("line 1")
        ctx.log("line 2")
        job = store.get(job_id)
        self.assertIn("line 1", job["result"])
        self.assertIn("line 2", job["result"])
        tmp.cleanup()

    def test_progress_update(self):
        from modelctl_web.jobs import JobStore, JobContext
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        store = JobStore(Path(tmp.name) / "jobs.db")
        job_id = store.create("test", "test job")
        ctx = JobContext(store, job_id)
        ctx.set_progress(0.5, "halfway")
        job = store.get(job_id)
        self.assertAlmostEqual(job["progress"], 0.5)
        self.assertEqual(job["detail"], "halfway")
        tmp.cleanup()

    def test_progress_detail_only(self):
        """submit_pull's file_start callback passes detail with no value;
        this exact call TypeError'd and killed every web-submitted pull."""
        from modelctl_web.jobs import JobStore, JobContext
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        store = JobStore(Path(tmp.name) / "jobs.db")
        job_id = store.create("test", "test job")
        ctx = JobContext(store, job_id)
        ctx.set_progress(detail="model-00001.gguf")
        job = store.get(job_id)
        self.assertEqual(job["detail"], "model-00001.gguf")
        tmp.cleanup()

    def test_cancellation_check(self):
        from modelctl_web.jobs import JobStore, JobContext, CancelledError
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        store = JobStore(Path(tmp.name) / "jobs.db")
        job_id = store.create("test", "test job")
        ctx = JobContext(store, job_id)
        self.assertFalse(ctx.is_cancelled())
        ctx.set_cancelled()
        self.assertTrue(ctx.is_cancelled())
        with self.assertRaises(CancelledError):
            ctx.raise_if_cancelled()
        tmp.cleanup()


class TestSSE(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()

    def test_sse_stream_ends_on_done(self):
        from modelctl_web.jobs import JobStore, JobManager, sse_job_stream
        store = JobStore(Path(self.tmp.name) / "jobs2.db")
        mgr = JobManager(store)

        def quick_fn(ctx):
            return {"ok": True}

        job_id = mgr.submit("test", "quick", quick_fn, lane="mutation")
        time.sleep(1)

        events = list(sse_job_stream(store, job_id, timeout=5))
        self.assertTrue(len(events) >= 1)
        last_event = events[-1]
        self.assertIn("done", last_event)

    def test_sse_revisions(self):
        from modelctl_web.jobs import JobStore, sse_job_stream
        store = JobStore(Path(self.tmp.name) / "jobs.db")
        job_id = store.create("test", "test job")

        def updater():
            time.sleep(0.3)
            store.update(job_id, status="running", started=time.time())
            time.sleep(0.3)
            store.update(job_id, progress=0.5)
            time.sleep(0.3)
            store.update(job_id, status="done", progress=1.0, finished=time.time())

        import threading
        t = threading.Thread(target=updater)
        t.start()
        events = list(sse_job_stream(store, job_id, timeout=10))
        t.join(timeout=5)
        import json
        revisions = []
        for e in events:
            data = e.strip().removeprefix("data: ")
            if data:
                revisions.append(json.loads(data).get("revision"))
        self.assertTrue(len(revisions) >= 2)
        self.assertTrue(revisions[-1] > revisions[0])


class TestHardware(WebTestBase):
    def test_settings_roundtrip(self):
        import modelctl_hardware
        settings_path = Path(self.tmp.name) / "hardware.json"
        with mock.patch.object(modelctl_hardware, "SETTINGS_PATH", settings_path):
            data = {"version": 1, "devices": {"SYCL0": {"reserve_bytes": 2147483648}},
                    "ram": {"reserve_bytes": 0}, "storage": {}}
            modelctl_hardware.save_settings(data)
            loaded = modelctl_hardware.load_settings()
        self.assertEqual(loaded["version"], 1)
        self.assertEqual(loaded["devices"]["SYCL0"]["reserve_bytes"], 2147483648)

    def test_apply_settings(self):
        from modelctl_hardware import HardwareSnapshot, GpuSnapshot, apply_settings
        import time as _t
        gpus = (GpuSnapshot("SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "", 608),
                GpuSnapshot("SYCL1", "B580", 12 << 30, 10 << 30, 0, True, "", 456))
        snap = HardwareSnapshot(captured_at=_t.time(), fingerprint="abc",
                                gpus=gpus, ram_total_bytes=64 << 30,
                                ram_available_bytes=50 << 30, ram_reserve_bytes=0,
                                storage=(), backend_fingerprints={})
        settings = {"devices": {"SYCL0": {"reserve_bytes": 2 << 30, "role": "primary",
                                          "enabled": True}},
                    "ram": {"reserve_bytes": 4 << 30}}
        result = apply_settings(snap, settings)
        self.assertEqual(result.gpu_by_device("SYCL0").reserve_bytes, 2 << 30)
        self.assertEqual(result.gpu_by_device("SYCL0").role, "primary")
        self.assertEqual(result.ram_reserve_bytes, 4 << 30)
        self.assertEqual(result.gpu_by_device("SYCL1").reserve_bytes, 0)

    def test_snapshot_fingerprint(self):
        from modelctl_hardware import GpuSnapshot, _fingerprint
        gpus = [GpuSnapshot("SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "", 608)]
        fp1 = _fingerprint(gpus, {"llama-server": "v1"})
        fp2 = _fingerprint(gpus, {"llama-server": "v2"})
        fp3 = _fingerprint(gpus, {"llama-server": "v1"})
        self.assertNotEqual(fp1, fp2)
        self.assertEqual(fp1, fp3)

    def test_settings_api_carries_device_policy(self):
        """The hardware page folded into /v2/settings; its device rows are
        the `hardware` section of GET /api/v2/settings -- machine facts
        (device id, name, total) beside the reserve being edited."""
        import modelctl_hardware
        fake_snap = modelctl_hardware.HardwareSnapshot(
            captured_at=time.time(), fingerprint="abc123",
            gpus=(modelctl_hardware.GpuSnapshot(
                "SYCL0", "B70", 32 << 30, 30 << 30, 2 << 30, True, "primary", 608),),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=4 << 30, storage=(),
            backend_fingerprints={"llama-server": "v1"})
        with mock.patch("modelctl_hardware.capture_hardware_snapshot",
                        return_value=fake_snap), \
             mock.patch("modelctl_hardware.load_settings",
                        return_value={"version": 1, "devices": {}, "ram": {}, "storage": {}}), \
             self.resolvable_backend():
            resp = self.client.get("/api/v2/settings", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        hw = resp.json()["hardware"]
        self.assertEqual(hw["error"], "")
        devices = {d["device"]: d for d in hw["devices"]}
        self.assertIn("SYCL0", devices)
        self.assertEqual(devices["SYCL0"]["name"], "B70")
        self.assertEqual(devices["SYCL0"]["total_bytes"], 32 << 30)
        self.assertEqual(devices["SYCL0"]["reserve_bytes"], 2 << 30)
        self.assertEqual(hw["ram"]["reserve_bytes"], 4 << 30)

    def test_hardware_api(self):
        import modelctl_hardware
        fake_snap = modelctl_hardware.HardwareSnapshot(
            captured_at=time.time(), fingerprint="abc123",
            gpus=(modelctl_hardware.GpuSnapshot(
                "SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "", 608),),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})
        with mock.patch("modelctl_hardware.capture_hardware_snapshot",
                        return_value=fake_snap):
            resp = self.client.get("/api/hardware", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fingerprint"], "abc123")
        self.assertEqual(len(resp.json()["gpus"]), 1)

    def test_settings_api_shows_storage(self):
        import modelctl_hardware
        storage = (modelctl_hardware.StorageSnapshot(
            path="/home/x/models", kind="ext4", allow_mmap=True,
            mount_point="/home", filesystem="ext4",
            block_devices=("/dev/nvme0n1p2",), transport="nvme",
            rotational=False, raid_level=None,
            total_bytes=1 << 40, free_bytes=(1 << 40) // 2),)
        fake_snap = modelctl_hardware.HardwareSnapshot(
            captured_at=time.time(), fingerprint="abc123",
            gpus=(), ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=storage, backend_fingerprints={})
        with mock.patch("modelctl_hardware.capture_hardware_snapshot",
                        return_value=fake_snap), \
             mock.patch("modelctl_hardware.load_settings",
                        return_value={"version": 1, "devices": {}, "ram": {}, "storage": {}}), \
             self.resolvable_backend():
            resp = self.client.get("/api/v2/settings", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()["hardware"]["storage"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mount_point"], "/home")
        self.assertEqual(rows[0]["transport"], "nvme")
        # An uncalibrated mount has no measurement; the page rendered that
        # as "not calibrated" and must never show it as a zero rate.
        self.assertIsNone(rows[0]["measured_sequential_read_bps"])

        with mock.patch("modelctl_hardware.capture_hardware_snapshot",
                        return_value=fake_snap):
            resp = self.client.get("/api/hardware", headers=self.auth)
        self.assertEqual(len(resp.json()["storage"]), 1)
        self.assertEqual(resp.json()["storage"][0]["transport"], "nvme")

    def test_calibrate_endpoint_submits_job(self):
        with mock.patch("modelctl_web.mutate.submit_calibrate_storage",
                        return_value="job-cal") as sub:
            resp = self.client.post("/api/v2/settings/hardware/calibrate",
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"job_id": "job-cal"})
        sub.assert_called_once()

    def test_calibrate_job_runs_end_to_end(self):
        from modelctl_services import hardware_service
        with mock.patch.object(hardware_service, "pick_calibration_file",
                               return_value="/nonexistent/model.gguf"):
            resp = self.client.post("/api/v2/settings/hardware/calibrate",
                                    headers=self.auth)
            job = self.wait_job(resp.json()["job_id"])
        # No real model file in this test env -- calibration fails cleanly,
        # it doesn't crash the job.
        self.assertEqual(job["status"], "failed")


class TestPlans(WebTestBase):
    def _mock_hardware(self, gpus=None):
        from modelctl_hardware import HardwareSnapshot, GpuSnapshot
        if gpus is None:
            gpus = [GpuSnapshot("SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "primary", 608)]
        return HardwareSnapshot(
            captured_at=1000.0, fingerprint="test",
            gpus=tuple(gpus), ram_total_bytes=64 << 30,
            ram_available_bytes=50 << 30, ram_reserve_bytes=0,
            storage=(), backend_fingerprints={"llama-server": "v1"})

    def test_baseline_always_generated(self):
        import modelctl_plans
        profile = {"name": "m1", "backend": "llama-cpp", "model_path": "/nonexistent",
                   "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
                              "ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                              "fit": "on", "extra": "", "flash_attn": "auto", "ttl": 3600,
                              "mtp": "off"}}
        hw = self._mock_hardware()
        with mock.patch("modelctl.build_server_args", return_value=["llama-server"]):
            plans = modelctl_plans.compile_launch_plans(profile, hw)
        sources = {p.source for p in plans}
        self.assertIn("current-profile", sources)

    def test_stable_plan_ids(self):
        import modelctl_plans
        profile = {"name": "m1", "backend": "llama-cpp", "model_path": "",
                   "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
                              "ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                              "fit": "on", "extra": ""}}
        hw = self._mock_hardware()
        with mock.patch("modelctl.build_server_args", return_value=["llama-server"]):
            p1 = modelctl_plans.compile_launch_plans(profile, hw)
            p2 = modelctl_plans.compile_launch_plans(profile, hw)
        ids1 = [p.id for p in p1]
        ids2 = [p.id for p in p2]
        self.assertEqual(ids1, ids2)

    def test_two_gpu_generates_split(self):
        import modelctl_plans
        from modelctl_hardware import GpuSnapshot
        gpus = [GpuSnapshot("SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "primary", 608),
                GpuSnapshot("SYCL1", "B580", 12 << 30, 10 << 30, 0, True, "secondary", 456)]
        hw = self._mock_hardware(gpus)
        profile = {"name": "m1", "backend": "llama-cpp", "model_path": "",
                   "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
                              "ctx": 32768, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                              "fit": "on", "extra": ""}}
        with mock.patch("modelctl.build_server_args", return_value=["llama-server"]), \
             mock.patch("modelctl.get_gpu_inventory", return_value=[
                 {"device": "SYCL0", "name": "B70", "total_bytes": 32 << 30, "free_bytes": 30 << 30},
                 {"device": "SYCL1", "name": "B580", "total_bytes": 12 << 30, "free_bytes": 10 << 30}]), \
             mock.patch("modelctl.load_defaults",
                        return_value={"vram_limit_pct": 90, "primary_gpu": "SYCL0", "ctx": 32768,
                                      "cache_type_k": "q8_0", "cache_type_v": "q8_0"}):
            plans = modelctl_plans.compile_launch_plans(profile, hw)
        sources = {p.source for p in plans}
        self.assertIn("current-profile", sources)
        self.assertTrue(len(plans) >= 2)

    def test_rank_balanced(self):
        import modelctl_plans
        from modelctl_plans import LaunchPlan, ResourceClaim, RuntimePolicy, rank_plans
        plans = [
            LaunchPlan(id="a", profile_name="m", backend="llama-cpp", label="fast",
                       argv=(), env={}, claim=ResourceClaim({"S0": 20 << 30}, 0, "none", 32768),
                       estimated={}, source="test", warnings=(), decision_data={}),
            LaunchPlan(id="b", profile_name="m", backend="llama-cpp", label="big ctx",
                       argv=(), env={}, claim=ResourceClaim({"S0": 28 << 30}, 4 << 30, "none", 131072),
                       estimated={}, source="test", warnings=(), decision_data={}),
        ]
        policy = RuntimePolicy("balanced", None, True, True, 8192, None, 3)
        obs = {"a": {"generation_tps": 10.0}, "b": {"generation_tps": 5.0}}
        ranked = rank_plans(plans, policy, obs)
        self.assertEqual(ranked[0][0].id, "a")

    def test_plan_api_route(self):
        import modelctl_hardware
        fake_snap = modelctl_hardware.HardwareSnapshot(
            captured_at=time.time(), fingerprint="f",
            gpus=(modelctl_hardware.GpuSnapshot(
                "SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "", 608),),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})
        with mock.patch("modelctl_hardware.capture_hardware_snapshot",
                        return_value=fake_snap), \
             mock.patch.object(modelctl, "load_profile", return_value=PROFILE), \
             mock.patch.object(modelctl, "build_server_args", return_value=["llama-server"]):
            resp = self.client.get("/api/profiles/m1/plans", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        plans = resp.json()
        self.assertTrue(len(plans) >= 1)
        self.assertIn("id", plans[0])


class TestRuntimeDB(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "runtime.db"

    def _make_db(self):
        import modelctl_runtime
        return modelctl_runtime.RuntimeDB(self.db_path)

    def test_acquire_and_release(self):
        db = self._make_db()
        claim = {"vram_bytes": {"SYCL0": 10 << 30}, "ram_bytes": 0}
        res = db.acquire_reservation("m1", "plan1", claim, os.getpid())
        self.assertIsNotNone(res)
        self.assertEqual(res["state"], "pending")
        self.assertEqual(res["profile_name"], "m1")
        db.release_reservation(res["id"])
        self.assertEqual(len(db.get_reservations()), 0)

    def test_conflict_detection(self):
        import modelctl_runtime
        db = self._make_db()
        claim = {"vram_bytes": {"SYCL0": 10 << 30}, "ram_bytes": 0}
        # Mock _pid_alive to treat all PIDs as alive
        with mock.patch("modelctl_runtime._pid_alive", return_value=True):
            res1 = db.acquire_reservation("m1", "plan1", claim, 99999)
            self.assertIsNotNone(res1)
            # Second acquisition on same device from different PID conflicts
            res2 = db.acquire_reservation("m2", "plan2", claim, 99998)
            self.assertIsNone(res2)
        db.release_reservation(res1["id"])

    def test_same_pid_allowed(self):
        db = self._make_db()
        claim = {"vram_bytes": {"SYCL0": 10 << 30}, "ram_bytes": 0}
        pid = os.getpid()
        res1 = db.acquire_reservation("m1", "plan1", claim, pid)
        res2 = db.acquire_reservation("m2", "plan2", claim, pid)
        self.assertIsNotNone(res1)
        self.assertIsNotNone(res2)
        db.release_reservation(res1["id"])
        db.release_reservation(res2["id"])

    def test_state_transitions(self):
        db = self._make_db()
        claim = {"vram_bytes": {"SYCL0": 10 << 30}, "ram_bytes": 0}
        res = db.acquire_reservation("m1", "plan1", claim, os.getpid())
        db.update_reservation(res["id"], state="starting")
        r = db.get_reservations(profile_name="m1")[0]
        self.assertEqual(r["state"], "starting")
        db.update_reservation(res["id"], state="active")
        r = db.get_reservations(profile_name="m1")[0]
        self.assertEqual(r["state"], "active")

    def test_events(self):
        db = self._make_db()
        db.record_event("test_event", "m1", {"key": "value"})
        db.record_event("other_event", "m2")
        events = db.get_events(profile_name="m1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "test_event")
        all_events = db.get_events()
        self.assertEqual(len(all_events), 2)

    def test_pending_claims(self):
        db = self._make_db()
        claim = {"vram_bytes": {"SYCL0": 10 << 30}, "ram_bytes": 0}
        pid = os.getpid()
        db.acquire_reservation("m1", "plan1", claim, pid)
        # Should exclude own PID
        claims = db.pending_claims(exclude_pid=pid)
        self.assertEqual(len(claims), 0)
        # Should include when not excluded
        claims = db.pending_claims()
        self.assertEqual(len(claims), 1)


class TestWorker(unittest.TestCase):
    def test_build_command_injects_port(self):
        from modelctl_worker import _build_command
        profile = {"config": {}}
        from modelctl_plans import LaunchPlan, ResourceClaim
        plan = LaunchPlan(
            id="test", profile_name="m", backend="llama-cpp", label="t",
            argv=("--model", "/m.gguf", "--port", "8080"),
            env={}, claim=ResourceClaim({}, 0, "none", None),
            estimated={}, source="test", warnings=(), decision_data={})
        cmd = _build_command(profile, plan, 5000, binary="/x/llama-server")
        self.assertEqual(cmd[0], "/x/llama-server")
        self.assertIn("--port", cmd)
        port_idx = cmd.index("--port")
        self.assertEqual(cmd[port_idx + 1], "5000")

    def test_build_command_adds_port_if_missing(self):
        from modelctl_worker import _build_command
        profile = {"config": {}}
        from modelctl_plans import LaunchPlan, ResourceClaim
        plan = LaunchPlan(
            id="test", profile_name="m", backend="llama-cpp", label="t",
            argv=("--model", "/m.gguf"),
            env={}, claim=ResourceClaim({}, 0, "none", None),
            estimated={}, source="test", warnings=(), decision_data={})
        cmd = _build_command(profile, plan, 5000, binary="/x/llama-server")
        self.assertEqual(cmd[0], "/x/llama-server")
        self.assertIn("--port", cmd)
        port_idx = cmd.index("--port")
        self.assertEqual(cmd[port_idx + 1], "5000")


if __name__ == "__main__":
    unittest.main()


class TestRuntimePolicy(WebTestBase):
    # Phase-3 cutover: the managed-runtime editor is gone. POST
    # /profiles/{name}/runtime-policy (write the runtime section) and
    # POST /profiles/{name}/plans/{id}/{select,disable} (pin / disable a
    # plan) had no /api/v2 successor -- mutate.submit_runtime_policy and
    # submit_plan_select survive but nothing calls them, so managed mode
    # can only be configured by editing the profile JSON or via the CLI.
    # The three write tests that covered them went with the routes; the
    # read below is the only part of the surface that survived.

    def test_policy_defaults_to_fixed(self):
        resp = self.client.get("/api/profiles/m1/runtime-policy", headers=self.auth)
        self.assertEqual(resp.json(), {"mode": "fixed"})


class TestPlanTesting(WebTestBase):
    # Phase-3 cutover: autotune has no /api/v2 successor. POST
    # /profiles/{name}/tune was mutate.submit_autotune's only caller, so
    # sweeping every candidate plan in one job is CLI-only now; its route
    # test went with the route.

    def test_plan_test_job_and_history(self):
        """A selected plan is measured through the wizard's test step,
        which is where submit_plan_test is reachable from now."""
        from modelctl_web.wizard import WizardState, WizardStore
        fake_run = {"profile_name": "m1", "plan_id": "p1", "success": True,
                    "generation_tps": 42.0, "prompt_tps": 100.0,
                    "load_seconds": 3.2, "log_path": "/x/log"}
        state = WizardState(step="test", profile_name="m1",
                            selected_plan_id="p1")
        WizardStore().save(state)
        with mock.patch("modelctl_tune.test_launch_plan", return_value=fake_run) as tlp:
            resp = self.client.post(
                f"/api/v2/wizard/{state.wizard_id}/test/run", headers=self.auth)
            self.assertEqual(resp.status_code, 200)
            job = self.wait_job(resp.json()["test_job_id"])
        self.assertEqual(job["status"], "done")
        tlp.assert_called_once()
        self.assertEqual(tlp.call_args[0][:2], ("m1", "p1"))

    def test_history_endpoint(self):
        # WebTestBase redirects default RuntimeDB() paths into self.tmp,
        # so this writes where the endpoint reads -- and nowhere else.
        rdb = modelctl_runtime.RuntimeDB()
        self.assertEqual(Path(rdb.db_path), self.runtime_db_path,
                         "test wrote plan runs to the real runtime.db")
        rdb.record_plan_run({"profile_name": "m1", "plan_id": "p1",
                             "success": True, "generation_tps": 42.0})
        resp = self.client.get("/api/profiles/m1/history", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        # Exactly the one row written above: an isolated store makes the
        # count assertable, and a leak to shared state fails loudly here.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["generation_tps"], 42.0)


class TestManagedMatrix(unittest.TestCase):
    def test_merge_preserves_hand_authored(self):
        import modelctl_matrix
        existing = {"vars": {"q27": "big-qwen", "mc_old": "stale"},
                    "evict_costs": {"q27": 1, "mc_old": 5},
                    "sets": {"qwen_stack": "q27 & f7", "mc_old_set": "mc_old"}}
        generated = {"vars": {"mc_a": "a"}, "evict_costs": {"mc_a": 2},
                     "sets": {"mc_a": "a"}}
        merged = modelctl_matrix.merge_matrix(existing, generated)
        self.assertEqual(merged["vars"], {"q27": "big-qwen", "mc_a": "a"})
        self.assertEqual(merged["evict_costs"], {"q27": 1, "mc_a": 2})
        self.assertIn("qwen_stack", merged["sets"])
        self.assertNotIn("mc_old", merged["vars"])
        self.assertNotIn("mc_old_set", merged["sets"])

    def test_compatibility_math(self):
        import modelctl_matrix
        budgets = {"SYCL0": 30 << 30, "RAM": 20 << 30}
        claims = {
            "big": {"SYCL0": 25 << 30},
            "small": {"SYCL0": 4 << 30},
            "toobig": {"SYCL0": 28 << 30},
        }
        self.assertTrue(modelctl_matrix._fits(claims, ["big", "small"], budgets))
        self.assertFalse(modelctl_matrix._fits(claims, ["big", "toobig"], budgets))
        self.assertTrue(modelctl_matrix._fits(claims, ["big"], budgets))


class TestByteReservations(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import modelctl_runtime
        self.rdb = modelctl_runtime.RuntimeDB(Path(self.tmp.name) / "rt.db")
        import os
        self.pid = os.getpid()

    def test_byte_based_admission(self):
        budgets = {"SYCL0": 10 << 30, "RAM": 8 << 30}
        r1 = self.rdb.acquire_reservation("a", "p1", {"vram_bytes": {"SYCL0": 6 << 30},
                                                      "ram_bytes": 0},
                                          self.pid, budgets=budgets)
        self.assertIsNotNone(r1)
        # same device, would exceed: rejected (other pid but alive=False? use
        # same pid excluded from conflict -- use a second live pid)
        r2 = self.rdb.acquire_reservation("b", "p2", {"vram_bytes": {"SYCL0": 5 << 30},
                                                      "ram_bytes": 0},
                                          self.pid, budgets=budgets)
        self.assertIsNotNone(r2)  # same owner is excluded from pending sum
        r3 = self.rdb.acquire_reservation("c", "p3", {"vram_bytes": {"SYCL0": 0},
                                                      "ram_bytes": 9 << 30},
                                          self.pid, budgets=budgets)
        self.assertIsNone(r3)  # RAM exceeds budget

    def test_pending_sum_blocks_second_worker(self):
        import modelctl_runtime
        budgets = {"SYCL0": 10 << 30, "RAM": 0}
        r1 = self.rdb.acquire_reservation("a", "p1", {"vram_bytes": {"SYCL0": 6 << 30},
                                                      "ram_bytes": 0},
                                          self.pid, budgets=budgets)
        self.assertIsNotNone(r1)
        with mock.patch.object(modelctl_runtime, "_pid_alive", return_value=True):
            r2 = self.rdb.acquire_reservation("b", "p2", {"vram_bytes": {"SYCL0": 5 << 30},
                                                          "ram_bytes": 0},
                                              self.pid + 1, budgets=budgets)
        self.assertIsNone(r2)  # 6+5 > 10
        with mock.patch.object(modelctl_runtime, "_pid_alive", return_value=True):
            r3 = self.rdb.acquire_reservation("b", "p3", {"vram_bytes": {"SYCL0": 4 << 30},
                                                          "ram_bytes": 0},
                                              self.pid + 1, budgets=budgets)
        self.assertIsNotNone(r3)  # 6+4 == 10 fits


class TestPolicyEnforcement(unittest.TestCase):
    def _plans(self):
        import modelctl_plans as mp
        def mk(pid, ctx=65536, ram=0, storage="none"):
            return mp.LaunchPlan(
                id=pid, profile_name="m", backend="llama-cpp", label=pid,
                argv=(), env={}, claim=mp.ResourceClaim({"SYCL0": 1 << 30}, ram, storage, ctx),
                estimated={}, source="t", warnings=(), decision_data={})
        return [mk("a"), mk("b", ctx=4096), mk("c", ram=6 << 30, storage="mmap")]

    def test_max_cpu_bytes(self):
        import modelctl_plans as mp
        pol = mp.RuntimePolicy("balanced", None, True, True, 1, 4 << 30, 3)
        ranked = mp.rank_plans(self._plans(), pol)
        self.assertNotIn("c", [p.id for p, _ in ranked])

    def test_allow_untested_false_keeps_validated_only(self):
        import modelctl_plans as mp
        pol = mp.RuntimePolicy("balanced", None, True, False, 1, None, 3)
        obs = {"a": {"generation_tps": 10, "stale": False}}
        ranked = mp.rank_plans(self._plans(), pol, obs)
        self.assertEqual([p.id for p, _ in ranked], ["a"])

    def test_allow_untested_false_bootstraps_baseline_only(self):
        import modelctl_plans as mp
        pol = mp.RuntimePolicy("balanced", None, True, False, 1, None, 3)
        # no baseline among the candidates -> nothing may launch untested
        ranked = mp.rank_plans(self._plans(), pol, {})
        self.assertEqual(ranked, [])
        # with the exact current-profile baseline present, it may bootstrap
        plans = self._plans()
        baseline = mp.LaunchPlan(
            id="base", profile_name="m", backend="llama-cpp", label="base",
            argv=(), env={}, claim=mp.ResourceClaim({"SYCL0": 1 << 30}, 0, "none", 65536),
            estimated={}, source="current-profile", warnings=(), decision_data={})
        ranked = mp.rank_plans(plans + [baseline], pol, {})
        self.assertEqual([p.id for p, _ in ranked], ["base"])

    def test_pinned_promotes_not_deletes(self):
        import modelctl_plans as mp
        pol = mp.RuntimePolicy("balanced", "c", True, True, 1, None, 3)
        ranked = mp.rank_plans(self._plans(), pol)
        self.assertEqual(ranked[0][0].id, "c")
        self.assertEqual(len(ranked), 3)  # pinned first, alternatives kept

    def test_failure_suppression(self):
        import modelctl_plans as mp
        plans = self._plans()
        obs = {"a": {"generation_tps": 10, "stale": False}}
        fails = {"a": ["unsupported_architecture"]}
        ranked = mp.rank_plans(plans, mp.DEFAULT_POLICY, obs, fails)
        self.assertNotIn("a", [p.id for p, _ in ranked])  # removed, not scored down


class TestClaimAccuracy(unittest.TestCase):
    def test_moe_cpu_expert_offload_claim(self):
        import modelctl_plans as mp
        GiB = 1 << 30
        layout = {"is_moe": True, "weight_bytes": 60 * GiB,
                  "non_expert_bytes": 6 * GiB,
                  "expert_bytes_per_layer": {i: GiB for i in range(1, 51)},
                  "other_bytes": 2 * GiB, "layer_bytes": 4 * GiB,
                  "meta": {"general.architecture": "m", "m.block_count": 50,
                           "m.attention.head_count_kv": 4,
                           "m.embedding_length": 512,
                           "m.attention.head_count": 8},
                  "block_count": 50}
        profile = {"model_path": "/x/m.gguf",
                   "config": {"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                              "extra": "-ot blk.(10|[1-9]).ffn_.*_exps=SYCL0,ffn_.*_exps=CPU",
                              "device": "SYCL0", "tensor_split": ""}}
        with mock.patch("modelctl_plans.os.path.exists", return_value=True), \
             mock.patch("modelctl_plans.modelctl_vram.gguf_model_layout",
                        return_value=layout):
            claim = mp._make_claim(profile, profile["config"], None)
        self.assertEqual(claim.ram_bytes, 40 * GiB)  # layers 11-50 to CPU
        self.assertEqual(claim.storage_mode, "mmap")
        # SYCL0: non-expert + 10 expert layers + KV + overhead
        self.assertGreater(claim.vram_bytes["SYCL0"], 16 * GiB)
        self.assertLess(claim.vram_bytes["SYCL0"], 20 * GiB)


class TestJobStoreRaces(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = JobStore(Path(self.tmp.name) / "jobs.db")

    def test_revision_increments_atomically(self):
        jid = self.store.create("t", "t")
        self.store.update(jid, detail="a")
        self.store.update(jid, detail="b")
        self.assertEqual(self.store.get(jid)["revision"], 2)

    def test_concurrent_log_appends_not_lost(self):
        jid = self.store.create("t", "t")
        self.store.append_result_line(jid, "line-one")
        self.store.append_result_line(jid, "line-two")
        result = self.store.get(jid)["result"]
        self.assertIn("line-one", result)
        self.assertIn("line-two", result)
        self.assertEqual(self.store.get(jid)["revision"], 2)


class TestMatrixMergePreserves(unittest.TestCase):
    def test_unknown_matrix_fields_survive(self):
        import modelctl_matrix
        existing = {"vars": {"q27": "big-qwen"},
                    "evict_costs": {"q27": 1},
                    "sets": {"s": "q27"},
                    "custom_future_field": {"keep": "me"}}
        generated = {"vars": {"mc_a": "a"}, "evict_costs": {"mc_a": 1},
                     "sets": {"mc_a": "a"}}
        merged = modelctl_matrix.merge_matrix(existing, generated)
        self.assertEqual(merged["custom_future_field"], {"keep": "me"})
        self.assertEqual(merged["vars"]["q27"], "big-qwen")


class TestComputeNgl(unittest.TestCase):
    def test_uses_actual_block_count(self):
        import modelctl_plans as mp
        GiB = 1 << 30
        layout = {"block_count": 60, "weight_bytes": 62 * GiB,
                  "other_bytes": 2 * GiB,
                  "meta": {"general.architecture": "m",
                           "m.block_count": 60,
                           "m.attention.head_count_kv": 4,
                           "m.embedding_length": 512,
                           "m.attention.head_count": 8}}
        profile = {"model_path": "/x/m.gguf", "config": {"ctx": 8192}}
        with mock.patch("modelctl_plans.os.path.exists", return_value=True), \
             mock.patch("modelctl_plans.modelctl_vram.gguf_model_layout",
                        return_value=layout):
            ngl = mp._compute_ngl(62 * GiB, 25 * GiB, profile, None)
        # per_layer = 1 GiB; fixed = 2 + KV + 3.1 overhead; ~25 layers*... 25 GiB budget
        self.assertGreater(ngl, 10)
        self.assertLessEqual(ngl, 59)

    def test_missing_layout_returns_zero(self):
        import modelctl_plans as mp
        with mock.patch("modelctl_plans.os.path.exists", return_value=False):
            self.assertEqual(mp._compute_ngl(10 << 30, 20 << 30,
                                             {"model_path": "/x", "config": {}}, None), 0)


class TestSseStream(unittest.TestCase):
    def test_terminal_event_then_close(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = JobStore(Path(tmp.name) / "jobs.db")
        jid = store.create("t", "t")
        store.update(jid, status="done", finished=1.0)
        from modelctl_web.jobs import sse_job_stream
        events = list(sse_job_stream(store, jid, timeout=5))
        self.assertEqual(len(events), 1)
        self.assertIn("done", events[0])


class TestSwapClientPlainText(unittest.TestCase):
    def test_plain_text_ok_is_not_an_error(self):
        from modelctl_web.swap import LlamaSwapClient
        import io

        class FakeResp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch("urllib.request.urlopen", return_value=FakeResp(b"OK")):
            client = LlamaSwapClient()
            result = client.unload("some-model")
        self.assertEqual(result, {"text": "OK"})


class _FakeTextResp:
    """Minimal urlopen stand-in returning fixed text."""
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestMoeCacheMetrics(WebTestBase):
    METRICS = "\n".join([
        "# HELP llamacpp:moe_cache_hits_total MoE expert cache hits",
        "# TYPE llamacpp:moe_cache_hits_total counter",
        'llamacpp:moe_cache_hits_total{device="0"} 42',
        'llamacpp:moe_cache_misses_total{device="0"} 7',
        'llamacpp:moe_cache_hit_ratio{device="1"} 0.75',
        'llamacpp:moe_cache_slots_used{device="0",layer="3"} 100',
        "llamacpp:moe_cache_slots 128",
        "llamacpp:prompt_tokens_total 999",
    ])

    def _runtime_state(self, port=5000):
        return mock.patch("modelctl_web.swap.LlamaSwapClient.runtime_state",
                          return_value={
                              "m1": {"model_id": "m1", "state": "ready",
                                     "registered": True, "running": True,
                                     "pid": 1234, "port": port,
                                     "started": time.time() - 60,
                                     "state_class": "ok"}})

    def _enable_moe_cache(self):
        p = dict(PROFILE)
        p["moe_cache"] = {"mode": "auto",
                          "gpu": {"budgets_bytes": {"SYCL0": 8 << 30}}}
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))

    # Phase-3 cutover: POST /models/{name}/cache/reset is gone and has no
    # /api/v2 successor, so clearing a live server's MoE cache from the
    # console is no longer possible (it was the only caller; the two
    # tests that covered the port-present and port-absent paths went with
    # the route). The scrape itself moved to modelctl.scrape_moe_cache_metrics,
    # which the web app no longer wraps.

    def test_scrape_parses_metrics_label_agnostic(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeTextResp(self.METRICS)):
            stats = modelctl.scrape_moe_cache_metrics(5000)
        self.assertEqual(stats["0"]["hits_total"], "42")
        self.assertEqual(stats["0"]["misses_total"], "7")
        # extra labels beyond device= still parse
        self.assertEqual(stats["0"]["slots_used"], "100")
        self.assertEqual(stats["1"]["hit_ratio"], "0.75")
        # no label block at all -> "" device key
        self.assertEqual(stats[""]["slots"], "128")
        # non-moe_cache metrics are ignored
        self.assertNotIn("prompt_tokens_total",
                         [k for dev in stats.values() for k in dev])

    def test_scrape_returns_none_on_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            self.assertIsNone(modelctl.scrape_moe_cache_metrics(5000))

    def test_scrape_returns_none_without_moe_metrics(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeTextResp("llamacpp:prompt_tokens_total 9\n")):
            self.assertIsNone(modelctl.scrape_moe_cache_metrics(5000))

    def test_models_api_reports_cache_stats(self):
        """The runtime page's per-model cache line is now the operate
        row's `cache` summary, scraped from the running worker's own
        /metrics port."""
        self._enable_moe_cache()
        with self._runtime_state(), \
             mock.patch("urllib.request.urlopen",
                        return_value=_FakeTextResp(self.METRICS)):
            resp = self.client.get("/api/v2/models", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        row = next(r for r in resp.json() if r["name"] == "m1")
        self.assertEqual(row["moe_cache_mode"], "auto")
        # Aggregate ratio from the raw counters, not the server's own
        # hit_ratio gauge: 42 hits / 49 lookups.
        self.assertAlmostEqual(row["cache"]["hit_ratio"], 42 / 49)
        self.assertEqual(row["cache"]["hits"], 42)
        self.assertEqual(row["cache"]["misses"], 7)
        # Both labelled devices survive the parse.
        self.assertEqual(row["cache"]["devices"], ["0", "1"])


class TestCacheVariants(WebTestBase):
    """compile_launch_plans cache variants: structured budgets (never a
    duplicate --moe-cache-bytes), configured limits, feasibility."""

    def _mock_hardware(self):
        from modelctl_hardware import HardwareSnapshot, GpuSnapshot
        return HardwareSnapshot(
            captured_at=1000.0, fingerprint="test",
            gpus=(GpuSnapshot("SYCL0", "B70", 32 << 30, 30 << 30, 0, True,
                              "primary", 608),),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})

    def _profile(self, model_path="/nonexistent"):
        return {"name": "m1", "backend": "llama-cpp", "model_path": model_path,
                "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
                           "ctx": 32768, "cache_type_k": "q8_0",
                           "cache_type_v": "q8_0", "fit": "on", "extra": "",
                           "flash_attn": "auto", "ttl": 3600, "mtp": "off"},
                "moe_cache": {"mode": "auto",
                              "gpu": {"budgets_bytes": {"SYCL0": 4 << 30},
                                      "policy": "slru", "admission_misses": 2}}}

    def _compile(self, profile, extra_patches=()):
        import modelctl_plans
        patches = [
            mock.patch("modelctl_capabilities.probe_backend",
                       return_value={"features": {"moe_weight_transfer_cache": True}}),
            mock.patch("modelctl.load_defaults",
                       return_value={"vram_limit_pct": 90, "primary_gpu": "SYCL0"}),
            mock.patch("modelctl.get_gpu_inventory", return_value=[]),
            mock.patch("modelctl_hardware.load_settings", return_value={}),
            *extra_patches,
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return modelctl_plans.compile_launch_plans(profile, self._mock_hardware(),
                                                   include_experimental=True)

    def test_variants_emit_single_structured_budget(self):
        plans = self._compile(self._profile())
        variants = [p for p in plans if p.source.startswith("cache-")]
        self.assertEqual({p.source for p in variants},
                         {"cache-small", "cache-balanced", "cache-large"})
        # distinct plan IDs -- the cache budget is part of plan identity
        self.assertEqual(len({p.id for p in variants}), 3)
        usable = (32 << 30) * 0.9
        for p in variants:
            argv = list(p.argv)
            # exactly one --moe-cache-bytes, from the structured path
            self.assertEqual(argv.count("--moe-cache-bytes"), 1, p.source)
            self.assertIn("--moe-cache-policy", argv)
        small = next(p for p in variants if p.source == "cache-small")
        i = list(small.argv).index("--moe-cache-bytes")
        self.assertEqual(small.argv[i + 1], str(int(usable * 0.20)))

    def test_infeasible_variants_skipped(self):
        # GPU-resident floor (KV 20 GiB + overhead 2 GiB) on a 28.8
        # GiB-usable card: only the small (20%) budget leaves room.
        model = Path(self.tmp.name) / "big.gguf"
        model.write_bytes(b"x")
        plans = self._compile(self._profile(str(model)), extra_patches=[
            mock.patch("modelctl_vram.weights_bytes_on_disk",
                       return_value=20 << 30),
            mock.patch("modelctl_vram.estimate_from_parts",
                       return_value={"total": 24 << 30,
                                     "kv_bytes": 20 << 30,
                                     "overhead": 2 << 30}),
        ])
        sources = {p.source for p in plans if p.source.startswith("cache-")}
        self.assertEqual(sources, {"cache-small"})

    def test_oversized_spill_model_still_gets_variants(self):
        # Regression: a model whose FULL static footprint exceeds one card
        # must still get cache variants -- weights spill to CPU, so the bar
        # is KV + overhead + budget, not total + budget. This is the MoE
        # spill case the cache exists for.
        model = Path(self.tmp.name) / "huge.gguf"
        model.write_bytes(b"x")
        plans = self._compile(self._profile(str(model)), extra_patches=[
            mock.patch("modelctl_vram.weights_bytes_on_disk",
                       return_value=54 << 30),
            mock.patch("modelctl_vram.estimate_from_parts",
                       return_value={"total": 76 << 30,
                                     "kv_bytes": 6 << 30,
                                     "overhead": 6 << 30}),
        ])
        variants = [p for p in plans if p.source.startswith("cache-")]
        self.assertTrue(variants, "spill model lost all cache variants")
        self.assertTrue(all(p.label.endswith(p.source) for p in variants))
        for p in variants:
            self.assertEqual(list(p.argv).count("--moe-cache-bytes"), 1)

    def test_variants_honor_vram_limit_pct(self):
        import modelctl_plans
        with             mock.patch("modelctl_capabilities.probe_backend",
                        return_value={"features": {"moe_weight_transfer_cache": True}}), \
             mock.patch("modelctl.load_defaults",
                        return_value={"vram_limit_pct": 50, "primary_gpu": "SYCL0"}), \
             mock.patch("modelctl.get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_hardware.load_settings", return_value={}):
            plans = modelctl_plans.compile_launch_plans(self._profile(),
                                                        self._mock_hardware(),
                                                        include_experimental=True)
        small = next(p for p in plans if p.source == "cache-small")
        i = list(small.argv).index("--moe-cache-bytes")
        # 50% limit, not the old hardcoded 0.9
        self.assertEqual(small.argv[i + 1], str(int((32 << 30) * 0.5 * 0.20)))

    def test_tier_candidate_passes_cache_request(self):
        import modelctl_plans
        import modelctl_tiers
        profile = self._profile()
        with mock.patch.object(modelctl_tiers, "plan_tiers",
                               return_value=None) as pt, \
             mock.patch("modelctl.get_gpu_inventory", return_value=[
                 {"device": "SYCL0", "name": "B70", "total_bytes": 32 << 30,
                  "free_bytes": 30 << 30}]), \
             mock.patch("modelctl.load_defaults",
                        return_value={"vram_limit_pct": 90, "primary_gpu": "SYCL0",
                                      "ctx": 32768, "cache_type_k": "q8_0",
                                      "cache_type_v": "q8_0"}):
            modelctl_plans.compile_launch_plans(profile, self._mock_hardware())
        pt.assert_called_once()
        self.assertEqual(pt.call_args.kwargs.get("cache_request"),
                         profile["moe_cache"])


class TestTierApplyCacheWriteback(WebTestBase):
    """M2b: tier-apply writes the planner's effective cache budgets back to
    the profile so the server allocates exactly what the planner reserved."""

    PLAN = {"tier": 3, "config": {"device": "", "split_mode": "layer",
                                  "tensor_split": "8,3", "extra": "--fit off"},
            "layout": [("SYCL0", 10.0, "experts")],
            "warnings": [], "analysis": {}}

    def _profile_with_cache(self, budgets):
        p = dict(PROFILE)
        p["moe_cache"] = {"mode": "auto", "gpu": {"budgets_bytes": budgets}}
        (self.profiles_dir / "m1.json").write_text(json.dumps(p))

    def _apply(self, plan):
        import modelctl_tiers
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[
                {"device": "SYCL0", "name": "big", "total_bytes": 32 << 30,
                 "free_bytes": 30 << 30}]), \
             mock.patch.object(modelctl_tiers, "plan_tiers", return_value=plan), \
             mock.patch.object(modelctl, "save_profile") as save, \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"):
            # These plans change split/tier structurally, which the replan
            # gate refuses by default -- acceptance is explicit.
            resp = self.client.post(
                "/api/tiers/m1/apply?accept_tier_change=true",
                headers=self.auth)
            job = self.wait_job(resp.json()["job"])
        self.assertEqual(job["status"], "done")
        return save.call_args[0][0]

    def test_oversize_request_is_cleared(self):
        self._profile_with_cache({"SYCL0": 20 << 30})
        plan = dict(self.PLAN, cache_budgets=None,
                    warnings=["cache budget 20.0 GiB exceeds usable VRAM"])
        saved = self._apply(plan)
        self.assertEqual(saved["moe_cache"]["gpu"]["budgets_bytes"], {})

    def test_request_normalized_to_planner_budgets(self):
        self._profile_with_cache({"SYCL0": 4 << 30, "SYCL1": 6 << 30})
        effective = {"SYCL0": 6 << 30, "SYCL1": 6 << 30}
        plan = dict(self.PLAN, cache_budgets=effective)
        saved = self._apply(plan)
        self.assertEqual(saved["moe_cache"]["gpu"]["budgets_bytes"], effective)


class TestMoeCacheRouteValidation(WebTestBase):
    """submit_moe_cache validates against probed capabilities (fail closed)
    before saving the profile.

    Re-homed onto POST /api/v2/models/{name}/config: flipping the cache
    mode is a structural change, so it also has to carry the explicit
    accept before the job is even submitted."""

    def _post_mode(self, mode, accept=True):
        return self.client.post(
            "/api/v2/models/m1/config", headers=self.auth,
            json={"moe_mode": mode, "accept_structural": accept})

    def test_mode_change_needs_the_structural_confirm(self):
        resp = self._post_mode("manual", accept=False)
        self.assertEqual(resp.status_code, 409)
        self.assertTrue(resp.json()["gate"]["requires_accept"])
        self.assertTrue(any("moe_cache mode" in c
                            for c in resp.json()["gate"]["changes"]))

    def test_manual_cache_rejected_on_incapable_backend(self):
        with mock.patch("modelctl_capabilities.probe_backend",
                        return_value={"schema": 2, "features": {}, "cli": {}}), \
             mock.patch.object(modelctl, "save_profile") as save, \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"):
            resp = self._post_mode("manual")
            self.assertEqual(resp.status_code, 200)
            job = self.wait_job(resp.json()["jobs"]["moe_cache"])
        self.assertEqual(job["status"], "failed")
        save.assert_not_called()

    def test_auto_cache_saves_despite_incapable_backend(self):
        # auto mode omits the candidate with a warning, it does not block.
        with mock.patch("modelctl_capabilities.probe_backend",
                        return_value={"schema": 2, "features": {}, "cli": {}}), \
             mock.patch.object(modelctl, "save_profile") as save, \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"):
            resp = self._post_mode("auto")
            self.assertEqual(resp.status_code, 200)
            job = self.wait_job(resp.json()["jobs"]["moe_cache"])
        self.assertEqual(job["status"], "done")
        save.assert_called_once()


class TestBaselinePlanCapabilities(WebTestBase):
    """M1 regression: baseline plan argv is compiled against the same
    probed capabilities that launch-time validation will see."""

    def _mock_hardware(self):
        from modelctl_hardware import HardwareSnapshot, GpuSnapshot
        return HardwareSnapshot(
            captured_at=1000.0, fingerprint="test",
            gpus=(GpuSnapshot("SYCL0", "B70", 32 << 30, 30 << 30, 0, True,
                              "primary", 608),),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})

    def test_current_profile_plan_includes_cache_flags_when_capable(self):
        import modelctl_plans
        profile = {"name": "m1", "backend": "llama-cpp",
                   "model_path": "/nonexistent",
                   "config": {"device": "SYCL0", "split_mode": "",
                              "tensor_split": "", "ctx": 32768,
                              "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                              "fit": "on", "extra": "", "flash_attn": "auto",
                              "ttl": 3600, "mtp": "off"},
                   "moe_cache": {"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 4 << 30},
                                         "policy": "slru",
                                         "admission_misses": 2}}}
        with mock.patch("modelctl_capabilities.probe_backend",
                        return_value={"features":
                                      {"moe_weight_transfer_cache": True}}), \
             mock.patch("modelctl.load_defaults",
                        return_value={"vram_limit_pct": 90,
                                      "primary_gpu": "SYCL0"}), \
             mock.patch("modelctl.get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_hardware.load_settings", return_value={}):
            plans = modelctl_plans.compile_launch_plans(
                profile, self._mock_hardware())
        base = next(p for p in plans if p.source == "current-profile")
        self.assertEqual(list(base.argv).count("--moe-cache-bytes"), 1)


class TestPlansPageDecisionTrace(WebTestBase):
    """Task H3: the justification has to survive the transport, not just
    dataclass construction -- a field the client never receives explains
    nothing.

    The old plans page rendered the whole DecisionTrace; nothing on the
    /api/v2 surface carries one (modelctl_evidence.build_decision_trace
    is now only exercised directly, in test_release_a.py). What replaced
    it is the per-plan `reason` on GET /api/v2/models/{name}/plans, which
    is what this asserts.
    """

    def test_plans_api_carries_the_per_plan_justification(self):
        import modelctl_hardware
        profile = {"name": "m1", "backend": "llama-cpp", "model_path": "",
                   "config": {"device": "SYCL0", "split_mode": "",
                              "tensor_split": "", "ctx": 32768,
                              "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                              "fit": "on", "extra": ""}}
        snap = modelctl_hardware.HardwareSnapshot(
            captured_at=time.time(), fingerprint="hwfingerprint01",
            gpus=(modelctl_hardware.GpuSnapshot(
                "SYCL0", "B70", 32 << 30, 30 << 30, 0, True, "", 608),),
            ram_total_bytes=64 << 30, ram_available_bytes=50 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})
        with mock.patch("modelctl_hardware.capture_hardware_snapshot",
                        return_value=snap), \
             mock.patch.object(modelctl, "load_profile", return_value=profile), \
             mock.patch.object(modelctl, "build_server_args",
                               return_value=["llama-server"]):
            resp = self.client.get("/api/v2/models/m1/plans", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["reason"],
                            f"plan {row['id']} carries no justification")
        # An unmeasured profile must say so rather than showing a blank
        # reason or an invented one.
        baseline = next(r for r in rows if r["category"] == "baseline")
        self.assertFalse(baseline["tested"])
        self.assertIsNone(baseline["measured"])
        self.assertIn("the profile as saved", baseline["reason"])


class TestOwnGroupKillGuard(unittest.TestCase):
    """The 2026-08-04 console crash: smoke_test_profile spawned its
    llama-server in the service's own process group, register_process
    recorded that pgid, and the job runner's completion cleanup killpg'd
    the whole web service. A registered child must never carry OUR pgid,
    and the smoke child must be detached at spawn."""

    def _ctx(self):
        from modelctl_web.jobs import JobContext
        from tempfile import TemporaryDirectory
        from modelctl_web.jobs import JobStore
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = JobStore(Path(tmp.name) / "jobs.db")
        job_id = store.create("smoke", "t", {})
        return JobContext(store, job_id)

    def test_never_records_our_own_process_group(self):
        from types import SimpleNamespace
        ctx = self._ctx()
        proc = SimpleNamespace(pid=os.getpid(), poll=lambda: 0)
        ctx.register_process(proc)
        self.assertFalse(hasattr(proc, "pgid"),
                         "recorded the service's own process group")
        with mock.patch("modelctl_web.jobs.os.killpg") as kp:
            ctx.cancel_all()
        kp.assert_not_called()

    def test_records_a_detached_childs_group(self):
        from types import SimpleNamespace
        from modelctl_web import jobs as jobs_mod
        ctx = self._ctx()
        proc = SimpleNamespace(pid=777777, poll=lambda: 0)
        own = os.getpgid(0)
        with mock.patch.object(jobs_mod.os, "getpgid",
                               side_effect=lambda p: 4242 if p == 777777
                               else own):
            ctx.register_process(proc)
        self.assertEqual(getattr(proc, "pgid", None), 4242)

    def test_smoke_child_spawns_detached(self):
        import modelctl
        from types import SimpleNamespace
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "m").mkdir()
        launch = SimpleNamespace(argv=("llama-server", "--port", "1"),
                                 environment={}, is_valid=True)
        seen = {}

        def fake_popen(cmd, **kw):
            seen.update(kw)
            raise FileNotFoundError("stop here")

        with mock.patch.object(modelctl, "load_profile",
                               return_value={"name": "m"}), \
             mock.patch.object(modelctl, "canonical_launch_command",
                               return_value=(launch, True, [])), \
             mock.patch.object(modelctl, "PROFILES_DIR", Path(tmp.name)), \
             mock.patch.object(modelctl.subprocess, "Popen",
                               side_effect=fake_popen):
            res = modelctl.smoke_test_profile("m")
        self.assertFalse(res["ok"])
        self.assertTrue(seen.get("start_new_session"),
                        "smoke llama-server must start in its own session")
