"""Task C2: the add-model wizard as a durable workflow.

These cover the states a happy-path test never reaches: a step that is
still running, a step that failed, a resume after restart, and a
registration that failed after the profile was already created.

The phase-3 cutover demolished the server-rendered /add/{id}/<step>
pages; every guard below now lives on /api/v2/wizard/{id}/*, which
answers JSON (409 for a blocked advance, 502 for a failed registration)
where the old flow answered a redirect plus rendered HTML. The
assertions follow the facts, not the surface.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
from modelctl_web import wizard as wiz
from modelctl_web.app import create_app
from modelctl_web.jobs import JobStore, JobRunner

TOKEN = "test-token"


class TestStepOutcomes(unittest.TestCase):
    def test_outcome_records_structured_state(self):
        s = wiz.WizardState()
        s.record_outcome("download", ok=False, job_id="j1", status="failed",
                         messages=["boom"], error="traceback")
        o = s.outcome("download")
        self.assertFalse(o["ok"])
        self.assertEqual(o["job_id"], "j1")
        self.assertEqual(o["error"], "traceback")
        self.assertTrue(s.step_failed("download"))

    def test_running_step_blocks_advancing(self):
        # Advancing over a running download produces a wizard that looks
        # finished while its download is still going.
        s = wiz.WizardState()
        s.record_outcome("download", ok=False, status="running")
        self.assertTrue(s.step_running("download"))
        self.assertFalse(s.step_failed("download"))
        self.assertFalse(s.can_advance("download"))
        self.assertIn("still running", s.blocking_reason("download"))

    def test_failed_step_blocks_advancing_with_the_reason(self):
        s = wiz.WizardState()
        s.record_outcome("download", ok=False, status="failed",
                         error="no space left")
        self.assertFalse(s.can_advance("download"))
        self.assertIn("no space left", s.blocking_reason("download"))

    def test_unrun_step_does_not_block(self):
        self.assertTrue(wiz.WizardState().can_advance("download"))

    def test_clear_outcome_allows_a_retry(self):
        s = wiz.WizardState()
        s.record_outcome("download", ok=False, status="failed")
        s.clear_outcome("download")
        self.assertEqual(s.outcome("download"), {})
        self.assertTrue(s.can_advance("download"))

    def test_new_fields_survive_a_persistence_roundtrip(self):
        s = wiz.WizardState(profile_name="m1", command_fingerprint="fp")
        s.record_outcome("register", ok=True, messages=["done"])
        s.source_verification = {"ok": True, "warnings": ["dup"]}
        back = wiz.WizardState.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(back.command_fingerprint, "fp")
        self.assertTrue(back.outcome("register")["ok"])
        self.assertEqual(back.source_verification["warnings"], ["dup"])


class TestOutcomeFromJob(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = JobStore(Path(self.tmp.name) / "jobs.db")

    def test_missing_job_is_distinguishable(self):
        o = wiz.outcome_from_job(self.store, "nope")
        self.assertFalse(o["ok"])
        self.assertEqual(o["status"], "missing")

    def test_no_job_id_returns_empty(self):
        self.assertEqual(wiz.outcome_from_job(self.store, ""), {})

    def test_structured_outcome_is_read_not_scraped_from_logs(self):
        job_id = self.store.create("import", "t", lane="mutation")
        self.store.update(job_id, status="done",
                          outcome=json.dumps({"profile": "m1",
                                              "warnings": ["dup"]}))
        o = wiz.outcome_from_job(self.store, job_id)
        self.assertTrue(o["ok"])
        self.assertEqual(o["data"]["profile"], "m1")
        self.assertEqual(o["warnings"], ["dup"])

    def test_failed_job_carries_its_error(self):
        job_id = self.store.create("import", "t", lane="mutation")
        self.store.update(job_id, status="failed", error="disk full")
        o = wiz.outcome_from_job(self.store, job_id)
        self.assertFalse(o["ok"])
        self.assertEqual(o["status"], "failed")
        self.assertIn("disk full", o["error"])

    def test_running_job_is_not_reported_as_success(self):
        job_id = self.store.create("import", "t", lane="mutation")
        self.store.update(job_id, status="running")
        o = wiz.outcome_from_job(self.store, job_id)
        self.assertFalse(o["ok"])
        self.assertEqual(o["status"], "running")


class WizardWebBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        self.wizards = self.root / "wizards"
        self.wizards.mkdir()
        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles),
                  mock.patch.object(wiz, "WIZARD_DIR", self.wizards)):
            p.start()
            self.addCleanup(p.stop)
        # WizardStore defaults its base_dir at call time from WIZARD_DIR's
        # value when the module was imported, so patch the default too.
        self.store_patch = mock.patch.object(
            wiz.WizardStore, "__init__",
            lambda s, base_dir=self.wizards: (
                setattr(s, "base_dir", Path(base_dir)),
                Path(base_dir).mkdir(parents=True, exist_ok=True))[0])
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)

        self.jobs = JobStore(self.root / "jobs.db")
        runner = JobRunner(self.jobs)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        self.client = TestClient(create_app(token=TOKEN, store=self.jobs,
                                            runner=runner))
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    def new_wizard(self):
        resp = self.client.post("/api/v2/wizards", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        return resp.json()["wizard_id"]


class TestSourceVerification(WizardWebBase):
    def test_bad_local_path_is_rejected_before_a_profile_exists(self):
        wid = self.new_wizard()
        bad = self.root / "notes.txt"
        bad.write_bytes(b"not a model" * 200)
        resp = self.client.post(
            f"/api/v2/wizard/{wid}/source", headers=self.auth,
            json={"source_type": "local_file", "local_path": str(bad)})
        # Refused on the source step rather than walking into an import.
        self.assertEqual(resp.status_code, 422)
        self.assertIn("not a GGUF", resp.json()["error"])
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.step, "source")
        # The verdict is persisted, not only reported: a reload of the
        # wizard has to show why the file was refused.
        self.assertFalse(state.source_verification["ok"])
        self.assertTrue(state.errors)
        self.assertEqual(len(list(self.profiles.glob("*.json"))), 0)

    def test_valid_gguf_advances_and_records_verification(self):
        wid = self.new_wizard()
        good = self.root / "model.gguf"
        good.write_bytes(b"GGUF" + b"\0" * 8192)
        # The source POST submits the import job itself (submission moved
        # out of the download step); stub it so the test doesn't run a
        # real import of this fake GGUF in a background thread that can
        # outlive the patches.
        with mock.patch("modelctl_web.mutate.submit_import_local",
                        return_value="job-import-1") as sub:
            resp = self.client.post(
                f"/api/v2/wizard/{wid}/source", headers=self.auth,
                json={"source_type": "local_file", "local_path": str(good)})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["step"], "download")
        state = wiz.WizardStore().load(wid)
        self.assertTrue(state.source_verification["ok"])
        # The submission happened exactly once, at POST time.
        sub.assert_called_once()
        self.assertEqual(state.download_job_id, "job-import-1")


class TestAdvanceGuards(WizardWebBase):
    def _wizard_with_download(self, status, error=""):
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        job_id = self.jobs.create("import", "t", lane="mutation")
        self.jobs.update(job_id, status=status, error=error)
        state.source_type = "local_file"
        state.local_path = str(self.root / "m.gguf")
        state.download_job_id = job_id
        state.advance("download")
        wiz.WizardStore().save(state)
        return wid

    def test_cannot_advance_while_the_download_runs(self):
        wid = self._wizard_with_download("running")
        resp = self.client.post(f"/api/v2/wizard/{wid}/download/next",
                                headers=self.auth)
        self.assertEqual(resp.status_code, 409)
        self.assertIn("still running", resp.json()["error"])
        self.assertEqual(wiz.WizardStore().load(wid).step, "download")

    def test_cannot_advance_after_a_failed_download(self):
        wid = self._wizard_with_download("failed", error="disk full")
        resp = self.client.post(f"/api/v2/wizard/{wid}/download/next",
                                headers=self.auth)
        self.assertEqual(resp.status_code, 409)
        self.assertIn("disk full", resp.json()["error"])
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.step, "download")
        self.assertFalse(state.download_complete)
        # The job's own error is folded into the step outcome and persisted,
        # so the reason survives a reload instead of living in one response.
        self.assertIn("disk full", state.outcome("download")["error"])
        self.assertIn("disk full", state.blocking_reason("download"))

    def test_successful_download_advances_and_captures_the_profile(self):
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        job_id = self.jobs.create("import", "t", lane="mutation")
        self.jobs.update(job_id, status="done",
                         outcome=json.dumps({"profile": "m1"}))
        state.source_type = "local_file"
        state.download_job_id = job_id
        state.advance("download")
        wiz.WizardStore().save(state)

        resp = self.client.post(f"/api/v2/wizard/{wid}/download/next",
                                headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["step"], "analyze")
        state = wiz.WizardStore().load(wid)
        self.assertTrue(state.download_complete)
        self.assertEqual(state.profile_name, "m1")


class TestRetry(WizardWebBase):
    def test_retry_clears_the_failed_step_and_returns_to_it(self):
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        job_id = self.jobs.create("import", "t", lane="mutation")
        self.jobs.update(job_id, status="failed", error="boom")
        state.source_type = "local_file"
        state.local_path = str(self.root / "m.gguf")
        state.download_job_id = job_id
        state.record_outcome("download", ok=False, status="failed",
                             error="boom")
        state.download_complete = False
        state.set_error("boom")
        state.advance("download")
        wiz.WizardStore().save(state)

        with mock.patch("modelctl_web.mutate.submit_import_local",
                        return_value="job-retry-1"):
            resp = self.client.post(f"/api/v2/wizard/{wid}/retry/download",
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["step"], "download")
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.outcome("download"), {})
        # The failed job is let go of: retry re-issues the step's own job
        # rather than re-reading the corpse of the old one.
        self.assertNotEqual(state.download_job_id, job_id)
        self.assertEqual(state.download_job_id, "job-retry-1")
        self.assertFalse(state.download_complete)
        self.assertFalse(state.errors)
        # The wizard survives: the same instance continues.
        self.assertEqual(resp.json()["wizard_id"], wid)
        self.assertEqual(state.step, "download")

    def test_retry_of_an_unknown_step_is_refused(self):
        wid = self.new_wizard()
        resp = self.client.post(f"/api/v2/wizard/{wid}/retry/teleport",
                                headers=self.auth)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("step", resp.json()["error"])
        # Nothing was invented for the unknown step.
        self.assertEqual(wiz.WizardStore().load(wid).step_outcomes, {})


class TestRegistrationFailure(WizardWebBase):
    def _ready_wizard(self):
        wid = self.new_wizard()
        (self.profiles / "m1.json").write_text(json.dumps({
            "name": "m1", "model_path": "/x/m.gguf", "backend": "llama-cpp",
            "config": {"ctx": 8192}}))
        state = wiz.WizardStore().load(wid)
        state.profile_name = "m1"
        state.selected_plan_id = "planX"
        state.advance("register")
        wiz.WizardStore().save(state)
        return wid

    def test_failed_registration_leaves_the_profile_and_offers_retry(self):
        # This previously set registration_complete=True regardless and
        # walked on to a done page reporting success.
        from modelctl_services import plan_service
        wid = self._ready_wizard()
        with mock.patch.object(plan_service, "apply_plan",
                               return_value=plan_service.PlanResult(
                                   ok=False, messages=["plan planX not found"])):
            resp = self.client.post(f"/api/v2/wizard/{wid}/register",
                                    headers=self.auth, json={})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("planX", resp.json()["error"])
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.step, "register")
        self.assertFalse(state.registration_complete)
        self.assertIn("planX", state.registration_error)
        self.assertFalse(state.outcome("register")["ok"])
        # The profile is untouched and still valid.
        self.assertTrue((self.profiles / "m1.json").exists())

        # The failure is a retry, not a dead end: what the old register
        # page rendered as a "Registration failed" banner plus a retry
        # link is the persisted registration_error plus this endpoint.
        retry = self.client.post(f"/api/v2/wizard/{wid}/retry/register",
                                 headers=self.auth)
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["step"], "register")
        self.assertEqual(retry.json()["registration_error"], "")
        self.assertEqual(retry.json()["step_gates"]["register"]["outcome"], {})

    def test_successful_registration_records_provenance(self):
        from modelctl_services import plan_service
        wid = self._ready_wizard()
        with mock.patch.object(plan_service, "apply_plan",
                               return_value=plan_service.PlanResult(ok=True)), \
             mock.patch.object(modelctl, "canonical_launch_command",
                               return_value=(mock.Mock(command_fingerprint="fp99"),
                                             True, [])), \
             mock.patch("modelctl_web.mutate.submit_load", return_value="job9"):
            resp = self.client.post(f"/api/v2/wizard/{wid}/register",
                                    headers=self.auth, json={})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["step"], "done")
        state = wiz.WizardStore().load(wid)
        self.assertTrue(state.registration_complete)
        self.assertEqual(state.command_fingerprint, "fp99")
        self.assertIn("/v1/", state.endpoint)
        self.assertTrue(state.outcome("register")["ok"])

    def test_register_without_a_profile_refuses(self):
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        state.advance("register")
        wiz.WizardStore().save(state)
        resp = self.client.post(f"/api/v2/wizard/{wid}/register",
                                headers=self.auth, json={})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("no profile", resp.json()["error"])
        self.assertFalse(wiz.WizardStore().load(wid).registration_complete)


class TestDoneEvidence(WizardWebBase):
    """The done page is now a SPA view over GET /api/v2/wizard/{id}.

    Every number it shows has to be a field in that payload -- a console
    that re-derives them client-side is how a done page reports a
    measurement nobody took.
    """

    def test_wizard_detail_carries_the_done_evidence(self):
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        state.profile_name = "m1"
        state.registration_complete = True
        state.command_fingerprint = "fp99"
        state.endpoint = "http://127.0.0.1:9292/v1/chat/completions"
        state.measured = {"generation_tps": 21.5, "cache_state": "warm"}
        state.record_outcome("download", ok=True, messages=["imported"])
        state.advance("done")
        wiz.WizardStore().save(state)

        d = self.client.get(f"/api/v2/wizard/{wid}", headers=self.auth).json()
        self.assertEqual(d["command_fingerprint"], "fp99")
        self.assertIn("chat/completions", d["endpoint"])
        self.assertEqual(d["measured"]["generation_tps"], 21.5)
        self.assertEqual(d["measured"]["cache_state"], "warm")
        self.assertEqual(d["step_gates"]["download"]["outcome"]["messages"],
                         ["imported"])

    def test_wizard_detail_admits_an_unmeasured_plan(self):
        # An empty `measured` is what the console renders as "not
        # measured"; anything non-empty here would be presented as a
        # number the wizard actually observed.
        wid = self.new_wizard()
        state = wiz.WizardStore().load(wid)
        state.profile_name = "m1"
        state.registration_complete = True
        state.advance("done")
        wiz.WizardStore().save(state)
        d = self.client.get(f"/api/v2/wizard/{wid}", headers=self.auth).json()
        self.assertEqual(d["measured"], {})
        self.assertEqual(d["step_gates"]["test"]["outcome"], {})


class TestAcquisitionIsConsolidatedOnAdd(WizardWebBase):
    """P1: /add owns acquisition. The legacy /pull and /import routes are
    compatibility shims that land the user inside a pre-populated add
    wizard -- there is no second path with its own validation, job state,
    or registration.

    Phase 3 moved the destination from the server-rendered /add pages to
    /v2/add, so the shims now redirect there; the consolidation itself is
    unchanged. Two cases moved out of this file entirely: the plain
    /pull -> /v2/add and /import -> /v2/add 301s are rows in
    test_console_phase3.py::TestDemolition.REDIRECTS, which asserts the
    same status and target for the whole redirect table, so keeping them
    here would only be a second copy.
    """

    def test_pull_search_is_served_by_the_one_add_search_api(self):
        # /pull?q= used to render its own search page. It is now a 301
        # into the add surface, carrying the search term along so a
        # bookmarked search does not land on an empty box -- and search
        # itself is one API the add wizard calls, not a second
        # acquisition surface.
        resp = self.client.get("/pull?q=qwen", headers=self.auth,
                               follow_redirects=False)
        self.assertEqual(resp.status_code, 301)
        self.assertEqual(resp.headers["location"], "/v2/add?q=qwen")
        with mock.patch.object(modelctl, "search_models",
                               return_value=[{"repo_id": "org/qwen-GGUF",
                                              "downloads": 5, "is_gguf": True,
                                              "has_mtp": False}]) as sm:
            body = self.client.get("/api/v2/hf/search?q=qwen",
                                   headers=self.auth).json()
        sm.assert_called_once_with("qwen")
        self.assertEqual(body["results"][0]["repo_id"], "org/qwen-GGUF")

    def test_pull_repo_becomes_a_prepopulated_wizard(self):
        resp = self.client.get("/pull/unsloth/Some-GGUF", headers=self.auth,
                               follow_redirects=False)
        self.assertEqual(resp.status_code, 303)
        loc = resp.headers["location"]
        self.assertRegex(loc, r"^/v2/add/[0-9a-f]+$")
        wid = loc.split("/")[3]
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.repo_id, "unsloth/Some-GGUF")
        self.assertEqual(state.source_type, "hf_repo")
        self.assertEqual(state.step, "inspect")

    def test_search_result_start_link_prepopulates_a_wizard(self):
        resp = self.client.get("/add/start/unsloth/Some-GGUF",
                               headers=self.auth, follow_redirects=False)
        loc = resp.headers["location"]
        wid = loc.split("/")[3]
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.repo_id, "unsloth/Some-GGUF")
        self.assertEqual(state.step, "inspect")

    def test_import_post_is_gone_and_local_files_enter_through_source(self):
        # POST /import minted a wizard with the path pre-filled. It could
        # not be redirected (a 301 turns the POST into a GET and drops the
        # body), so it was removed outright: a local path is now typed
        # into the wizard's own source step.
        resp = self.client.post("/import", headers=self.auth,
                                data={"file_path": "/models/x.gguf"},
                                follow_redirects=False)
        self.assertIn(resp.status_code, (404, 405))

        wid = self.new_wizard()
        resp = self.client.post(
            f"/api/v2/wizard/{wid}/source", headers=self.auth,
            json={"source_type": "local_file",
                  "local_path": "/models/x.gguf"})
        state = wiz.WizardStore().load(wid)
        self.assertEqual(state.local_path, "/models/x.gguf")
        self.assertEqual(state.source_type, "local_file")
        # Deliberately NOT advanced past source: the source step runs
        # local-file verification on submit, so no acquisition path can
        # skip the checks the wizard exists to enforce.
        self.assertEqual(resp.status_code, 422)
        self.assertFalse(state.source_verification["ok"])
        self.assertEqual(state.step, "source")

    def test_primary_nav_has_one_acquisition_entry(self):
        # The nav moved into the SPA shell at the phase-3 cutover, so the
        # assertion follows it: the shell is where a second acquisition
        # entry would reappear.
        shell = (Path(__file__).resolve().parent / "console" / "src"
                 / "app.tsx").read_text()
        nav = shell.split('<aside class="side">')[1].split("</aside>")[0]
        self.assertIn('href="/v2/add"', nav)
        self.assertNotIn('href="/pull"', nav)
        self.assertNotIn('href="/import"', nav)

    def test_add_search_results_are_startable(self):
        with mock.patch.object(modelctl, "search_models",
                               return_value=[{"repo_id": "org/m-GGUF",
                                              "downloads": 5, "is_gguf": True,
                                              "has_mtp": False}]) as sm:
            body = self.client.get("/api/v2/hf/search?q=m-GGUF",
                                   headers=self.auth).json()
        sm.assert_called_once_with("m-GGUF")
        self.assertEqual(body["error"], "")
        self.assertEqual(body["results"][0]["repo_id"], "org/m-GGUF")
        # A result is actionable, which is what the old "/add/start/..."
        # link on each row bought: its repo id drives a wizard to inspect.
        wid = self.new_wizard()
        started = self.client.post(
            f"/api/v2/wizard/{wid}/source", headers=self.auth,
            json={"source_type": "hf_repo",
                  "repo_id": body["results"][0]["repo_id"]})
        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.json()["repo_id"], "org/m-GGUF")
        self.assertEqual(started.json()["step"], "inspect")


if __name__ == "__main__":
    unittest.main()
