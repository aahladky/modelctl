"""Console phase 3: the settings API, the old-console demolition, and the
redirect layer that replaced it.

Carries its own isolation (tmp state, patched dirs/globs) so single-file
runs never touch real state -- the full suite additionally rides
test__hermeticity's bootstrap.
"""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_hardware
import modelctl_setup
from modelctl_services import settings_service
from modelctl_web import settings as web_settings
from modelctl_web import wizard as wiz
from modelctl_web.app import create_app
from modelctl_web.jobs import JobRunner, JobStore

TOKEN = "test-token"


class SettingsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        self.defaults_path = self.root / "defaults.json"
        self.hardware_path = self.root / "hardware.json"
        self.token_path = self.root / "web_token"
        no_binaries = str(self.root / "no-binaries" / "*")

        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
                  mock.patch.object(modelctl, "DEFAULTS_PATH", self.defaults_path),
                  mock.patch.object(modelctl, "COMMON_LLAMA_SERVER_GLOBS",
                                    [no_binaries]),
                  mock.patch.object(modelctl, "COMMON_ENV_SCRIPT_GLOBS",
                                    [no_binaries]),
                  mock.patch.object(modelctl_hardware, "SETTINGS_PATH",
                                    self.hardware_path),
                  mock.patch.object(wiz, "WIZARD_DIR", self.root / "wizards")):
            p.start()
            self.addCleanup(p.stop)

        # A fixed two-GPU machine, so hardware policy is deterministic.
        self.snapshot = modelctl_hardware.HardwareSnapshot(
            captured_at=0.0, fingerprint="fixture",
            gpus=(
                modelctl_hardware.GpuSnapshot(
                    device="SYCL0", name="Fixture GPU 0",
                    total_bytes=8 << 30, free_bytes=7 << 30,
                    reserve_bytes=0, enabled=True, role="",
                    bandwidth_gbs=400.0),
                modelctl_hardware.GpuSnapshot(
                    device="SYCL1", name="Fixture GPU 1",
                    total_bytes=4 << 30, free_bytes=4 << 30,
                    reserve_bytes=0, enabled=True, role="",
                    bandwidth_gbs=200.0),
            ),
            ram_total_bytes=32 << 30, ram_available_bytes=24 << 30,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})

        def _fixture_capture(settings=None):
            return modelctl_hardware.apply_settings(
                self.snapshot, settings or modelctl_hardware.load_settings())

        p = mock.patch.object(modelctl_hardware, "capture_hardware_snapshot",
                              _fixture_capture)
        p.start()
        self.addCleanup(p.stop)

        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.store = store
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        p = mock.patch("modelctl_web.app.TOKEN_PATH", self.token_path)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(create_app(token=TOKEN, store=store,
                                            runner=runner))
        self.auth = {"Authorization": f"Bearer {TOKEN}"}


class TestSettingsOverview(SettingsBase):
    def test_overview_has_every_section(self):
        r = self.client.get("/api/v2/settings", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(
            set(body),
            {"readiness", "defaults", "hardware", "access", "paths",
             "diagnostics"})
        json.dumps(body)  # the endpoint serializes this verbatim

    def test_every_settable_default_has_a_form_descriptor(self):
        """The form is generated from the service's own key lists.

        A key added to settings_service without a descriptor here would be
        writable by hand and uneditable in the console -- exactly the kind
        of drift a settings page is supposed to remove.
        """
        described = {f["name"] for f in web_settings.default_fields()}
        settable = set(settings_service.STRING_KEYS) | set(settings_service.INT_KEYS)
        self.assertEqual(described, settable)

    def test_int_field_bounds_are_the_validator_s_own(self):
        by_name = {f["name"]: f for f in web_settings.default_fields()}
        for key, (low, high) in settings_service.BOUNDS.items():
            self.assertEqual((by_name[key]["min"], by_name[key]["max"]),
                             (low, high),
                             f"{key}: the form would offer a range the write "
                             f"rejects")

    def test_paths_report_their_source(self):
        r = self.client.get("/api/v2/settings", headers=self.auth)
        rows = r.json()["paths"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(set(row), {"label", "value", "source", "overridden"})


class TestDefaultsWrite(SettingsBase):
    def test_valid_write_persists_and_reports_what_applied(self):
        r = self.client.post("/api/v2/settings/defaults", headers=self.auth,
                             json={"updates": {"ctx": "4096", "flash_attn": "on"}})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["warnings"], [])
        self.assertEqual(body["applied"], {"ctx": 4096, "flash_attn": "on"})
        self.assertEqual(json.loads(self.defaults_path.read_text())["ctx"], 4096)

    def test_out_of_range_is_rejected_with_the_server_s_own_wording(self):
        r = self.client.post("/api/v2/settings/defaults", headers=self.auth,
                             json={"updates": {"ctx": "10"}})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["applied"], {})
        self.assertEqual(len(body["warnings"]), 1)
        self.assertIn("ctx", body["warnings"][0])
        self.assertIn("outside", body["warnings"][0])

    def test_unknown_key_is_rejected_not_stored(self):
        r = self.client.post("/api/v2/settings/defaults", headers=self.auth,
                             json={"updates": {"rm_rf": "1"}})
        self.assertEqual(r.json()["applied"], {})
        self.assertIn("not a settable default", r.json()["warnings"][0])
        self.assertFalse(self.defaults_path.exists())

    def test_a_bad_body_is_a_400_not_a_silent_no_op(self):
        r = self.client.post("/api/v2/settings/defaults", headers=self.auth,
                             json={"ctx": 4096})
        self.assertEqual(r.status_code, 400)

    def test_env_shadowed_field_says_the_save_will_not_take_effect(self):
        """MODELCTL_DEFAULT_* wins over defaults.json in load_defaults().

        Saving such a field writes the file and changes nothing observable;
        reporting a clean save there is how a value 'mysteriously does not
        stick'.
        """
        with mock.patch.dict(os.environ, {"MODELCTL_DEFAULT_CTX": "8192"}):
            r = self.client.post("/api/v2/settings/defaults", headers=self.auth,
                                 json={"updates": {"ctx": "4096"}})
            body = r.json()
            self.assertEqual(body["shadowed"], {"ctx": "MODELCTL_DEFAULT_CTX"})
            self.assertTrue(any("MODELCTL_DEFAULT_CTX" in w
                                for w in body["warnings"]),
                            body["warnings"])


class TestHardwareWrite(SettingsBase):
    def test_reserve_and_role_persist(self):
        r = self.client.post("/api/v2/settings/hardware", headers=self.auth,
                             json={"devices": {"SYCL0": {
                                 "reserve_bytes": 2 << 30, "role": "primary",
                                 "enabled": True}}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["warnings"], [])
        stored = json.loads(self.hardware_path.read_text())
        self.assertEqual(stored["devices"]["SYCL0"]["reserve_bytes"], 2 << 30)
        self.assertEqual(stored["devices"]["SYCL0"]["role"], "primary")

    def test_a_fractional_reserve_is_reported_not_silently_dropped(self):
        """The old form route coerced with `except ValueError: pass`.

        Its own template rendered the value as "0.0", so every save of an
        untouched reserve was silently discarded -- the reason no GPU in
        the live hardware.json ever had a reserve_bytes key.
        """
        r = self.client.post("/api/v2/settings/hardware", headers=self.auth,
                             json={"devices": {"SYCL0": {"reserve_bytes": "1.5"}}})
        body = r.json()
        self.assertEqual(body["applied"], 0)
        self.assertEqual(len(body["warnings"]), 1)
        self.assertIn("SYCL0 reserve", body["warnings"][0])

    def test_reserve_larger_than_the_card_is_refused(self):
        r = self.client.post("/api/v2/settings/hardware", headers=self.auth,
                             json={"devices": {"SYCL1": {"reserve_bytes": 64 << 30}}})
        self.assertEqual(r.json()["applied"], 0)
        self.assertIn("SYCL1 reserve", r.json()["warnings"][0])

    def test_unknown_device_is_refused(self):
        r = self.client.post("/api/v2/settings/hardware", headers=self.auth,
                             json={"devices": {"SYCL9": {"enabled": False}}})
        self.assertEqual(r.json()["applied"], 0)
        self.assertIn("not a device on this machine", r.json()["warnings"][0])

    def test_role_outside_the_planner_s_vocabulary_is_refused(self):
        r = self.client.post("/api/v2/settings/hardware", headers=self.auth,
                             json={"devices": {"SYCL0": {"role": "secondary"}}})
        self.assertEqual(r.json()["applied"], 0)
        self.assertIn("role", r.json()["warnings"][0])

    def test_bandwidth_override_writes_the_key_apply_settings_reads(self):
        """hardware_service.update_device_settings writes bandwidth_gbs,
        which apply_settings never reads -- a dead write. The typed
        endpoint must use memory_bandwidth_gbs_override."""
        self.client.post("/api/v2/settings/hardware", headers=self.auth,
                         json={"devices": {"SYCL0": {
                             "memory_bandwidth_gbs_override": 512}}})
        stored = json.loads(self.hardware_path.read_text())
        self.assertEqual(
            stored["devices"]["SYCL0"]["memory_bandwidth_gbs_override"], 512.0)
        r = self.client.get("/api/v2/settings", headers=self.auth)
        dev = next(d for d in r.json()["hardware"]["devices"]
                   if d["device"] == "SYCL0")
        self.assertTrue(dev["bandwidth_overridden"])
        self.assertEqual(dev["bandwidth_gbs"], 512.0)

    def test_clearing_the_override_returns_to_the_probed_value(self):
        self.client.post("/api/v2/settings/hardware", headers=self.auth,
                         json={"devices": {"SYCL0": {
                             "memory_bandwidth_gbs_override": 512}}})
        self.client.post("/api/v2/settings/hardware", headers=self.auth,
                         json={"devices": {"SYCL0": {
                             "memory_bandwidth_gbs_override": ""}}})
        r = self.client.get("/api/v2/settings", headers=self.auth)
        dev = next(d for d in r.json()["hardware"]["devices"]
                   if d["device"] == "SYCL0")
        self.assertFalse(dev["bandwidth_overridden"])
        self.assertEqual(dev["bandwidth_gbs"], 400.0)


class TestTokenRotation(SettingsBase):
    def test_rotation_needs_an_explicit_confirm(self):
        r = self.client.post("/api/v2/settings/token/rotate", headers=self.auth,
                             json={})
        self.assertEqual(r.status_code, 409)
        self.assertTrue(r.json()["gate"]["changes"])
        self.assertFalse(self.token_path.exists())

    def test_confirmed_rotation_replaces_the_token_and_resigns_this_session(self):
        r = self.client.post("/api/v2/settings/token/rotate", headers=self.auth,
                             json={"confirm": True})
        self.assertEqual(r.status_code, 200)
        new_token = self.token_path.read_text().strip()
        self.assertTrue(new_token)
        self.assertNotEqual(new_token, TOKEN)
        # never echoed in the body
        self.assertNotIn(new_token, r.text)
        # the rotating session keeps working: the cookie was re-issued
        self.assertEqual(r.cookies.get("modelctl_web_token"), new_token)
        # and the old bearer token is now rejected in-process, without a
        # restart -- a stale closure would keep honouring it
        self.assertEqual(
            self.client.get("/api/v2/settings",
                            headers={"Authorization": f"Bearer {TOKEN}"}).status_code,
            401)
        self.assertEqual(
            self.client.get("/api/v2/settings",
                            headers={"Authorization": f"Bearer {new_token}"}).status_code,
            200)

    def test_an_env_supplied_token_is_not_rotatable_here(self):
        with mock.patch.dict(os.environ, {"MODELCTL_WEB_TOKEN": "from-env"}):
            r = self.client.get("/api/v2/settings", headers=self.auth)
            self.assertFalse(r.json()["access"]["token_rotatable"])
            r = self.client.post("/api/v2/settings/token/rotate",
                                 headers=self.auth, json={"confirm": True})
            self.assertEqual(r.status_code, 400)
            self.assertIn("MODELCTL_WEB_TOKEN", r.json()["error"])


class TestReadiness(SettingsBase):
    def test_readiness_carries_the_setup_checks(self):
        r = self.client.get("/api/v2/settings", headers=self.auth)
        readiness = r.json()["readiness"]
        self.assertEqual(set(readiness), {"ready", "first_run", "checks", "error"})
        ids = {c["id"] for c in readiness["checks"]}
        self.assertIn("models_dir", ids)
        self.assertIn("backend", ids)

    def test_no_check_links_at_a_demolished_page(self):
        """modelctl_setup emits fix_url values that render as links.

        They pointed at /settings, /hardware and /runtime/routing -- three
        pages the cutover removed.
        """
        status = modelctl_setup.probe_setup(probe_backend=False)
        for c in status.checks:
            if c.fix_url:
                self.assertTrue(c.fix_url.startswith("/v2/"),
                                f"{c.id} links at {c.fix_url}, which no "
                                f"longer exists")


class TestDemolition(SettingsBase):
    """The old server-rendered console is gone and every URL it published
    still resolves."""

    REDIRECTS = {
        "/": "/v2/",
        "/setup": "/v2/settings",
        "/settings": "/v2/settings",
        "/hardware": "/v2/settings",
        "/tiers": "/v2/models",
        "/runtime": "/v2/",
        "/runtime/routing": "/v2/settings",
        "/jobs": "/v2/jobs",
        "/jobs/abc123": "/v2/jobs",
        "/add": "/v2/add",
        "/add/w1/register": "/v2/add/w1",
        "/pull": "/v2/add",
        "/import": "/v2/add",
        "/profiles/m1": "/v2/models/m1",
        "/profiles/m1/plans": "/v2/models/m1",
        "/profiles/m1/history": "/v2/models/m1",
        "/profiles/m1/tiers": "/v2/models/m1",
        "/profiles/m1/run.sh": "/v2/models/m1",
        "/runtime/logs/m1": "/v2/models/m1",
        "/settings/support-bundle": "/api/v2/settings/support-bundle",
        "/events/jobs/j1": "/api/v2/events",
    }

    def test_every_old_url_permanently_redirects_to_its_v2_equivalent(self):
        for old, new in self.REDIRECTS.items():
            with self.subTest(old=old):
                r = self.client.get(old, headers=self.auth,
                                    follow_redirects=False)
                self.assertEqual(r.status_code, 301, old)
                self.assertEqual(r.headers["location"], new, old)

    # Two targets cannot simply be GET-ed in a test: /api/v2/events is an
    # endless SSE stream, and the support bundle zips the whole install.
    # They are covered by their own tests instead.
    UNFOLLOWABLE = {"/api/v2/events", "/api/v2/settings/support-bundle"}

    def test_no_redirect_target_is_itself_a_redirect(self):
        """A 301 chain is cached forever by browsers; one hop or none."""
        for old, new in self.REDIRECTS.items():
            if new in self.UNFOLLOWABLE:
                continue
            with self.subTest(old=old):
                r = self.client.get(new, headers=self.auth,
                                    follow_redirects=False)
                self.assertNotIn(r.status_code, (301, 302, 303, 307, 308),
                                 f"{old} -> {new} -> another redirect")

    def test_the_unfollowable_targets_are_real_routes(self):
        """Both redirect into endpoints a GET cannot casually exercise; a
        301 into a 404 would be a dead end nobody notices."""
        paths = {r.path for r in self.client.app.routes
                 if hasattr(r, "path")}
        for target in self.UNFOLLOWABLE:
            self.assertIn(target, paths)

    def test_the_support_bundle_moved_rather_than_disappearing(self):
        with mock.patch("modelctl_diagnostics.support_bundle_bytes",
                        return_value=b"PK\x05\x06" + b"\0" * 18) as build:
            r = self.client.get("/api/v2/settings/support-bundle",
                                headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/zip")
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertTrue(build.called)

    def test_the_repo_shims_still_mint_a_wizard_and_keep_the_repo_id(self):
        """These two GETs created a wizard and only then redirected.

        A plain 301 to /v2/add would drop the repo id, so they stayed as
        shims.
        """
        for path in ("/pull/org/model-name", "/add/start/org/model-name"):
            with self.subTest(path=path):
                r = self.client.get(path, headers=self.auth,
                                    follow_redirects=False)
                self.assertEqual(r.status_code, 303)
                self.assertTrue(r.headers["location"].startswith("/v2/add/"))
                wizard_id = r.headers["location"].rsplit("/", 1)[1]
                detail = self.client.get(f"/api/v2/wizard/{wizard_id}",
                                         headers=self.auth).json()
                self.assertEqual(detail["repo_id"], "org/model-name")

    def test_the_pull_search_term_survives_the_redirect(self):
        """A bookmarked /pull?q=... must not land on an empty add page."""
        r = self.client.get("/pull?q=qwen%20gguf", headers=self.auth,
                            follow_redirects=False)
        self.assertEqual(r.status_code, 301)
        self.assertEqual(r.headers["location"], "/v2/add?q=qwen%20gguf")

    def test_old_mutating_posts_are_gone_not_redirected(self):
        """A 301 turns a POST into a GET and the /api/v2 replacements take
        JSON where these took form encoding, so redirecting them would
        silently drop the write."""
        for path in ("/settings/save", "/hardware/save", "/hardware/calibrate",
                     "/profiles/m1", "/profiles/m1/delete", "/profiles/m1/bench",
                     "/models/m1/load", "/models/m1/unload",
                     "/runtime/unload-all", "/setup/reprobe",
                     "/jobs/j1/cancel"):
            with self.subTest(path=path):
                r = self.client.post(path, headers=self.auth,
                                     follow_redirects=False)
                self.assertIn(r.status_code, (404, 405), path)

    def test_the_static_mount_is_gone(self):
        r = self.client.get("/static/htmx-2.0.4.min.js", headers=self.auth,
                            follow_redirects=False)
        self.assertEqual(r.status_code, 404)

    def test_no_route_renders_a_removed_template(self):
        web_dir = Path(create_app.__module__ and
                       __import__("modelctl_web").__file__).parent
        present = {p.name for p in (web_dir / "templates").glob("*.html")}
        self.assertEqual(present, {"base.html", "error.html", "login.html"})

    def test_login_defaults_into_the_console_not_the_dead_dashboard(self):
        r = self.client.get("/login", follow_redirects=False)
        self.assertEqual(r.status_code, 200)
        self.assertIn('value="/v2/"', r.text)

    def test_the_spa_and_its_api_survive(self):
        for path in ("/v2/", "/v2/settings", "/api/v2/models", "/api/v2/jobs",
                     "/api/v2/settings", "/healthz"):
            with self.subTest(path=path):
                r = self.client.get(path, headers=self.auth,
                                    follow_redirects=False)
                self.assertEqual(r.status_code, 200, path)


class TestModelActionsRehomed(SettingsBase):
    """The SPA's operate buttons used to POST to the old console's
    /models/{name}/{verb} routes; the cutover removed them, so the typed
    lane has to carry the same submissions."""

    def test_load_and_unload_answer_with_a_job_id(self):
        for verb, target in (("load", "modelctl_web.mutate.submit_load"),
                             ("unload", "modelctl_web.mutate.submit_unload")):
            with self.subTest(verb=verb):
                with mock.patch(target, return_value=f"job-{verb}") as sub:
                    r = self.client.post(f"/api/v2/models/m1/{verb}",
                                         headers=self.auth)
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(r.json(), {"job_id": f"job-{verb}"})
                    self.assertEqual(sub.call_args[0][1], "m1")


class TestScratchSafeCoversSettings(unittest.TestCase):
    """Every new mutating endpoint has to be refused by a scratch
    instance, like every other write."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.dict(os.environ, {"MODELCTL_WEB_SCRATCH": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        store = JobStore(self.root / "jobs.db", scratch_safe=True)
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        self.client = TestClient(create_app(token=TOKEN, store=store,
                                            runner=runner))
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    def test_settings_writes_are_refused_with_the_reason(self):
        for path, body in (
                ("/api/v2/settings/defaults", {"updates": {"ctx": "4096"}}),
                ("/api/v2/settings/hardware", {"devices": {}}),
                ("/api/v2/settings/hardware/calibrate", {}),
                ("/api/v2/settings/capabilities/reprobe", {}),
                ("/api/v2/settings/token/rotate", {"confirm": True}),
                ("/api/v2/models/m1/load", {}),
                ("/api/v2/models/m1/unload", {}),
        ):
            with self.subTest(path=path):
                r = self.client.post(path, headers=self.auth, json=body)
                self.assertEqual(r.status_code, 405, path)
                self.assertIn("scratch-safe mode", r.json()["reason"])

    def test_settings_reads_behave_exactly_as_live(self):
        r = self.client.get("/api/v2/settings", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertIn("defaults", r.json())


if __name__ == "__main__":
    unittest.main()
