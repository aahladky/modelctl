"""Task C1: first-run readiness, and the setup page built on it."""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_setup
from modelctl_web.app import create_app
from modelctl_web.jobs import JobStore, JobRunner

TOKEN = "test-token"


class SetupBase(unittest.TestCase):
    """A machine with nothing configured, unless a test says otherwise."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.models = self.root / "models"
        self.profiles = self.root / "profiles"
        self.swap_config = self.root / "llama-swap" / "config.yaml"

        for p in (mock.patch.object(modelctl, "DEFAULT_MODELS_DIR", self.models),
                  mock.patch.object(modelctl, "PROFILES_DIR", self.profiles),
                  mock.patch.object(modelctl, "LLAMA_SWAP_CONFIG_PATH",
                                    self.swap_config),
                  # Point llama-swap at a port nothing is listening on, so
                  # the check does not depend on the developer's machine.
                  mock.patch.object(modelctl, "LLAMA_SWAP_BASE_URL",
                                    "http://127.0.0.1:9/v1/")):
            p.start()
            self.addCleanup(p.stop)

    def check(self, status, check_id):
        for c in status.checks:
            if c.id == check_id:
                return c
        self.fail(f"no check with id {check_id!r}")


class TestModelsDirCheck(SetupBase):
    def test_missing_model_dir_blocks(self):
        status = modelctl_setup.probe_setup()
        c = self.check(status, "models_dir")
        self.assertEqual(c.severity, "error")
        self.assertFalse(status.ready)
        self.assertTrue(status.first_run)

    def test_present_model_dir_passes(self):
        self.models.mkdir()
        # Free space is mocked because the temp dir may live on a small
        # tmpfs, which would legitimately trip the low-space warning.
        usage = mock.Mock(free=500 << 30, total=1000 << 30, used=500 << 30)
        with mock.patch("shutil.disk_usage", return_value=usage):
            c = self.check(modelctl_setup.probe_setup(), "models_dir")
        self.assertEqual(c.severity, "ok")

    def test_low_space_warns_without_blocking(self):
        self.models.mkdir()
        usage = mock.Mock(free=1 << 30, total=2 << 30, used=1 << 30)
        with mock.patch("shutil.disk_usage", return_value=usage):
            c = self.check(modelctl_setup.probe_setup(), "models_dir")
        self.assertEqual(c.severity, "warning")


class TestBackendCheck(SetupBase):
    def test_no_binary_blocks(self):
        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN", "/nonexistent/llama-server"), \
             mock.patch.object(modelctl, "find_llama_server_candidates", return_value=[]):
            status = modelctl_setup.probe_setup()
        c = self.check(status, "backend")
        self.assertEqual(c.severity, "error")
        self.assertTrue(c.fix_command)
        self.assertTrue(status.first_run)

    def test_binary_found_passes(self):
        binary = self.root / "llama-server"
        binary.write_text("#!/bin/sh\nexit 1\n")
        binary.chmod(0o755)
        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN", str(binary)), \
             mock.patch.object(modelctl, "find_llama_server_candidates", return_value=[]):
            c = self.check(modelctl_setup.probe_setup(), "backend")
        self.assertEqual(c.severity, "ok")
        self.assertIn(str(binary), c.detail)

    def test_multiple_builds_are_reported(self):
        # Several builds side by side is the normal state; "no MoE cache"
        # must not read as a property of the machine when it is a property
        # of which build happens to be the default.
        first = self.root / "a-server"
        second = self.root / "b-server"
        for b in (first, second):
            b.write_text("#!/bin/sh\nexit 1\n")
            b.chmod(0o755)
        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN", str(first)), \
             mock.patch.object(modelctl, "find_llama_server_candidates",
                               return_value=[str(second)]):
            c = self.check(modelctl_setup.probe_setup(), "backend")
        self.assertIn("other build", c.detail)


class TestLlamaSwapCheck(SetupBase):
    def test_unreachable_swap_warns_but_does_not_block(self):
        self.models.mkdir()
        binary = self.root / "llama-server"
        binary.write_text("#!/bin/sh\nexit 1\n")
        binary.chmod(0o755)
        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN", str(binary)), \
             mock.patch.object(modelctl, "find_llama_server_candidates", return_value=[]):
            status = modelctl_setup.probe_setup()
        c = self.check(status, "llama_swap")
        self.assertEqual(c.severity, "warning")
        # A model can still be added, planned and tested without llama-swap;
        # only registration needs it. That must not present as "blocked".
        self.assertTrue(status.ready)
        self.assertFalse(status.first_run)


class TestSetupPage(SetupBase):
    def web(self):
        store = JobStore(self.root / "jobs.db")
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        client = TestClient(create_app(token=TOKEN, store=store, runner=runner))
        return client, {"Authorization": f"Bearer {TOKEN}"}

    def test_setup_page_lists_every_check(self):
        client, auth = self.web()
        resp = client.get("/setup", headers=auth)
        self.assertEqual(resp.status_code, 200)
        for title in ("model directory", "profile store", "llama.cpp runtime",
                      "llama-swap"):
            self.assertIn(title, resp.text)

    def test_api_setup_reports_readiness(self):
        client, auth = self.web()
        body = client.get("/api/setup", headers=auth).json()
        self.assertFalse(body["ready"])
        self.assertTrue(body["first_run"])
        self.assertTrue(any(c["id"] == "models_dir" for c in body["checks"]))

    def test_first_run_redirects_dashboard_to_setup(self):
        client, auth = self.web()
        resp = client.get("/", headers=auth, follow_redirects=False)
        # 303, matching every other redirect in the app: 307 preserves
        # the method, which is never what a navigation redirect wants.
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(resp.headers["location"], "/setup")

    def test_configured_machine_keeps_its_dashboard(self):
        self.models.mkdir()
        self.profiles.mkdir(parents=True)
        binary = self.root / "llama-server"
        binary.write_text("#!/bin/sh\nexit 1\n")
        binary.chmod(0o755)
        client, auth = self.web()
        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN", str(binary)), \
             mock.patch.object(modelctl, "find_llama_server_candidates", return_value=[]), \
             mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]):
            resp = client.get("/", headers=auth, follow_redirects=False)
        self.assertEqual(resp.status_code, 200)


class TestWebEntryPointCommands(unittest.TestCase):
    def test_console_url_resolves_wildcard_bind(self):
        # 0.0.0.0 is not a URL anyone can open.
        url = modelctl.web_console_url("0.0.0.0:9293")
        self.assertNotIn("0.0.0.0", url)
        self.assertTrue(url.startswith("http://"))
        self.assertTrue(url.endswith(":9293/"))

    def test_console_url_keeps_explicit_host(self):
        self.assertEqual(modelctl.web_console_url("127.0.0.1:8080"),
                         "http://127.0.0.1:8080/")

    def test_install_writes_unit_and_prints_url(self):
        with TemporaryDirectory() as d:
            unit = Path(d) / "systemd" / "user" / "modelctl-web.service"
            calls = []

            def fake_run(cmd, **kw):
                calls.append(cmd)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(modelctl, "WEB_UNIT_PATH", unit), \
                 mock.patch.object(modelctl.subprocess, "run", fake_run), \
                 mock.patch.object(modelctl, "web_token", return_value="tok"), \
                 mock.patch("builtins.print") as pr:
                rc = modelctl.cmd_web_install(
                    mock.Mock(bind="127.0.0.1:9293"))

            self.assertEqual(rc, 0)
            text = unit.read_text()
            self.assertIn("ExecStart=", text)
            self.assertIn("-m modelctl_web", text)
            self.assertIn("MODELCTL_WEB_BIND=127.0.0.1:9293", text)
            self.assertIn(["systemctl", "--user", "daemon-reload"], calls)
            printed = " ".join(str(c.args[0]) for c in pr.call_args_list if c.args)
            self.assertIn("http://127.0.0.1:9293/", printed)
            self.assertIn("tok", printed)

    def test_install_reports_systemctl_failure(self):
        with TemporaryDirectory() as d:
            unit = Path(d) / "modelctl-web.service"
            with mock.patch.object(modelctl, "WEB_UNIT_PATH", unit), \
                 mock.patch.object(modelctl.subprocess, "run",
                                   return_value=mock.Mock(
                                       returncode=1, stdout="", stderr="nope")), \
                 mock.patch.object(modelctl, "web_token", return_value="tok"), \
                 mock.patch("builtins.print"):
                rc = modelctl.cmd_web_install(mock.Mock(bind=None))
        self.assertEqual(rc, 1)


class TestNetworkExposure(unittest.TestCase):
    """P1: the console's reach is always visible -- loopback vs LAN, on
    which address, with the plain-HTTP caveat stated where the user looks."""

    def test_default_bind_is_lan_accessible_and_says_so(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODELCTL_WEB_BIND", None)
            exp = modelctl.web_exposure()
        self.assertEqual(exp["mode"], "LAN-accessible")
        self.assertEqual(exp["bind"], "0.0.0.0:9293")
        self.assertIn("plain HTTP", exp["warning"])
        self.assertIn("unsupported", exp["warning"])

    def test_loopback_bind_reports_loopback_and_no_warning(self):
        exp = modelctl.web_exposure("127.0.0.1:9293")
        self.assertEqual(exp["mode"], "loopback-only")
        self.assertEqual(exp["warning"], "")

    def test_setup_check_warns_on_lan_exposure(self):
        import modelctl_setup
        with mock.patch.dict(os.environ,
                             {"MODELCTL_WEB_BIND": "0.0.0.0:9293"}):
            check = modelctl_setup._check_network_exposure()
        self.assertEqual(check.severity, "warning")
        self.assertIn("LAN-accessible", check.detail)
        self.assertIn("0.0.0.0:9293", check.detail)
        self.assertIn("unsupported", check.fix)
        self.assertIn("127.0.0.1:9293", check.fix_command)

    def test_setup_check_ok_on_loopback(self):
        import modelctl_setup
        with mock.patch.dict(os.environ,
                             {"MODELCTL_WEB_BIND": "127.0.0.1:9293"}):
            check = modelctl_setup._check_network_exposure()
        self.assertEqual(check.severity, "ok")
        self.assertIn("loopback-only", check.detail)

    def test_exposure_check_is_part_of_setup_probe(self):
        import modelctl_setup
        status = modelctl_setup.probe_setup()
        self.assertIn("network_exposure", [c.id for c in status.checks])

    def test_lan_exposure_never_blocks_the_workflow(self):
        # A warning, not an error: LAN binding is the intended personal
        # default, and it must not flip the setup page to "not ready".
        import modelctl_setup
        with mock.patch.dict(os.environ,
                             {"MODELCTL_WEB_BIND": "0.0.0.0:9293"}):
            check = modelctl_setup._check_network_exposure()
        self.assertNotEqual(check.severity, "error")


if __name__ == "__main__":
    unittest.main()
