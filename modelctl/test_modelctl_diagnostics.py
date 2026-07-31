"""Task C4: settings and diagnostics.

The load-bearing behaviour here is redaction and truthful drift
reporting. A support bundle exists to be sent to someone else, so a
credential that leaks into one has leaked permanently.
"""
import io
import json
import subprocess
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_diagnostics as diag
from modelctl_web.app import create_app
from modelctl_web.jobs import JobStore, JobRunner

TOKEN = "test-token"


class TestRedaction(unittest.TestCase):
    def test_credential_named_variables_are_redacted(self):
        out = diag.redact({
            "HF_TOKEN": "hf_realsecret",
            "MY_API_KEY": "abcd",
            "DB_PASSWORD": "hunter2",
            "AUTH_HEADER": "Bearer x",
            "MODELCTL_MODELS_DIR": "/home/u/models",
        })
        self.assertEqual(out["HF_TOKEN"], diag.REDACTED)
        self.assertEqual(out["MY_API_KEY"], diag.REDACTED)
        self.assertEqual(out["DB_PASSWORD"], diag.REDACTED)
        self.assertEqual(out["AUTH_HEADER"], diag.REDACTED)
        self.assertEqual(out["MODELCTL_MODELS_DIR"], "/home/u/models")

    def test_token_shaped_values_are_redacted_whatever_the_name(self):
        # A variable innocently named "SETTING" still must not ship a
        # Hugging Face token.
        out = diag.redact({"SETTING": "hf_abcdef", "OTHER": "sk-abcdef",
                           "FINE": "plain-value"})
        self.assertEqual(out["SETTING"], diag.REDACTED)
        self.assertEqual(out["OTHER"], diag.REDACTED)
        self.assertEqual(out["FINE"], "plain-value")


class TestCommitComparison(unittest.TestCase):
    def test_short_sha_matches_full_sha(self):
        self.assertTrue(diag._commits_agree(
            "2dbf948", "2dbf948015d3476ce94e3fe0c3b38b941a32a685"))

    def test_different_commits_do_not_match(self):
        self.assertFalse(diag._commits_agree(
            "2dbf948", "9b2a08881ffff76ce94e3fe0c3b38b941a32a685"))

    def test_empty_and_stub_values_never_match(self):
        # A blank manifest field must not read as "matches everything".
        self.assertFalse(diag._commits_agree("", ""))
        self.assertFalse(diag._commits_agree("abc", "abc"))
        self.assertFalse(diag._commits_agree("2dbf948", ""))


class ManifestBase(unittest.TestCase):
    """A throwaway git repo with a manifest, so drift can be simulated."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "t"],
                       cwd=self.root, check=True)
        (self.root / "seed").write_text("x")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "seed"], cwd=self.root, check=True)
        self.head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.root,
            capture_output=True, text=True).stdout.strip()
        p = mock.patch.object(diag, "REPO_ROOT", self.root)
        p.start()
        self.addCleanup(p.stop)

    def write_manifest(self, **overrides):
        data = {
            "modelctl_commit": self.head,
            "llama_cpp_commit": "a" * 40,
            "capability_schema": 2,
            "profile_schema": 2,
            "release_channel": "experimental",
        }
        data.update(overrides)
        (self.root / diag.MANIFEST_NAME).write_text(json.dumps(data))
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "manifest"], cwd=self.root, check=True)
        self.head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.root,
            capture_output=True, text=True).stdout.strip()


class TestManifestStatus(ManifestBase):
    def test_missing_manifest_is_reported_not_raised(self):
        status = diag.manifest_status()
        self.assertFalse(status.present)
        self.assertIn("integration-manifest.json", status.error)

    def test_malformed_manifest_is_reported(self):
        (self.root / diag.MANIFEST_NAME).write_text("{not json")
        status = diag.manifest_status()
        self.assertFalse(status.present)
        self.assertIn("unreadable", status.error)

    def test_profile_schema_drift_is_a_mismatch(self):
        # Bumping the schema in code without updating the manifest means
        # the manifest describes a wire format that no longer exists.
        self.write_manifest(profile_schema=99)
        status = diag.manifest_status()
        self.assertTrue(status.present)
        self.assertTrue(any("profile_schema" in m for m in status.mismatches))
        self.assertFalse(status.ok)

    def test_capability_schema_drift_is_a_mismatch(self):
        self.write_manifest(capability_schema=7)
        status = diag.manifest_status()
        self.assertTrue(any("capability_schema" in m for m in status.mismatches))

    def test_control_plane_drift_is_a_note_not_a_mismatch(self):
        # modelctl moving ahead of the manifest is the normal state between
        # runtime promotions. Reporting it as a mismatch would leave the
        # page permanently red and train the user to ignore it.
        self.write_manifest(modelctl_commit="0" * 40)
        status = diag.manifest_status()
        self.assertTrue(any("newer than the last hardware-validated" in n
                            for n in status.notes))
        self.assertFalse(any("modelctl" in m for m in status.mismatches))

    def test_newer_than_validated_is_flagged_with_a_rollback_target(self):
        # The distinction the manifest exists for: which pair was last
        # hardware-validated together, vs what is running now.
        self.write_manifest(validated_modelctl_commit="0" * 40)
        status = diag.manifest_status()
        self.assertTrue(status.newer_than_validated)
        self.assertTrue(any("rollback target" in n for n in status.notes))

    def test_running_the_validated_pair_is_not_flagged(self):
        # Written WITHOUT committing (committing the manifest would move
        # HEAD past the commit it records), so HEAD == validated.
        (self.root / diag.MANIFEST_NAME).write_text(json.dumps({
            "validated_modelctl_commit": self.head,
            "validated_llama_commit": "a" * 40,
            "capability_schema": 2, "profile_schema": 2,
        }))
        status = diag.manifest_status()
        self.assertFalse(status.newer_than_validated)

    def test_validated_fields_fall_back_to_legacy_names(self):
        self.write_manifest(modelctl_commit="1" * 40,
                            llama_cpp_commit="b" * 40,
                            llama_cpp_upstream_base="c" * 9,
                            acceptance_report="docs/report.md")
        status = diag.manifest_status()
        self.assertEqual(status.validated_modelctl_commit, "1" * 40)
        self.assertEqual(status.validated_llama_commit, "b" * 40)
        self.assertEqual(status.upstream_base, "c" * 9)
        self.assertEqual(status.validation_report, "docs/report.md")

    def test_explicit_validated_fields_win_over_legacy(self):
        self.write_manifest(validated_llama_commit="d" * 40,
                            llama_cpp_commit="b" * 40)
        status = diag.manifest_status()
        self.assertEqual(status.validated_llama_commit, "d" * 40)

    def test_dirty_working_tree_is_noted(self):
        self.write_manifest()
        (self.root / "scratch").write_text("uncommitted")
        status = diag.manifest_status()
        self.assertTrue(status.working_tree_dirty)
        self.assertTrue(any("uncommitted" in n for n in status.notes))


class TestSubmodulePinDrift(unittest.TestCase):
    """The pin is what a fresh clone builds; the working tree is what runs.

    ManifestBase has no submodule, so every check that compares the two
    was passing vacuously. Building a real one is the only way to observe
    that they were being read from the same place: `git submodule status`
    reports the *checked-out* commit, so parsing it made the pinned and
    checked-out values identical by construction and the drift mismatch
    could never fire -- while the repo sat two Phase F commits ahead of
    its own pin.
    """

    def _run(self, args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True,
                       capture_output=True, text=True)

    def _init(self, path):
        path.mkdir(parents=True)
        self._run(["init", "-q"], path)
        self._run(["config", "user.email", "t@example.com"], path)
        self._run(["config", "user.name", "t"], path)

    def _commit(self, path, message):
        self._run(["add", "-A"], path)
        self._run(["commit", "-qm", message], path)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                              capture_output=True, text=True).stdout.strip()

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)

        runtime = base / "runtime"
        self._init(runtime)
        (runtime / "src").write_text("promoted")
        self.promoted = self._commit(runtime, "promoted runtime")
        (runtime / "src").write_text("newer")
        self.newer = self._commit(runtime, "newer runtime")

        self.root = base / "umbrella"
        self._init(self.root)
        (self.root / "seed").write_text("x")
        self._commit(self.root, "seed")
        # Local-path submodules need the file transport enabled explicitly
        # on git >= 2.38.
        self._run(["-c", "protocol.file.allow=always", "submodule", "add",
                   "-q", str(runtime), "llama.cpp"], self.root)
        self._run(["checkout", "-q", self.promoted], self.root / "llama.cpp")

        (self.root / diag.MANIFEST_NAME).write_text(json.dumps({
            "modelctl_commit": "0" * 40,   # a note, not a mismatch
            "llama_cpp_commit": self.promoted,
            "capability_schema": 2,
            "profile_schema": 2,
        }))
        self.head = self._commit(self.root, "pin the promoted runtime")

        p = mock.patch.object(diag, "REPO_ROOT", self.root)
        p.start()
        self.addCleanup(p.stop)

    def test_pin_is_read_from_the_commit_tree(self):
        status = diag.manifest_status()
        self.assertEqual(status.submodule_pinned, self.promoted)
        self.assertEqual(status.submodule_checked_out, self.promoted)

    def test_agreeing_pin_and_working_tree_raise_no_mismatch(self):
        status = diag.manifest_status()
        self.assertEqual(status.mismatches, ())
        self.assertTrue(status.ok)

    def test_working_tree_ahead_of_the_pin_is_a_mismatch(self):
        # Exactly the live condition this was written for: the runtime has
        # moved on, the umbrella still pins the promoted commit, and the
        # binary being measured is not the one a clone would build.
        self._run(["checkout", "-q", self.newer], self.root / "llama.cpp")
        status = diag.manifest_status()
        self.assertEqual(status.submodule_pinned, self.promoted)
        self.assertEqual(status.submodule_checked_out, self.newer)
        self.assertTrue(any("working tree is at" in m
                            for m in status.mismatches), status.mismatches)
        self.assertFalse(status.ok)

    def test_drifted_working_tree_is_not_blamed_on_the_manifest(self):
        # The manifest agrees with the pin here, so it must not be
        # implicated -- the remedy is promoting or reverting the runtime,
        # not regenerating the manifest.
        self._run(["checkout", "-q", self.newer], self.root / "llama.cpp")
        status = diag.manifest_status()
        self.assertFalse(any("validated llama.cpp" in m
                             for m in status.mismatches),
                         status.mismatches)

    def test_manifest_behind_the_pin_is_reported_against_the_pin(self):
        stale = self.root / diag.MANIFEST_NAME
        stale.write_text(json.dumps({
            "modelctl_commit": "0" * 40,
            "llama_cpp_commit": self.newer,   # never promoted
            "capability_schema": 2,
            "profile_schema": 2,
        }))
        self._commit(self.root, "stale manifest")
        status = diag.manifest_status()
        self.assertTrue(any("validated llama.cpp" in m
                            for m in status.mismatches), status.mismatches)
        # The message must name the real pin, not the checked-out commit.
        self.assertTrue(any(self.promoted[:12] in m
                            for m in status.mismatches), status.mismatches)


class TestSupportBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        (self.profiles / "m1.json").write_text(json.dumps({
            "name": "m1", "model_path": "/x/m.gguf",
            "config": {"ctx": 8192},
            "env": ["LD_LIBRARY_PATH=/opt/lib", "HF_TOKEN=hf_supersecret"],
        }))
        (self.profiles / "m1").mkdir()
        (self.profiles / "m1" / "llama-swap.log").write_text(
            "\n".join(f"line {i}" for i in range(2000)))
        p = mock.patch.object(modelctl, "PROFILES_DIR", self.profiles)
        p.start()
        self.addCleanup(p.stop)

    def bundle(self):
        return zipfile.ZipFile(io.BytesIO(diag.support_bundle_bytes()))

    def test_bundle_contains_the_diagnostic_documents(self):
        names = self.bundle().namelist()
        for expected in ("integration-manifest.json", "manifest-status.json",
                         "environment.json", "capabilities.json",
                         "hardware.json", "setup.json"):
            self.assertIn(expected, names)

    def test_profile_env_secrets_are_redacted(self):
        z = self.bundle()
        text = z.read("profiles/m1.json").decode()
        self.assertNotIn("hf_supersecret", text)
        self.assertIn(diag.REDACTED, text)
        # Non-secret env entries survive, or the bundle stops being useful.
        self.assertIn("/opt/lib", text)

    def test_bundle_never_contains_the_console_token(self):
        z = self.bundle()
        blob = b"".join(z.read(n) for n in z.namelist())
        self.assertNotIn(b"web_token", blob)

    def test_logs_are_tailed_not_shipped_whole(self):
        z = self.bundle()
        log = z.read("logs/m1-llama-swap.log").decode()
        self.assertIn("line 1999", log)
        self.assertNotIn("line 0\n", log)

    def test_write_support_bundle_creates_a_readable_zip(self):
        out = self.root / "nested" / "bundle.zip"
        diag.write_support_bundle(out)
        self.assertTrue(out.exists())
        self.assertTrue(zipfile.ZipFile(out).namelist())


class TestSettingsPage(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        profiles = self.root / "profiles"
        profiles.mkdir()
        p = mock.patch.object(modelctl, "PROFILES_DIR", profiles)
        p.start()
        self.addCleanup(p.stop)
        store = JobStore(self.root / "jobs.db")
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        self.client = TestClient(create_app(token=TOKEN, store=store, runner=runner))
        self.auth = {"Authorization": f"Bearer {TOKEN}"}

    def test_settings_shows_manifest_and_capabilities(self):
        resp = self.client.get("/settings", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("integration manifest", resp.text)
        self.assertIn("runtime builds and capabilities", resp.text)
        self.assertIn("oneAPI environment", resp.text)
        self.assertIn("support bundle", resp.text)

    def test_api_diagnostics_returns_structured_state(self):
        body = self.client.get("/api/diagnostics", headers=self.auth).json()
        self.assertIn("manifest", body)
        self.assertIn("capabilities", body)
        self.assertIn("environment", body)
        self.assertIn("paths", body["environment"])

    def test_support_bundle_downloads_as_a_zip(self):
        resp = self.client.get("/settings/support-bundle", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "application/zip")
        self.assertIn("attachment", resp.headers["content-disposition"])
        self.assertTrue(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())

    def test_diagnostics_redacts_modelctl_environment(self):
        with mock.patch.dict("os.environ", {"MODELCTL_WEB_TOKEN": "supersecret"}):
            body = self.client.get("/api/diagnostics", headers=self.auth).json()
        env = body["environment"]["modelctl_env"]
        self.assertEqual(env.get("MODELCTL_WEB_TOKEN"), diag.REDACTED)


if __name__ == "__main__":
    unittest.main()
