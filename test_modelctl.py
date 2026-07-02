import argparse
import builtins
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl


class TestGetRouterBaseUrl(unittest.TestCase):
    def test_default_when_no_override(self):
        # Empty string is falsy, so this masks any value set in the real shell.
        with mock.patch.dict("os.environ", {"MODELCTL_ROUTER_BASE_URL": ""}):
            self.assertEqual(
                modelctl.get_router_base_url(),
                "http://127.0.0.1:7071/v1",
            )

    def test_env_override_wins(self):
        with mock.patch.dict("os.environ", {"MODELCTL_ROUTER_BASE_URL": "http://elsewhere:9000/v1"}):
            self.assertEqual(
                modelctl.get_router_base_url(),
                "http://elsewhere:9000/v1/",
            )


class TestRouterRootUrl(unittest.TestCase):
    def test_strips_v1_suffix_with_trailing_slash(self):
        with mock.patch.object(modelctl, "get_router_base_url", return_value="http://host:7071/v1/"):
            self.assertEqual(modelctl.router_root_url(), "http://host:7071/")

    def test_strips_v1_suffix_without_trailing_slash(self):
        with mock.patch.object(modelctl, "get_router_base_url", return_value="http://host:7071/v1"):
            self.assertEqual(modelctl.router_root_url(), "http://host:7071/")

    def test_leaves_non_v1_base_url_alone(self):
        # A custom MODELCTL_ROUTER_BASE_URL override might not use /v1 at all.
        with mock.patch.object(modelctl, "get_router_base_url", return_value="http://host:9000/"):
            self.assertEqual(modelctl.router_root_url(), "http://host:9000/")


class TestRouterStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        (self.profiles_dir / "qwen.json").write_text(json.dumps({
            "name": "qwen",
            "config": {"split_mode": "layer", "tensor_split": "3.5,1"},
        }))
        (self.profiles_dir / "gemma.json").write_text(json.dumps({
            "name": "gemma",
            "config": {"device": "SYCL0"},
        }))

    @staticmethod
    def _fake_response(payload):
        cm = mock.MagicMock()
        cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        return cm

    def test_merges_live_status_with_profile_gpu_placement(self):
        payload = {"data": [
            {"id": "qwen", "status": {"value": "loaded"}, "source": "preset"},
            {"id": "gemma", "status": {"value": "unloaded"}, "source": "preset"},
            {"id": "unsloth/bge-small-en-v1.5-GGUF:F16", "status": {"value": "unloaded"}, "source": "cache"},
        ]}
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", return_value=self._fake_response(payload)):
            rows = modelctl.router_status()
        by_name = {r["name"]: r for r in rows}
        self.assertEqual(by_name["qwen"]["status"], "loaded")
        self.assertEqual(by_name["qwen"]["gpu"], "split 3.5,1 (layer)")
        self.assertEqual(by_name["gemma"]["gpu"], "SYCL0")
        self.assertFalse(by_name["unsloth/bge-small-en-v1.5-GGUF:F16"]["from_preset"])

    def test_flags_failed_models(self):
        payload = {"data": [
            {"id": "gemma", "status": {"value": "unloaded", "failed": True, "exit_code": 1}, "source": "preset"},
        ]}
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", return_value=self._fake_response(payload)):
            rows = modelctl.router_status()
        self.assertTrue(rows[0]["failed"])
        self.assertEqual(rows[0]["exit_code"], 1)

    def test_raises_runtimeerror_when_router_unreachable(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", side_effect=OSError("refused")):
            with self.assertRaises(RuntimeError):
                modelctl.router_status()


class TestRouterLoadUnload(unittest.TestCase):
    @staticmethod
    def _success_response():
        cm = mock.MagicMock()
        cm.__enter__.return_value.read.return_value = json.dumps({"success": True}).encode()
        return cm

    def test_load_success(self):
        with mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", return_value=self._success_response()) as mock_open:
            ok, msg = modelctl.router_load("qwen")
        self.assertTrue(ok)
        req = mock_open.call_args[0][0]
        self.assertEqual(req.full_url, "http://host:7071/models/load")

    def test_unload_success(self):
        with mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", return_value=self._success_response()) as mock_open:
            ok, msg = modelctl.router_unload("qwen")
        self.assertTrue(ok)
        req = mock_open.call_args[0][0]
        self.assertEqual(req.full_url, "http://host:7071/models/unload")

    def test_load_failure_http_error(self):
        err = modelctl.urllib.error.HTTPError(
            url="http://host:7071/models/load", code=400, msg="bad request", hdrs=None,
            fp=io.BytesIO(b'{"error":"model not found"}'),
        )
        with mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", side_effect=err):
            ok, msg = modelctl.router_load("nope")
        self.assertFalse(ok)
        self.assertIn("model not found", msg)

    def test_unload_failure_connection_error(self):
        with mock.patch.object(modelctl, "router_root_url", return_value="http://host:7071/"), \
             mock.patch("modelctl.urllib.request.urlopen", side_effect=OSError("refused")):
            ok, msg = modelctl.router_unload("qwen")
        self.assertFalse(ok)


class TestSyncHermesCustomProviders(unittest.TestCase):
    """Covers the current confirmed on-disk Hermes schema: a `providers:`
    dict keyed by name, each with name/api/models -- not the older
    `custom_providers:` list this function originally targeted. Hermes has
    changed this shape at least twice already; these tests read from the
    live config's actual current structure, not the public docs, since
    those have proven to lag behind what a running Hermes instance does."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        self.hermes_config = Path(self.tmp.name) / "hermes_config.yaml"

        for name, ctx in [("Qwythos-9B-Q4", 64000), ("Qwythos-9B-Q5", 64000), ("gemma4-26b", 64000)]:
            (self.profiles_dir / f"{name}.json").write_text(json.dumps({
                "name": name,
                "config": {"ctx": str(ctx)},
            }))

    def _sync(self, initial_yaml, dry_run=False):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "HERMES_CONFIG", self.hermes_config), \
             mock.patch.object(modelctl, "get_router_base_url", return_value="http://192.168.0.184:7071/v1/"):
            self.hermes_config.write_text(initial_yaml)
            modelctl.sync_hermes_custom_providers(dry_run=dry_run)
        return modelctl.yaml.safe_load(self.hermes_config.read_text())

    def test_syncs_router_provider(self):
        cfg = self._sync("model:\n  default: gemma4-26b\n")
        providers = cfg["providers"]

        expected_models = {
            "Qwythos-9B-Q4": {"context_length": 64000},
            "Qwythos-9B-Q5": {"context_length": 64000},
            "gemma4-26b": {"context_length": 64000},
        }
        self.assertIn("local-router", providers)
        self.assertEqual(providers["local-router"]["api"], "http://192.168.0.184:7071/v1/")
        self.assertEqual(providers["local-router"]["models"], expected_models)

    def test_preserves_unrelated_providers_like_compression_cuda(self):
        initial = (
            "providers:\n"
            "  compression-cuda:\n"
            "    name: compression-cuda\n"
            "    api: http://192.168.0.157:8090/v1/\n"
            "    models:\n"
            "      llama3.2-3b-cuda:\n"
            "        context_length: 65536\n"
        )
        cfg = self._sync(initial)
        self.assertIn("compression-cuda", cfg["providers"])
        self.assertEqual(
            cfg["providers"]["compression-cuda"]["models"],
            {"llama3.2-3b-cuda": {"context_length": 65536}},
        )

    def test_preserves_extra_fields_on_existing_provider(self):
        initial = (
            "providers:\n"
            "  local-swap:\n"
            "    name: local-swap\n"
            "    api: http://192.168.0.184:7070/v1/\n"
            "    transport: chat_completions\n"
            "    models: {}\n"
        )
        cfg = self._sync(initial)
        self.assertEqual(cfg["providers"]["local-swap"]["transport"], "chat_completions")

    def test_preserves_extra_per_model_fields(self):
        initial = (
            "providers:\n"
            "  local-router:\n"
            "    name: local-router\n"
            "    api: http://192.168.0.184:7071/v1/\n"
            "    models:\n"
            "      gemma4-26b:\n"
            "        context_length: 32000\n"
            "        reasoning_effort: medium\n"
        )
        cfg = self._sync(initial)
        gemma_entry = cfg["providers"]["local-router"]["models"]["gemma4-26b"]
        self.assertEqual(gemma_entry["reasoning_effort"], "medium")
        self.assertEqual(gemma_entry["context_length"], 64000)  # updated from the profile

    def test_removed_profile_drops_out_of_router_provider(self):
        # An unrelated provider (e.g. a leftover llama-swap entry) is the
        # user's to manage -- only the router provider is rebuilt.
        initial = (
            "providers:\n"
            "  local-swap:\n"
            "    name: local-swap\n"
            "    api: http://192.168.0.184:7070/v1/\n"
            "    models:\n"
            "      some-deleted-model:\n"
            "        context_length: 8192\n"
            "  local-router:\n"
            "    name: local-router\n"
            "    api: http://192.168.0.184:7071/v1/\n"
            "    models:\n"
            "      some-deleted-model:\n"
            "        context_length: 8192\n"
        )
        cfg = self._sync(initial)
        self.assertNotIn("some-deleted-model", cfg["providers"]["local-router"]["models"])
        self.assertIn("some-deleted-model", cfg["providers"]["local-swap"]["models"])

    def test_migrates_legacy_custom_providers_list_format(self):
        initial = (
            "custom_providers:\n"
            "  - name: LocalLlama\n"
            "    base_url: http://192.168.0.184:7070/v1/\n"
            "    models:\n"
            "      gemma4-26b:\n"
            "        context_length: 32000\n"
        )
        cfg = self._sync(initial)
        self.assertNotIn("custom_providers", cfg)
        self.assertIn("LocalLlama", cfg["providers"])
        self.assertEqual(cfg["providers"]["LocalLlama"]["api"], "http://192.168.0.184:7070/v1/")

    def test_dry_run_does_not_write(self):
        original = "model:\n  default: gemma4-26b\n"
        self.hermes_config.write_text(original)
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "HERMES_CONFIG", self.hermes_config), \
             mock.patch.object(modelctl, "get_router_base_url", return_value="http://192.168.0.184:7071/v1/"):
            modelctl.sync_hermes_custom_providers(dry_run=True)
        self.assertEqual(self.hermes_config.read_text(), original)


class TestBuildServerArgs(unittest.TestCase):
    def test_jinja_and_parallel_always_present(self):
        profile = {
            "model_path": "/home/aaron/models/test.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto",
                "ctx": "64000",
                "split_mode": "layer",
                "tensor_split": "4,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "",
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertIn("--jinja", args)
        self.assertIn("--parallel", args)
        self.assertEqual(args[args.index("--parallel") + 1], "1")

    def test_mtp_on_adds_spec_type_draft_mtp(self):
        profile = {
            "model_path": "/home/aaron/models/test.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto",
                "ctx": "64000",
                "split_mode": "layer",
                "tensor_split": "4,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "",
                "mtp": "on",
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertIn("--spec-type", args)
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")

    def test_mtp_off_by_default_omits_spec_type(self):
        profile = {
            "model_path": "/home/aaron/models/test.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto",
                "ctx": "64000",
                "split_mode": "layer",
                "tensor_split": "4,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "",
                # no "mtp" key at all -- matches every profile saved before this feature existed
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertNotIn("--spec-type", args)

    def test_mtp_on_bundled_omits_spec_draft_model(self):
        """Most current MTP GGUFs (Qwen3.5/3.6) bundle the draft heads in the
        same file as the main model -- no companion file, so no mtp_path."""
        profile = {
            "model_path": "/home/aaron/models/test.gguf",
            "mmproj_path": None,
            "mtp_path": None,
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600",
                "extra": "", "mtp": "on",
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertIn("--spec-type", args)
        self.assertNotIn("--spec-draft-model", args)

    def test_mtp_on_with_companion_file_adds_spec_draft_model(self):
        """Gemma-style MTP ships the draft heads as a separate companion
        GGUF -- llama-server needs --spec-draft-model pointed at it, in
        addition to --spec-type draft-mtp."""
        profile = {
            "model_path": "/home/aaron/models/gemma4-26b.gguf",
            "mmproj_path": None,
            "mtp_path": "/home/aaron/models/gemma4-26b-mtp.gguf",
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600",
                "extra": "", "mtp": "on",
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertIn("--spec-type", args)
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertIn("--spec-draft-model", args)
        self.assertEqual(args[args.index("--spec-draft-model") + 1], "/home/aaron/models/gemma4-26b-mtp.gguf")

    def test_mtp_off_with_companion_file_present_still_omits_both_flags(self):
        """Having mtp_path saved on the profile doesn't mean it's active --
        the mtp on/off toggle is still the deciding switch."""
        profile = {
            "model_path": "/home/aaron/models/gemma4-26b.gguf",
            "mmproj_path": None,
            "mtp_path": "/home/aaron/models/gemma4-26b-mtp.gguf",
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600",
                "extra": "", "mtp": "off",
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertNotIn("--spec-type", args)
        self.assertNotIn("--spec-draft-model", args)


class TestRenderRouterPreset(unittest.TestCase):
    def test_emits_ini_section_with_device_and_ctx(self):
        profile = {
            "name": "Qwythos-9B-Q4",
            "model_path": "/home/aaron/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
            "mmproj_path": "/home/aaron/models/mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf",
            "config": {
                "flash_attn": "auto",
                "ctx": "64000",
                "split_mode": "layer",
                "tensor_split": "4,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "",
            },
        }
        text, ok, messages = modelctl.render_router_preset(profile)
        self.assertIn("[Qwythos-9B-Q4]", text)
        self.assertIn("ctx-size = 64000", text)
        self.assertIn("split-mode = layer", text)
        self.assertIn("tensor-split = 4,1", text)
        self.assertIn("jinja = true", text)
        self.assertIn("ngl = 999", text)
        self.assertIn("mmproj = /home/aaron/models/mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf", text)

    def test_emits_extra_args_line_when_extra_configured(self):
        profile = {
            "name": "Qwythos-9B-Q4",
            "model_path": "/home/aaron/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto",
                "ctx": "64000",
                "split_mode": "layer",
                "tensor_split": "4,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "--some-flag value",
            },
        }
        text, ok, messages = modelctl.render_router_preset(profile)
        self.assertIn("extra-args = --some-flag value", text)

    def test_mtp_on_emits_spec_type_line(self):
        profile = {
            "name": "llama3.2-3b",
            "model_path": "/home/aaron/models/llama32-3b.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto",
                "ctx": "128000",
                "split_mode": "layer",
                "tensor_split": "3,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "",
                "mtp": "on",
            },
        }
        text, ok, messages = modelctl.render_router_preset(profile)
        self.assertIn("spec-type = draft-mtp", text)

    def test_mtp_off_by_default_omits_spec_type_line(self):
        profile = {
            "name": "llama3.2-3b",
            "model_path": "/home/aaron/models/llama32-3b.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto",
                "ctx": "128000",
                "split_mode": "layer",
                "tensor_split": "3,1",
                "kv_quant": "q8_0",
                "ttl": "3600",
                "extra": "",
            },
        }
        text, ok, messages = modelctl.render_router_preset(profile)
        self.assertNotIn("spec-type", text)

    def test_mtp_on_with_companion_file_emits_spec_draft_model_line(self):
        profile = {
            "name": "gemma4-26b",
            "model_path": "/home/aaron/models/gemma4-26b.gguf",
            "mmproj_path": None,
            "mtp_path": "/home/aaron/models/gemma4-26b-mtp.gguf",
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600",
                "extra": "", "mtp": "on",
            },
        }
        text, ok, messages = modelctl.render_router_preset(profile)
        self.assertIn("spec-type = draft-mtp", text)
        self.assertIn("spec-draft-model = /home/aaron/models/gemma4-26b-mtp.gguf", text)

    def test_mtp_on_bundled_omits_spec_draft_model_line(self):
        profile = {
            "name": "qwen36-35b",
            "model_path": "/home/aaron/models/qwen36-35b.gguf",
            "mmproj_path": None,
            "mtp_path": None,
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600",
                "extra": "", "mtp": "on",
            },
        }
        text, ok, messages = modelctl.render_router_preset(profile)
        self.assertIn("spec-type = draft-mtp", text)
        self.assertNotIn("spec-draft-model", text)


class TestGenerateArtifactsRunSh(unittest.TestCase):
    """Regression test: args_to_shell_line() returns ONE joined string, but
    generate_artifacts() was doing `" \\\n  ".join(args_to_shell_line(args))`
    -- .join() on a string iterates its characters, so every argument was
    getting split one character per line in the generated run.sh."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()

    def test_run_sh_keeps_multi_character_flags_intact(self):
        profile = {
            "name": "test-profile",
            "model_path": "/home/aaron/models/test.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600", "extra": "",
            },
        }
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])):
            modelctl.generate_artifacts(profile)

        run_sh = (self.profiles_dir / "test-profile" / "run.sh").read_text()
        # a correctly-joined script contains "--model" as one token; the bug
        # produced "- \\\n  - \\\n  m \\\n  o \\\n  d ..." instead.
        self.assertIn("--model", run_sh)
        self.assertNotIn("- \\\n  - \\\n  m", run_sh)
        self.assertIn("/home/aaron/models/test.gguf", run_sh)


class TestSyncRouterPreset(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        self.router_path = Path(self.tmp.name) / "router.preset.ini"

        (self.profiles_dir / "Qwythos-9B-Q4.json").write_text(json.dumps({
            "name": "Qwythos-9B-Q4",
            "model_path": "/home/aaron/models/q4.gguf",
            "mmproj_path": None,
            "config": {
                "flash_attn": "auto", "ctx": "64000", "split_mode": "layer",
                "tensor_split": "4,1", "kv_quant": "q8_0", "ttl": "3600", "extra": "",
            },
        }))

    def test_writes_all_profile_sections(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "ROUTER_PRESET_PATH", self.router_path), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch.object(modelctl, "restart_router_service"):
            modelctl.sync_router_preset()

        content = self.router_path.read_text()
        self.assertIn("[Qwythos-9B-Q4]", content)

    def test_backs_up_existing_preset_before_overwrite(self):
        self.router_path.write_text("[stale-old-content]\n")

        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "ROUTER_PRESET_PATH", self.router_path), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch.object(modelctl, "restart_router_service"):
            modelctl.sync_router_preset()

        backup_path = self.router_path.with_suffix(self.router_path.suffix + ".bak")
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.read_text(), "[stale-old-content]\n")
        self.assertIn("[Qwythos-9B-Q4]", self.router_path.read_text())

    def test_restarts_router_service_when_preset_changes(self):
        """Router mode has no --watch-config hot-reload, so a preset rewrite
        is a no-op until something restarts the process. That restart should
        happen automatically -- the whole point is one less manual step."""
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "ROUTER_PRESET_PATH", self.router_path), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch.object(modelctl, "restart_router_service") as mock_restart:
            modelctl.sync_router_preset()
        mock_restart.assert_called_once()

    def test_does_not_restart_when_preset_unchanged(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "ROUTER_PRESET_PATH", self.router_path), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch.object(modelctl, "restart_router_service") as mock_restart:
            modelctl.sync_router_preset()  # first call writes it
            mock_restart.reset_mock()
            modelctl.sync_router_preset()  # second call: no change, no restart
        mock_restart.assert_not_called()

    def test_restart_false_skips_restart_even_when_changed(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "ROUTER_PRESET_PATH", self.router_path), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch.object(modelctl, "restart_router_service") as mock_restart:
            modelctl.sync_router_preset(restart=False)
        mock_restart.assert_not_called()


class TestRestartRouterService(unittest.TestCase):
    def test_returns_true_on_successful_restart(self):
        fake_result = mock.Mock(returncode=0, stderr="")
        with mock.patch.object(modelctl.subprocess, "run", return_value=fake_result) as mock_run:
            self.assertTrue(modelctl.restart_router_service())
        mock_run.assert_called_once_with(
            ["systemctl", "--user", "restart", modelctl.ROUTER_SERVICE_NAME],
            capture_output=True, text=True, timeout=30,
        )

    def test_returns_false_when_systemctl_exits_nonzero(self):
        fake_result = mock.Mock(returncode=1, stderr="Unit llama-router.service not found.")
        with mock.patch.object(modelctl.subprocess, "run", return_value=fake_result):
            self.assertFalse(modelctl.restart_router_service())

    def test_returns_false_when_systemctl_missing(self):
        with mock.patch.object(modelctl.subprocess, "run", side_effect=FileNotFoundError()):
            self.assertFalse(modelctl.restart_router_service())

    def test_returns_false_on_timeout(self):
        with mock.patch.object(modelctl.subprocess, "run",
                                side_effect=modelctl.subprocess.TimeoutExpired(cmd="systemctl", timeout=30)):
            self.assertFalse(modelctl.restart_router_service())


class TestRepoFileSizes(unittest.TestCase):
    def setUp(self):
        # _repo_file_sizes is lru_cached per repo_id; clear it so each test's
        # mocked api.model_info is actually consulted.
        modelctl._repo_file_sizes.cache_clear()

    def test_returns_size_map_from_model_info(self):
        fake_sibling_a = mock.Mock(rfilename="model-Q4_K_M.gguf", size=1000)
        fake_sibling_b = mock.Mock(rfilename="mmproj-F16.gguf", size=500)
        fake_info = mock.Mock(siblings=[fake_sibling_a, fake_sibling_b])
        with mock.patch.object(modelctl.api, "model_info", return_value=fake_info):
            sizes = modelctl._repo_file_sizes("some/repo")
        self.assertEqual(sizes, {"model-Q4_K_M.gguf": 1000, "mmproj-F16.gguf": 500})

    def test_returns_empty_dict_on_api_failure(self):
        with mock.patch.object(modelctl.api, "model_info", side_effect=Exception("network error")):
            self.assertEqual(modelctl._repo_file_sizes("some/repo"), {})

    def test_skips_siblings_with_no_size(self):
        fake_sibling = mock.Mock(rfilename="README.md", size=None)
        fake_info = mock.Mock(siblings=[fake_sibling])
        with mock.patch.object(modelctl.api, "model_info", return_value=fake_info):
            self.assertEqual(modelctl._repo_file_sizes("some/repo"), {})


class TestGroupFiles(unittest.TestCase):
    def test_strips_directory_prefix_from_label(self):
        """unsloth/Qwen3.5-35B-A3B-GGUF ships a BF16/ subfolder -- the label
        must not leak that path component, since it's used both for display
        and as the stored profile 'file' field."""
        groups = modelctl.group_files(["BF16/Qwen3.5-35B-A3B-BF16.gguf"])
        self.assertEqual(groups[0]["label"], "Qwen3.5-35B-A3B-BF16")

    def test_sharded_files_in_subfolder_also_strip_prefix(self):
        groups = modelctl.group_files([
            "BF16/model-00001-of-00002.gguf",
            "BF16/model-00002-of-00002.gguf",
        ])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["label"], "model")

    def test_flat_files_unaffected(self):
        groups = modelctl.group_files(["model-Q4_K_M.gguf"])
        self.assertEqual(groups[0]["label"], "model-Q4_K_M")


class TestGetRepoContents(unittest.TestCase):
    def test_separates_quants_mmproj_and_mtp_with_sizes(self):
        files = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "mmproj-F16.gguf",
            "model-mtp.gguf",
        ]
        sizes = {
            "model-Q4_K_M.gguf": 1000,
            "model-Q8_0.gguf": 2000,
            "mmproj-F16.gguf": 500,
            "model-mtp.gguf": 100,
        }
        with mock.patch.object(modelctl, "list_gguf_files", return_value=files), \
             mock.patch.object(modelctl, "_repo_file_sizes", return_value=sizes):
            contents = modelctl.get_repo_contents("some/repo")

        quant_labels = {g["label"] for g in contents["quant_groups"]}
        self.assertEqual(quant_labels, {"model-Q4_K_M", "model-Q8_0"})
        self.assertEqual(contents["mmproj_files"], [{"name": "mmproj-F16.gguf", "size": 500}])
        self.assertEqual(contents["mtp_files"], [{"name": "model-mtp.gguf", "size": 100}])

    def test_quant_group_total_size_sums_all_shards(self):
        files = [
            "model-Q4_K_M-00001-of-00002.gguf",
            "model-Q4_K_M-00002-of-00002.gguf",
        ]
        sizes = {
            "model-Q4_K_M-00001-of-00002.gguf": 1000,
            "model-Q4_K_M-00002-of-00002.gguf": 1500,
        }
        with mock.patch.object(modelctl, "list_gguf_files", return_value=files), \
             mock.patch.object(modelctl, "_repo_file_sizes", return_value=sizes):
            contents = modelctl.get_repo_contents("some/repo")

        self.assertEqual(len(contents["quant_groups"]), 1)
        self.assertEqual(contents["quant_groups"][0]["total_size"], 2500)

    def test_missing_size_data_yields_none_total(self):
        with mock.patch.object(modelctl, "list_gguf_files", return_value=["model-Q4_K_M.gguf"]), \
             mock.patch.object(modelctl, "_repo_file_sizes", return_value={}):
            contents = modelctl.get_repo_contents("some/repo")
        self.assertIsNone(contents["quant_groups"][0]["total_size"])


class TestQuantSuffixes(unittest.TestCase):
    def test_strips_common_prefix_between_same_family_quants(self):
        groups = [{"label": "Qwen3.5-35B-A3B-UD-Q4_K_M"}, {"label": "Qwen3.5-35B-A3B-UD-Q8_0"}]
        self.assertEqual(modelctl._quant_suffixes(groups), ["Q4_K_M", "Q8_0"])

    def test_preserves_ud_as_a_distinguishing_variant(self):
        """The real case this exists for: Qwen3.5-35B-A3B-GGUF ships both a
        plain Q4_K_M and an Unsloth-Dynamic UD-Q4_K_M -- these must not
        collapse to the same displayed label."""
        groups = [{"label": "Qwen3.5-35B-A3B-Q4_K_M"}, {"label": "Qwen3.5-35B-A3B-UD-Q4_K_M"}]
        self.assertEqual(modelctl._quant_suffixes(groups), ["Q4_K_M", "UD-Q4_K_M"])

    def test_single_label_uses_strip_quant_from_label(self):
        groups = [{"label": "Qwen3.5-35B-A3B-Q4_K_M"}]
        self.assertEqual(modelctl._quant_suffixes(groups), ["Q4_K_M"])

    def test_empty_list(self):
        self.assertEqual(modelctl._quant_suffixes([]), [])


class TestSearchModels(unittest.TestCase):
    def _fake_hit(self, repo_id, downloads=100, likes=5, tags=None):
        return mock.Mock(id=repo_id, downloads=downloads, likes=likes, tags=tags or ["gguf"])

    def test_basic_search_returns_plain_results_without_contents_by_default_disabled(self):
        hits = [self._fake_hit("unsloth/model-a-GGUF")]
        with mock.patch.object(modelctl.api, "list_models", return_value=hits):
            results = modelctl.search_models("model", enrich=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["repo_id"], "unsloth/model-a-GGUF")
        self.assertIsNone(results[0]["contents"])

    def test_enrich_true_populates_contents(self):
        hits = [self._fake_hit("unsloth/model-a-GGUF")]
        fake_contents = {"quant_groups": [], "mmproj_files": [], "mtp_files": []}
        with mock.patch.object(modelctl.api, "list_models", return_value=hits), \
             mock.patch.object(modelctl, "get_repo_contents", return_value=fake_contents):
            results = modelctl.search_models("model", enrich=True)
        self.assertEqual(results[0]["contents"], fake_contents)

    def test_mtp_tag_filters_out_non_mtp_repos(self):
        hits = [
            self._fake_hit("unsloth/model-a-GGUF"),
            self._fake_hit("unsloth/model-a-MTP-GGUF"),
        ]
        with mock.patch.object(modelctl.api, "list_models", return_value=hits):
            results = modelctl.search_models("model", tags=["mtp"], enrich=False)
        self.assertEqual([r["repo_id"] for r in results], ["unsloth/model-a-MTP-GGUF"])

    def test_size_filter_keeps_only_repos_with_a_quant_in_range(self):
        hits = [
            self._fake_hit("repo/small"),
            self._fake_hit("repo/big"),
        ]
        small_contents = {"quant_groups": [{"label": "x", "total_size": 2 * 1024**3}]}
        big_contents = {"quant_groups": [{"label": "x", "total_size": 20 * 1024**3}]}

        def fake_get_contents(repo_id):
            return small_contents if repo_id == "repo/small" else big_contents

        with mock.patch.object(modelctl.api, "list_models", return_value=hits), \
             mock.patch.object(modelctl, "get_repo_contents", side_effect=fake_get_contents):
            results = modelctl.search_models("x", min_gb=15, max_gb=25)

        self.assertEqual([r["repo_id"] for r in results], ["repo/big"])


class TestDownloadIfNeeded(unittest.TestCase):
    """Regression coverage for the cross-repo filename collision: two
    different HF repos can both ship a file named e.g. 'mmproj-F16.gguf'.
    Downloads are namespaced under dest_dir/<repo_id>/ so two repos sharing
    a filename can never land at the same path -- previously they did,
    which meant a same-sized coincidence could serve the wrong file, and a
    different-sized one would overwrite an earlier repo's file on disk out
    from under any profile still pointing at it."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest_dir = Path(self.tmp.name)

    def test_skips_when_local_size_matches_remote(self):
        target = self.dest_dir / "repo/a" / "mmproj-F16.gguf"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 100)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=100), \
             mock.patch.object(modelctl, "hf_hub_download") as mock_dl:
            result = modelctl.download_if_needed("repo/a", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_not_called()
        self.assertEqual(result, str(target))

    def test_different_repos_with_same_filename_never_share_a_path(self):
        """The actual bug this fixes: repo/a's mmproj-F16.gguf must survive
        untouched, and repo/b's pull of a same-named file must land
        somewhere else entirely -- not merely trigger a same-path
        re-download that clobbers repo/a's copy."""
        target_a = self.dest_dir / "repo/a" / "mmproj-F16.gguf"
        target_a.parent.mkdir(parents=True)
        target_a.write_bytes(b"a" * 100)

        def fake_download(repo_id, filename, local_dir):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"b" * 900)
            return str(path)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=900), \
             mock.patch.object(modelctl, "hf_hub_download", side_effect=fake_download) as mock_dl:
            result_b = modelctl.download_if_needed("repo/b", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_called_once_with(
            repo_id="repo/b", filename="mmproj-F16.gguf", local_dir=str(self.dest_dir / "repo/b"),
        )
        self.assertNotEqual(str(target_a), result_b)
        self.assertEqual(target_a.read_bytes(), b"a" * 100)  # untouched
        self.assertEqual(Path(result_b).read_bytes(), b"b" * 900)

    def test_redownloads_when_local_size_mismatches_remote(self):
        """Re-pulling the SAME repo after a truncated/partial download
        still redownloads, in place, for that repo's own subdirectory."""
        target = self.dest_dir / "repo/a" / "mmproj-F16.gguf"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 100)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=899283648), \
             mock.patch.object(modelctl, "hf_hub_download", return_value=str(target)) as mock_dl:
            modelctl.download_if_needed("repo/a", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_called_once_with(
            repo_id="repo/a", filename="mmproj-F16.gguf", local_dir=str(self.dest_dir / "repo/a"),
        )

    def test_falls_back_to_trusting_local_file_when_remote_size_unknown(self):
        """If the HF API call fails (offline, rate-limited, etc.) this
        should degrade to the old behavior rather than force a redundant
        multi-GB re-download every time the network hiccups."""
        target = self.dest_dir / "repo/a" / "mmproj-F16.gguf"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 100)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=None), \
             mock.patch.object(modelctl, "hf_hub_download") as mock_dl:
            result = modelctl.download_if_needed("repo/a", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_not_called()
        self.assertEqual(result, str(target))

    def test_downloads_when_file_not_present_at_all(self):
        with mock.patch.object(modelctl, "hf_hub_download", return_value="/downloaded/path") as mock_dl:
            result = modelctl.download_if_needed("repo/a", "model.gguf", self.dest_dir)

        mock_dl.assert_called_once_with(
            repo_id="repo/a", filename="model.gguf", local_dir=str(self.dest_dir / "repo/a"),
        )
        self.assertEqual(result, "/downloaded/path")


class TestPullTuiFlag(unittest.TestCase):
    def test_tui_flag_makes_repo_id_optional(self):
        parser = modelctl.build_arg_parser()
        args = parser.parse_args(["pull", "--tui"])
        self.assertTrue(args.tui)
        self.assertIsNone(args.repo_id)

    def test_repo_id_still_required_without_tui(self):
        parser = modelctl.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["pull"])

    def test_cmd_pull_dispatches_to_tui_when_flag_set(self):
        args = argparse.Namespace(tui=True, repo_id=None, no_hermes=False, no_router_restart=False)
        with mock.patch.object(modelctl, "run_pull_wizard") as mock_wizard:
            modelctl.cmd_pull(args)
        mock_wizard.assert_called_once_with(no_hermes=False, no_router_restart=False)

    def test_cmd_pull_passes_no_hermes_and_no_router_restart_to_wizard(self):
        # Regression test for the silent-ignore bug: `modelctl pull --tui
        # --no-hermes --no-router-restart` must actually thread those flags
        # through to run_pull_wizard(), not silently drop them.
        args = argparse.Namespace(tui=True, repo_id=None, no_hermes=True, no_router_restart=True)
        with mock.patch.object(modelctl, "run_pull_wizard") as mock_wizard:
            modelctl.cmd_pull(args)
        mock_wizard.assert_called_once_with(no_hermes=True, no_router_restart=True)

    def test_cmd_pull_tui_flags_default_false_when_absent(self):
        # getattr(args, ..., False) fallback -- args objects that lack these
        # attributes entirely (shouldn't happen via the real parser, but
        # cheap to guard) must not crash cmd_pull.
        args = argparse.Namespace(tui=True, repo_id=None)
        with mock.patch.object(modelctl, "run_pull_wizard") as mock_wizard:
            modelctl.cmd_pull(args)
        mock_wizard.assert_called_once_with(no_hermes=False, no_router_restart=False)

    def test_run_pull_wizard_constructs_app_with_flags(self):
        mock_app_instance = mock.MagicMock()
        with mock.patch("modelctl_tui.PullWizardApp", return_value=mock_app_instance) as mock_app_cls:
            modelctl.run_pull_wizard(no_hermes=True, no_router_restart=True)
        mock_app_cls.assert_called_once_with(no_hermes=True, no_router_restart=True)
        mock_app_instance.run.assert_called_once()

    def test_run_pull_wizard_defaults_to_false(self):
        mock_app_instance = mock.MagicMock()
        with mock.patch("modelctl_tui.PullWizardApp", return_value=mock_app_instance) as mock_app_cls:
            modelctl.run_pull_wizard()
        mock_app_cls.assert_called_once_with(no_hermes=False, no_router_restart=False)
        mock_app_instance.run.assert_called_once()

    def test_run_pull_wizard_shows_friendly_message_when_textual_missing(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "modelctl_tui":
                # Python's real import machinery raises ModuleNotFoundError
                # (a subclass of ImportError) when a module genuinely can't
                # be found, as opposed to a plain ImportError for other
                # reasons (e.g. a name that doesn't exist inside an
                # otherwise-importable module).
                raise ModuleNotFoundError("No module named 'textual'", name="textual")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                with self.assertRaises(SystemExit):
                    modelctl.run_pull_wizard()
        self.assertIn("requires the 'textual' package", mock_stderr.getvalue())

    def test_run_pull_wizard_reraises_unrelated_import_errors(self):
        # Regression test: an ImportError raised *inside* modelctl_tui.py for
        # a reason unrelated to textual being missing (e.g. a bug in that
        # module) must not be swallowed and misreported as "install textual".
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "modelctl_tui":
                raise ImportError(
                    "cannot import name 'PullWizardApp' from 'modelctl_tui'",
                    name="modelctl_tui",
                )
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError) as ctx:
                modelctl.run_pull_wizard()
        self.assertEqual(ctx.exception.name, "modelctl_tui")

    def test_run_pull_wizard_reraises_plain_importerror_named_like_textual(self):
        # Regression test: a plain ImportError (NOT ModuleNotFoundError) can
        # also have e.name == "textual.widgets" -- e.g. `from textual.widgets
        # import NonExistentWidgetTypo` (a typo, or an API renamed between
        # textual versions) raises ImportError with e.name == "textual.widgets"
        # even though textual itself IS installed. Checking e.name alone
        # would wrongly match this and misreport it as "textual isn't
        # installed" instead of surfacing the real error. Only a genuine
        # ModuleNotFoundError for textual should get the friendly message.
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "modelctl_tui":
                raise ImportError(
                    "cannot import name 'NonExistentWidgetTypo' from 'textual.widgets'",
                    name="textual.widgets",
                )
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError) as ctx:
                modelctl.run_pull_wizard()
        self.assertNotIsInstance(ctx.exception, ModuleNotFoundError)
        self.assertEqual(ctx.exception.name, "textual.widgets")


class TestVramDefaults(unittest.TestCase):
    def test_vram_keys_present_with_defaults(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")):
            d = modelctl.load_defaults()
        self.assertEqual(d["vram_limit_pct"], 90)
        self.assertEqual(d["primary_gpu"], "")

    def test_env_override(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_VRAM_LIMIT_PCT": "80",
                                            "MODELCTL_DEFAULT_PRIMARY_GPU": "SYCL1"}):
            d = modelctl.load_defaults()
        self.assertEqual(d["vram_limit_pct"], 80)
        self.assertEqual(d["primary_gpu"], "SYCL1")


class TestLocalWeightsBytes(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_single_file(self):
        p = self.dir / "model.gguf"
        p.write_bytes(b"x" * 100)
        self.assertEqual(modelctl._local_weights_bytes(p), 100)

    def test_sharded_sums_all_parts(self):
        for i in (1, 2, 3):
            (self.dir / f"model-0000{i}-of-00003.gguf").write_bytes(b"x" * 10 * i)
        first = self.dir / "model-00001-of-00003.gguf"
        self.assertEqual(modelctl._local_weights_bytes(first), 60)

    def test_prefix_sibling_model_not_counted(self):
        for i in (1, 2):
            (self.dir / f"model-0000{i}-of-00002.gguf").write_bytes(b"x" * 10)
            (self.dir / f"model-instruct-0000{i}-of-00002.gguf").write_bytes(b"y" * 100)
        first = self.dir / "model-00001-of-00002.gguf"
        self.assertEqual(modelctl._local_weights_bytes(first), 20)


class TestEstimateVramFootprint(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "model.gguf"
        self.model.write_bytes(b"x" * 1000)

    def test_missing_model_returns_none(self):
        profile = {"model_path": str(Path(self.tmp.name) / "gone.gguf"),
                   "config": {"ctx": 4096, "kv_quant": "q8_0"}}
        self.assertIsNone(modelctl.estimate_vram_footprint(profile))

    def test_heuristic_when_header_unparseable(self):
        profile = {"model_path": str(self.model),
                   "config": {"ctx": 4096, "kv_quant": "q8_0"}}
        est = modelctl.estimate_vram_footprint(profile)
        self.assertEqual(est["quality"], "heuristic")
        self.assertEqual(est["weights"], 1000)

    def test_mmproj_counted(self):
        mmproj = Path(self.tmp.name) / "mmproj.gguf"
        mmproj.write_bytes(b"y" * 500)
        profile = {"model_path": str(self.model), "mmproj_path": str(mmproj),
                   "config": {"ctx": 4096, "kv_quant": "q8_0"}}
        est = modelctl.estimate_vram_footprint(profile)
        self.assertEqual(est["weights"], 1500)


class TestGetGpuInventory(unittest.TestCase):
    XPU = [
        {"xpu_id": 0, "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
        {"xpu_id": 1, "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.map_path = Path(self.tmp.name) / "gpu_map.json"

    def test_no_xpu_smi_returns_empty(self):
        with mock.patch.object(modelctl.modelctl_vram, "xpu_devices", return_value=[]):
            self.assertEqual(modelctl.get_gpu_inventory(), [])

    def test_builds_and_caches_map(self):
        sycl = [{"device": "SYCL0", "name": "big", "total_mib": 32657},
                {"device": "SYCL1", "name": "small", "total_mib": 12215}]
        with mock.patch.object(modelctl, "GPU_MAP_PATH", self.map_path), \
             mock.patch.object(modelctl.modelctl_vram, "xpu_devices",
                               return_value=self.XPU), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               return_value=sycl) as mock_list:
            inv = modelctl.get_gpu_inventory()
            inv2 = modelctl.get_gpu_inventory()  # second call: cached map
        self.assertEqual(mock_list.call_count, 1)
        self.assertEqual(inv[0]["device"], "SYCL0")  # sorted, biggest first
        self.assertEqual(inv[0]["free_bytes"], 30 << 30)
        self.assertEqual(inv, inv2)
        self.assertTrue(self.map_path.exists())

    def test_fallback_map_when_list_devices_fails(self):
        with mock.patch.object(modelctl, "GPU_MAP_PATH", self.map_path), \
             mock.patch.object(modelctl.modelctl_vram, "xpu_devices",
                               return_value=self.XPU), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               return_value=[]):
            inv = modelctl.get_gpu_inventory()
        self.assertEqual([d["device"] for d in inv], ["SYCL0", "SYCL1"])


class TestCmdPlace(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
        {"device": "SYCL1", "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        model = Path(self.tmp.name) / "model.gguf"
        model.write_bytes(b"x" * 1000)
        (self.profiles_dir / "small-model.json").write_text(json.dumps({
            "name": "small-model", "repo_id": "r/a", "file": "f",
            "model_path": str(model),
            "config": {"ctx": 4096, "kv_quant": "q8_0", "flash_attn": "auto",
                       "split_mode": "layer", "tensor_split": "3,1",
                       "ttl": 3600, "mtp": "off", "extra": ""},
            "env": [], "enabled": True,
        }))

    def _args(self, **kw):
        defaults = {"name": None, "apply": False, "remap": False,
                    "no_hermes": True, "no_router_restart": True}
        defaults.update(kw)
        m = mock.Mock()
        m.configure_mock(**defaults)
        return m

    def test_report_only_does_not_touch_profile(self):
        before = (self.profiles_dir / "small-model.json").read_text()
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY):
            modelctl.cmd_place(self._args())
        self.assertEqual((self.profiles_dir / "small-model.json").read_text(), before)

    def test_apply_rewrites_placement_and_syncs(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "generate_artifacts") as mock_gen, \
             mock.patch.object(modelctl, "sync_all_backends") as mock_sync:
            modelctl.cmd_place(self._args(apply=True))
        saved = json.loads((self.profiles_dir / "small-model.json").read_text())
        # tiny model fits the primary card alone -> pinned, split cleared
        self.assertEqual(saved["config"]["device"], "SYCL0")
        self.assertEqual(saved["config"]["split_mode"], "")
        self.assertEqual(saved["config"]["tensor_split"], "")
        mock_gen.assert_called_once()
        mock_sync.assert_called_once()

    def test_no_gpu_inventory_exits_nonzero(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]):
            with self.assertRaises(SystemExit):
                modelctl.cmd_place(self._args())


if __name__ == "__main__":
    unittest.main()
