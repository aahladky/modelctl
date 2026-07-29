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
             mock.patch.object(modelctl, "LLAMA_SWAP_BASE_URL", "http://192.168.0.184:9292/v1/"):
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
        self.assertIn("local-swap-managed", providers)
        self.assertEqual(providers["local-swap-managed"]["api"], "http://192.168.0.184:9292/v1/")
        self.assertEqual(providers["local-swap-managed"]["models"], expected_models)

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
            "  local-swap-managed:\n"
            "    name: local-swap-managed\n"
            "    api: http://192.168.0.184:9292/v1/\n"
            "    models:\n"
            "      gemma4-26b:\n"
            "        context_length: 32000\n"
            "        reasoning_effort: medium\n"
        )
        cfg = self._sync(initial)
        gemma_entry = cfg["providers"]["local-swap-managed"]["models"]["gemma4-26b"]
        self.assertEqual(gemma_entry["reasoning_effort"], "medium")
        self.assertEqual(gemma_entry["context_length"], 64000)  # updated from the profile

    def test_removed_profile_drops_out_of_router_provider(self):
        # 'local-swap' is the hand-authored provider bucket (OVMS models,
        # not modelctl profiles) that shares llama-swap's exact URL with the
        # modelctl-managed 'local-swap-managed' bucket -- sync must never merge
        # into or drop from it, even though upsert()'s normal URL-reuse
        # matching would otherwise treat them as the same provider.
        initial = (
            "providers:\n"
            "  local-swap:\n"
            "    name: local-swap\n"
            "    api: http://192.168.0.184:9292/v1/\n"
            "    models:\n"
            "      some-deleted-model:\n"
            "        context_length: 8192\n"
            "  local-swap-managed:\n"
            "    name: local-swap-managed\n"
            "    api: http://192.168.0.184:9292/v1/\n"
            "    models:\n"
            "      some-deleted-model:\n"
            "        context_length: 8192\n"
        )
        cfg = self._sync(initial)
        self.assertNotIn("some-deleted-model", cfg["providers"]["local-swap-managed"]["models"])
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
             mock.patch.object(modelctl, "LLAMA_SWAP_BASE_URL", "http://192.168.0.184:9292/v1/"):
            modelctl.sync_hermes_custom_providers(dry_run=True)
        self.assertEqual(self.hermes_config.read_text(), original)


class TestSyncHermesCustomProvidersOvms(unittest.TestCase):
    """ovms-backend profiles (added 2026-07-09, replacing the retired
    OpenArc backend) fold into the SAME 'local-swap-managed' bucket as
    llama-cpp profiles -- both run behind llama-swap now, so there's no
    reason for modelctl to split them across two Hermes providers the way
    OpenArc (a genuinely separate process/API) used to need."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        self.hermes_config = Path(self.tmp.name) / "hermes_config.yaml"

        (self.profiles_dir / "big-qwen.json").write_text(json.dumps({
            "name": "big-qwen", "backend": "ovms",
            "config": {"target_device": "GPU.0", "task": "text_generation"},
        }))
        (self.profiles_dir / "gemma4-26b.json").write_text(json.dumps({
            "name": "gemma4-26b", "config": {"ctx": "64000"},
        }))

    def _sync(self, initial_yaml):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "HERMES_CONFIG", self.hermes_config), \
             mock.patch.object(modelctl, "LLAMA_SWAP_BASE_URL", "http://127.0.0.1:9292/v1/"):
            self.hermes_config.write_text(initial_yaml)
            modelctl.sync_hermes_custom_providers()
        return modelctl.yaml.safe_load(self.hermes_config.read_text())

    def test_ovms_and_llama_cpp_profiles_share_one_provider(self):
        cfg = self._sync("model:\n  default: x\n")
        providers = cfg["providers"]
        self.assertIn("local-swap-managed", providers)
        models = providers["local-swap-managed"]["models"]
        self.assertIn("big-qwen", models)
        self.assertIn("gemma4-26b", models)
        self.assertEqual(models["gemma4-26b"]["context_length"], 64000)

    def test_drops_stale_local_openarc_and_ovms_providers(self):
        initial = (
            "providers:\n"
            "  local-openarc:\n"
            "    name: local-openarc\n"
            "    api: http://127.0.0.1:8000/v1/\n"
            "    models:\n"
            "      Qwen3.6-27B-int4-ov: {}\n"
            "  ovms-qwen3-6-27b-int4-ov:\n"
            "    name: ovms-qwen3-6-27b-int4-ov\n"
            "    api: http://127.0.0.1:46957/v3/\n"
            "    models:\n"
            "      OpenVINO/Qwen3.6-27B-int4-ov: {}\n"
        )
        cfg = self._sync(initial)
        self.assertNotIn("local-openarc", cfg["providers"])
        self.assertNotIn("ovms-qwen3-6-27b-int4-ov", cfg["providers"])


class TestPreflightOvms(unittest.TestCase):
    def _profile(self, **overrides):
        p = {"name": "big-qwen", "repo_id": "OpenVINO/Qwen3.6-27B-int4-ov",
             "config": {"target_device": "GPU.0"}}
        p.update(overrides)
        return p

    def test_ok_when_script_docker_and_device_present(self):
        with mock.patch.object(modelctl, "OVMS_PROXY_SCRIPT", Path(__file__)), \
             mock.patch.object(modelctl.shutil, "which", return_value="/usr/bin/docker"), \
             mock.patch.object(modelctl, "OVMS_MODEL_REPOSITORY", Path(__file__).parent):
            ok, messages = modelctl.preflight_ovms(self._profile())
        self.assertTrue(ok)

    def test_missing_proxy_script_fails(self):
        with mock.patch.object(modelctl, "OVMS_PROXY_SCRIPT", Path("/nonexistent/ovms-proxy.py")), \
             mock.patch.object(modelctl.shutil, "which", return_value="/usr/bin/docker"):
            ok, messages = modelctl.preflight_ovms(self._profile())
        self.assertFalse(ok)
        self.assertTrue(any("ovms-proxy.py" in m or "not found" in m for m in messages))

    def test_missing_docker_fails(self):
        with mock.patch.object(modelctl, "OVMS_PROXY_SCRIPT", Path(__file__)), \
             mock.patch.object(modelctl.shutil, "which", return_value=None):
            ok, messages = modelctl.preflight_ovms(self._profile())
        self.assertFalse(ok)

    def test_missing_target_device_fails(self):
        with mock.patch.object(modelctl, "OVMS_PROXY_SCRIPT", Path(__file__)), \
             mock.patch.object(modelctl.shutil, "which", return_value="/usr/bin/docker"):
            ok, messages = modelctl.preflight_ovms(self._profile(config={}))
        self.assertFalse(ok)

    def test_missing_local_weights_is_a_note_not_an_error(self):
        with mock.patch.object(modelctl, "OVMS_PROXY_SCRIPT", Path(__file__)), \
             mock.patch.object(modelctl.shutil, "which", return_value="/usr/bin/docker"), \
             mock.patch.object(modelctl, "OVMS_MODEL_REPOSITORY", Path("/nonexistent/ovms-models")):
            ok, messages = modelctl.preflight_ovms(self._profile())
        self.assertTrue(ok)
        self.assertTrue(any("NOTE" in m for m in messages))


class TestRenderOvmsLlamaSwapEntry(unittest.TestCase):
    def _profile(self, **config_overrides):
        config = {"target_device": "GPU.0", "task": "text_generation", "ttl": 1800}
        config.update(config_overrides)
        return {"name": "big-qwen", "repo_id": "OpenVINO/Qwen3.6-27B-int4-ov", "config": config}

    def test_basic_entry_shape(self):
        with mock.patch.object(modelctl, "preflight_ovms", return_value=(True, [])):
            entry_text, ok, messages = modelctl.render_ovms_llama_swap_entry(self._profile())
        self.assertTrue(ok)
        parsed = modelctl.yaml.safe_load(entry_text)
        self.assertIn("big-qwen", parsed)
        cmd = parsed["big-qwen"]["cmd"]
        self.assertIn("ovms-proxy.py", cmd)
        self.assertIn("--source-model", cmd)
        self.assertIn("OpenVINO/Qwen3.6-27B-int4-ov", cmd)
        self.assertIn("--ovms-model-name", cmd)
        self.assertIn("big-qwen", cmd)
        self.assertIn("--target-device", cmd)
        self.assertIn("GPU.0", cmd)
        self.assertEqual(parsed["big-qwen"]["checkEndpoint"], "/v2/health/ready")
        self.assertEqual(parsed["big-qwen"]["cmdStop"], "docker stop ${MODEL_ID}")
        self.assertEqual(parsed["big-qwen"]["ttl"], 1800)

    def test_cache_size_and_tool_parser_included_when_set(self):
        with mock.patch.object(modelctl, "preflight_ovms", return_value=(True, [])):
            entry_text, _, _ = modelctl.render_ovms_llama_swap_entry(
                self._profile(cache_size=6, tool_parser="hermes3"))
        cmd = modelctl.yaml.safe_load(entry_text)["big-qwen"]["cmd"]
        self.assertIn("--cache-size", cmd)
        self.assertIn("6", cmd)
        self.assertIn("--tool-parser", cmd)
        self.assertIn("hermes3", cmd)

    def test_reasoning_parser_included_when_set(self):
        with mock.patch.object(modelctl, "preflight_ovms", return_value=(True, [])):
            entry_text, _, _ = modelctl.render_ovms_llama_swap_entry(
                self._profile(tool_parser="hermes3", reasoning_parser="qwen3"))
        cmd = modelctl.yaml.safe_load(entry_text)["big-qwen"]["cmd"]
        self.assertIn("--reasoning-parser", cmd)
        self.assertIn("qwen3", cmd)

    def test_omitted_when_unset(self):
        with mock.patch.object(modelctl, "preflight_ovms", return_value=(True, [])):
            entry_text, _, _ = modelctl.render_ovms_llama_swap_entry(self._profile())
        cmd = modelctl.yaml.safe_load(entry_text)["big-qwen"]["cmd"]
        self.assertNotIn("--cache-size", cmd)
        self.assertNotIn("--tool-parser", cmd)
        self.assertNotIn("--reasoning-parser", cmd)


class TestSyncLlamaSwapConfigOvms(unittest.TestCase):
    """sync_llama_swap_config() merges ovms-backend profiles into the same
    config.yaml as llama-cpp ones, and must never touch hand-authored
    top-level keys (the matrix: section) or hand-authored models entries
    (they have no logFile pointing under PROFILES_DIR, unlike anything
    modelctl generates)."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        self.config_path = Path(self.tmp.name) / "config.yaml"

        (self.profiles_dir / "big-qwen.json").write_text(json.dumps({
            "name": "big-qwen", "backend": "ovms", "enabled": True,
            "repo_id": "OpenVINO/Qwen3.6-27B-int4-ov",
            "config": {"target_device": "GPU.0", "task": "text_generation", "ttl": 1800},
        }))

    def _sync(self, initial_yaml=""):
        self.config_path.write_text(initial_yaml)
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "LLAMA_SWAP_CONFIG_PATH", self.config_path), \
             mock.patch.object(modelctl, "preflight_ovms", return_value=(True, [])), \
             mock.patch.object(modelctl, "restart_llama_swap_service"):
            modelctl.sync_llama_swap_config()
        return modelctl.yaml.safe_load(self.config_path.read_text())

    def test_adds_ovms_entry(self):
        config = self._sync()
        self.assertIn("big-qwen", config["models"])
        self.assertIn("ovms-proxy.py", config["models"]["big-qwen"]["cmd"])

    def test_preserves_matrix_section_and_hand_authored_model(self):
        initial = (
            "models:\n"
            "  fast-7b:\n"
            "    cmd: hand-authored-command\n"
            "matrix:\n"
            "  vars:\n"
            "    f7: fast-7b\n"
        )
        config = self._sync(initial)
        self.assertEqual(config["models"]["fast-7b"]["cmd"], "hand-authored-command")
        self.assertEqual(config["matrix"]["vars"]["f7"], "fast-7b")
        self.assertIn("big-qwen", config["models"])

    def test_removed_profile_is_dropped(self):
        # Simulate a previously-synced ovms entry (has a logFile under
        # PROFILES_DIR, the "this is ours" marker) whose profile no longer
        # exists -- must be dropped, same as a removed llama-cpp profile.
        initial = (
            f"models:\n"
            f"  stale-model:\n"
            f"    cmd: old\n"
            f'    logFile: "{self.profiles_dir / "stale-model" / "llama-swap.log"}"\n'
        )
        config = self._sync(initial)
        self.assertNotIn("stale-model", config["models"])


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

    def test_no_xpu_smi_falls_back_to_llama_probe(self):
        # No xpu-smi AND no llama devices -> empty; the llama fallback itself
        # is covered by TestGpuInventoryFromLlama.
        with mock.patch.object(modelctl.modelctl_vram, "xpu_devices", return_value=[]), \
             mock.patch.object(modelctl, "_gpu_inventory_from_llama", return_value=[]):
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

    def test_integrated_gpu_excluded(self):
        xpu = self.XPU + [{"xpu_id": 2, "name": "Intel(R) UHD Graphics 770",
                           "total_bytes": 33369731072, "free_bytes": 1 << 30}]
        sycl = [{"device": "SYCL0", "name": "big", "total_mib": 32657},
                {"device": "SYCL1", "name": "small", "total_mib": 12215},
                {"device": "SYCL2", "name": "Intel(R) UHD Graphics 770",
                 "total_mib": 31824}]
        with mock.patch.object(modelctl, "GPU_MAP_PATH", self.map_path), \
             mock.patch.object(modelctl.modelctl_vram, "xpu_devices",
                               return_value=xpu), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               return_value=sycl):
            inv = modelctl.get_gpu_inventory()
        self.assertEqual({d["device"] for d in inv}, {"SYCL0", "SYCL1"})


class TestGpuInventoryFromLlama(unittest.TestCase):
    """The xpu-smi-free fallback: probes llama binaries, SYCL-first, with
    env-script retries when the bare binary sees no devices."""
    DEVS = [{"device": "SYCL0", "name": "big", "total_mib": 32657, "free_mib": 30000},
            {"device": "SYCL1", "name": "small", "total_mib": 12215, "free_mib": 12000}]

    def test_sycl_binary_probed_before_vulkan(self):
        calls = []

        def fake_list(binary, env=None):
            calls.append(binary)
            return self.DEVS

        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN",
                               "/x/build-sycl/bin/llama-server"), \
             mock.patch.object(modelctl, "find_llama_server_candidates",
                               return_value=["/x/build-vulkan/bin/llama-server",
                                             "/x/build-sycl/bin/llama-server"]), \
             mock.patch.object(modelctl, "find_env_script_candidates", return_value=[]), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               side_effect=fake_list):
            inv = modelctl._gpu_inventory_from_llama()
        self.assertEqual(calls, ["/x/build-sycl/bin/llama-server"])
        self.assertEqual([d["device"] for d in inv], ["SYCL0", "SYCL1"])
        self.assertEqual(inv[0]["total_bytes"], 32657 << 20)
        self.assertEqual(inv[0]["free_bytes"], 30000 << 20)

    def test_env_script_retried_when_bare_binary_sees_nothing(self):
        envs = []

        def fake_list(binary, env=None):
            envs.append(env)
            return self.DEVS if env else []

        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN",
                               "/x/build-sycl/bin/llama-server"), \
             mock.patch.object(modelctl, "find_llama_server_candidates",
                               return_value=["/x/build-sycl/bin/llama-server"]), \
             mock.patch.object(modelctl, "find_env_script_candidates",
                               return_value=["/x/env.sh"]), \
             mock.patch.object(modelctl, "source_env_script",
                               return_value={"LD_LIBRARY_PATH": "/opt/intel"}), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               side_effect=fake_list):
            inv = modelctl._gpu_inventory_from_llama()
        self.assertEqual(envs, [None, {"LD_LIBRARY_PATH": "/opt/intel"}])
        self.assertEqual(len(inv), 2)

    def test_empty_when_everything_fails(self):
        with mock.patch.object(modelctl, "LLAMA_SERVER_BIN", "llama-server"), \
             mock.patch.object(modelctl, "find_llama_server_candidates", return_value=[]), \
             mock.patch.object(modelctl, "find_env_script_candidates", return_value=[]), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               return_value=[]):
            self.assertEqual(modelctl._gpu_inventory_from_llama(), [])


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
        defaults = {"name": None, "apply": False, "remap": False, "tiers": False,
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

    def test_apply_over_budget_still_applies_with_warning(self):
        tiny_inventory = [{"device": "SYCL0", "name": "tiny",
                           "total_bytes": 1 << 30, "free_bytes": 1 << 30}]
        big_estimate = {"weights": 10 << 30, "kv_bytes": 0, "overhead": 0,
                        "total": 10 << 30, "quality": "exact"}
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=tiny_inventory), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=big_estimate), \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"):
            modelctl.cmd_place(self._args(apply=True))
        saved = json.loads((self.profiles_dir / "small-model.json").read_text())
        # over-budget single-GPU recommendation keeps the pin; split cleared
        self.assertEqual(saved["config"]["device"], "SYCL0")
        self.assertEqual(saved["config"]["split_mode"], "")


class TestPullPlacementHint(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
    ]

    def test_hint_computed_from_remote_size(self):
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "DEFAULTS_PATH",
                               Path("/nonexistent/x.json")):
            hint = modelctl.compute_pull_placement_hint(18 << 30)
        self.assertEqual(hint["device"], "SYCL0")

    def test_no_inventory_returns_none(self):
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]), \
             mock.patch.object(modelctl, "DEFAULTS_PATH",
                               Path("/nonexistent/x.json")):
            self.assertIsNone(modelctl.compute_pull_placement_hint(18 << 30))

    def test_no_size_returns_none(self):
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "DEFAULTS_PATH",
                               Path("/nonexistent/x.json")):
            self.assertIsNone(modelctl.compute_pull_placement_hint(None))


class TestCheckVramForLoad(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 10 << 30},
        {"device": "SYCL1", "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def _profile(self, device="SYCL0"):
        return {"name": "m", "model_path": "/x/model.gguf",
                "config": {"ctx": 4096, "kv_quant": "q8_0", "device": device,
                           "split_mode": "", "tensor_split": ""}}

    def test_no_inventory_degrades_open(self):
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]):
            ok, msgs = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)
        self.assertTrue(any("skipping VRAM check" in m for m in msgs))

    def test_no_estimate_degrades_open(self):
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=None):
            ok, _ = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)

    def test_fits_passes(self):
        est = {"total": 5 << 30, "quality": "exact"}
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est):
            ok, _ = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)

    def test_exact_over_free_blocks(self):
        est = {"total": 20 << 30, "quality": "exact"}
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est), \
             mock.patch.object(modelctl, "router_status", return_value=[]):
            ok, msgs = modelctl.check_vram_for_load(self._profile())
        self.assertFalse(ok)
        self.assertTrue(any("--evict" in m for m in msgs))

    def test_heuristic_over_free_warns_but_passes(self):
        est = {"total": 20 << 30, "quality": "heuristic"}
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est):
            ok, msgs = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)
        self.assertTrue(any("heuristic" in m for m in msgs))

    def test_split_profile_targets_all_gpus(self):
        est = {"total": 20 << 30, "quality": "exact"}
        profile = self._profile()
        profile["config"]["split_mode"] = "layer"
        profile["config"]["tensor_split"] = "8,3"
        # 10 + 12 GiB free combined > 20 GiB estimate -> fits
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est):
            ok, _ = modelctl.check_vram_for_load(profile)
        self.assertTrue(ok)

    def test_evict_unloads_largest_first(self):
        est = {"total": 20 << 30, "quality": "exact"}
        rows = [
            {"name": "small-loaded", "status": "loaded", "failed": False,
             "exit_code": None, "gpu": "SYCL0", "from_preset": True},
            {"name": "big-loaded", "status": "loaded", "failed": False,
             "exit_code": None, "gpu": "SYCL0", "from_preset": True},
        ]
        estimates = {"m": est,
                     "small-loaded": {"total": 2 << 30, "quality": "exact"},
                     "big-loaded": {"total": 18 << 30, "quality": "exact"}}
        # After evicting big-loaded, free jumps enough to fit.
        inventories = [self.INVENTORY,
                       [dict(self.INVENTORY[0], free_bytes=28 << 30),
                        self.INVENTORY[1]]]

        def fake_est(profile):
            return estimates[profile["name"]]

        def fake_load_profile(name):
            return {"name": name, "model_path": "/x", "config": {"device": "SYCL0"}}

        with mock.patch.object(modelctl, "get_gpu_inventory",
                               side_effect=inventories), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               side_effect=fake_est), \
             mock.patch.object(modelctl, "load_profile",
                               side_effect=fake_load_profile), \
             mock.patch.object(modelctl, "router_status", return_value=rows), \
             mock.patch.object(modelctl, "router_unload",
                               return_value=(True, "ok")) as mock_unload, \
             mock.patch.object(modelctl, "_wait_for_router_model",
                               return_value="unloaded"):
            ok, _ = modelctl.check_vram_for_load(self._profile(), evict=True)
        self.assertTrue(ok)
        mock_unload.assert_called_once_with("big-loaded")


class TestWaitForRouterModel(unittest.TestCase):
    def _row(self, status, failed=False):
        return [{"name": "m", "status": status, "failed": failed,
                 "exit_code": 1 if failed else None, "gpu": "?",
                 "from_preset": True}]

    def test_stale_failed_flag_ignored_on_first_poll(self):
        # Previous attempt left failed=True; the new load succeeds.
        states = [self._row("unloaded", failed=True),
                  self._row("loading", failed=True),
                  self._row("loaded")]
        with mock.patch.object(modelctl, "router_status", side_effect=states), \
             mock.patch.object(modelctl.time, "sleep"):
            self.assertEqual(modelctl._wait_for_router_model("m", "loaded"),
                             "loaded")

    def test_genuine_failure_still_detected(self):
        states = [self._row("loading"),
                  self._row("unloaded", failed=True)]
        with mock.patch.object(modelctl, "router_status", side_effect=states), \
             mock.patch.object(modelctl.time, "sleep"):
            self.assertEqual(modelctl._wait_for_router_model("m", "loaded"),
                             "failed")

    def test_unload_wait_satisfied_despite_stale_failed(self):
        states = [self._row("unloaded", failed=True)]
        with mock.patch.object(modelctl, "router_status", side_effect=states), \
             mock.patch.object(modelctl.time, "sleep"):
            self.assertEqual(modelctl._wait_for_router_model("m", "unloaded"),
                             "unloaded")

    def test_router_unreachable_returns_none(self):
        with mock.patch.object(modelctl, "router_status",
                               side_effect=RuntimeError("down")), \
             mock.patch.object(modelctl.time, "sleep"):
            self.assertIsNone(modelctl._wait_for_router_model("m", "loaded"))


class TestDetectDefaultTask(unittest.TestCase):
    """The signal here is the one that actually caused a real, live failure
    earlier: a repo with vision_config is a VLM even when only used for text
    chat, and registering it as 'llm' fails at inference time with a cryptic
    port-mismatch error, not at conversion or load time."""

    def _config_file(self, tmp, config: dict) -> str:
        p = Path(tmp) / "config.json"
        p.write_text(json.dumps(config))
        return str(p)

    def test_vision_config_top_level_detected_as_vlm(self):
        with TemporaryDirectory() as tmp:
            path = self._config_file(tmp, {"vision_config": {"hidden_size": 1152}})
            with mock.patch.object(modelctl, "hf_hub_download", return_value=path):
                task, is_vlm = modelctl.detect_default_task("some/repo")
        self.assertTrue(is_vlm)
        self.assertEqual(task, "image-text-to-text")

    def test_vision_config_nested_in_text_config_detected_as_vlm(self):
        with TemporaryDirectory() as tmp:
            path = self._config_file(tmp, {"text_config": {"vision_config": {}}})
            with mock.patch.object(modelctl, "hf_hub_download", return_value=path):
                task, is_vlm = modelctl.detect_default_task("some/repo")
        self.assertTrue(is_vlm)

    def test_conditional_generation_with_image_token_detected_as_vlm(self):
        with TemporaryDirectory() as tmp:
            path = self._config_file(tmp, {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "image_token_id": 248056,
            })
            with mock.patch.object(modelctl, "hf_hub_download", return_value=path):
                task, is_vlm = modelctl.detect_default_task("some/repo")
        self.assertTrue(is_vlm)

    def test_plain_llm_not_detected_as_vlm(self):
        with TemporaryDirectory() as tmp:
            path = self._config_file(tmp, {"architectures": ["LlamaForCausalLM"]})
            with mock.patch.object(modelctl, "hf_hub_download", return_value=path):
                task, is_vlm = modelctl.detect_default_task("some/repo")
        self.assertFalse(is_vlm)
        self.assertEqual(task, "text-generation-with-past")

    def test_unreachable_repo_degrades_to_llm_default(self):
        with mock.patch.object(modelctl, "hf_hub_download", side_effect=OSError("network down")):
            task, is_vlm = modelctl.detect_default_task("some/repo")
        self.assertFalse(is_vlm)
        self.assertEqual(task, "text-generation-with-past")


class TestBuildOptimumExportArgs(unittest.TestCase):
    def test_int4_includes_group_size_and_ratio(self):
        args = modelctl.build_optimum_export_args("org/repo", "/out", {
            "weight_format": "int4", "group_size": 128, "ratio": 1.0,
            "sym": False, "task": "text-generation-with-past", "trust_remote_code": False,
        })
        self.assertEqual(args, [
            "export", "openvino", "-m", "org/repo",
            "--task", "text-generation-with-past",
            "--weight-format", "int4",
            "--group-size", "128",
            "--ratio", "1.0",
            "/out",
        ])

    def test_fp16_omits_group_size_and_ratio(self):
        args = modelctl.build_optimum_export_args("org/repo", "/out", {
            "weight_format": "fp16", "group_size": 128, "ratio": 1.0,
            "sym": False, "task": None, "trust_remote_code": False,
        })
        self.assertNotIn("--group-size", args)
        self.assertNotIn("--ratio", args)
        self.assertIn("--weight-format", args)

    def test_int8_includes_group_size_but_not_ratio(self):
        args = modelctl.build_optimum_export_args("org/repo", "/out", {
            "weight_format": "int8", "group_size": 64, "ratio": 1.0,
            "sym": False, "task": None, "trust_remote_code": False,
        })
        self.assertIn("--group-size", args)
        self.assertNotIn("--ratio", args)

    def test_sym_and_trust_remote_code_flags(self):
        args = modelctl.build_optimum_export_args("org/repo", "/out", {
            "weight_format": "int4", "group_size": 128, "ratio": 1.0,
            "sym": True, "task": None, "trust_remote_code": True,
        })
        self.assertIn("--sym", args)
        self.assertIn("--trust-remote-code", args)

    def test_output_dir_is_last_argument(self):
        args = modelctl.build_optimum_export_args("org/repo", "/some/out/dir", {
            "weight_format": "int4", "group_size": 128, "ratio": 1.0,
            "sym": False, "task": None, "trust_remote_code": False,
        })
        self.assertEqual(args[-1], "/some/out/dir")


class TestRunOptimumExport(unittest.TestCase):
    def _cfg(self):
        return {"weight_format": "int4", "group_size": 128, "ratio": 1.0,
                "sym": False, "task": None, "trust_remote_code": False}

    def test_missing_binary_fails_without_running_subprocess(self):
        with mock.patch.object(modelctl, "OPTIMUM_CLI_BIN", "/nonexistent/optimum-cli"), \
             mock.patch.object(modelctl.shutil, "which", return_value=None), \
             mock.patch("modelctl.subprocess.run") as mock_run:
            ok = modelctl.run_optimum_export("org/repo", "/out", self._cfg())
        self.assertFalse(ok)
        mock_run.assert_not_called()

    def test_success_returns_true(self):
        with mock.patch.object(modelctl, "OPTIMUM_CLI_BIN", "/usr/bin/optimum-cli"), \
             mock.patch("modelctl.Path.exists", return_value=True), \
             mock.patch("modelctl.subprocess.run", return_value=mock.Mock(returncode=0)):
            ok = modelctl.run_optimum_export("org/repo", "/out", self._cfg())
        self.assertTrue(ok)

    def test_nonzero_exit_returns_false(self):
        with mock.patch.object(modelctl, "OPTIMUM_CLI_BIN", "/usr/bin/optimum-cli"), \
             mock.patch("modelctl.Path.exists", return_value=True), \
             mock.patch("modelctl.subprocess.run", return_value=mock.Mock(returncode=1)):
            ok = modelctl.run_optimum_export("org/repo", "/out", self._cfg())
        self.assertFalse(ok)


class TestVramFooter(unittest.TestCase):
    def test_footer_lines(self):
        inventory = [{"device": "SYCL0", "name": "big",
                      "total_bytes": 32 << 30, "free_bytes": 10 << 30}]
        lines = modelctl.vram_footer_lines(inventory)
        self.assertEqual(len(lines), 1)
        self.assertIn("SYCL0", lines[0])
        self.assertIn("22.0GB", lines[0])   # used = total - free
        self.assertIn("32.0GB", lines[0])

    def test_empty_inventory_no_lines(self):
        self.assertEqual(modelctl.vram_footer_lines([]), [])


PROM_SAMPLE = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 12000
llamacpp:prompt_seconds_total 10
llamacpp:tokens_predicted_total 4500
llamacpp:tokens_predicted_seconds_total 300
llamacpp:n_decode_total 4600
llamacpp:prompt_tokens_seconds 1180.5
llamacpp:predicted_tokens_seconds 14.9
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
"""


class TestParsePrometheus(unittest.TestCase):
    def test_parses_values_skips_comments(self):
        m = modelctl.parse_prometheus_text(PROM_SAMPLE)
        self.assertEqual(m["llamacpp:prompt_tokens_total"], 12000.0)
        self.assertEqual(m["llamacpp:predicted_tokens_seconds"], 14.9)
        self.assertNotIn("# HELP llamacpp:prompt_tokens_total", m)

    def test_garbage_lines_ignored(self):
        m = modelctl.parse_prometheus_text("weird\nllamacpp:x notanumber\n")
        self.assertEqual(m, {})


class TestStatsRow(unittest.TestCase):
    def test_computed_columns(self):
        metrics = modelctl.parse_prometheus_text(PROM_SAMPLE)
        row = modelctl.stats_row_from_metrics(metrics)
        self.assertEqual(row["gen_tps"], "15.0")       # 4500/300
        self.assertEqual(row["prompt_tps"], "1200.0")  # 12000/10
        self.assertEqual(row["last_gen_tps"], "14.9")  # gauge
        self.assertEqual(row["requests"], "1/0")       # processing/deferred

    def test_missing_metrics_render_question_marks(self):
        row = modelctl.stats_row_from_metrics({})
        self.assertEqual(row["gen_tps"], "?")
        self.assertEqual(row["requests"], "?")


class TestResolveCacheTypes(unittest.TestCase):
    def test_both_new_fields(self):
        self.assertEqual(
            modelctl._resolve_cache_types({"cache_type_k": "q8_0", "cache_type_v": "q4_0"}),
            ("q8_0", "q4_0"))

    def test_legacy_only(self):
        self.assertEqual(modelctl._resolve_cache_types({"kv_quant": "q8_0"}),
                         ("q8_0", "q8_0"))

    def test_one_new_field_no_legacy(self):
        self.assertEqual(modelctl._resolve_cache_types({"cache_type_k": "q8_0"}),
                         ("q8_0", None))

    def test_neither(self):
        self.assertEqual(modelctl._resolve_cache_types({}), (None, None))


class TestSplitCacheTypeEmission(unittest.TestCase):
    def _profile(self, cfg_extra):
        cfg = {"flash_attn": "auto", "ctx": 4096, "split_mode": "",
               "tensor_split": "", "ttl": 3600, "mtp": "off", "extra": ""}
        cfg.update(cfg_extra)
        return {"name": "m", "model_path": "/x/m.gguf", "mmproj_path": None,
                "config": cfg}

    def test_build_server_args_distinct_types(self):
        args = modelctl.build_server_args(self._profile(
            {"cache_type_k": "q8_0", "cache_type_v": "q4_0"}))
        self.assertIn("q8_0", args[args.index("--cache-type-k") + 1])
        self.assertIn("q4_0", args[args.index("--cache-type-v") + 1])

    def test_build_server_args_legacy_kv_quant(self):
        args = modelctl.build_server_args(self._profile({"kv_quant": "q8_0"}))
        self.assertEqual(args[args.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(args[args.index("--cache-type-v") + 1], "q8_0")

    def test_build_server_args_k_only(self):
        args = modelctl.build_server_args(self._profile({"cache_type_k": "q8_0"}))
        self.assertIn("--cache-type-k", args)
        self.assertNotIn("--cache-type-v", args)

    def test_new_keys_default_q8_0(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": ""}):
            d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q8_0")
        self.assertEqual(d["cache_type_v"], "q8_0")

    def test_legacy_env_var_applies_to_both(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": "q5_1"}):
            d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q5_1")
        self.assertEqual(d["cache_type_v"], "q5_1")

    def test_new_env_vars_win_and_diverge(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": "q5_1",
                                            "MODELCTL_DEFAULT_CACHE_TYPE_V": "q4_0"}):
            d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q5_1")   # legacy still covers K
        self.assertEqual(d["cache_type_v"], "q4_0")   # new var wins for V

    def test_persisted_legacy_kv_quant_applies_to_both(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "defaults.json"
            p.write_text(json.dumps({"kv_quant": "q5_0"}))
            with mock.patch.object(modelctl, "DEFAULTS_PATH", p), \
                 mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": ""}):
                d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q5_0")
        self.assertEqual(d["cache_type_v"], "q5_0")


class TestPromptConfigLegacyOverlay(unittest.TestCase):
    def test_legacy_kv_quant_prefills_cache_type_k(self):
        current = {"kv_quant": "q5_1"}
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": ""}), \
             mock.patch("builtins.input", return_value=""):
            profile = modelctl.prompt_config(current=current)
        self.assertEqual(profile["cache_type_k"], "q5_1")
        self.assertEqual(profile["cache_type_v"], "q5_1")


class TestLlamaSwapModelIds(unittest.TestCase):
    @staticmethod
    def _response(payload):
        cm = mock.MagicMock()
        cm.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        return cm

    def test_success_returns_ids(self):
        payload = {"data": [{"id": "big-qwen"}, {"id": "fast-7b"}]}
        with mock.patch("modelctl.urllib.request.urlopen", return_value=self._response(payload)):
            ids = modelctl.llama_swap_model_ids()
        self.assertEqual(ids, ["big-qwen", "fast-7b"])

    def test_empty_data_returns_empty_list(self):
        with mock.patch("modelctl.urllib.request.urlopen", return_value=self._response({"data": []})):
            ids = modelctl.llama_swap_model_ids()
        self.assertEqual(ids, [])

    def test_connection_failure_returns_none(self):
        with mock.patch("modelctl.urllib.request.urlopen", side_effect=OSError("refused")):
            ids = modelctl.llama_swap_model_ids()
        self.assertIsNone(ids)

    def test_timeout_returns_none(self):
        with mock.patch("modelctl.urllib.request.urlopen", side_effect=TimeoutError()):
            ids = modelctl.llama_swap_model_ids()
        self.assertIsNone(ids)


class TestIsCodeEvalTask(unittest.TestCase):
    def test_humaneval_variants_are_code_eval(self):
        for name in ["humaneval", "humaneval_instruct", "humaneval_plus", "humaneval_64_instruct"]:
            self.assertTrue(modelctl.is_code_eval_task(name), name)

    def test_mbpp_variants_are_code_eval(self):
        for name in ["mbpp", "mbpp_instruct", "mbpp_plus", "mbpp_plus_instruct"]:
            self.assertTrue(modelctl.is_code_eval_task(name), name)

    def test_non_code_tasks_are_not_code_eval(self):
        for name in ["gsm8k", "ifeval", "mmlu_pro", "bbh_cot_fewshot_causal_judgement"]:
            self.assertFalse(modelctl.is_code_eval_task(name), name)


class TestBuildBenchEnv(unittest.TestCase):
    def test_maps_limit_think_max_gen_toks(self):
        with mock.patch.dict("os.environ", {"SOME_UNRELATED_VAR": "keepme"}):
            env = modelctl.build_bench_env(limit=50, think=True, max_gen_toks=4096, allow_code_eval=False)
        self.assertEqual(env["LIMIT"], "50")
        self.assertEqual(env["THINK"], "1")
        self.assertEqual(env["MAX_GEN_TOKS"], "4096")
        self.assertNotIn("HF_ALLOW_CODE_EVAL", env)
        self.assertEqual(env["SOME_UNRELATED_VAR"], "keepme")

    def test_limit_none_becomes_zero(self):
        env = modelctl.build_bench_env(limit=None, think=False, max_gen_toks=2048, allow_code_eval=False)
        self.assertEqual(env["LIMIT"], "0")

    def test_limit_zero_stays_zero(self):
        env = modelctl.build_bench_env(limit=0, think=False, max_gen_toks=2048, allow_code_eval=False)
        self.assertEqual(env["LIMIT"], "0")

    def test_think_false_maps_to_zero_string(self):
        env = modelctl.build_bench_env(limit=30, think=False, max_gen_toks=2048, allow_code_eval=False)
        self.assertEqual(env["THINK"], "0")

    def test_allow_code_eval_sets_env_var(self):
        env = modelctl.build_bench_env(limit=30, think=False, max_gen_toks=2048, allow_code_eval=True)
        self.assertEqual(env["HF_ALLOW_CODE_EVAL"], "1")


class TestBuildBenchCommand(unittest.TestCase):
    def test_basic_command(self):
        with mock.patch.object(modelctl, "BENCH_SH", Path("/x/bench.sh")):
            cmd = modelctl.build_bench_command("big-qwen", ["gsm8k", "ifeval"])
        self.assertEqual(cmd, ["/x/bench.sh", "big-qwen", "gsm8k,ifeval"])

    def test_code_eval_appends_confirm_flag(self):
        with mock.patch.object(modelctl, "BENCH_SH", Path("/x/bench.sh")):
            cmd = modelctl.build_bench_command("big-qwen", ["humaneval_instruct"], allow_code_eval=True)
        self.assertEqual(cmd, ["/x/bench.sh", "big-qwen", "humaneval_instruct", "--confirm_run_unsafe_code"])


class TestBuildSpeedCommand(unittest.TestCase):
    def test_basic_command(self):
        with mock.patch.object(modelctl, "SPEED_PY", Path("/x/speed.py")):
            cmd = modelctl.build_speed_command("fast-7b")
        self.assertEqual(cmd, [modelctl.sys.executable, "/x/speed.py", "fast-7b", "256", "3"])

    def test_custom_max_tokens_and_runs(self):
        with mock.patch.object(modelctl, "SPEED_PY", Path("/x/speed.py")):
            cmd = modelctl.build_speed_command("fast-7b", max_tokens=128, runs=5)
        self.assertEqual(cmd, [modelctl.sys.executable, "/x/speed.py", "fast-7b", "128", "5"])

    def test_think_appends_flag(self):
        with mock.patch.object(modelctl, "SPEED_PY", Path("/x/speed.py")):
            cmd = modelctl.build_speed_command("big-qwen", think=True)
        self.assertEqual(cmd, [modelctl.sys.executable, "/x/speed.py", "big-qwen", "256", "3", "--think"])


class TestRunProfileEvals(unittest.TestCase):
    @staticmethod
    def _args(evals, limit=30, think=False, max_gen_toks=2048, confirm_unsafe_code=False):
        ns = argparse.Namespace()
        ns.evals = evals
        ns.limit = limit
        ns.think = think
        ns.max_gen_toks = max_gen_toks
        ns.confirm_unsafe_code = confirm_unsafe_code
        return ns

    def test_llama_swap_unreachable_exits(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=None), \
             self.assertRaises(SystemExit) as cm:
            modelctl.run_profile_evals({"name": "big-qwen"}, self._args("gsm8k"))
        self.assertEqual(cm.exception.code, 1)

    def test_model_not_registered_exits(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["fast-7b"]), \
             self.assertRaises(SystemExit):
            modelctl.run_profile_evals({"name": "big-qwen"}, self._args("gsm8k"))

    def test_code_eval_without_confirm_exits_before_running_anything(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["big-qwen"]), \
             mock.patch("modelctl.subprocess.run") as mock_run, \
             self.assertRaises(SystemExit):
            modelctl.run_profile_evals({"name": "big-qwen"}, self._args("humaneval_instruct"))
        mock_run.assert_not_called()

    def test_code_eval_with_confirm_runs(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["big-qwen"]), \
             mock.patch("modelctl.Path.exists", return_value=True), \
             mock.patch("modelctl.subprocess.run", return_value=mock.Mock(returncode=0)) as mock_run:
            modelctl.run_profile_evals(
                {"name": "big-qwen"}, self._args("humaneval_instruct", confirm_unsafe_code=True))
        mock_run.assert_called_once()
        cmd, kwargs = mock_run.call_args[0][0], mock_run.call_args[1]
        self.assertIn("--confirm_run_unsafe_code", cmd)
        self.assertEqual(kwargs["env"]["HF_ALLOW_CODE_EVAL"], "1")

    def test_speed_pseudo_task_dispatches_to_speed_command(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["fast-7b"]), \
             mock.patch("modelctl.Path.exists", return_value=True), \
             mock.patch("modelctl.subprocess.run", return_value=mock.Mock(returncode=0)) as mock_run:
            modelctl.run_profile_evals({"name": "fast-7b"}, self._args("speed"))
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("speed.py", cmd[1])

    def test_speed_and_tasks_both_run(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["fast-7b"]), \
             mock.patch("modelctl.Path.exists", return_value=True), \
             mock.patch("modelctl.subprocess.run", return_value=mock.Mock(returncode=0)) as mock_run:
            modelctl.run_profile_evals({"name": "fast-7b"}, self._args("speed,gsm8k"))
        self.assertEqual(mock_run.call_count, 2)

    def test_nonzero_exit_propagates(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["fast-7b"]), \
             mock.patch("modelctl.Path.exists", return_value=True), \
             mock.patch("modelctl.subprocess.run", return_value=mock.Mock(returncode=7)), \
             self.assertRaises(SystemExit) as cm:
            modelctl.run_profile_evals({"name": "fast-7b"}, self._args("gsm8k"))
        self.assertEqual(cm.exception.code, 7)

    def test_blank_evals_exits(self):
        with mock.patch.object(modelctl, "llama_swap_model_ids", return_value=["fast-7b"]), \
             self.assertRaises(SystemExit):
            modelctl.run_profile_evals({"name": "fast-7b"}, self._args(" , , "))


class TestCmdTestEvalsDispatch(unittest.TestCase):
    def test_evals_flag_routes_to_run_profile_evals(self):
        profile = {"name": "big-qwen", "backend": "ovms"}
        args = argparse.Namespace(name="big-qwen", evals="gsm8k")
        with mock.patch.object(modelctl, "load_profile", return_value=profile), \
             mock.patch.object(modelctl, "run_profile_evals") as mock_run_evals:
            modelctl.cmd_test(args)
        mock_run_evals.assert_called_once_with(profile, args)

    def test_no_evals_flag_keeps_legacy_ovms_message(self):
        profile = {"name": "big-qwen", "backend": "ovms"}
        args = argparse.Namespace(name="big-qwen", evals=None)
        with mock.patch.object(modelctl, "load_profile", return_value=profile), \
             mock.patch.object(modelctl, "run_profile_evals") as mock_run_evals:
            modelctl.cmd_test(args)
        mock_run_evals.assert_not_called()


class TestPreflightPinnedBinary(unittest.TestCase):
    """A per-profile "binary" pin wins over global resolution, so env-less
    regen/sync runs can't clobber profiles that need a specific build."""

    def _profile(self, binary):
        return {"name": "m", "binary": binary,
                "model_path": "/x/model.gguf",
                "config": {"ctx": 4096, "device": "", "split_mode": "layer",
                           "tensor_split": "8,3"}}

    def test_pinned_binary_used(self):
        with mock.patch.object(modelctl.os.path, "exists", return_value=True), \
             mock.patch.object(modelctl.os, "access", return_value=True), \
             mock.patch.object(modelctl, "find_env_script_candidates", return_value=[]):
            ok, effective_bin, _env, messages = modelctl.preflight(self._profile("/fork/bin/llama-server"))
        self.assertEqual(effective_bin, "/fork/bin/llama-server")
        self.assertTrue(any("pinned binary" in m for m in messages))

    def test_missing_pinned_binary_is_error(self):
        with mock.patch.object(modelctl.os.path, "exists", return_value=True), \
             mock.patch.object(modelctl.os, "access", return_value=False), \
             mock.patch.object(modelctl, "find_env_script_candidates", return_value=[]):
            ok, _bin, _env, messages = modelctl.preflight(self._profile("/gone/bin/llama-server"))
        self.assertFalse(ok)
        self.assertTrue(any("pinned binary not executable" in m for m in messages))


if __name__ == "__main__":
    unittest.main()


class TestCmdPullYes(unittest.TestCase):
    """pull --yes: zero prompts, auto-quant, defaults config, env backfill."""
    GROUPS = [
        {"label": "m-Q8_0", "total_size": 28 << 30, "sharded": False, "files": ["m-Q8_0.gguf"]},
        {"label": "m-Q6_K", "total_size": 22 << 30, "sharded": False, "files": ["m-Q6_K.gguf"]},
        {"label": "m-Q4_K_M", "total_size": 16 << 30, "sharded": False, "files": ["m-Q4_K_M.gguf"]},
        {"label": "imatrix_x", "total_size": 1 << 20, "sharded": False, "files": ["imatrix.gguf"]},
    ]
    INVENTORY = [{"device": "SYCL0", "name": "big", "total_bytes": 32 << 30,
                  "free_bytes": 30 << 30}]

    def _args(self):
        m = mock.Mock()
        m.configure_mock(repo_id="r/m", tui=False, yes=True,
                         no_hermes=True, no_router_restart=True)
        return m

    def _run(self, contents):
        saved = []
        with mock.patch.object(modelctl, "get_repo_contents", return_value=contents), \
             mock.patch.object(modelctl, "get_gpu_inventory", return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "download_if_needed", return_value="/models/r/m/m.gguf"), \
             mock.patch.object(modelctl, "save_profile", side_effect=saved.append), \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"), \
             mock.patch.object(modelctl, "next_unique_profile_name", return_value="m"), \
             mock.patch.object(modelctl, "capture_env_passthrough", return_value=[]), \
             mock.patch.object(modelctl, "_env_from_scripts",
                               return_value=["LD_LIBRARY_PATH=/opt/intel"]), \
             mock.patch("builtins.input", side_effect=AssertionError("prompted in --yes mode")):
            modelctl.cmd_pull(self._args())
        return saved

    def test_zero_config_pull(self):
        saved = self._run({"quant_groups": self.GROUPS, "mmproj_files": [], "mtp_files": []})
        self.assertEqual(len(saved), 1)
        p = saved[0]
        # 28 GiB budget at 90% of 32 GiB: Q8_0 (28 + KV + overhead) doesn't
        # fit, Q6_K does
        self.assertEqual(p["file"], "m-Q6_K")
        self.assertEqual(p["name"], "m")
        self.assertEqual(p["env"], ["LD_LIBRARY_PATH=/opt/intel"])
        cfg = p["config"]
        self.assertIn("ctx", cfg)
        self.assertFalse(any(k.startswith("_prompt") for k in cfg))

    def test_mtp_auto_selected_when_repo_ships_one(self):
        contents = {"quant_groups": self.GROUPS, "mmproj_files": [],
                    "mtp_files": [{"name": "m-mtp.gguf", "size": 1 << 30}]}
        saved = self._run(contents)
        self.assertEqual(saved[0]["mtp_path"], "/models/r/m/m.gguf")
        self.assertEqual(saved[0]["config"]["mtp"], "on")


class TestBuildServerArgsFit(unittest.TestCase):
    BASE = {"model_path": "/x/m.gguf", "mmproj_path": None,
            "config": {"flash_attn": "auto", "ctx": 32768, "split_mode": "",
                       "tensor_split": "", "device": "SYCL0",
                       "cache_type_k": "q8_0", "cache_type_v": "q4_0",
                       "ttl": 3600, "extra": ""}}

    def test_fit_on_omits_ngl_and_ctx_emits_fit(self):
        p = {"name": "m", **self.BASE}
        p["config"] = dict(self.BASE["config"], fit="on")
        args = modelctl.build_server_args(p)
        self.assertIn("--fit", args)
        self.assertEqual(args[args.index("--fit") + 1], "on")
        self.assertNotIn("-ngl", args)
        self.assertNotIn("-c", args)

    def test_default_profile_unchanged(self):
        p = {"name": "m", **self.BASE}
        args = modelctl.build_server_args(p)
        self.assertIn("-ngl", args)
        self.assertIn("-c", args)
        self.assertNotIn("--fit", args)

    def test_fit_off_explicit_still_fixed(self):
        p = {"name": "m", **self.BASE}
        p["config"] = dict(self.BASE["config"], fit="off")
        args = modelctl.build_server_args(p)
        self.assertIn("-ngl", args)
        self.assertIn("-c", args)


class TestRenderManagedEntry(unittest.TestCase):
    def test_managed_profile_renders_worker_command(self):
        profile = {"name": "m", "model_path": "/x/m.gguf",
                   "config": {"ctx": 32768, "ttl": 3600, "flash_attn": "auto",
                              "cache_type_k": "q8_0", "cache_type_v": "q4_0",
                              "device": "SYCL0", "split_mode": "", "tensor_split": "",
                              "extra": ""},
                   "runtime": {"mode": "managed", "objective": "balanced"},
                   "env": []}
        with mock.patch.object(modelctl, "preflight",
                               return_value=(True, "/x/llama-server", {}, [])):
            text, ok, _ = modelctl.render_llama_swap_entry(profile)
        self.assertIn("_worker m --port ${PORT}", text)
        self.assertNotIn("llama-server --port", text)

    def test_fixed_profile_keeps_direct_command(self):
        profile = {"name": "m", "model_path": "/x/m.gguf",
                   "config": {"ctx": 32768, "ttl": 3600, "flash_attn": "auto",
                              "cache_type_k": "q8_0", "cache_type_v": "q4_0",
                              "device": "SYCL0", "split_mode": "", "tensor_split": "",
                              "extra": ""},
                   "env": []}
        with mock.patch.object(modelctl, "preflight",
                               return_value=(True, "/x/llama-server", {}, [])), \
             mock.patch.object(modelctl, "build_server_args", return_value=["--model", "/x/m.gguf"]):
            text, ok, _ = modelctl.render_llama_swap_entry(profile)
        self.assertIn("llama-server --port", text)
        self.assertNotIn("_worker", text)


class TestNormalizeProfile(unittest.TestCase):
    def test_legacy_kv_quant_maps_to_split_types(self):
        p = modelctl.normalize_profile({"config": {"kv_quant": "q8_0"}})
        self.assertEqual(p["config"]["cache_type_k"], "q8_0")
        self.assertEqual(p["config"]["cache_type_v"], "q8_0")

    def test_explicit_split_types_win(self):
        p = modelctl.normalize_profile(
            {"config": {"kv_quant": "q8_0", "cache_type_v": "q4_0"}})
        self.assertEqual(p["config"]["cache_type_k"], "q8_0")
        self.assertEqual(p["config"]["cache_type_v"], "q4_0")

    def test_defaults_and_int_coercion(self):
        p = modelctl.normalize_profile({"name": "x", "config": {"ctx": "32768"}})
        self.assertTrue(p["enabled"])
        self.assertEqual(p["backend"], "llama-cpp")
        self.assertEqual(p["env"], [])
        self.assertEqual(p["config"]["ctx"], 32768)

    def test_idempotent_and_preserves_unknown(self):
        original = {"name": "x", "custom_field": 42,
                    "config": {"cache_type_k": "q8_0"}}
        p1 = modelctl.normalize_profile(original)
        p2 = modelctl.normalize_profile(p1)
        self.assertEqual(p1, p2)
        self.assertEqual(p1["custom_field"], 42)
