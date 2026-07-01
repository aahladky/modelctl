import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl


class TestGetRouterBaseUrl(unittest.TestCase):
    def test_swaps_port_only_by_default(self):
        with mock.patch.object(modelctl, "ROUTER_PORT", "7071"):
            self.assertEqual(
                modelctl.get_router_base_url("http://192.168.0.184:7070/v1/"),
                "http://192.168.0.184:7071/v1/",
            )

    def test_env_override_wins(self):
        with mock.patch.dict("os.environ", {"MODELCTL_ROUTER_BASE_URL": "http://elsewhere:9000/v1"}):
            self.assertEqual(
                modelctl.get_router_base_url("http://192.168.0.184:7070/v1/"),
                "http://elsewhere:9000/v1/",
            )


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
             mock.patch.object(modelctl, "get_llama_swap_base_url", return_value="http://192.168.0.184:7070/v1/"), \
             mock.patch.object(modelctl, "ROUTER_PORT", "7071"):
            self.hermes_config.write_text(initial_yaml)
            modelctl.sync_hermes_custom_providers(dry_run=dry_run)
        return modelctl.yaml.safe_load(self.hermes_config.read_text())

    def test_syncs_both_swap_and_router_providers(self):
        cfg = self._sync("model:\n  default: gemma4-26b\n")
        providers = cfg["providers"]

        expected_models = {
            "Qwythos-9B-Q4": {"context_length": 64000},
            "Qwythos-9B-Q5": {"context_length": 64000},
            "gemma4-26b": {"context_length": 64000},
        }
        self.assertIn("local-swap", providers)
        self.assertIn("local-router", providers)
        self.assertEqual(providers["local-swap"]["api"], "http://192.168.0.184:7070/v1/")
        self.assertEqual(providers["local-router"]["api"], "http://192.168.0.184:7071/v1/")
        self.assertEqual(providers["local-swap"]["models"], expected_models)
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
            "  local-swap:\n"
            "    name: local-swap\n"
            "    api: http://192.168.0.184:7070/v1/\n"
            "    models:\n"
            "      gemma4-26b:\n"
            "        context_length: 32000\n"
            "        reasoning_effort: medium\n"
        )
        cfg = self._sync(initial)
        gemma_entry = cfg["providers"]["local-swap"]["models"]["gemma4-26b"]
        self.assertEqual(gemma_entry["reasoning_effort"], "medium")
        self.assertEqual(gemma_entry["context_length"], 64000)  # updated from the profile

    def test_removed_profile_drops_out_of_both_providers(self):
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
        self.assertNotIn("some-deleted-model", cfg["providers"]["local-swap"]["models"])
        self.assertNotIn("some-deleted-model", cfg["providers"]["local-router"]["models"])

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
             mock.patch.object(modelctl, "get_llama_swap_base_url", return_value="http://192.168.0.184:7070/v1/"), \
             mock.patch.object(modelctl, "ROUTER_PORT", "7071"):
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
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])):
            modelctl.sync_router_preset()

        content = self.router_path.read_text()
        self.assertIn("[Qwythos-9B-Q4]", content)

    def test_backs_up_existing_preset_before_overwrite(self):
        self.router_path.write_text("[stale-old-content]\n")

        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "ROUTER_PRESET_PATH", self.router_path), \
             mock.patch.object(modelctl, "preflight", return_value=(True, "llama-server", {}, [])):
            modelctl.sync_router_preset()

        backup_path = self.router_path.with_suffix(self.router_path.suffix + ".bak")
        self.assertTrue(backup_path.exists())
        self.assertEqual(backup_path.read_text(), "[stale-old-content]\n")
        self.assertIn("[Qwythos-9B-Q4]", self.router_path.read_text())


class TestDownloadIfNeeded(unittest.TestCase):
    """Regression coverage for the cross-repo filename collision: two
    different HF repos can both ship a file named e.g. 'mmproj-F16.gguf'.
    The old version only checked local filename + nonzero size before
    skipping the download, so a file left behind by one repo's pull would
    get silently reused for a completely different repo's model."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest_dir = Path(self.tmp.name)

    def test_skips_when_local_size_matches_remote(self):
        target = self.dest_dir / "mmproj-F16.gguf"
        target.write_bytes(b"x" * 100)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=100), \
             mock.patch.object(modelctl, "hf_hub_download") as mock_dl:
            result = modelctl.download_if_needed("repo/a", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_not_called()
        self.assertEqual(result, str(target))

    def test_redownloads_when_local_size_mismatches_remote(self):
        """The actual bug this fixes: repo/a's file is sitting there, but
        repo/b (a different repo, same filename) is what's being pulled --
        sizes differ, so it must NOT be silently reused."""
        target = self.dest_dir / "mmproj-F16.gguf"
        target.write_bytes(b"x" * 100)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=899283648), \
             mock.patch.object(modelctl, "hf_hub_download", return_value=str(target)) as mock_dl:
            modelctl.download_if_needed("repo/b", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_called_once_with(repo_id="repo/b", filename="mmproj-F16.gguf", local_dir=str(self.dest_dir))

    def test_falls_back_to_trusting_local_file_when_remote_size_unknown(self):
        """If the HF API call fails (offline, rate-limited, etc.) this
        should degrade to the old behavior rather than force a redundant
        multi-GB re-download every time the network hiccups."""
        target = self.dest_dir / "mmproj-F16.gguf"
        target.write_bytes(b"x" * 100)

        with mock.patch.object(modelctl, "_remote_file_size", return_value=None), \
             mock.patch.object(modelctl, "hf_hub_download") as mock_dl:
            result = modelctl.download_if_needed("repo/a", "mmproj-F16.gguf", self.dest_dir)

        mock_dl.assert_not_called()
        self.assertEqual(result, str(target))

    def test_downloads_when_file_not_present_at_all(self):
        with mock.patch.object(modelctl, "hf_hub_download", return_value="/downloaded/path") as mock_dl:
            result = modelctl.download_if_needed("repo/a", "model.gguf", self.dest_dir)

        mock_dl.assert_called_once()
        self.assertEqual(result, "/downloaded/path")


if __name__ == "__main__":
    unittest.main()
