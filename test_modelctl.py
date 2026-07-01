import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl


class TestSyncHermesCustomProviders(unittest.TestCase):
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

    def test_writes_one_provider_entry_with_nested_models_map(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "HERMES_CONFIG", self.hermes_config), \
             mock.patch.object(modelctl, "get_llama_swap_base_url", return_value="http://192.168.0.184:7070/v1/"):
            self.hermes_config.write_text("model:\n  default: gemma4-26b\n")
            modelctl.sync_hermes_custom_providers()

        cfg = modelctl.yaml.safe_load(self.hermes_config.read_text())
        providers = cfg["custom_providers"]

        self.assertEqual(len(providers), 1)
        provider = providers[0]
        self.assertEqual(provider["base_url"], "http://192.168.0.184:7070/v1/")
        self.assertNotIn("model", provider)

        self.assertEqual(
            provider["models"],
            {
                "Qwythos-9B-Q4": {"context_length": 64000},
                "Qwythos-9B-Q5": {"context_length": 64000},
                "gemma4-26b": {"context_length": 64000},
            },
        )


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


if __name__ == "__main__":
    unittest.main()
