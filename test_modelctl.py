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


if __name__ == "__main__":
    unittest.main()
