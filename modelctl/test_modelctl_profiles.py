import unittest

import modelctl_profiles


class TestNormalizeProfile(unittest.TestCase):
    def test_v1_profile_gets_defaults(self):
        raw = {"name": "test", "config": {"ctx": 4096}}
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["profile_version"], 2)
        self.assertEqual(p["backend"], "llama-cpp")
        self.assertEqual(p["enabled"], True)
        self.assertIn("moe_cache", p)
        self.assertEqual(p["moe_cache"]["mode"], "off")
        self.assertEqual(p["config"]["cache_type_k"], "q8_0")
        self.assertEqual(p["config"]["cache_type_v"], "q8_0")

    def test_legacy_kv_quant_migrated(self):
        raw = {"name": "test", "config": {"ctx": 4096, "kv_quant": "q4_0"}}
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["config"]["cache_type_k"], "q4_0")
        self.assertEqual(p["config"]["cache_type_v"], "q4_0")

    def test_string_ctx_coerced_to_int(self):
        raw = {"name": "test", "config": {"ctx": "32768", "ttl": "7200"}}
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["config"]["ctx"], 32768)
        self.assertEqual(p["config"]["ttl"], 7200)

    def test_existing_moe_cache_preserved(self):
        raw = {
            "name": "test",
            "config": {},
            "moe_cache": {"mode": "manual", "gpu": {"budgets_bytes": {"SYCL0": 999}}},
        }
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["moe_cache"]["mode"], "manual")
        self.assertEqual(p["moe_cache"]["gpu"]["budgets_bytes"]["SYCL0"], 999)
        # Missing sub-sections are filled
        self.assertIn("ram", p["moe_cache"])
        self.assertIn("storage", p["moe_cache"])

    def test_partial_moe_cache_filled(self):
        raw = {
            "name": "test",
            "config": {},
            "moe_cache": {"mode": "auto", "gpu": {"policy": "lru"}},
        }
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["moe_cache"]["gpu"]["policy"], "lru")
        self.assertEqual(p["moe_cache"]["gpu"]["admission_misses"], 2)
        self.assertIn("decode", p["moe_cache"])

    def test_idempotent(self):
        raw = {"name": "test", "config": {}}
        n1 = modelctl_profiles.normalize_profile(raw)
        n2 = modelctl_profiles.normalize_profile(n1)
        self.assertEqual(n1["moe_cache"], n2["moe_cache"])
        self.assertEqual(n1["profile_version"], n2["profile_version"])

    def test_unknown_fields_preserved(self):
        raw = {"name": "test", "config": {}, "custom_field": "value"}
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["custom_field"], "value")

    def test_runtime_int_coercion(self):
        raw = {
            "name": "test",
            "config": {},
            "runtime": {"minimum_context": "8192", "maximum_storage_tier": "2"},
        }
        p = modelctl_profiles.normalize_profile(raw)
        self.assertEqual(p["runtime"]["minimum_context"], 8192)
        self.assertEqual(p["runtime"]["maximum_storage_tier"], 2)


class TestValidateProfile(unittest.TestCase):
    def test_valid_profile(self):
        p = modelctl_profiles.normalize_profile({"name": "test", "config": {}})
        msgs = modelctl_profiles.validate_profile(p)
        codes = [m.code for m in msgs]
        self.assertIn("profile_valid", codes)
        self.assertFalse(any(m.severity == "error" for m in msgs))

    def test_missing_name(self):
        p = {"name": "", "config": {"ctx": 4096}}
        msgs = modelctl_profiles.validate_profile(p)
        errors = [m for m in msgs if m.severity == "error"]
        self.assertTrue(any("name" in m.summary.lower() for m in errors))

    def test_invalid_backend(self):
        p = {"name": "test", "backend": "unknown", "config": {"ctx": 4096}}
        msgs = modelctl_profiles.validate_profile(p)
        errors = [m for m in msgs if m.severity == "error"]
        self.assertTrue(any("backend" in m.summary.lower() for m in errors))

    def test_invalid_context(self):
        p = {"name": "test", "config": {"ctx": 0}}
        msgs = modelctl_profiles.validate_profile(p)
        errors = [m for m in msgs if m.severity == "error"]
        self.assertTrue(any("context" in m.summary.lower() for m in errors))

    def test_invalid_cache_mode(self):
        p = modelctl_profiles.normalize_profile({
            "name": "test",
            "config": {},
            "moe_cache": {"mode": "invalid"},
        })
        msgs = modelctl_profiles.validate_profile(p)
        errors = [m for m in msgs if m.severity == "error"]
        self.assertTrue(any("moe_cache" in m.summary.lower() for m in errors))

    def test_invalid_objective(self):
        p = modelctl_profiles.normalize_profile({
            "name": "test",
            "config": {},
            "runtime": {"mode": "managed", "objective": "invalid"},
        })
        msgs = modelctl_profiles.validate_profile(p)
        errors = [m for m in msgs if m.severity == "error"]
        self.assertTrue(any("objective" in m.summary.lower() for m in errors))

    def test_missing_model_path_warns(self):
        p = modelctl_profiles.normalize_profile({
            "name": "test",
            "model_path": "/nonexistent/model.gguf",
            "config": {},
        })
        msgs = modelctl_profiles.validate_profile(p)
        warnings = [m for m in msgs if m.severity == "warning"]
        self.assertTrue(any("not found" in m.summary.lower() for m in warnings))


class TestProfileSchemaVersion(unittest.TestCase):
    def test_no_version_returns_zero(self):
        self.assertEqual(modelctl_profiles.profile_schema_version({"name": "t"}), 0)

    def test_v2(self):
        self.assertEqual(
            modelctl_profiles.profile_schema_version({"profile_version": 2}), 2)


class TestMigrateDryRun(unittest.TestCase):
    def test_shows_changes(self):
        raw = {"name": "test", "config": {"ctx": "4096", "kv_quant": "q4_0"}}
        result = modelctl_profiles.migrate_profile_dry_run(raw)
        self.assertTrue(result["changes"])
        self.assertEqual(result["migrated"]["profile_version"], 2)
        self.assertEqual(result["migrated"]["config"]["cache_type_k"], "q4_0")

    def test_no_changes_for_current(self):
        raw = modelctl_profiles.normalize_profile({"name": "test", "config": {}})
        result = modelctl_profiles.migrate_profile_dry_run(raw)
        self.assertEqual(result["changes"], [])
