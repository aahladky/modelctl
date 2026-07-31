"""Regression tripwires for the capability cache (Phase 6 remediation).

The failure mode these guard: one bad probe run (crash, timeout, wrong
env) permanently branding a fork binary "unsupported" -- every cache
feature silently stripped from every plan while everything reports
healthy, with no recovery path short of rebuilding the binary or
hand-deleting undocumented files.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_capabilities as mc

CAPS_JSON = json.dumps({
    "schema": 2, "backend": "llama.cpp", "build": "test",
    "features": {"moe_weight_transfer_cache": True},
    "cli": {"moe_cache_bytes": "--moe-cache-bytes"},
})

# Answers the probe only when the flag file exists next to it -- the same
# binary content can first crash (no env) and later answer (env fixed),
# which is exactly the transient class that must never be persisted.
FLAKY_SERVER = f"""#!/bin/sh
case "$1" in
  --version) echo "version: test"; exit 0 ;;
esac
[ -f "$(dirname "$0")/healthy" ] || exit 134
echo '{CAPS_JSON}'
"""

REJECTING_SERVER = """#!/bin/sh
case "$1" in
  --version) echo "version: stock"; exit 0 ;;
  *) echo "error: invalid argument: $1" >&2; exit 1 ;;
esac
"""


class CapCacheBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cap_dir = Path(self.tmp.name) / "caps"
        patcher = mock.patch.object(mc, "CAPABILITIES_DIR", self.cap_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        mc._session_cache.clear()
        self.addCleanup(mc._session_cache.clear)

    def _write_binary(self, text, name="fake-server"):
        path = Path(self.tmp.name) / name
        path.write_text(text)
        path.chmod(0o755)
        return str(path)


class TestProbeFailureIsNeverPersisted(CapCacheBase):
    def test_crash_then_recovery_without_rebuild(self):
        binary = self._write_binary(FLAKY_SERVER)

        # First probe: the binary aborts (missing env analogue).
        caps = mc.probe_backend(binary)
        self.assertEqual(caps["_probe_status"], "error")
        self.assertFalse(mc.is_cache_capable(caps))
        self.assertEqual(list(self.cap_dir.glob("*.json")), [],
                         "a transient probe failure was persisted to disk")

        # Environment fixed; a NEW process (fresh session cache) must see
        # the real features -- same binary content, no rebuild.
        (Path(self.tmp.name) / "healthy").touch()
        mc._session_cache.clear()
        caps = mc.probe_backend(binary)
        self.assertEqual(caps["_probe_status"], "ok")
        self.assertTrue(mc.is_cache_capable(caps))

    def test_clean_rejection_is_persisted(self):
        binary = self._write_binary(REJECTING_SERVER)
        caps = mc.probe_backend(binary)
        self.assertEqual(caps["_probe_status"], "unsupported")
        entries = list(self.cap_dir.glob("*.json"))
        self.assertEqual(len(entries), 1,
                         "a coherent stock-binary rejection should cache")


class TestRawStorageAndFreshNormalization(CapCacheBase):
    def test_cache_stores_raw_and_normalizes_on_read(self):
        binary = self._write_binary(FLAKY_SERVER)
        (Path(self.tmp.name) / "healthy").touch()
        mc.probe_backend(binary)

        entry_path = next(self.cap_dir.glob("*.json"))
        entry = json.loads(entry_path.read_text())
        self.assertIn("raw", entry, "cache must store the raw probe output")
        self.assertNotIn("_probe_status", entry.get("raw", {}).get("features", {}))

        # A normalizer improvement reaches the cached (pinned, never
        # rebuilt) binary: simulate by editing the raw features and
        # confirming a fresh read reflects it.
        entry["raw"]["features"]["moe_cache_metrics"] = True
        entry_path.write_text(json.dumps(entry))
        mc._session_cache.clear()
        caps = mc.probe_backend(binary)
        self.assertTrue(caps["features"]["moe_cache_metrics"],
                        "read path did not re-normalize from raw")

    def test_legacy_normalized_entries_are_reprobed(self):
        binary = self._write_binary(FLAKY_SERVER)
        (Path(self.tmp.name) / "healthy").touch()
        # Legacy format: normalized caps at the OLD un-suffixed path.
        self.cap_dir.mkdir(parents=True)
        fp = mc._binary_fingerprint(binary)
        legacy = mc._classify_probe_failure(binary)
        legacy["_version"] = mc._version_string(binary)
        (self.cap_dir / f"{fp}.json").write_text(json.dumps(legacy))

        caps = mc.probe_backend(binary)
        self.assertEqual(caps["_probe_status"], "ok",
                         "legacy poisoned entry outlived the migration")


class TestEnvKeyedEntries(CapCacheBase):
    def test_envless_probe_does_not_shadow_the_launch_env(self):
        # The binary reports a device only when PROBE_DEV is set --
        # modelling a SYCL build that sees no GPUs under the bare env.
        script = f"""#!/bin/sh
case "$1" in
  --version) echo v; exit 0 ;;
esac
if [ -n "$PROBE_DEV" ]; then
  echo '{{"schema": 2, "features": {{"moe_weight_transfer_cache": true}}, "devices": [{{"type": "SYCL", "name": "d", "index": 0}}], "cli": {{}}}}'
else
  echo '{{"schema": 2, "features": {{"moe_weight_transfer_cache": true}}, "devices": [], "cli": {{}}}}'
fi
"""
        binary = self._write_binary(script)
        bare = mc.probe_backend(binary)
        with_env = mc.probe_backend(binary, env={"PROBE_DEV": "1"})
        self.assertEqual(len(bare["devices"]), 0)
        self.assertEqual(len(with_env["devices"]), 1,
                         "launch-env probe was served the bare-env answer")

    def test_clear_cache_for_one_binary_drops_all_env_variants(self):
        binary = self._write_binary(FLAKY_SERVER)
        (Path(self.tmp.name) / "healthy").touch()
        mc.probe_backend(binary)
        mc.probe_backend(binary, env={"X": "1"})
        self.assertEqual(len(list(self.cap_dir.glob("*.json"))), 2)
        mc.clear_cache(binary)
        self.assertEqual(list(self.cap_dir.glob("*.json")), [])
        self.assertEqual(mc._session_cache, {})


class TestReprobeCommand(unittest.TestCase):
    def test_cli_reprobe_clears_then_probes(self):
        import modelctl
        with mock.patch.object(mc, "clear_cache") as clear, \
             mock.patch.object(mc, "probe_backend",
                               return_value={"_probe_status": "ok",
                                             "features": {}}), \
             mock.patch("sys.stdout"):
            from argparse import Namespace
            rc = modelctl.cmd_capabilities(
                Namespace(binary="/x/llama-server", reprobe=True))
        clear.assert_called_once_with("/x/llama-server")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
