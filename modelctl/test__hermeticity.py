"""Process-wide isolation for the modelctl test suite.

unittest discovery (``python -m unittest discover -p "test_*.py"``)
imports test modules in sorted filename order, and this file sorts before
every other test module ('_' < any letter), so its import side effects
run first -- before any test module imports a modelctl module and
freezes state paths at import time (modelctl_paths.STATE_DIR and every
constant derived from it).

Two live leaks this exists to stop (both observed on 2026-07-31):

* Tests ran against the real ``~/.local/share/modelctl``: plan-run rows
  written to the production runtime.db (216 junk m1/p1 rows), the real
  ``.state.lock`` taken under the feet of the live web console, real
  backend_capabilities entries deleted.
* Hardware/capability sweeps (``_backend_fingerprints``, preflight
  auto-fix, setup checks) globbed COMMON_LLAMA_SERVER_GLOBS, found the
  real llama.cpp SYCL builds, and probed them. A bare-env SYCL probe
  SIGABRTs on this box (oneAPI 2026.1), so every full-suite run dumped
  dozens of llama-server cores.

The bootstrap redirects MODELCTL_HOME to a throwaway directory for the
whole test process and empties the binary/env-script discovery globs.
It also removes the two remaining machine couplings measured on
2026-07-31: setup banners GET the live llama-swap service on page
renders (redirected to an unroutable port), and GPU discovery shells
out to the real xpu-smi (shimmed to answer like a box with no Intel
GPUs). The TestCase below is a tripwire that fails the suite loudly if
any of these redirects ever stops holding.

Single-file runs (``python -m unittest test_modelctl_web``) do not
import this module; each test file still owns its local isolation. This
is the suite-wide backstop, not a substitute for per-test hygiene.
"""
import atexit
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_REAL_DEFAULT = Path.home() / ".local" / "share" / "modelctl"

if "modelctl_paths" in sys.modules:
    _premature = Path(sys.modules["modelctl_paths"].STATE_DIR)
    if _premature == _REAL_DEFAULT:
        raise RuntimeError(
            "modelctl_paths was imported before test__hermeticity could "
            "redirect MODELCTL_HOME, so the suite would run against the "
            f"real state dir {_premature}. This file must be imported "
            "first; check discovery ordering (its name must sort before "
            "every other test_*.py).")

if not os.environ.get("MODELCTL_HOME"):
    _home = tempfile.mkdtemp(prefix="modelctl-test-home-")
    atexit.register(shutil.rmtree, _home, ignore_errors=True)
    os.environ["MODELCTL_HOME"] = _home
# The legacy alias would override the redirect for modules that honor it.
os.environ.pop("MODELCTL_STATE_DIR", None)

# Setup banners GET <base>/models on page renders; point them at the
# discard port (nothing listens there, connect fails instantly) instead
# of the production llama-swap service. Must precede the modelctl and
# modelctl_web.app imports, which both freeze this at import time.
os.environ.setdefault("MODELCTL_LLAMA_SWAP_BASE_URL", "http://127.0.0.1:9/v1/")

# GPU discovery shells out to xpu-smi (modelctl_vram.xpu_devices,
# modelctl_hardware._pci_address). A shim that fails like the tool is
# absent makes every machine answer "no Intel GPUs" -- code degrades
# open to the CPU-only paths by design -- instead of the suite's
# hardware answers depending on the box it runs on.
_shim_dir = Path(os.environ["MODELCTL_HOME"]) / "shims"
_shim_dir.mkdir(parents=True, exist_ok=True)
(_shim_dir / "xpu-smi").write_text("#!/bin/sh\nexit 1\n")
(_shim_dir / "xpu-smi").chmod(0o755)
os.environ["PATH"] = f"{_shim_dir}{os.pathsep}" + os.environ.get("PATH", "")

import modelctl  # noqa: E402  -- must import after the env redirect
import modelctl_paths  # noqa: E402
import modelctl_runtime  # noqa: E402

# Point discovery at a path that cannot match. In-place mutation, not
# rebinding: modules keep references to these exact list objects, and
# tests that mock.patch.object() the names restore to these lists.
_NO_MATCH = os.path.join(os.environ["MODELCTL_HOME"], "no-binaries", "*")
modelctl.COMMON_LLAMA_SERVER_GLOBS[:] = [_NO_MATCH]
modelctl.COMMON_ENV_SCRIPT_GLOBS[:] = [_NO_MATCH]


class TestSuiteHermeticity(unittest.TestCase):
    """Tripwires: fail the suite if it can reach real state again."""

    def test_state_dir_is_redirected(self):
        self.assertNotEqual(Path(modelctl_paths.STATE_DIR), _REAL_DEFAULT,
                            "test suite is pointed at the real MODELCTL_HOME")
        self.assertEqual(str(modelctl_paths.STATE_DIR),
                         os.environ["MODELCTL_HOME"])

    def test_runtime_db_default_lands_in_redirected_home(self):
        # RuntimeDB's default db_path was frozen from DB_PATH at class
        # definition; this catches any import that beat the redirect.
        db = modelctl_runtime.RuntimeDB()
        self.assertEqual(Path(db.db_path).parent,
                         Path(modelctl_paths.STATE_DIR))

    def test_no_real_llama_server_is_discoverable(self):
        self.assertEqual(modelctl.find_llama_server_candidates(), [],
                         "candidate sweep can still reach real binaries; "
                         "probing them SIGABRTs bare-env on SYCL boxes")

    def test_no_real_env_script_is_discoverable(self):
        self.assertEqual(modelctl.find_env_script_candidates(), [])

    def test_llama_swap_url_is_unroutable(self):
        # Setup banners (_check_llama_swap) GET <base>/models on page
        # renders; without this redirect every web-page test polls the
        # production llama-swap service at 127.0.0.1:9292.
        base = os.environ.get("MODELCTL_LLAMA_SWAP_BASE_URL", "")
        self.assertTrue(base.startswith("http://127.0.0.1:9/"),
                        "suite would GET the live llama-swap service")
        self.assertEqual(modelctl.LLAMA_SWAP_BASE_URL, base)
        import modelctl_web.app as web_app
        self.assertEqual(web_app.LLAMA_SWAP_BASE, base)

    def test_xpu_smi_is_stubbed_out(self):
        # get_gpu_inventory / capture_hardware_snapshot shell out to
        # xpu-smi; the suite must see this machine's shim, not the real
        # tool, so hardware answers don't depend on the box.
        which = shutil.which("xpu-smi")
        self.assertIsNotNone(which, "xpu-smi shim missing from PATH")
        self.assertTrue(which.startswith(os.environ["MODELCTL_HOME"]),
                        f"GPU discovery would query the real machine via {which}")
        import modelctl_vram
        self.assertEqual(modelctl_vram.xpu_devices(timeout=5), [])


if __name__ == "__main__":
    unittest.main()
