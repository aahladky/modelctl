"""First-run readiness checks for modelctl.

The browser is the product entry point, so it has to be
able to say what this machine is still missing *before* the user starts a
download that will fail inside a job twenty minutes later.

Every check is read-only and cheap enough to run on a page load.  Nothing
here mutates configuration -- it reports, and points at the page or
command that fixes the problem.

Public API:
    SetupCheck dataclass
    SetupStatus dataclass
    probe_setup(probe_backend=False) -> SetupStatus
"""
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import modelctl


# A model directory with less than this free is reported as a warning: the
# smallest useful quantized MoE checkpoints are already larger.
LOW_SPACE_BYTES = 20 << 30


@dataclass(frozen=True)
class SetupCheck:
    """One thing that has to be true before the browser workflow works."""
    id: str
    title: str
    severity: str          # "ok" | "warning" | "error"
    detail: str
    fix: str = ""          # what the user should do
    fix_url: str = ""      # page in the console that does it
    fix_command: str = ""  # CLI equivalent, for headless/recovery use

    @property
    def ok(self) -> bool:
        return self.severity == "ok"


@dataclass(frozen=True)
class SetupStatus:
    checks: tuple = field(default_factory=tuple)

    @property
    def blocking(self) -> tuple:
        """Checks that stop the browser workflow from completing at all."""
        return tuple(c for c in self.checks if c.severity == "error")

    @property
    def warnings(self) -> tuple:
        return tuple(c for c in self.checks if c.severity == "warning")

    @property
    def ready(self) -> bool:
        """True when a user can actually add and serve a model from here."""
        return not self.blocking

    @property
    def first_run(self) -> bool:
        """True when this looks like a machine nobody has configured yet.

        Distinct from `not ready`: an established install with llama-swap
        temporarily down is not a first run, and should not be dropped onto
        a setup wizard.
        """
        return any(c.id in ("models_dir", "backend") and c.severity == "error"
                   for c in self.checks)


def _check_models_dir() -> SetupCheck:
    path = Path(modelctl.DEFAULT_MODELS_DIR)
    if not path.is_dir():
        return SetupCheck(
            id="models_dir", title="model directory", severity="error",
            detail=f"{path} does not exist",
            fix="Create it, or point MODELCTL_MODELS_DIR at your model store.",
            fix_url="/v2/settings",
            fix_command=f"mkdir -p {path}")
    if not os.access(path, os.W_OK):
        return SetupCheck(
            id="models_dir", title="model directory", severity="error",
            detail=f"{path} is not writable",
            fix="Downloads and imports need write access here.",
            fix_url="/v2/settings")
    free = shutil.disk_usage(path).free
    if free < LOW_SPACE_BYTES:
        return SetupCheck(
            id="models_dir", title="model directory", severity="warning",
            detail=f"{path} — only {modelctl._format_size(free)} free",
            fix="Most quantized MoE checkpoints are larger than this.",
            fix_url="/v2/settings")
    return SetupCheck(
        id="models_dir", title="model directory", severity="ok",
        detail=f"{path} — {modelctl._format_size(free)} free")


def _check_state_dir() -> SetupCheck:
    path = Path(modelctl.PROFILES_DIR)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return SetupCheck(
            id="state_dir", title="profile store", severity="error",
            detail=f"cannot create {path}: {e}",
            fix="Check permissions on the modelctl state directory.")
    if not os.access(path, os.W_OK):
        return SetupCheck(
            id="state_dir", title="profile store", severity="error",
            detail=f"{path} is not writable",
            fix="Profiles cannot be saved.")
    count = len(list(path.glob("*.json")))
    return SetupCheck(
        id="state_dir", title="profile store", severity="ok",
        detail=f"{path} — {count} profile{'s' if count != 1 else ''}")


def _check_backend(probe=False) -> SetupCheck:
    """Is there a llama-server this machine can actually run?

    Resolution deliberately mirrors the launch path rather than just
    testing MODELCTL_LLAMA_SERVER: the configured default is frequently
    absent on a fresh machine while a real build sits in one of the known
    locations, and the launch path would find it.
    """
    candidates = []
    try:
        candidates = [c for c in
                      [modelctl.LLAMA_SERVER_BIN] + modelctl.find_llama_server_candidates()
                      if c]
    except Exception:
        pass
    binary = next((c for c in candidates if c and os.access(c, os.X_OK)), None)
    if not binary:
        return SetupCheck(
            id="backend", title="llama.cpp runtime", severity="error",
            detail="no llama-server build found",
            fix="Build the pinned runtime, or set MODELCTL_LLAMA_SERVER to an "
                "existing llama-server binary.",
            fix_url="/v2/settings",
            fix_command="export MODELCTL_LLAMA_SERVER=/path/to/llama-server")

    # Several builds side by side (build-sycl, build-vulkan, a fork) is the
    # normal state on this kind of workstation, and only one of them is the
    # default. Saying so prevents "no MoE cache" from reading as a property
    # of the machine when it is a property of which build got picked.
    others = len([c for c in dict.fromkeys(candidates) if os.access(c, os.X_OK)]) - 1
    also = ("" if others == 0 else
        f" \u00b7 {others} other build{'s' if others > 1 else ''} available \u2014 a profile can pin one")

    if not probe:
        return SetupCheck(
            id="backend", title="llama.cpp runtime", severity="ok",
            detail=binary + also)

    # The probe is what makes this truthful rather than "a file exists":
    # it also tells the user whether this build has the MoE cache feature.
    try:
        import modelctl_capabilities
        caps = modelctl_capabilities.probe_backend(binary)
    except Exception as e:
        return SetupCheck(
            id="backend", title="llama.cpp runtime", severity="warning",
            detail=f"{binary} — build-features probe failed: {e}",
            fix="Experimental cache features stay disabled until the probe "
                "succeeds.",
            fix_url="/v2/settings")
    status = caps.get("_probe_status", "unknown")
    if status == "unsupported":
        return SetupCheck(
            id="backend", title="llama.cpp runtime", severity="ok",
            detail=f"{binary} — stock build, no expert cache{also}",
            fix="Experimental cache plans need the moe-serving fork; stock "
                "builds serve normally without them.",
            fix_url="/v2/settings")
    features = [k for k, v in (caps.get("features") or {}).items() if v]
    return SetupCheck(
        id="backend", title="llama.cpp runtime", severity="ok",
        detail=f"{binary} — expert cache supported"
               + (f" ({len(features)} cache features)" if features else "")
               + also,
        fix_url="/v2/settings")


def _check_hardware() -> SetupCheck:
    try:
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
    except Exception as e:
        return SetupCheck(
            id="hardware", title="GPUs", severity="warning",
            detail=f"hardware probe failed: {e}",
            fix="Placement falls back to CPU-only estimates.",
            fix_url="/v2/settings")
    gpus = [g for g in snap.gpus if g.enabled]
    if not gpus:
        return SetupCheck(
            id="hardware", title="GPUs", severity="warning",
            detail="no GPUs detected or all disabled",
            fix="Models will be planned for CPU/RAM execution only.",
            fix_url="/v2/settings")
    names = ", ".join(f"{g.device} {g.name}" for g in gpus)
    return SetupCheck(
        id="hardware", title="GPUs", severity="ok",
        detail=f"{len(gpus)} enabled — {names}",
        fix_url="/v2/settings")


def _check_llama_swap() -> SetupCheck:
    """llama-swap is the front door models are served through."""
    config = Path(modelctl.LLAMA_SWAP_CONFIG_PATH)
    base = modelctl.LLAMA_SWAP_BASE_URL.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=2) as r:
            if r.status == 200:
                return SetupCheck(
                    id="llama_swap", title="llama-swap", severity="ok",
                    detail=f"reachable at {base}", fix_url="/v2/settings")
    except Exception:
        pass
    if not config.exists():
        return SetupCheck(
            id="llama_swap", title="llama-swap", severity="warning",
            detail=f"not running, and no config at {config}",
            fix="Profiles can still be created and tested; registering them "
                "for serving needs llama-swap.",
            fix_url="/v2/settings",
            fix_command="modelctl sync")
    return SetupCheck(
        id="llama_swap", title="llama-swap", severity="warning",
        detail=f"config at {config}, but {base} is not responding",
        fix="Start the service to load and serve registered models.",
        fix_url="/v2/settings",
        fix_command=f"systemctl --user start {modelctl.LLAMA_SWAP_SERVICE_NAME}")


def _check_runtime_env() -> SetupCheck:
    """SYCL builds need an oneAPI environment to find their libraries."""
    try:
        scripts = modelctl.find_env_script_candidates()
    except Exception:
        scripts = []
    if scripts:
        return SetupCheck(
            id="runtime_env", title="oneAPI environment", severity="ok",
            detail=f"{scripts[0]}")
    return SetupCheck(
        id="runtime_env", title="oneAPI environment", severity="warning",
        detail="no env script found",
        fix="SYCL builds may fail to start with a shared-library error. "
            "Save LD_LIBRARY_PATH into the profile's env instead.",
        fix_url="/v2/settings")


def _check_network_exposure() -> SetupCheck:
    """The console can launch processes and rewrite runtime config, so its
    reach must always be visible: loopback-only or LAN-accessible, and on
    what address. LAN over plain HTTP is an acceptable personal default on
    a trusted home network; exposure to untrusted networks is unsupported."""
    exposure = modelctl.web_exposure()
    if exposure["mode"] == "loopback-only":
        return SetupCheck(
            id="network_exposure", title="network exposure", severity="ok",
            detail=f"loopback-only on {exposure['bind']} -- reachable from "
                   "this machine only")
    return SetupCheck(
        id="network_exposure", title="network exposure", severity="warning",
        detail=f"LAN-accessible on {exposure['bind']} ({exposure['url']}), "
               "plain HTTP with token auth",
        fix="Fine on a trusted home LAN; the token travels unencrypted, so "
            "untrusted-network exposure is unsupported. For loopback-only, "
            "set MODELCTL_WEB_BIND=127.0.0.1:9293 in the service "
            "environment.",
        fix_command="systemctl --user edit modelctl-web  "
                    "# add Environment=MODELCTL_WEB_BIND=127.0.0.1:9293")


def probe_setup(probe_backend: bool = False) -> SetupStatus:
    """Run every readiness check.

    probe_backend=True additionally runs the capability probe, which spawns
    a subprocess.  The result is cached by binary hash, so the setup page
    passes True and cheaper callers (a nav banner on every page) pass
    False.
    """
    return SetupStatus(checks=(
        _check_models_dir(),
        _check_state_dir(),
        _check_backend(probe=probe_backend),
        _check_hardware(),
        _check_llama_swap(),
        _check_runtime_env(),
        _check_network_exposure(),
    ))
