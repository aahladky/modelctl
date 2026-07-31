"""Diagnostics: integration manifest, capability report, support bundle.

The settings page has to answer "what exactly is this
install running?" without the user shelling into the box: which runtime
commit, which schema contract, whether the checkout matches the manifest,
what the backend actually reported, and a bundle they can hand to someone
else.

Everything here is read-only. The support bundle is the one place that
writes, and it writes only to the path it is given.

Public API:
    manifest_status() -> ManifestStatus
    capability_report() -> dict
    environment_report() -> dict
    write_support_bundle(path) -> Path
    redact(mapping) -> dict
"""
import io
import json
import os
import platform
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import modelctl


# The umbrella repository: modelctl/ lives inside it, llama.cpp is a
# submodule beside it.
REPO_ROOT = Path(os.environ.get(
    "MODELCTL_REPO_ROOT", Path(__file__).resolve().parent.parent))
MANIFEST_NAME = "integration-manifest.json"

# Anything whose *name* looks like a credential has its value replaced.
# Matching on the name rather than the value is deliberate: a token is not
# distinguishable from a hash by inspection, and a bundle that leaks once
# has leaked permanently.
_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE)",
    re.IGNORECASE)

# Values that look like a bearer token regardless of the variable name.
_SECRET_VALUE = re.compile(r"^(hf_|sk-|ghp_|gho_|github_pat_)")

REDACTED = "<redacted>"


def redact(mapping):
    """Redact credential-looking entries from a name -> value mapping."""
    out = {}
    for k, v in (mapping or {}).items():
        text = str(v)
        if _SECRET_NAME.search(str(k)) or _SECRET_VALUE.match(text):
            out[k] = REDACTED
        else:
            out[k] = text
    return out


def _git(args, cwd=None):
    """Run a git command, returning stripped stdout or "" on any failure."""
    try:
        r = subprocess.run(["git", *args], cwd=str(cwd or REPO_ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class ManifestStatus:
    """The integration manifest joined against the live checkout."""
    path: str = ""
    manifest: dict = field(default_factory=dict)
    error: str = ""
    modelctl_commit: str = ""
    submodule_pinned: str = ""
    submodule_checked_out: str = ""
    working_tree_dirty: bool = False
    # Things that mean the running system is not the integration the
    # manifest describes.
    mismatches: tuple = ()
    # Expected drift during development -- worth showing, not worth
    # alarming about.
    notes: tuple = ()
    # True when the running control plane is newer than the last
    # hardware-validated modelctl+llama.cpp pair. Normal between
    # promotions; the UI and support bundle surface it so future-you
    # knows which commit to roll back to when an experiment goes wrong.
    newer_than_validated: bool = False

    @property
    def present(self) -> bool:
        return bool(self.manifest) and not self.error

    @property
    def ok(self) -> bool:
        return self.present and not self.mismatches

    @property
    def validated_modelctl_commit(self) -> str:
        return (self.manifest.get("validated_modelctl_commit")
                or self.manifest.get("modelctl_commit", ""))

    @property
    def validated_llama_commit(self) -> str:
        return (self.manifest.get("validated_llama_commit")
                or self.manifest.get("llama_cpp_commit", ""))

    @property
    def upstream_base(self) -> str:
        return (self.manifest.get("upstream_base")
                or self.manifest.get("llama_cpp_upstream_base", ""))

    @property
    def validation_report(self) -> str:
        return (self.manifest.get("validation_report")
                or self.manifest.get("acceptance_report", ""))


def _commits_agree(a: str, b: str) -> bool:
    """True when two commit strings name the same commit.

    The manifest records short SHAs for readability while git reports full
    ones, so compare by prefix -- but never treat an empty or absurdly
    short string as a match.
    """
    a, b = (a or "").strip(), (b or "").strip()
    if len(a) < 7 or len(b) < 7:
        return False
    n = min(len(a), len(b))
    return a[:n].lower() == b[:n].lower()


def manifest_status() -> ManifestStatus:
    """Load the integration manifest and diff it against this checkout."""
    path = REPO_ROOT / MANIFEST_NAME
    status = ManifestStatus(path=str(path))
    try:
        status.manifest = json.loads(path.read_text())
    except FileNotFoundError:
        status.error = f"no {MANIFEST_NAME} at {path}"
        return status
    except (OSError, json.JSONDecodeError) as e:
        status.error = f"{MANIFEST_NAME} is unreadable: {e}"
        return status

    m = status.manifest
    status.modelctl_commit = _git(["rev-parse", "HEAD"])
    status.working_tree_dirty = bool(_git(["status", "--porcelain"]))

    # The *pinned* commit is what the umbrella repo records at HEAD, which
    # is not necessarily what is checked out in the submodule working tree.
    # Both matter, and for different reasons, so report both.
    #
    # Read the pin out of the commit tree. `git submodule status` cannot
    # supply it: that command prints the *checked-out* commit, flagging it
    # with a leading '+' when it differs from the pin. Parsing its SHA
    # yields the working-tree commit wearing the pinned commit's name,
    # which makes the two identical by construction and leaves the drift
    # check below unable to fire in the one situation it exists for.
    status.submodule_pinned = _git(["rev-parse", "HEAD:llama.cpp"])
    status.submodule_checked_out = _git(["rev-parse", "HEAD"],
                                        cwd=REPO_ROOT / "llama.cpp")

    mismatches = []
    notes = []
    declared_runtime = status.validated_llama_commit
    if declared_runtime and status.submodule_pinned:
        if not _commits_agree(declared_runtime, status.submodule_pinned):
            mismatches.append(
                f"manifest's validated llama.cpp is {declared_runtime[:12]} "
                f"but the submodule is pinned at "
                f"{status.submodule_pinned[:12]}")
    if status.submodule_pinned and status.submodule_checked_out:
        if not _commits_agree(status.submodule_pinned,
                              status.submodule_checked_out):
            mismatches.append(
                f"llama.cpp working tree is at "
                f"{status.submodule_checked_out[:12]}, not the pinned "
                f"{status.submodule_pinned[:12]} — the running binary may not "
                f"be the promoted runtime")

    # Control-plane drift is the normal state between promotions -- the
    # manifest is regenerated when a runtime is promoted, not on every
    # commit. Reporting it as a mismatch would leave the page permanently
    # red and train the user to ignore it. It IS flagged distinctly
    # (newer_than_validated) so rollback always has a named target.
    declared_ctl = status.validated_modelctl_commit
    if declared_ctl and status.modelctl_commit:
        if not _commits_agree(declared_ctl, status.modelctl_commit):
            status.newer_than_validated = True
            notes.append(
                f"running modelctl {status.modelctl_commit[:12]} is newer "
                f"than the last hardware-validated pair "
                f"(validated: {declared_ctl[:12]}); expected between "
                f"promotions -- that commit is the rollback target")
    if status.working_tree_dirty:
        notes.append("working tree has uncommitted changes")

    # Schema contracts: a bump in the code that the manifest never learned
    # about means the manifest is describing a wire format that no longer
    # exists.
    try:
        import modelctl_profiles
        declared = m.get("profile_schema")
        actual = modelctl_profiles.PROFILE_SCHEMA_VERSION
        if isinstance(declared, int) and declared != actual:
            mismatches.append(
                f"manifest declares profile_schema {declared}, code writes {actual}")
    except Exception:
        pass
    declared_caps = m.get("capability_schema")
    if isinstance(declared_caps, int) and declared_caps != 2:
        mismatches.append(
            f"manifest declares capability_schema {declared_caps}, "
            f"modelctl normalizes to 2")

    status.mismatches = tuple(mismatches)
    status.notes = tuple(notes)
    return status


def capability_report() -> dict:
    """What the resolved backend binary actually reports about itself."""
    report = {"binary": "", "candidates": [], "probe": None, "error": ""}
    try:
        candidates = [c for c in
                      [modelctl.LLAMA_SERVER_BIN] + modelctl.find_llama_server_candidates()
                      if c]
        report["candidates"] = [c for c in dict.fromkeys(candidates)
                                if os.access(c, os.X_OK)]
    except Exception as e:
        report["error"] = f"candidate search failed: {e}"
        return report

    binary = report["candidates"][0] if report["candidates"] else ""
    report["binary"] = binary
    if not binary:
        report["error"] = "no executable llama-server found"
        return report
    try:
        import modelctl_capabilities
        caps = modelctl_capabilities.probe_backend(binary)
        report["probe"] = caps
        report["capability_fingerprint"] = \
            modelctl_capabilities.capability_fingerprint(caps)
    except Exception as e:
        report["error"] = f"probe failed: {e}"
    return report


def environment_report() -> dict:
    """Paths, policy, and platform identity -- the boring facts a bug
    report needs and nobody remembers to include."""
    try:
        defaults = modelctl.load_defaults()
    except Exception:
        defaults = {}
    try:
        env_scripts = modelctl.find_env_script_candidates()
    except Exception:
        env_scripts = []
    try:
        import modelctl_benchmark
        modes = [m.value for m in modelctl_benchmark.BenchmarkMode]
    except Exception:
        modes = []
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "paths": {
            "models_dir": str(modelctl.DEFAULT_MODELS_DIR),
            "profiles_dir": str(modelctl.PROFILES_DIR),
            "llama_server": modelctl.LLAMA_SERVER_BIN,
            "llama_swap_config": str(modelctl.LLAMA_SWAP_CONFIG_PATH),
            "llama_swap_service": modelctl.LLAMA_SWAP_SERVICE_NAME,
            "llama_swap_base_url": modelctl.LLAMA_SWAP_BASE_URL,
            "repo_root": str(REPO_ROOT),
        },
        "oneapi_env_scripts": list(env_scripts),
        "benchmark_modes": modes,
        "defaults": defaults,
        # Only MODELCTL_* vars, and redacted: dumping the whole environment
        # into a shareable bundle is how credentials escape.
        "modelctl_env": redact({k: v for k, v in os.environ.items()
                                if k.startswith("MODELCTL_")}),
    }


def _redacted_profile(profile: dict) -> dict:
    """A profile with its env passthrough redacted.

    Profiles are documented as hand-editable and people put API keys in
    the env list.
    """
    out = dict(profile)
    env = out.get("env") or []
    safe = []
    for entry in env:
        if "=" in str(entry):
            k, _, v = str(entry).partition("=")
            safe.append(f"{k}={REDACTED}" if (_SECRET_NAME.search(k)
                                              or _SECRET_VALUE.match(v))
                        else f"{k}={v}")
        else:
            safe.append(str(entry))
    out["env"] = safe
    return out


def support_bundle_bytes(log_tail_lines: int = 500) -> bytes:
    """Build the support bundle in memory and return the zip bytes."""
    buf = io.BytesIO()
    status = manifest_status()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("integration-manifest.json",
                   json.dumps(status.manifest, indent=2))
        z.writestr("manifest-status.json", json.dumps({
            "path": status.path,
            "error": status.error,
            "modelctl_commit": status.modelctl_commit,
            "validated_modelctl_commit": status.validated_modelctl_commit,
            "validated_llama_commit": status.validated_llama_commit,
            "upstream_base": status.upstream_base,
            "validation_report": status.validation_report,
            "newer_than_validated": status.newer_than_validated,
            "submodule_pinned": status.submodule_pinned,
            "submodule_checked_out": status.submodule_checked_out,
            "working_tree_dirty": status.working_tree_dirty,
            "mismatches": list(status.mismatches),
            "notes": list(status.notes),
        }, indent=2))
        z.writestr("environment.json",
                   json.dumps(environment_report(), indent=2, default=str))
        z.writestr("capabilities.json",
                   json.dumps(capability_report(), indent=2, default=str))

        try:
            import modelctl_hardware
            snap = modelctl_hardware.capture_hardware_snapshot()
            from dataclasses import asdict
            z.writestr("hardware.json", json.dumps({
                "fingerprint": snap.fingerprint,
                "gpus": [asdict(g) for g in snap.gpus],
                "ram_total_bytes": snap.ram_total_bytes,
                "ram_available_bytes": snap.ram_available_bytes,
                "storage": [asdict(s) for s in snap.storage],
                "backend_fingerprints": snap.backend_fingerprints,
            }, indent=2, default=str))
        except Exception as e:
            z.writestr("hardware.json", json.dumps({"error": str(e)}))

        try:
            import modelctl_setup
            from dataclasses import asdict
            z.writestr("setup.json", json.dumps(
                [asdict(c) for c in modelctl_setup.probe_setup().checks],
                indent=2))
        except Exception as e:
            z.writestr("setup.json", json.dumps({"error": str(e)}))

        for path in sorted(Path(modelctl.PROFILES_DIR).glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            z.writestr(f"profiles/{path.name}",
                       json.dumps(_redacted_profile(data), indent=2))

        # Log tails only: a full llama-swap log can be hundreds of MB, and
        # the tail is what carries the failure.
        for log in sorted(Path(modelctl.PROFILES_DIR).glob("*/*.log")):
            try:
                lines = log.read_text(errors="replace").splitlines()
            except OSError:
                continue
            tail = "\n".join(lines[-log_tail_lines:])
            z.writestr(f"logs/{log.parent.name}-{log.name}", tail)

    return buf.getvalue()


def write_support_bundle(path) -> Path:
    """Write the support bundle to `path`. Never includes the web token."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(support_bundle_bytes())
    return path
