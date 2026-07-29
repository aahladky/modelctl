"""Backend capability probing and caching for modelctl.

Probes a llama-server binary for feature support (MoE expert cache,
hybrid CPU miss execution, cache metrics, etc.) by running:

    llama-server --modelctl-capabilities

and caching the JSON response keyed by binary content hash.  Stock
llama.cpp binaries don't support this flag; the probe classifies the
result accordingly.

The capability dict is consumed by:
  - modelctl.build_moe_cache_args() for flag name resolution
  - modelctl.preflight_moe_cache() for launch-time validation
  - modelctl_plans.compile_launch_plans() for cache-variant filtering
  - modelctl_backends.LlamaCppAdapter for adapter-level methods

Public API:
    probe_backend(binary_path) -> dict
    get_cached_capabilities(binary_path) -> dict | None
    clear_cache()
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

STATE_DIR = Path(os.environ.get(
    "MODELCTL_STATE_DIR",
    Path.home() / ".local" / "share" / "modelctl"))
CAPABILITIES_DIR = STATE_DIR / "backend_capabilities"

# The probe is fast but binary hashing is not free; cache for the session.
_session_cache: dict[str, dict] = {}


def _binary_fingerprint(binary_path: str) -> str:
    """Content hash of the binary, used as cache key."""
    h = hashlib.sha256()
    try:
        with open(binary_path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        # Binary unreadable; return a hash of the path itself so we can
        # still cache the "probe failed" result.
        h.update(binary_path.encode())
    return h.hexdigest()[:16]


def _version_string(binary_path: str) -> str:
    """Extract --version output for cache invalidation."""
    try:
        r = subprocess.run([binary_path, "--version"],
                           capture_output=True, text=True, timeout=5)
        return (r.stdout + r.stderr).strip()[:200]
    except Exception:
        return ""


def _probe_raw(binary_path: str) -> dict | None:
    """Run --modelctl-capabilities and parse JSON, or None on failure."""
    try:
        r = subprocess.run([binary_path, "--modelctl-capabilities"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        return None


def _classify_probe_failure(binary_path: str) -> dict:
    """Build a minimal capability dict for a binary that doesn't support
    the probe command (stock llama.cpp or non-llama.cpp backend)."""
    return {
        "schema": 0,
        "backend": "unknown",
        "build": "",
        "features": {},
        "cli": {},
        "_probe_status": "unsupported",
    }


def _cache_path(binary_path: str) -> Path:
    fp = _binary_fingerprint(binary_path)
    return CAPABILITIES_DIR / f"{fp}.json"


def _read_cache(binary_path: str) -> dict | None:
    """Read cached capabilities if the binary fingerprint and probe schema
    still match."""
    path = _cache_path(binary_path)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
        # Invalidate if the binary version changed.
        if cached.get("_version") != _version_string(binary_path):
            return None
        # Invalidate if the probe schema itself changed.
        if cached.get("schema", 0) < 0:
            return None
        return cached
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(binary_path: str, caps: dict):
    """Persist capabilities to disk.  Best-effort; failures are silent."""
    CAPABILITIES_DIR.mkdir(parents=True, exist_ok=True)
    caps["_version"] = _version_string(binary_path)
    caps["_binary"] = binary_path
    try:
        _cache_path(binary_path).write_text(json.dumps(caps, indent=2))
    except OSError:
        pass


def probe_backend(binary_path: str) -> dict:
    """Probe a binary for backend capabilities.  Returns a dict with at
    minimum: schema, backend, build, features, cli, _probe_status.

    _probe_status is one of:
      "ok"         — binary responded with valid JSON
      "unsupported" — binary does not support --modelctl-capabilities

    Results are cached on disk by binary content hash and invalidated
    when the binary content, version string, or probe schema changes.
    """
    # Session cache first.
    fp = _binary_fingerprint(binary_path)
    if fp in _session_cache:
        return _session_cache[fp]

    # Disk cache next.
    cached = _read_cache(binary_path)
    if cached is not None:
        _session_cache[fp] = cached
        return cached

    # Live probe.
    raw = _probe_raw(binary_path)
    if raw is not None:
        raw.setdefault("schema", 1)
        raw.setdefault("backend", "unknown")
        raw.setdefault("build", "")
        raw.setdefault("features", {})
        raw.setdefault("cli", {})
        raw["_probe_status"] = "ok"
        _write_cache(binary_path, raw)
        _session_cache[fp] = raw
        return raw

    # Unsupported binary.
    caps = _classify_probe_failure(binary_path)
    _write_cache(binary_path, caps)
    _session_cache[fp] = caps
    return caps


def get_cached_capabilities(binary_path: str) -> dict | None:
    """Return cached capabilities if available, without running a probe."""
    fp = _binary_fingerprint(binary_path)
    if fp in _session_cache:
        return _session_cache[fp]
    cached = _read_cache(binary_path)
    if cached is not None:
        _session_cache[fp] = cached
    return cached


def clear_cache():
    """Remove all cached capability files and the session cache."""
    _session_cache.clear()
    if CAPABILITIES_DIR.exists():
        for f in CAPABILITIES_DIR.glob("*.json"):
            try:
                f.unlink()
            except OSError:
                pass


def is_cache_capable(caps: dict) -> bool:
    """True if the backend supports MoE expert caching at all."""
    return bool(caps.get("features", {}).get("moe_expert_cache"))


def is_sycl_cache_capable(caps: dict) -> bool:
    """True if the backend supports GPU-side expert caching on SYCL."""
    return bool(caps.get("features", {}).get("moe_cache_sycl"))


def supports_hybrid_miss(caps: dict) -> bool:
    """True if cache misses can execute on CPU without blocking GPU."""
    return bool(caps.get("features", {}).get("moe_hybrid_cpu_miss"))


def supports_metrics(caps: dict) -> bool:
    """True if the backend exposes cache hit/miss/eviction metrics."""
    return bool(caps.get("features", {}).get("moe_cache_metrics"))


def supports_prefetch(caps: dict) -> bool:
    """True if the backend supports expert prefetching."""
    return bool(caps.get("features", {}).get("moe_cache_prefetch"))
