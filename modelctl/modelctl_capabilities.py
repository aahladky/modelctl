"""Backend capability probing and caching for modelctl.

Probes a llama-server binary for feature support (MoE weight transfer
cache, hybrid CPU miss execution, cache metrics, etc.) by running:

    llama-server --modelctl-capabilities

and caching the JSON response keyed by binary content hash.  Stock
llama.cpp binaries don't support this flag; the probe classifies the
result accordingly.

Schema versions:
  0 — stock binary that rejects the probe (no cache support)
  1 — early fork response (moe_expert_cache, moe_cache_sycl, etc.)
  2 — canonical schema with moe_weight_transfer_cache, device features,
      constraints, and cli flag declarations

normalize_capabilities() converts schema 0/1 to the internal canonical
form so all consumers see a consistent representation.  The raw probe
output is preserved in _raw_probe for debugging.

Public API:
    probe_backend(binary_path) -> dict
    get_cached_capabilities(binary_path) -> dict | None
    clear_cache()
    normalize_capabilities(raw_caps) -> dict
    is_cache_capable(caps) -> bool
    is_weight_transfer_cache_capable(caps) -> bool
    supports_hybrid_miss(caps) -> bool
    supports_metrics(caps) -> bool
    supports_prefetch(caps) -> bool
    backend_features(caps) -> dict
    backend_constraints(caps) -> dict
    backend_cli(caps) -> dict
    capability_fingerprint(caps) -> str
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


def _run_probe(binary_path: str, extra_env: dict | None) -> dict | None:
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    try:
        r = subprocess.run([binary_path, "--modelctl-capabilities"],
                           capture_output=True, text=True, timeout=10, env=env)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        return None


def _probe_raw(binary_path: str) -> dict | None:
    """Run --modelctl-capabilities and parse JSON, or None on failure.

    SYCL builds initialize the backend registry (and thus the GPU driver)
    even for the probe, so they crash without their oneAPI env. Retry with
    the env scripts the launch path uses before giving up."""
    raw = _run_probe(binary_path, None)
    if raw is not None:
        return raw
    try:
        import modelctl
        for script in modelctl.find_env_script_candidates():
            env = modelctl.source_env_script(script)
            if env:
                modelctl.ensure_binary_ld_library_path(env, binary_path)
                raw = _run_probe(binary_path, env)
                if raw is not None:
                    return raw
    except ImportError:
        pass
    return None


def _classify_probe_failure(binary_path: str) -> dict:
    """Build a minimal capability dict for a binary that doesn't support
    the probe command (stock llama.cpp or non-llama.cpp backend)."""
    return {
        "schema": 2,
        "backend": "unknown",
        "build": {"commit": "", "compiler": "", "dynamic_backends": False},
        "devices": [],
        "features": {
            "moe_weight_transfer_cache": False,
            "moe_hybrid_cpu_miss": False,
            "moe_cache_metrics": False,
            "moe_cache_prefill_policy": False,
            "moe_cache_reset": False,
            "moe_cache_prefetch": False,
            "moe_offload_threshold_control": False,
        },
        "constraints": {
            "moe_cache_backend": "",
            "moe_cache_min_batch": 0,
            "moe_cache_supported_projections": [],
        },
        "cli": {},
        "_probe_status": "unsupported",
    }


def normalize_capabilities(raw_caps: dict) -> dict:
    """Convert any schema version to the canonical internal representation.

    Schema 0 (unsupported probe): all features false.
    Schema 1 (early fork): map old names to canonical names.
    Schema 2+: pass through with defaults filled.

    Never invents support — missing or unknown fields evaluate false.
    """
    schema = raw_caps.get("schema", 0)
    status = raw_caps.get("_probe_status", "ok")

    if schema == 0 or status == "unsupported":
        return _classify_probe_failure(raw_caps.get("_binary", ""))

    features = raw_caps.get("features", {})
    cli = raw_caps.get("cli", {})
    build = raw_caps.get("build", "")
    devices = raw_caps.get("devices", [])

    if schema == 1:
        # Map schema 1 names to canonical names.
        # moe_expert_cache was the generic "cache works" flag.
        # moe_cache_sycl was the SYCL-specific flag.
        # Together they mean: weight transfer cache on SYCL.
        has_cache = bool(features.get("moe_expert_cache"))
        has_sycl = bool(features.get("moe_cache_sycl"))

        canonical_features = {
            "moe_weight_transfer_cache": has_cache and has_sycl,
            "moe_hybrid_cpu_miss": bool(features.get("moe_hybrid_cpu_miss")),
            "moe_cache_metrics": bool(features.get("moe_cache_metrics")),
            "moe_cache_prefill_policy": bool(features.get("moe_cache_prefill_policy")),
            "moe_cache_reset": has_cache,
            "moe_cache_prefetch": False,
            # Schema 1 predates a per-op-type offload threshold.
            "moe_offload_threshold_control": False,
        }

        # Map schema 1 CLI names to canonical names.
        canonical_cli = {}
        if cli.get("cache_bytes"):
            canonical_cli["moe_cache_bytes"] = cli["cache_bytes"]
        if cli.get("cache_policy"):
            canonical_cli["moe_cache_policy"] = cli["cache_policy"]
        if cli.get("admission_misses"):
            canonical_cli["moe_cache_admission"] = cli["admission_misses"]
        if cli.get("prefill_admission"):
            canonical_cli["moe_cache_prefill"] = cli["prefill_admission"]

        # Normalize build to structured form.
        if isinstance(build, str):
            build = {"commit": build, "compiler": "", "dynamic_backends": False}

        # Normalize devices: schema 1 has string list, schema 2 has objects.
        normalized_devices = []
        for d in devices:
            if isinstance(d, str):
                normalized_devices.append({
                    "type": "SYCL" if "SYCL" in d else "CPU",
                    "name": d,
                    "index": len(normalized_devices),
                    "features": {"moe_weight_transfer_cache": has_cache and has_sycl},
                })
            elif isinstance(d, dict):
                normalized_devices.append(d)

        return {
            "schema": 2,
            "backend": raw_caps.get("backend", "unknown"),
            "build": build,
            "devices": normalized_devices,
            "features": canonical_features,
            "constraints": {
                "moe_cache_backend": "SYCL" if has_sycl else "",
                "moe_cache_min_batch": 0,
                "moe_cache_supported_projections": [],
            },
            "cli": canonical_cli,
            "_probe_status": "ok",
            "_raw_schema": 1,
        }

    # Schema 2+: fill defaults for missing fields.
    if isinstance(build, str):
        build = {"commit": build, "compiler": "", "dynamic_backends": False}

    canonical_features = {
        "moe_weight_transfer_cache": bool(features.get("moe_weight_transfer_cache")),
        "moe_hybrid_cpu_miss": bool(features.get("moe_hybrid_cpu_miss")),
        "moe_cache_metrics": bool(features.get("moe_cache_metrics")),
        "moe_cache_prefill_policy": bool(features.get("moe_cache_prefill_policy")),
        "moe_cache_reset": bool(features.get("moe_cache_reset")),
        "moe_cache_prefetch": False,  # Not implemented until Phase 9
        # Routed MoE ops honour their own offload minimum. Passed
        # through from the backend rather than defaulted, because the
        # acceptance matrix gates its offload sweep on it: dropping the
        # key made those cells skip against a runtime that does support
        # it, which is the opposite failure to the one the gate exists
        # to prevent but just as misleading.
        "moe_offload_threshold_control": bool(
            features.get("moe_offload_threshold_control")),
    }

    # Force prefetch false until implemented. Hybrid is now allowed through
    # when the backend reports it (Phase 7 implements the control plane).
    raw_features = dict(features)

    constraints = raw_caps.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    canonical_constraints = {
        "moe_cache_backend": constraints.get("moe_cache_backend", ""),
        "moe_cache_min_batch": int(constraints.get("moe_cache_min_batch", 0)),
        "moe_cache_supported_projections": list(
            constraints.get("moe_cache_supported_projections", [])),
        "moe_hybrid_supported_archs": list(
            constraints.get("moe_hybrid_supported_archs", [])),
        "moe_hybrid_supported_quant": list(
            constraints.get("moe_hybrid_supported_quant", [])),
        "moe_hybrid_can_overlap": bool(
            constraints.get("moe_hybrid_can_overlap", False)),
    }

    canonical_cli = {}
    for key in ("moe_cache_bytes", "moe_cache_policy", "moe_cache_admission",
                "moe_cache_prefill", "moe_cache_reset", "moe_hybrid_mode"):
        if cli.get(key):
            canonical_cli[key] = cli[key]

    normalized_devices = []
    for d in devices:
        if isinstance(d, dict):
            # Ensure device features also fail closed.
            dev_features = d.get("features", {})
            normalized_devices.append({
                "type": d.get("type", "unknown"),
                "name": d.get("name", ""),
                "index": d.get("index", 0),
                "features": {
                    "moe_weight_transfer_cache": bool(
                        dev_features.get("moe_weight_transfer_cache")),
                },
            })

    return {
        "schema": 2,
        "backend": raw_caps.get("backend", "unknown"),
        "build": build,
        "devices": normalized_devices,
        "features": canonical_features,
        "constraints": canonical_constraints,
        "cli": canonical_cli,
        "_probe_status": "ok",
        "_raw_schema": schema,
        "_raw_features": raw_features,
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
    """Probe a binary for backend capabilities.  Returns a normalized dict
    with schema 2 representation.

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
        # Re-normalize in case we upgraded from schema 1 to 2.
        if cached.get("schema") != 2:
            cached = normalize_capabilities(cached)
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
        normalized = normalize_capabilities(raw)
        _write_cache(binary_path, normalized)
        _session_cache[fp] = normalized
        return normalized

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
        if cached.get("schema") != 2:
            cached = normalize_capabilities(cached)
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


# --- Canonical feature queries (consumers should use these) ---

def is_cache_capable(caps: dict) -> bool:
    """True if the backend supports any form of MoE expert caching.

    This is the broad gate: if false, no cache variant is generated.
    """
    return bool(caps.get("features", {}).get("moe_weight_transfer_cache"))


def is_weight_transfer_cache_capable(caps: dict) -> bool:
    """True if the backend supports GPU-side weight transfer caching."""
    return bool(caps.get("features", {}).get("moe_weight_transfer_cache"))


def has_moe_offload_threshold_control(caps: dict) -> bool:
    """True if routed MoE ops have their own offload threshold.

    Without this the runtime honours only the global
    GGML_OP_OFFLOAD_MIN_BATCH, so a cell setting
    GGML_OP_OFFLOAD_MOE_MIN_BATCH would run identically to the baseline and
    report a difference of zero -- a measurement of nothing that reads as a
    real result. The offload sweep is gated on it rather than trusting that
    setting a variable did something.
    """
    return bool(caps.get("features", {}).get("moe_offload_threshold_control"))


def is_sycl_cache_capable(caps: dict) -> bool:
    """True if the backend supports GPU-side expert caching on SYCL.

    Alias for is_weight_transfer_cache_capable — the only current
    implementation is SYCL-based.
    """
    return bool(caps.get("features", {}).get("moe_weight_transfer_cache"))


def supports_hybrid_miss(caps: dict) -> bool:
    """True if cache misses can execute on CPU without blocking GPU.

    Always false until Phase 7 implements true hybrid execution.
    """
    return bool(caps.get("features", {}).get("moe_hybrid_cpu_miss"))


def supports_metrics(caps: dict) -> bool:
    """True if the backend exposes cache hit/miss/eviction metrics."""
    return bool(caps.get("features", {}).get("moe_cache_metrics"))


def supports_prefetch(caps: dict) -> bool:
    """True if the backend supports expert prefetching.

    Always false until Phase 9 implements prefetch.
    """
    return bool(caps.get("features", {}).get("moe_cache_prefetch"))


def supports_prefill_policy(caps: dict) -> bool:
    """True if the backend supports prefill/decode phase admission."""
    return bool(caps.get("features", {}).get("moe_cache_prefill_policy"))


def supports_reset(caps: dict) -> bool:
    """True if the backend supports cache reset."""
    return bool(caps.get("features", {}).get("moe_cache_reset"))


def backend_features(caps: dict) -> dict:
    """Return the normalized features dict."""
    return caps.get("features", {})


def backend_constraints(caps: dict) -> dict:
    """Return the normalized constraints dict."""
    return caps.get("constraints", {})


def backend_cli(caps: dict) -> dict:
    """Return the canonical CLI flag name mapping."""
    return caps.get("cli", {})


def backend_build(caps: dict) -> dict:
    """Return the build info dict."""
    return caps.get("build", {})


def capability_fingerprint(caps: dict) -> str:
    """Stable digest of a capability response.

    Observations record this so a runtime whose *reported* capabilities
    changed -- a rebuilt fork that gained or lost a feature, or a backend
    loaded with different devices visible -- invalidates measurements taken
    under the old contract, even when the binary path is unchanged.

    Internal bookkeeping keys (probe cache metadata, the raw
    pre-normalization copy) are excluded so re-probing an unchanged binary
    yields an unchanged digest.  _probe_status is kept: "unsupported" and
    "ok" are genuinely different contracts.
    """
    material = {k: v for k, v in (caps or {}).items()
                if not k.startswith("_") or k == "_probe_status"}
    text = json.dumps(material, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
