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
  2 — moe_weight_transfer_cache, device features, constraints, and cli
      flag declarations; one uniform per-GPU cache budget
  3 — adds moe_cache_per_device_budgets: --moe-cache-bytes accepts a
      device map (SYCL0=N,SYCL1=N), so each card gets the budget the
      planner reserved for it

normalize_capabilities() converts every schema to the current canonical
form so all consumers see a consistent representation; features absent
from an older schema normalize to false, never to an assumption.

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

from modelctl_paths import STATE_DIR
CAPABILITIES_DIR = STATE_DIR / "backend_capabilities"

# The canonical schema every consumer sees after normalization. One
# constant, not six scattered literals: a schema-3 migration must not
# have to hunt for magic numbers.
CAPABILITY_SCHEMA_VERSION = 3

# Bumped whenever normalize_capabilities() changes meaning. Cache entries
# store the RAW probe output and are re-normalized on every read, so a
# normalizer fix reaches even the pinned, never-rebuilt binary; this
# stamp exists for observability, not gating.
NORMALIZER_VERSION = 3

# SYCL driver init under load can exceed the old 10s. Overridable for
# slow machines without patching code.
PROBE_TIMEOUT = int(os.environ.get("MODELCTL_PROBE_TIMEOUT", "10") or 10)

# The probe is fast but binary hashing is not free; cache for the session.
# Keyed by (binary fingerprint, env fingerprint): a probe answered under
# the bare environment (possibly seeing zero SYCL devices) must never be
# served to the launch path that asked under its resolved env.
_session_cache: dict[tuple, dict] = {}

# stderr signatures of a binary that RAN its argument parser and rejected
# the flag -- the only outcome that may be persisted as "unsupported".
# Crashes, timeouts, and garbage output are transient ("error"): caching
# them silently stripped every fork feature until the binary was rebuilt,
# with no reprobe command to recover.
_REJECTION_MARKERS = ("invalid argument", "unrecognized argument",
                      "unknown argument", "unknown option", "invalid option")


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
    """Extract --version output for cache invalidation.

    Probed through modelctl's env candidates: a SYCL binary aborts without
    the oneAPI env, and its crash banner must not be stamped into the cache
    as a version identity. A probe that never exits 0 has no version."""
    try:
        import modelctl
        r = modelctl.run_llama_probe(binary_path, ["--version"], timeout=5)
    except Exception:
        return ""
    if r is None or r.returncode != 0:
        return ""
    return (r.stdout + r.stderr).strip()[:200]


def _run_probe(binary_path: str, extra_env: dict | None) -> tuple:
    """One probe attempt. Returns (verdict, raw):

    "ok"       -- exit 0 with a JSON object on stdout
    "rejected" -- the binary ran its argument parser and refused the flag
                  (stock llama.cpp); the only verdict safe to persist as
                  "this binary has no fork features"
    "error"    -- crash, timeout, unreadable binary, or garbage output;
                  transient by definition, must never be persisted
    """
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    try:
        r = subprocess.run([binary_path, "--modelctl-capabilities"],
                           capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT, env=env)
    except (subprocess.TimeoutExpired, OSError):
        return "error", None
    if r.returncode != 0:
        # A clean rejection exits via the arg parser with a recognizable
        # message and a small positive code; a crash exits via a signal
        # (negative returncode) or an abort, message or not.
        err = (r.stderr or "").lower()
        if r.returncode > 0 and any(m in err for m in _REJECTION_MARKERS):
            return "rejected", None
        return "error", None
    try:
        raw = json.loads(r.stdout)
    except json.JSONDecodeError:
        return "error", None
    if not isinstance(raw, dict):
        # Valid JSON that is not an object ([] or "ok") used to crash
        # resolution with AttributeError further down.
        return "error", None
    return "ok", raw


def _probe_raw(binary_path: str, env: dict | None = None) -> tuple:
    """Probe under the caller's launch env, the bare env, then the env
    scripts the launch path uses (SYCL builds initialize the GPU driver
    even for the probe, so they crash without their oneAPI env).

    Returns (verdict, raw). "ok" and "rejected" are decisive -- the
    binary answered coherently -- and short-circuit; only crashes keep
    trying the next environment."""
    for attempt_env in ([env] if env is not None else []) + [None]:
        verdict, raw = _run_probe(binary_path, attempt_env)
        if verdict in ("ok", "rejected"):
            return verdict, raw
    # Still undecided (crashes only): gather env scripts lazily -- they
    # cost subprocess spawns and must not run when the bare probe answers.
    try:
        import modelctl
        for script in modelctl.find_env_script_candidates():
            script_env = modelctl.source_env_script(script)
            if script_env:
                modelctl.ensure_binary_ld_library_path(script_env, binary_path)
                verdict, raw = _run_probe(binary_path, script_env)
                if verdict in ("ok", "rejected"):
                    return verdict, raw
    except ImportError:
        pass
    return "error", None


def _classify_probe_failure(binary_path: str, status: str = "unsupported") -> dict:
    """Build a minimal all-false capability dict.

    status "unsupported": the binary coherently rejected the probe (stock
    llama.cpp) -- cacheable. status "error": the probe itself failed
    (crash/timeout/garbage) -- fail closed for this call, session-cached
    only, retried next process."""
    return {
        "schema": CAPABILITY_SCHEMA_VERSION,
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
            "moe_cache_per_device_budgets": False,
        },
        "constraints": {
            "moe_cache_backend": "",
            "moe_cache_min_batch": 0,
            "moe_cache_supported_projections": [],
        },
        "cli": {},
        "_probe_status": status,
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
            # Schema 1/2 backends take a single uniform budget; the
            # per-device map form arrived with schema 3.
            "moe_cache_per_device_budgets": False,
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
            "schema": CAPABILITY_SCHEMA_VERSION,
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
        "moe_cache_prefetch": False,  # not implemented; forced false
        # Routed MoE ops honour their own offload minimum. Passed
        # through from the backend rather than defaulted, because the
        # acceptance matrix gates its offload sweep on it: dropping the
        # key made those cells skip against a runtime that does support
        # it, which is the opposite failure to the one the gate exists
        # to prevent but just as misleading.
        "moe_offload_threshold_control": bool(
            features.get("moe_offload_threshold_control")),
        # Schema 3: --moe-cache-bytes accepts SYCL0=N,SYCL1=N. Absent on
        # schema 1/2 backends, which take one uniform budget -- the
        # planner and build_moe_cache_args collapse the map for those.
        "moe_cache_per_device_budgets": bool(
            features.get("moe_cache_per_device_budgets")),
    }

    # Force prefetch false until implemented. Hybrid is allowed through
    # when the backend reports it.
    raw_features = dict(features)

    constraints = raw_caps.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    def _int_or(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _list_or(value):
        return list(value) if isinstance(value, (list, tuple)) else []

    # Guarded conversions: a malformed constraint value from a newer or
    # foreign fork must degrade fail-closed, not raise out of every
    # launch/preview path for the profile.
    canonical_constraints = {
        "moe_cache_backend": constraints.get("moe_cache_backend", ""),
        "moe_cache_min_batch": _int_or(constraints.get("moe_cache_min_batch", 0)),
        "moe_cache_supported_projections": _list_or(
            constraints.get("moe_cache_supported_projections")),
        "moe_hybrid_supported_archs": _list_or(
            constraints.get("moe_hybrid_supported_archs")),
        "moe_hybrid_supported_quant": _list_or(
            constraints.get("moe_hybrid_supported_quant")),
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
        "schema": CAPABILITY_SCHEMA_VERSION,
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


def _env_fingerprint(env: dict | None) -> str:
    """Digest of the CALLER's requested probe environment.

    Cache entries are keyed by it: a SYCL build probed under the bare env
    can truthfully answer with zero SYCL devices, and that answer must
    never be served to the launch path asking under its resolved env.
    """
    if not env:
        return "default"
    text = json.dumps(sorted(env.items()), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _cache_path(binary_path: str, env: dict | None = None) -> Path:
    fp = _binary_fingerprint(binary_path)
    return CAPABILITIES_DIR / f"{fp}-{_env_fingerprint(env)}.json"


def _read_cache(binary_path: str, env: dict | None = None) -> dict | None:
    """Read a cache entry and normalize it FRESH from the stored raw probe
    output. Entries in the legacy format (normalized caps, no "raw" key)
    are ignored so they get re-probed once: they were written by whatever
    normalizer was current then, and its bugs were frozen in with them.
    """
    path = _cache_path(binary_path, env)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(entry, dict) or "raw" not in entry:
        return None
    if entry.get("_version") != _version_string(binary_path):
        return None
    raw = entry["raw"]
    if not isinstance(raw, dict):
        return None
    if raw.get("_probe_status") == "unsupported":
        return _classify_probe_failure(binary_path)
    return normalize_capabilities(raw)


def _write_cache(binary_path: str, raw: dict, env: dict | None = None):
    """Persist the RAW probe output (not the normalized form -- see
    _read_cache). Best-effort; failures are silent."""
    CAPABILITIES_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "_version": _version_string(binary_path),
        "_binary": binary_path,
        "_normalizer_version": NORMALIZER_VERSION,
        "raw": raw,
    }
    try:
        _cache_path(binary_path, env).write_text(json.dumps(entry, indent=2))
    except OSError:
        pass


def probe_backend(binary_path: str, env: dict | None = None) -> dict:
    """Probe a binary for backend capabilities. Returns a normalized dict
    in the canonical schema.

    `env` is the environment the binary will actually launch under
    (resolve_backend passes its resolved one); results are cached per
    (binary content, requested env) so an env-less diagnostics probe can
    never shadow the launch path's answer.

    _probe_status is one of:
      "ok"          -- binary responded with a valid JSON object
      "unsupported" -- binary ran its arg parser and rejected the flag
                       (stock llama.cpp); persisted
      "error"       -- the probe crashed, timed out, or produced garbage;
                       fail-closed for this call, session-cached only,
                       retried next process. Never persisted: one bad
                       probe used to brand a fork binary "unsupported"
                       until it was rebuilt.
    """
    fp = _binary_fingerprint(binary_path)
    key = (fp, _env_fingerprint(env))
    if key in _session_cache:
        return _session_cache[key]

    cached = _read_cache(binary_path, env)
    if cached is not None:
        _session_cache[key] = cached
        return cached

    verdict, raw = _probe_raw(binary_path, env=env)
    if verdict == "ok":
        raw.setdefault("schema", 1)
        raw.setdefault("backend", "unknown")
        raw.setdefault("build", "")
        raw.setdefault("features", {})
        raw.setdefault("cli", {})
        raw["_probe_status"] = "ok"
        _write_cache(binary_path, raw, env)
        normalized = normalize_capabilities(raw)
        _session_cache[key] = normalized
        return normalized

    if verdict == "rejected":
        _write_cache(binary_path, {"_probe_status": "unsupported"}, env)
        caps = _classify_probe_failure(binary_path)
        _session_cache[key] = caps
        return caps

    caps = _classify_probe_failure(binary_path, status="error")
    _session_cache[key] = caps
    return caps


def get_cached_capabilities(binary_path: str, env: dict | None = None) -> dict | None:
    """Return cached capabilities if available, without running a probe.

    Falls back to any env variant's entry for this binary (display
    callers ask without an env; serving them the launch env's answer
    beats serving nothing)."""
    fp = _binary_fingerprint(binary_path)
    key = (fp, _env_fingerprint(env))
    if key in _session_cache:
        return _session_cache[key]
    cached = _read_cache(binary_path, env)
    if cached is None and CAPABILITIES_DIR.exists():
        candidates = sorted(CAPABILITIES_DIR.glob(f"{fp}-*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                entry = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            raw = entry.get("raw")
            if isinstance(raw, dict):
                if raw.get("_probe_status") == "unsupported":
                    cached = _classify_probe_failure(binary_path)
                else:
                    cached = normalize_capabilities(raw)
                break
    if cached is not None:
        _session_cache[key] = cached
    return cached


def clear_cache(binary_path: str | None = None):
    """Remove cached capability entries and their session copies.

    With a binary_path, clears only that binary's entries (every env
    variant) -- the reprobe escape hatch `modelctl capabilities
    --reprobe` and the /setup button use this."""
    if binary_path is None:
        _session_cache.clear()
        if CAPABILITIES_DIR.exists():
            for f in CAPABILITIES_DIR.glob("*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
        return
    fp = _binary_fingerprint(binary_path)
    for key in [k for k in _session_cache if k[0] == fp]:
        _session_cache.pop(key, None)
    if CAPABILITIES_DIR.exists():
        for f in list(CAPABILITIES_DIR.glob(f"{fp}-*.json")) \
                + list(CAPABILITIES_DIR.glob(f"{fp}.json")):
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

    Implemented by the fork's Tasks G4/G5 (2026-07-31): declined misses
    skip staging entirely and their rows run on CPU inside mul_mat_id,
    token-identical to the non-hybrid path. Measured SLOWER than the
    plain transfer cache on the 122B IQ1_M target (2.5 vs 4.3 t/s), so
    it stays opt-in; the experimental-margin guardrail keeps it from
    outranking measured baselines automatically.
    """
    return bool(caps.get("features", {}).get("moe_hybrid_cpu_miss"))


def supports_metrics(caps: dict) -> bool:
    """True if the backend exposes cache hit/miss/eviction metrics."""
    return bool(caps.get("features", {}).get("moe_cache_metrics"))


def supports_prefetch(caps: dict) -> bool:
    """True if the backend supports expert prefetching.

    Always false; prefetch is not implemented and normalization forces it off.
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
