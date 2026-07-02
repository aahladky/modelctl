#!/usr/bin/env python3
"""
modelctl - search, pull, and configure local GGUF models from Hugging Face.

Backend-agnostic: generates a run.sh (raw llama-server command), a
router preset.ini, and an Ollama-style Modelfile from one
saved profile. Profiles are plain JSON so they're easy to inspect,
edit by hand, or regenerate later.
"""
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import modelctl_vram

try:
    import yaml
except ImportError:
    yaml = None

STATE_DIR = Path(os.environ.get("MODELCTL_HOME", Path.home() / ".local/share/modelctl"))
PROFILES_DIR = STATE_DIR / "profiles"
DEFAULT_MODELS_DIR = Path(os.environ.get("MODELCTL_MODELS_DIR", Path.home() / "models"))

_resolved = os.environ.get("MODELCTL_LLAMA_SERVER") or shutil.which("llama-server")
LLAMA_SERVER_BIN = _resolved or "llama-server"
LLAMA_SERVER_RESOLVED = _resolved is not None
ROUTER_PRESET_PATH = Path(os.environ.get("MODELCTL_ROUTER_PRESET", Path.home() / "llama-router" / "router.preset.ini"))
# Not consumed by modelctl itself -- read by the operator when launching the router
# process by hand, e.g. `llama-server --port $MODELCTL_ROUTER_PORT --models-preset ...`.
ROUTER_PORT = os.environ.get("MODELCTL_ROUTER_PORT", "7071")
# The systemd --user unit running the router-mode llama-server. sync_router_preset()
# restarts this after writing a changed preset, since router mode has no
# --watch-config hot-reload -- without a restart, an updated preset just sits there
# unused until someone notices and restarts the process by hand.
ROUTER_SERVICE_NAME = os.environ.get("MODELCTL_ROUTER_SERVICE", "llama-router.service")

# Hermes Agent integration: modelctl can keep Hermes' custom_providers list
# in sync with saved profiles so pulled models show up in `hermes model`.
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_CONFIG = Path(os.environ.get("MODELCTL_HERMES_CONFIG", HERMES_HOME / "config.yaml"))
# Base URL of the router / OpenAI-compatible endpoint Hermes should talk to.
ROUTER_BASE_URL = os.environ.get("MODELCTL_ROUTER_BASE_URL")

# Defaults tuned for serving local GGUF models to Hermes Agent. Hermes sends
# long contexts and benefits from KV cache quantization for memory headroom.
DEFAULT_DEVICE = os.environ.get("MODELCTL_DEFAULT_DEVICE", "SYCL0")
DEFAULT_CTX = int(os.environ.get("MODELCTL_DEFAULT_CTX", "32768"))
DEFAULT_KV_QUANT = os.environ.get("MODELCTL_DEFAULT_KV_QUANT", "q8_0")
DEFAULT_FLASH_ATTN = os.environ.get("MODELCTL_DEFAULT_FLASH_ATTN", "auto")
DEFAULT_TTL = int(os.environ.get("MODELCTL_DEFAULT_TTL", "3600"))
# Default GPU split strategy: load across both GPUs instead of pinning to one.
DEFAULT_SPLIT_MODE = os.environ.get("MODELCTL_DEFAULT_SPLIT_MODE", "layer")
DEFAULT_TENSOR_SPLIT = os.environ.get("MODELCTL_DEFAULT_TENSOR_SPLIT", "3,1")
# Multi-token prediction (self-speculative decoding via --spec-type draft-mtp).
# Off by default: it only works if the checkpoint was trained with MTP draft
# heads AND the specific GGUF conversion preserved them -- not universal, and
# not something modelctl can detect from the profile alone. Opt in per model
# once you've confirmed the GGUF actually has them.
DEFAULT_MTP = os.environ.get("MODELCTL_DEFAULT_MTP", "off")

# Persisted user defaults live next to profiles so they survive re-installs.
DEFAULTS_PATH = STATE_DIR / "defaults.json"

# Cached SYCL-name -> xpu-smi-id mapping, built by matching total VRAM
# between `llama-server --list-devices` and `xpu-smi discovery`. Rebuild
# with `modelctl place --remap` after hardware changes.
GPU_MAP_PATH = STATE_DIR / "gpu_map.json"
DEFAULT_VRAM_LIMIT_PCT = int(os.environ.get("MODELCTL_DEFAULT_VRAM_LIMIT_PCT", "90"))
DEFAULT_PRIMARY_GPU = os.environ.get("MODELCTL_DEFAULT_PRIMARY_GPU", "")


# Env vars worth carrying into generated configs if they're set in the
# shell modelctl is run from (e.g. after sourcing llama-sycl-env.sh).
# Override the list with MODELCTL_PASSTHROUGH_ENV="VAR1,VAR2,...".
DEFAULT_PASSTHROUGH = ["LD_LIBRARY_PATH", "SYCL_CACHE_PERSISTENT", "ONEAPI_DEVICE_SELECTOR", "GGML_SYCL_DISABLE_OPT"]
PASSTHROUGH_VARS = [
    v.strip()
    for v in os.environ.get("MODELCTL_PASSTHROUGH_ENV", ",".join(DEFAULT_PASSTHROUGH)).split(",")
    if v.strip()
]

api = HfApi()


def load_defaults() -> dict:
    """Load persisted user defaults. Env vars always take precedence."""
    persisted = {}
    if DEFAULTS_PATH.exists():
        try:
            persisted = json.loads(DEFAULTS_PATH.read_text())
        except json.JSONDecodeError:
            persisted = {}

    def pick(key, fallback):
        env_key = f"MODELCTL_DEFAULT_{key.upper()}"
        if os.environ.get(env_key) is not None:
            return os.environ[env_key]
        return persisted.get(key, fallback)

    return {
        "device": pick("device", DEFAULT_DEVICE),
        "ctx": int(pick("ctx", DEFAULT_CTX)),
        "kv_quant": pick("kv_quant", DEFAULT_KV_QUANT),
        "flash_attn": pick("flash_attn", DEFAULT_FLASH_ATTN),
        "ttl": int(pick("ttl", DEFAULT_TTL)),

        "split_mode": pick("split_mode", DEFAULT_SPLIT_MODE),
        "tensor_split": pick("tensor_split", DEFAULT_TENSOR_SPLIT),
        "mtp": pick("mtp", DEFAULT_MTP),
        "vram_limit_pct": int(pick("vram_limit_pct", DEFAULT_VRAM_LIMIT_PCT)),
        "primary_gpu": pick("primary_gpu", DEFAULT_PRIMARY_GPU),
    }


def save_defaults(defaults: dict):
    """Persist user defaults to disk."""
    DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULTS_PATH.write_text(json.dumps(defaults, indent=2) + "\n")


def prompt_defaults():
    """Interactively configure default runtime settings for new profiles."""
    current = load_defaults()
    print("\n--- modelctl defaults (blank = leave unchanged) ---")
    print(f"Current defaults: {json.dumps(current, indent=2)}\n")

    device = input(f"GPU device, '-' to clear [{current.get('device', '') or '(none)'}]: ").strip() or current.get("device", "")
    if device == "-":
        device = ""
    split_mode = input(f"Split mode [{current['split_mode']}]: ").strip() or current["split_mode"]
    tensor_split = input(f"Tensor split weights [{current['tensor_split']}]: ").strip() or current["tensor_split"]
    ctx = prompt_int("Context length", current["ctx"])
    kv_quant = input(f"KV cache quant [{current['kv_quant']}]: ").strip() or current["kv_quant"]
    flash_attn = input(f"Flash attention [{current['flash_attn']}]: ").strip() or current["flash_attn"]

    ttl = prompt_int("llama-swap idle TTL in seconds", current["ttl"])

    mtp = input(f"Multi-token prediction, on/off -- only if the GGUF has MTP heads [{current.get('mtp', DEFAULT_MTP)}]: ").strip() or current.get("mtp", DEFAULT_MTP)

    defaults = {
        "device": device,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "ctx": ctx,
        "kv_quant": kv_quant,
        "flash_attn": flash_attn,
        "ttl": ttl,
        "mtp": mtp,
    }
    save_defaults(defaults)
    print(f"\nDefaults saved to {DEFAULTS_PATH}")
    print(json.dumps(defaults, indent=2))


def hermes_config_path() -> Path:
    """Return the Hermes config.yaml path, respecting HERMES_HOME."""
    return HERMES_CONFIG


def read_hermes_config() -> dict:
    """Load Hermes config.yaml if it exists and pyyaml is available."""
    if yaml is None:
        raise RuntimeError("pyyaml is required for Hermes config editing; install it with `pip install pyyaml`")
    path = hermes_config_path()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def write_hermes_config(cfg: dict):
    """Write Hermes config.yaml, backing up the previous version first."""
    if yaml is None:
        raise RuntimeError("pyyaml is required for Hermes config editing; install it with `pip install pyyaml`")
    path = hermes_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(path, backup)
        print(f"(Hermes config backed up to {backup})")
    # Preserve a trailing newline and use the same 2-space indent Hermes itself writes.
    path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False, indent=2).rstrip() + "\n")


def get_router_base_url() -> str:
    """Figure out the router-mode OpenAI-compatible base URL for Hermes.

    Defaults to http://127.0.0.1:7071/v1 -- override with
    MODELCTL_ROUTER_BASE_URL if router mode lives somewhere else entirely.
    """
    override = os.environ.get("MODELCTL_ROUTER_BASE_URL")
    if override:
        return override.rstrip("/") + "/"
    return "http://127.0.0.1:7071/v1"


def router_root_url() -> str:
    """Router base URL without the /v1 suffix. get_router_base_url() returns
    the OpenAI-compatible base (.../v1/) that Hermes and friends talk to, but
    the router's own control endpoints (/v1/models is the exception -- /models/
    load, /models/unload) live at the router's root, not under /v1."""
    v1_url = get_router_base_url()
    root = re.sub(r"v1/?$", "", v1_url)
    if not root.endswith("/"):
        root += "/"
    return root


def router_status() -> list:
    """Query the router-mode llama-server's live /v1/models endpoint and
    merge in each profile's static GPU placement (device, or split-mode +
    tensor-split) so 'what's loaded, and on which GPU(s)' is one call. Pure
    data, no printing -- same reasoning as search_models(): keep this usable
    by a future TUI without dragging a CLI-shaped API along with it.

    Raises RuntimeError if the router can't be reached at all (unlike the
    HF-lookup helpers, there's no sane empty-result fallback here -- "the
    router is down" is worth surfacing distinctly from "it's up with no
    models registered")."""
    url = router_root_url().rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise RuntimeError(f"couldn't reach router at {url}: {e}")

    gpu_by_name = {}
    for p in sorted(PROFILES_DIR.glob("*.json")):
        profile = json.loads(p.read_text())
        cfg = profile.get("config", {})
        if cfg.get("split_mode") and cfg.get("tensor_split"):
            gpu_by_name[profile["name"]] = f"split {cfg['tensor_split']} ({cfg['split_mode']})"
        elif cfg.get("device"):
            gpu_by_name[profile["name"]] = cfg["device"]

    rows = []
    for m in data.get("data", []):
        status = m.get("status", {})
        rows.append({
            "name": m["id"],
            "status": status.get("value", "unknown"),
            "failed": bool(status.get("failed")),
            "exit_code": status.get("exit_code"),
            "gpu": gpu_by_name.get(m["id"], "?"),
            # "preset" = one of modelctl's own profiles; "cache" = something
            # the router picked up on its own (e.g. a prior --hf-repo pull),
            # not something modelctl manages.
            "from_preset": m.get("source") == "preset",
        })
    return rows


def _router_models_action(action: str, name: str, timeout: int = 120):
    """POST to the router's /models/load or /models/unload, per
    tools/server/README.md's router-mode API. Returns (ok, message) instead
    of raising, so cmd_router_load/unload can print and set an exit code
    without a try/except of their own -- same pattern as preflight()."""
    url = router_root_url().rstrip("/") + f"/models/{action}"
    body = json.dumps({"model": name}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return False, f"router returned HTTP {e.code} for {action} '{name}': {detail}"
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        return False, f"couldn't reach router at {url}: {e}"
    except json.JSONDecodeError as e:
        return False, f"router returned a non-JSON response for {action} '{name}': {e}"
    if resp.get("success"):
        return True, f"{action} requested for '{name}'."
    return False, f"router responded without success for {action} '{name}': {resp}"


def router_load(name: str, timeout: int = 300):
    """Load a model now instead of waiting for the first request to trigger
    it. Loading a large model can take a while, hence the generous default
    timeout (vs. unload's, which should be near-instant)."""
    return _router_models_action("load", name, timeout=timeout)


def router_unload(name: str, timeout: int = 30):
    """Unload a model to free its GPU memory immediately, rather than
    waiting for --models-max's LRU eviction to decide for you."""
    return _router_models_action("unload", name, timeout=timeout)


def _wait_for_router_model(name, want, timeout=300, poll=2):
    """Poll router_status() until model `name` reaches `want` ('loaded' or
    'unloaded'), returns 'failed' if it fails, or None on timeout/router
    unreachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            rows = router_status()
        except RuntimeError:
            return None
        row = next((r for r in rows if r["name"] == name), None)
        if row and row["failed"]:
            return "failed"
        status = row["status"] if row else "unloaded"
        if want == "loaded" and status == "loaded":
            return "loaded"
        if want == "unloaded" and status in ("unloaded", "stopped"):
            return "unloaded"
        time.sleep(poll)
    return None


def _load_target_gpus(profile, inventory):
    """The inventory entries a profile's placement actually lands on:
    its pinned device, or every GPU when split (or unplaced)."""
    cfg = profile.get("config", {})
    if cfg.get("split_mode") and cfg.get("tensor_split"):
        return inventory
    device = cfg.get("device")
    if device:
        matched = [d for d in inventory if d["device"] == device]
        if matched:
            return matched
    return inventory


def check_vram_for_load(profile, evict=False):
    """VRAM guard for explicit router loads. Returns (ok, messages).

    Degrades open (ok=True with a warning) when xpu-smi, the estimate, or
    exact GGUF metadata is unavailable -- the guard should never make a
    load impossible just because tooling is missing. Only a confident
    (exact) over-budget estimate blocks; --evict unloads the largest
    loaded profiles until the target fits."""
    inventory = get_gpu_inventory()
    if not inventory:
        return True, ["WARNING: xpu-smi unavailable -- skipping VRAM check."]
    est = estimate_vram_footprint(profile)
    if est is None:
        return True, ["WARNING: couldn't estimate footprint (model file "
                      "missing?) -- skipping VRAM check."]

    targets = _load_target_gpus(profile, inventory)
    free = sum(d["free_bytes"] for d in targets)
    target_names = ", ".join(d["device"] for d in targets)
    msgs = [f"Estimated footprint: ~{_format_size(est['total'])} "
            f"({est['quality']}); free on {target_names}: {_format_size(free)}"]

    if est["total"] <= free:
        return True, msgs
    if est["quality"] == "heuristic":
        msgs.append("WARNING: heuristic estimate exceeds free VRAM -- "
                    "proceeding anyway (couldn't parse GGUF header for an "
                    "exact number).")
        return True, msgs

    if not evict:
        try:
            loaded = [r for r in router_status()
                      if r["status"] == "loaded" and r["from_preset"]]
        except RuntimeError:
            loaded = []
        if loaded:
            msgs.append("Currently loaded: " + ", ".join(r["name"] for r in loaded))
        msgs.append("Not enough free VRAM. Re-run with --evict to unload "
                    "loaded models until it fits, or --force to skip this check.")
        return False, msgs

    # Evict largest-estimate first until the target fits.
    try:
        rows = [r for r in router_status()
                if r["status"] == "loaded" and r["from_preset"]
                and r["name"] != profile["name"]]
    except RuntimeError as e:
        msgs.append(f"ERROR: can't evict -- {e}")
        return False, msgs

    def loaded_estimate(row):
        try:
            e = estimate_vram_footprint(load_profile(row["name"]))
        except SystemExit:
            return 0
        return e["total"] if e else 0

    rows.sort(key=loaded_estimate, reverse=True)
    for row in rows:
        msgs.append(f"Evicting '{row['name']}' to free VRAM ...")
        ok, unload_msg = router_unload(row["name"])
        msgs.append(f"  {unload_msg}")
        if not ok:
            continue
        _wait_for_router_model(row["name"], "unloaded", timeout=60)
        inventory = get_gpu_inventory()
        targets = _load_target_gpus(profile, inventory)
        free = sum(d["free_bytes"] for d in targets)
        if est["total"] <= free:
            msgs.append(f"Free VRAM now {_format_size(free)} -- proceeding.")
            return True, msgs
    msgs.append("Still not enough free VRAM after evicting everything "
                "evictable. Use --force to try anyway.")
    return False, msgs


def sync_hermes_custom_providers(dry_run: bool = False):
    """Rebuild Hermes' provider entries from all saved modelctl profiles,
    for the router-mode endpoint.

    Hermes's own config format for this has moved at least twice already: a
    `custom_providers:` list initially, now a `providers:` dict keyed by name
    with name/api/models fields. Hermes appears to normalize whatever you
    feed it and rewrite the file in its current shape. This function reads
    whichever shape is on disk (migrating the legacy list into the dict) and
    always WRITES the current `providers:` dict shape, since that's what's
    been directly confirmed against a live Hermes instance. If Hermes
    changes shape again, check ~/.hermes/config.yaml directly rather than
    trust this docstring, and update accordingly.
    """
    if yaml is None:
        print("WARNING: pyyaml not installed -- cannot sync Hermes providers. "
              "Install it with `pip install pyyaml` and re-run `modelctl sync`.", file=sys.stderr)
        return False

    if not HERMES_CONFIG.exists():
        print(f"NOTE: Hermes config not found at {HERMES_CONFIG} -- skipping Hermes sync.")
        return False

    cfg = read_hermes_config()
    router_url = get_router_base_url()

    profiles = sorted(PROFILES_DIR.glob("*.json"))
    models_map = {}
    for p in profiles:
        profile = json.loads(p.read_text())
        name = profile.get("name")
        if not name or not profile.get("enabled", True):
            continue
        # Both backends expose models by profile name at /v1/models.
        ctx = profile.get("config", {}).get("ctx")
        models_map[name] = {"context_length": int(ctx)} if ctx else {}

    if not models_map:
        print("No saved profiles -- leaving Hermes providers unchanged.")
        return False

    # Normalize whatever's currently on disk into one dict keyed by provider
    # name, so unrelated providers (compression-cuda, an Ollama entry, etc.)
    # and any legacy list-format entries all survive the round-trip.
    providers = dict(cfg.get("providers") or {})
    for cp in (cfg.get("custom_providers") or []):
        pname = cp.get("name")
        if pname and pname not in providers:
            entry = dict(cp)
            if "base_url" in entry and "api" not in entry:
                entry["api"] = entry.pop("base_url")
            providers[pname] = entry
    had_legacy_key = "custom_providers" in cfg
    cfg.pop("custom_providers", None)

    before = dict(providers)

    def upsert(url, label):
        # Reuse an existing entry's name if one already targets this exact
        # base URL (so a manually-renamed provider keeps its name).
        target_name = None
        for pname, pdata in providers.items():
            existing_url = pdata.get("api") or pdata.get("base_url") or ""
            if existing_url.rstrip("/") + "/" == url:
                target_name = pname
                break
        if not target_name:
            target_name = label

        existing_entry = providers.get(target_name, {})
        existing_models = existing_entry.get("models") or {}
        # Merge per-model so extras Hermes itself adds (e.g. reasoning_effort)
        # survive, while context_length always reflects the current profile.
        # Models no longer in models_map (deleted or disabled profiles) are dropped.
        merged_models = {
            mname: {**existing_models.get(mname, {}), **mctx}
            for mname, mctx in models_map.items()
        }

        providers[target_name] = {
            **existing_entry,
            "name": target_name,
            "api": url,
            "models": merged_models,
        }
        return target_name

    router_name = upsert(router_url, "local-router")

    if providers == before and not had_legacy_key:
        print("Hermes providers already up to date.")
        return True

    cfg["providers"] = providers

    if dry_run:
        print("Would write the following providers to Hermes config:")
        print(f"  - {router_name} @ {router_url}: models = [{', '.join(models_map.keys())}]")
        return True

    write_hermes_config(cfg)
    print(f"Synced {len(models_map)} model(s) to Hermes providers @ {HERMES_CONFIG}")
    print(f"  {router_name} -> {router_url}")
    return True


def strip_quant_from_label(label: str) -> str:
    """Remove common GGUF quantization suffixes from a model label so each
    quant is registered as a distinct model in Hermes without a redundant
    '-q4_k_m' suffix in the model name.

    Examples:
      'model-q4_k_m.gguf' -> 'model'
      'model-Q5_K_M'       -> 'model'
      'model-f16'          -> 'model'
    """
    # Remove .gguf extension first.
    name = re.sub(r"\.gguf$", "", label, flags=re.IGNORECASE)
    # Strip well-known quant suffixes: -q4_k_m, -Q5_K_M, -f16, -bf16, -iq4_xs, etc.
    # The leading I/Q is required so version-ish suffixes like '-3_1' survive.
    name = re.sub(r"-I?Q[0-9]_[A-Za-z0-9_]+(?:-\d+)?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"-(?:f16|f32|bf16|q4_0|q4_k|q5_k|q6_k|q8_0|q8_k|q2_k|iq4_xs|iq4_nl|q4km|q5km|q6k|q8k)$", "", name, flags=re.IGNORECASE)
    # Remove common bracketed quant forms like [Q4_K_M].
    name = re.sub(r"\s*[\[(](?:Q|F|IQ|BF)[0-9][A-Za-z0-9_]*[\])]\s*$", "", name, flags=re.IGNORECASE)
    return name.strip("-_. ")


def next_unique_profile_name(base_name: str) -> str:
    """Return a profile name that doesn't collide with an existing profile.

    If `base_name` is taken, try appending '-2', '-3', etc. The user is
    prompted with the result and can always override.
    """
    candidate = base_name
    if not (PROFILES_DIR / f"{candidate}.json").exists():
        return candidate
    n = 2
    while (PROFILES_DIR / f"{candidate}-{n}.json").exists():
        n += 1
    return f"{candidate}-{n}"


def capture_env_passthrough():
    """Snapshot the current shell's relevant env vars into the profile at
    pull-time, so `sync`/`regen` later don't silently lose them if you run
    those commands from a shell that never sourced your SYCL env script."""
    return [f"{k}={os.environ[k]}" for k in PASSTHROUGH_VARS if k in os.environ and os.environ[k] != ""]


COMMON_LLAMA_SERVER_GLOBS = [
    "~/workspace/llama.cpp/build*/bin/llama-server",
    "~/llama.cpp/build*/bin/llama-server",
    "~/.local/bin/llama-server",
    "/opt/llama.cpp/build*/bin/llama-server",
    "/usr/local/bin/llama-server",
]

COMMON_ENV_SCRIPT_GLOBS = [
    "~/workspace/llama.cpp/*sycl*env*.sh",
    "~/workspace/llama.cpp/*env*.sh",
    "~/llama.cpp/*sycl*env*.sh",
    "/opt/intel/oneapi/setvars.sh",
]


def find_llama_server_candidates():
    import glob as _glob
    found = []
    for pattern in COMMON_LLAMA_SERVER_GLOBS:
        found.extend(sorted(_glob.glob(os.path.expanduser(pattern)), reverse=True))
    return [f for f in found if os.access(f, os.X_OK)]


def binary_supports_device(binary_path: str, device: str, timeout=15) -> bool:
    """Run --list-devices and check whether the requested device's backend
    prefix (SYCL, CUDA, VULKAN, ...) actually shows up. A Vulkan-only build
    and a SYCL-only build both satisfy 'a llama-server exists here' but only
    one of them can actually serve a SYCL0 device, so existence alone isn't
    enough to call a candidate usable."""
    backend = re.match(r"[A-Za-z]+", device)
    backend = backend.group(0).upper() if backend else device.upper()
    try:
        result = subprocess.run(
            [binary_path, "--list-devices"],
            capture_output=True, timeout=timeout, text=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    output = (result.stdout or "") + (result.stderr or "")
    return backend in output.upper()


def find_env_script_candidates():
    import glob as _glob
    found = []
    for pattern in COMMON_ENV_SCRIPT_GLOBS:
        found.extend(sorted(_glob.glob(os.path.expanduser(pattern))))
    return found


def source_env_script(script_path: str, timeout=30):
    """Run a shell script in a subshell and capture what it exports, without
    polluting modelctl's own process env. Returns a dict of relevant vars."""
    try:
        result = subprocess.run(
            ["bash", "-c", f"source {shlex.quote(script_path)} >/dev/null 2>&1 && env -0"],
            capture_output=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}
    if result.returncode != 0:
        return {}
    out = {}
    for pair in result.stdout.decode(errors="replace").split("\0"):
        if "=" in pair:
            k, _, v = pair.partition("=")
            out[k] = v
    return {k: out[k] for k in PASSTHROUGH_VARS if k in out and out[k] != ""}


def preflight(profile, auto_fix=True):
    """Single source of truth for 'is this profile actually runnable right
    now'. Checks everything a launch needs, auto-fixes what it safely can
    (resolving the binary, sourcing a found env script), and returns
    (ok: bool, effective_bin: str, effective_env: dict, messages: list[str]).
    Never launches anything itself -- just reports.
    """
    messages = []
    ok = True
    device = profile.get("config", {}).get("device", "")

    # 1. model file must exist
    model_path = Path(profile["model_path"])
    if not model_path.exists():
        messages.append(f"ERROR: model file not found on disk: {model_path}")
        ok = False

    mmproj_path = profile.get("mmproj_path")
    if mmproj_path and not Path(mmproj_path).exists():
        messages.append(f"ERROR: mmproj file not found on disk: {mmproj_path}")
        ok = False

    # 2. llama-server binary -- must both exist AND actually support the
    #    backend this profile needs. Multiple builds (e.g. build-vulkan and
    #    build-sycl side by side) means "a binary exists" isn't enough --
    #    the wrong one accepts the same --device flag and fails at runtime
    #    with a much more confusing error than a clear preflight check.
    effective_bin = LLAMA_SERVER_BIN
    if LLAMA_SERVER_RESOLVED and device and not binary_supports_device(LLAMA_SERVER_BIN, device):
        messages.append(f"WARNING: configured llama-server ({LLAMA_SERVER_BIN}) doesn't appear to "
                         f"support device '{device}' -- checking for an alternative build ...")
        effective_bin = None  # force the search path below

    if not LLAMA_SERVER_RESOLVED or effective_bin is None:
        if auto_fix:
            candidates = find_llama_server_candidates()
            matching = [c for c in candidates if not device or binary_supports_device(c, device)]
            if matching:
                effective_bin = matching[0]
                messages.append(f"AUTO-FIXED: using {effective_bin}, confirmed to support "
                                 f"'{device}'.")
                messages.append(f"  Make this permanent: export MODELCTL_LLAMA_SERVER={effective_bin}")
            elif candidates:
                messages.append(f"ERROR: found {len(candidates)} llama-server build(s) "
                                 f"({', '.join(candidates)}), but none of them support device "
                                 f"'{device}'. Is this the right device for this build, or do you "
                                 f"need a SYCL/CUDA/Vulkan build that matches it?")
                ok = False
            else:
                messages.append("ERROR: 'llama-server' isn't on PATH, and no build was found in "
                                 "the usual locations (~/workspace/llama.cpp/build*/bin/, etc).")
                messages.append("  Fix: export MODELCTL_LLAMA_SERVER=/full/path/to/llama-server")
                ok = False
        else:
            messages.append("ERROR: 'llama-server' isn't on PATH or doesn't support this device.")
            ok = False

    # 3. env vars -- only actually required if this profile targets a SYCL
    #    device, since that's the case that's bitten us (libiomp5.so etc).
    # Profiles are documented as hand-editable -- tolerate a malformed env
    # entry (no '=') with a warning instead of an opaque unpacking error.
    effective_env = {}
    for e in profile.get("env") or []:
        if "=" in e:
            k, _, v = e.partition("=")
            effective_env[k] = v
        else:
            messages.append(f"WARNING: ignoring malformed env entry (no '='): {e!r}")
    needs_sycl_env = device.upper().startswith("SYCL")
    if needs_sycl_env and "LD_LIBRARY_PATH" not in effective_env:
        if auto_fix:
            scripts = find_env_script_candidates()
            sourced = {}
            used_script = None
            for script in scripts:
                sourced = source_env_script(script)
                if "LD_LIBRARY_PATH" in sourced:
                    used_script = script
                    break
            if sourced:
                effective_env.update(sourced)
                messages.append(f"AUTO-FIXED: this profile targets {device} but had no "
                                 f"LD_LIBRARY_PATH saved. Sourced {used_script} and recovered it "
                                 f"for this run only -- the saved profile is unchanged.")
                messages.append(f"  Make this permanent: modelctl edit {profile['name']} "
                                 f"(after sourcing {used_script} in your shell first)")
            else:
                messages.append(f"WARNING: this profile targets {device} but has no LD_LIBRARY_PATH "
                                 f"saved, and no env script was found to auto-source. If launch fails "
                                 f"with a 'shared libraries' error, that's why.")
        else:
            messages.append(f"WARNING: this profile targets {device} but has no LD_LIBRARY_PATH saved.")

    return ok, effective_bin, effective_env, messages


def warn_if_llama_server_unresolved():
    if not LLAMA_SERVER_RESOLVED:
        print(
            f"WARNING: couldn't find 'llama-server' on PATH, so this command was written as the "
            f"bare name '{LLAMA_SERVER_BIN}' -- it will fail wherever it runs unless llama-server "
            f"is actually on THAT process's PATH too. Fix it by either adding its directory to PATH "
            f"before running modelctl, or setting MODELCTL_LLAMA_SERVER=/full/path/to/llama-server, "
            f"then re-run `modelctl sync`.",
            file=sys.stderr,
        )


def warn_if_env_empty(env):
    if not env:
        print(
            f"WARNING: none of {PASSTHROUGH_VARS} were set in this shell, so this profile is "
            f"saving an EMPTY env list. If llama-server needs LD_LIBRARY_PATH or similar to find "
            f"its shared libraries (e.g. libiomp5.so), source your env script BEFORE running "
            f"modelctl -- then `modelctl edit <name>` to recapture it, or fix it now and re-pull.",
            file=sys.stderr,
        )


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def search_models(query, tags=None, min_gb=None, max_gb=None, sort="downloads", limit=15, enrich=True):
    """Search Hugging Face for candidate model repos. Returns a list of
    plain dicts (repo_id, downloads, likes, is_gguf, has_mtp, contents) --
    no printing or prompting here, so this is safe to call from any UI (the
    current CLI, or a future TUI) without dragging print()/input() along.

    tags: currently only "mtp" is recognized -- a cheap repo-name filter
    (no extra API calls), requiring "mtp" to appear in the repo id.
    min_gb/max_gb: filters to repos with at least one quant whose total
    size falls in this range. Requires fetching each candidate's file
    listing (via get_repo_contents), so this is the expensive path --
    only paid for when actually requested.
    enrich: when True (the default, since "see quants before pulling" is
    the whole point of this function existing), populates `contents` via
    get_repo_contents() for every result. Set False for a fast, cheap
    search (e.g. live-filter-as-you-type in a future TUI), then enrich
    only the one item actually selected.
    """
    tags = tags or []
    want_mtp = "mtp" in tags
    need_contents = enrich or min_gb is not None or max_gb is not None

    hits = api.list_models(search=query, limit=limit, sort=sort)
    results = []
    for m in hits:
        model_tags = m.tags or []
        is_gguf = any("gguf" in t.lower() for t in model_tags) or "gguf" in m.id.lower()
        has_mtp = "mtp" in m.id.lower()
        if want_mtp and not has_mtp:
            continue

        contents = None
        if need_contents:
            try:
                contents = get_repo_contents(m.id)
            except Exception:
                contents = None

        if min_gb is not None or max_gb is not None:
            if not _has_quant_in_range(contents, min_gb, max_gb):
                continue

        results.append({
            "repo_id": m.id,
            "downloads": m.downloads or 0,
            "likes": m.likes or 0,
            "is_gguf": is_gguf,
            "has_mtp": has_mtp,
            "contents": contents if enrich else None,
        })
    return results


def _has_quant_in_range(contents, min_gb, max_gb) -> bool:
    if not contents:
        return False
    min_bytes = min_gb * 1024 ** 3 if min_gb is not None else 0
    max_bytes = max_gb * 1024 ** 3 if max_gb is not None else float("inf")
    for g in contents.get("quant_groups", []):
        size = g.get("total_size")
        if size is not None and min_bytes <= size <= max_bytes:
            return True
    return False


def _quant_suffixes(quant_groups):
    """Best-effort short quant labels for a compact table column -- strips
    the longest common prefix across THIS repo's own quant labels, so
    'Qwen3.5-35B-A3B-Q4_K_M' / 'Qwen3.5-35B-A3B-UD-Q4_K_M' display as
    'Q4_K_M' / 'UD-Q4_K_M' rather than repeating the model name in every
    column -- while still distinguishing UD (Unsloth Dynamic) variants
    from plain ones of the same quant, since a common-prefix-only strip
    (rather than assuming any fixed quant naming pattern) can't accidentally
    swallow a distinguishing token the way pattern matching against known
    quant suffixes can."""
    labels = [g["label"] for g in quant_groups]
    if not labels:
        return []
    if len(labels) == 1:
        base = strip_quant_from_label(labels[0])
        if base and labels[0].startswith(base):
            return [labels[0][len(base):].strip("-_. ") or labels[0]]
        return [labels[0]]

    prefix = os.path.commonprefix(labels)
    cut = prefix.rfind("-")
    prefix = prefix[:cut + 1] if cut != -1 else ""
    return [label[len(prefix):] or label for label in labels]


def _format_size(num_bytes):
    if num_bytes is None:
        return "?"
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.1f}GB"


def cmd_search(args):
    query = " ".join(args.query)
    extras = []
    if args.tag:
        extras.append(f"{'/'.join(args.tag)}-tagged")
    if args.min_gb is not None or args.max_gb is not None:
        extras.append(f"{args.min_gb or 0}-{args.max_gb if args.max_gb is not None else '∞'}GB")
    suffix = f" ({', '.join(extras)})" if extras else ""
    print(f"Searching Hugging Face for: {query}{suffix}\n")

    results = search_models(
        query, tags=args.tag, min_gb=args.min_gb, max_gb=args.max_gb,
        sort=args.sort, limit=args.limit, enrich=True,
    )

    if not results:
        print("No results.")
        return

    print(f"{'REPO':<50} {'DOWNLOADS':>10} {'LIKES':>6}  {'QUANTS AVAILABLE':<32} {'MTP':<4} VISION")
    for r in results:
        contents = r["contents"] or {}
        quant_groups = contents.get("quant_groups", [])
        quant_col = ", ".join(_quant_suffixes(quant_groups)[:4])
        if len(quant_groups) > 4:
            quant_col += f" (+{len(quant_groups) - 4} more)"
        mtp_col = "yes" if r["has_mtp"] else "no"
        vision_col = "yes" if contents.get("mmproj_files") else "no"
        print(f"{r['repo_id']:<50} {r['downloads']:>10,} {r['likes']:>6,}  {quant_col:<32} {mtp_col:<4} {vision_col}")
    print(f"\n{len(results)} results. Use: modelctl pull <repo_id>")


SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def list_gguf_files(repo_id: str):
    files = api.list_repo_files(repo_id)
    return [f for f in files if f.endswith(".gguf")]


def get_repo_contents(repo_id: str) -> dict:
    """Fetch and categorize a repo's GGUF files -- quant groups, mmproj
    files, and MTP companion files -- with sizes where available. Pure
    data, no printing/prompting, so both cmd_pull's interactive picker and
    search_models()'s enrichment can share this instead of duplicating the
    same mmproj/MTP/quant separation logic in two places."""
    gguf_files = list_gguf_files(repo_id)
    sizes = _repo_file_sizes(repo_id)

    # Three strictly separate lists -- a file can only ever appear in one of
    # these. MTP checked first since "mtp" and "mmproj" filenames don't
    # overlap in practice, but excluding whichever runs first from the
    # other keeps it that way regardless.
    mmproj_files = [f for f in gguf_files if "mmproj" in f.lower()]
    mtp_files = [f for f in gguf_files if "mtp" in f.lower() and f not in mmproj_files]
    quant_files = [f for f in gguf_files if f not in mmproj_files and f not in mtp_files]
    quant_groups = group_files(quant_files)

    for g in quant_groups:
        file_sizes = [sizes[f] for f in g["files"] if f in sizes]
        g["total_size"] = sum(file_sizes) if len(file_sizes) == len(g["files"]) else None

    return {
        "quant_groups": quant_groups,
        "mmproj_files": [{"name": f, "size": sizes.get(f)} for f in mmproj_files],
        "mtp_files": [{"name": f, "size": sizes.get(f)} for f in mtp_files],
    }


def group_files(gguf_files):
    """Group multi-part split GGUFs (e.g. -00001-of-00002.gguf) into one
    selectable unit so they don't show up as confusing duplicate entries.
    Everything else stays as its own single-item group."""
    groups, order = {}, []
    for f in gguf_files:
        m = SHARD_RE.match(f)
        if m:
            key, idx = m.group(1), int(m.group(2))
            groups.setdefault(key, {})[idx] = f
            if key not in order:
                order.append(key)
        else:
            groups[f] = {0: f}
            order.append(f)

    result = []
    for key in order:
        parts = [groups[key][i] for i in sorted(groups[key])]
        label = key if len(parts) > 1 else re.sub(r"\.gguf$", "", parts[0], flags=re.IGNORECASE)
        # Some repos (e.g. unsloth/Qwen3.5-35B-A3B-GGUF's BF16/ subfolder)
        # ship files inside a subdirectory -- strip that prefix so it doesn't
        # leak into the display label or the stored profile's "file" field.
        label = label.rsplit("/", 1)[-1]
        result.append({"label": label, "files": parts, "sharded": len(parts) > 1})
    return result


def parse_selection(s: str, max_index: int):
    s = s.strip().lower()
    if s == "all":
        return list(range(max_index))
    indices = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(token))
    seen, out = set(), []
    for i in indices:
        if i < 0 or i >= max_index:
            raise ValueError(f"index {i} out of range (0-{max_index - 1})")
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


@functools.lru_cache(maxsize=None)
def _repo_file_sizes(repo_id: str) -> dict:
    """Best-effort map of {filename: size_bytes} for every file in a repo,
    fetched in a single HF API call. Returns {} on any failure so callers
    degrade gracefully (missing sizes, not a crash) rather than forcing a
    hard dependency on API availability for basic search/browse to work.

    Cached per repo_id for the life of the process: pulling a sharded model
    calls this once per shard, and `verify` once per file field, all for the
    same repo -- one API hit covers them all. (Failures cache as {} too;
    acceptable for a short-lived CLI.)"""
    try:
        info = api.model_info(repo_id, files_metadata=True)
        return {sib.rfilename: sib.size for sib in info.siblings if sib.size is not None}
    except Exception:
        return {}


def _remote_file_size(repo_id: str, filename: str):
    """Best-effort lookup of a single repo file's size via the HF API, used
    to verify a local file actually belongs to this repo before reusing it.
    Returns None (not False) on any failure so callers fall back to the old
    trust-it behavior rather than force a redundant download when offline."""
    return _repo_file_sizes(repo_id).get(filename)


def download_if_needed(repo_id: str, filename: str, dest_dir: Path) -> str:
    """Skip the download only if a file already sitting on disk actually
    matches THIS repo's copy (verified by size) -- avoids re-pulling a
    multi-GB mmproj/model file if it's already present from an earlier run
    or a profile that shares it.

    Downloads are namespaced under dest_dir/<repo_id>/ rather than landing
    flat in dest_dir. Many repos ship generically-named files (e.g. a
    'mmproj-F16.gguf' or 'model-mtp.gguf' with no repo-specific prefix), so
    a flat directory meant a second repo using the same filename would
    either get served the FIRST repo's file, or -- worse, if the size check
    below caught the mismatch -- overwrite it on disk out from under any
    earlier profile that still points at that path. Per-repo subdirectories
    make that collision structurally impossible instead of merely detected."""
    repo_dir = dest_dir / repo_id
    repo_dir.mkdir(parents=True, exist_ok=True)
    # hf_hub_download(local_dir=...) preserves the repo-relative path, so a
    # file in a subfolder (e.g. 'BF16/model.gguf') lands at
    # repo_dir/BF16/model.gguf -- check that same path, not just the basename,
    # or the "already present" fast path never matches subfolder files.
    target = repo_dir / filename
    if target.exists() and target.stat().st_size > 0:
        expected_size = _remote_file_size(repo_id, filename)
        if expected_size is None or target.stat().st_size == expected_size:
            print(f"  already present, skipping download: {target.name}")
            return str(target)
        print(f"  {target.name} exists but its size ({target.stat().st_size} bytes) doesn't "
              f"match {repo_id}'s current copy ({expected_size} bytes) -- stale or partial "
              f"download, re-downloading.")
    print(f"  downloading {filename} -> {repo_dir} ...")
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(repo_dir))


def _find_repo_filename_by_basename(repo_id: str, basename: str):
    """Recover the repo-relative filename for a locally-saved path.

    Profiles only ever recorded the local basename (model_path etc. --
    there's no separate 'repo filename' field), so verifying an existing
    download against its repo means matching on basename. Returns
    (rfilename, size), or (None, None) if the repo can't be reached, the
    basename doesn't appear at all, or it appears more than once (e.g. the
    same filename nested under two different subfolders) -- ambiguous cases
    are left alone rather than guessed at.
    """
    sizes = _repo_file_sizes(repo_id)
    matches = [(rfilename, size) for rfilename, size in sizes.items()
               if Path(rfilename).name == basename]
    if len(matches) == 1:
        return matches[0]
    return None, None


PROFILE_FILE_FIELDS = ("model_path", "mmproj_path", "mtp_path")


def verify_profile_files(profile: dict) -> list:
    """Check model_path/mmproj_path/mtp_path against this profile's repo_id.

    This is the repair path for the pre-fix collision bug: before per-repo
    download directories, two repos sharing a generic filename (e.g. a
    'mmproj-F16.gguf' with no repo-specific prefix) could end up pointing a
    profile at a file that actually belongs to a *different* repo. Returns
    a list of {field, path, status, repo_filename, expected_size,
    local_size} dicts, one per non-empty field, with status one of "ok",
    "mismatch", "missing", or "unknown" (repo unreachable, or basename
    isn't uniquely identifiable in the repo -- e.g. this happens for the
    non-primary parts of a sharded model, since only the first shard's path
    is saved on the profile).
    """
    repo_id = profile.get("repo_id")
    results = []
    if not repo_id:
        return results

    for field in PROFILE_FILE_FIELDS:
        raw_path = profile.get(field)
        if not raw_path:
            continue
        local_path = Path(raw_path)
        basename = local_path.name
        repo_filename, expected_size = _find_repo_filename_by_basename(repo_id, basename)
        local_size = local_path.stat().st_size if local_path.exists() else None

        if repo_filename is None:
            status = "unknown"
        elif local_size is None:
            status = "missing"
        elif expected_size is not None and local_size != expected_size:
            status = "mismatch"
        else:
            status = "ok"

        results.append({
            "field": field,
            "path": str(local_path),
            "status": status,
            "repo_filename": repo_filename,
            "expected_size": expected_size,
            "local_size": local_size,
        })
    return results


def prompt_pick(label: str, prompt: str) -> int:
    """Prompt for a single index with validation; -1 means 'skip/blank'."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            return -1
        if raw.isdigit():
            return int(raw)
        print(f"  '{raw}' isn't a valid {label} number -- try again, or press Enter to skip.")


def cmd_pull(args):
    if getattr(args, "tui", False):
        run_pull_wizard(
            no_hermes=getattr(args, "no_hermes", False),
            no_router_restart=getattr(args, "no_router_restart", False),
        )
        return

    repo_id = args.repo_id
    print(f"Fetching file list for {repo_id} ...")
    try:
        contents = get_repo_contents(repo_id)
    except Exception as e:
        print(f"Error: couldn't list files for '{repo_id}': {e}")
        sys.exit(1)

    groups = contents["quant_groups"]
    mmproj_files = contents["mmproj_files"]
    mtp_files = contents["mtp_files"]

    if not groups and not mmproj_files and not mtp_files:
        print("No .gguf files found in this repo.")
        sys.exit(1)
    if not groups:
        print("No selectable model/quant files found in this repo (only mmproj/MTP files present).")
        sys.exit(1)

    print(f"\n=== Model files ({len(groups)}) ===")
    for i, g in enumerate(groups):
        tag = f"  [{len(g['files'])}-part split]" if g["sharded"] else ""
        size = f"  ({_format_size(g['total_size'])})" if g.get("total_size") is not None else ""
        print(f"  [{i}] {g['label']}{tag}{size}")

    raw = input("\nPick model file number(s) -- comma list, range (3-5), or 'all': ").strip()
    try:
        chosen_idx = parse_selection(raw, len(groups))
    except ValueError as e:
        print(f"Invalid selection: {e}")
        sys.exit(1)
    if not chosen_idx:
        print("Nothing selected.")
        sys.exit(1)
    chosen_groups = [groups[i] for i in chosen_idx]

    mmproj_chosen = None
    if mmproj_files:
        print(f"\n=== Vision/mmproj files ({len(mmproj_files)}) -- separate list, pick at most one ===")
        for i, f in enumerate(mmproj_files):
            size = f"  ({_format_size(f['size'])})" if f.get("size") is not None else ""
            print(f"  [{i}] {f['name']}{size}")
        mi = prompt_pick("mmproj", "Pick mmproj number, or blank to skip: ")
        if mi != -1:
            if 0 <= mi < len(mmproj_files):
                mmproj_chosen = mmproj_files[mi]["name"]
            else:
                print(f"  index {mi} isn't in the mmproj list, skipping mmproj.")

    mtp_chosen = None
    if mtp_files:
        print(f"\n=== MTP draft-head files ({len(mtp_files)}) -- separate list, pick at most one, optional ===")
        print("  (most MTP-capable models bundle these in the main file -- this list is for")
        print("   repos that ship a separate companion file, e.g. Gemma 4's '-mtp.gguf' pairing)")
        for i, f in enumerate(mtp_files):
            size = f"  ({_format_size(f['size'])})" if f.get("size") is not None else ""
            print(f"  [{i}] {f['name']}{size}")
        ti = prompt_pick("MTP", "Pick MTP file number, or blank to skip: ")
        if ti != -1:
            if 0 <= ti < len(mtp_files):
                mtp_chosen = mtp_files[ti]["name"]
            else:
                print(f"  index {ti} isn't in the MTP list, skipping MTP.")

    dest_dir = Path(input(f"Download directory, a per-repo subfolder will be created under it "
                          f"[{DEFAULT_MODELS_DIR}]: ").strip() or DEFAULT_MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    local_mmproj_path = None
    if mmproj_chosen:
        print("Fetching mmproj (shared across all selected profiles):")
        local_mmproj_path = download_if_needed(repo_id, mmproj_chosen, dest_dir)

    local_mtp_path = None
    if mtp_chosen:
        print("Fetching MTP draft-head file (shared across all selected profiles):")
        local_mtp_path = download_if_needed(repo_id, mtp_chosen, dest_dir)

    if len(chosen_groups) > 1:
        print(f"\n{len(chosen_groups)} files selected -- config below is shared across all of them.")
    placement_hint = compute_pull_placement_hint(chosen_groups[0].get("total_size"))
    shared_config = prompt_config(repo_id, chosen_groups[0]["label"] if chosen_groups else "",
                                   mtp_file_chosen=bool(local_mtp_path),
                                   placement=placement_hint)
    env = capture_env_passthrough()
    warn_if_env_empty(env)

    for g in chosen_groups:
        for part in g["files"]:
            local_path = download_if_needed(repo_id, part, dest_dir)
            if part == g["files"][0]:
                primary_path = local_path  # point profile at the first shard if split

        # Strip the quant suffix from the label so the Hermes model name is clean
        # (e.g. "model-q4_k_m" becomes "model"). If that collides with an existing
        # profile (e.g. you already pulled Q5_K_M), append a number. The user can
        # still override.
        clean_label = strip_quant_from_label(g["label"])
        name_default = next_unique_profile_name(slugify(clean_label))
        name = input(f"Profile name for '{g['label']}' [{name_default}]: ").strip() or name_default

        config = dict(shared_config)

        profile = {
            "name": name,
            "repo_id": repo_id,
            "file": g["label"],
            "model_path": primary_path,
            "mmproj_path": local_mmproj_path,
            "mtp_path": local_mtp_path,
            "config": config,
            "env": env,
            "enabled": True,
        }
        save_profile(profile)
        generate_artifacts(profile)
        print(f"-> saved profile '{name}'")

    sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"\nDone. {len(chosen_groups)} profile(s) created and pushed to {ROUTER_PRESET_PATH}.")
    if args.no_router_restart:
        print("Router NOT restarted (--no-router-restart) -- the running router won't see the "
              f"new preset until you restart it, e.g. `systemctl --user restart {ROUTER_SERVICE_NAME}`.")


def prompt_int(prompt: str, default: int) -> int:
    """Prompt for an integer, re-asking on garbage input (e.g. a numpad
    sending escape codes instead of digits with NumLock off) instead of
    silently accepting whatever the terminal sent. Returns an int so
    profiles store ctx/ttl consistently typed (build_server_args and the
    renderers stringify at the point of use)."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        if raw.lstrip("-").isdigit():
            return int(raw)
        print(f"  '{raw}' isn't a number -- try again, or press Enter for the default ({default}).")


def prompt_config(repo_id: str = "", label: str = "", mtp_file_chosen: bool = False,
                  current: dict | None = None, placement: dict | None = None):
    """Prompt for a profile's runtime config. Defaults come from the saved
    user defaults, overlaid with `current` when editing an existing profile
    -- so pressing Enter through the prompts keeps that profile's values
    instead of silently resetting it to the global defaults."""
    print("\n--- runtime config (blank = keep the value shown) ---")
    d = {**load_defaults(), "extra": "", **(current or {})}
    if placement and not current:
        # Seed device/split defaults from the VRAM-fit recommendation for
        # new profiles; edits keep the profile's own values instead.
        d = {**d, "device": placement["device"],
             "split_mode": placement["split_mode"] or d["split_mode"],
             "tensor_split": placement["tensor_split"] or d["tensor_split"]}
    device = input(f"GPU device, '-' to clear [{d.get('device') or 'blank = use split strategy'}]: ").strip() or d.get("device", "")
    if device == "-":
        device = ""
    split_mode = input(f"Split mode [{d['split_mode']}]: ").strip() or d["split_mode"]
    tensor_split = input(f"Tensor split weights [{d['tensor_split']}]: ").strip() or d["tensor_split"]
    ctx = prompt_int("Context length", d["ctx"])
    kv_quant = input(f"KV cache quant, e.g. q8_0 [{d['kv_quant']}]: ").strip() or d["kv_quant"]
    flash_attn = input(f"Flash attention [{d['flash_attn']}]: ").strip() or d["flash_attn"]

    ttl = prompt_int("llama-swap idle TTL in seconds", d["ttl"])
    mtp_default = "on" if mtp_file_chosen else d.get("mtp", DEFAULT_MTP)
    mtp_hint = " (MTP file was just downloaded for this pull)" if mtp_file_chosen else ""
    mtp = input(f"Multi-token prediction, on/off -- only if this GGUF has MTP heads [{mtp_default}]{mtp_hint}: ").strip() or mtp_default
    extra = input(f"Any extra llama-server flags, '-' to clear [{d.get('extra') or 'none'}]: ").strip() or d.get("extra", "")
    if extra == "-":
        extra = ""
    return {
        "device": device,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "ctx": ctx,
        "kv_quant": kv_quant,
        "flash_attn": flash_attn,
        "ttl": ttl,
        "mtp": mtp,
        "extra": extra,
    }


def save_profile(profile):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{profile['name']}.json"
    path.write_text(json.dumps(profile, indent=2))


def load_profile(name):
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        print(f"No profile named '{name}'. Run `modelctl list` to see saved profiles.")
        sys.exit(1)
    return json.loads(path.read_text())


def build_server_args(profile):
    """Return a flat list of llama-server CLI tokens. Values with spaces are
    kept as single tokens so callers don't have to re-tokenize with shlex."""
    cfg = profile["config"]
    args = [
        "--model", str(profile['model_path']),
        "-ngl", "999",
        "--flash-attn", cfg['flash_attn'],
        # str() because ctx is an int in defaults/TUI-written profiles but a
        # string in older prompt-written ones -- subprocess argv needs str.
        "-c", str(cfg['ctx']),
        "--jinja",
        "--parallel", "1",
    ]
    # Prefer GPU split if configured. If not, fall back to explicit device.
    # Legacy profiles without either field will simply omit --device; the
    # backend's default is usually GPU0, which is the least surprising fallback.
    if cfg.get("split_mode") and cfg.get("tensor_split"):
        args.extend(["--split-mode", cfg['split_mode'], "--tensor-split", cfg['tensor_split']])
    elif cfg.get("device"):
        args.extend(["--device", cfg['device']])

    if profile.get("mmproj_path"):
        args.extend(["--mmproj", str(profile['mmproj_path'])])
    if cfg.get("kv_quant"):
        args.extend(["--cache-type-k", cfg['kv_quant'], "--cache-type-v", cfg['kv_quant']])
    # Multi-token prediction: self-speculative decoding using the model's own
    # MTP draft heads (no separate draft model needed). Only meaningful if the
    # checkpoint was trained with MTP heads and the GGUF conversion kept them --
    # opt-in per profile, see DEFAULT_MTP. --spec-draft-n-max etc. default to
    # sane values (n-max=3); override via the "extra" field if you need to tune them.
    if cfg.get("mtp", "off").strip().lower() == "on":
        args.extend(["--spec-type", "draft-mtp"])
        # Most current MTP GGUFs (e.g. Qwen3.5/3.6) bundle the draft heads in
        # the same file as the main model, so there's nothing else to point
        # at. Some (Gemma 4) ship them as a separate companion GGUF instead --
        # if cmd_pull captured one, tell llama-server to load it as the draft.
        if profile.get("mtp_path"):
            args.extend(["--spec-draft-model", str(profile["mtp_path"])])
    if cfg.get("extra"):
        # Extra is a raw string the user typed; split it safely to preserve
        # quoted values with spaces, but only if they provided it.
        args.extend(shlex.split(cfg["extra"]))
    return args


def render_llama_swap_entry(profile):
    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    args = build_server_args(profile)
    # The cmd: block scalar is a *shell* command line, so quote for the
    # shell (shlex), not YAML -- json.dumps-style double quotes would leave
    # `$` and backticks live inside the string.
    args_str = " \\\n      ".join(shlex.quote(str(a)) for a in args)
    log_path = PROFILES_DIR / profile["name"] / "llama-swap.log"
    lines = [
        f"{profile['name']}:",
        "  cmd: |",
        f"    {effective_bin} --port ${{PORT}} \\\n      {args_str}",
        f'  logFile: "{log_path}"',
    ]
    if effective_env:
        lines.append("  env:")
        lines.extend(f'    - "{k}={v}"' for k, v in effective_env.items())
    lines.append(f"  ttl: {profile['config']['ttl']}")
    # Expose context_length and tool support in /v1/models so clients
    # (Hermes, Open WebUI, etc.) can auto-detect model capabilities.
    ctx = int(profile['config'].get('ctx', 0))
    if ctx > 0:
        lines.append("  capabilities:")
        lines.append(f"    context: {ctx}")
        lines.append("    tools: true")
    return "\n".join(lines) + "\n", ok, messages


def render_router_preset(profile):
    """Render one [section] of the router-mode preset.ini for this profile.
    Mirrors render_llama_swap_entry but targets `llama-server --models-preset`
    instead of llama-swap. Runs through the same preflight() resolution so
    binary/env issues are caught the same way for both backends."""
    # Unlike render_llama_swap_entry, router mode shares one binary/env across
    # all spawned models (the router itself is the process llama-swap-style
    # YAML would otherwise point `cmd:` at), so effective_bin/effective_env
    # have nothing to attach to in a per-model INI section -- only ok/messages
    # (file existence, device support, etc.) are needed here.
    ok, _, _, messages = preflight(profile, auto_fix=True)
    cfg = profile["config"]

    lines = [f"[{profile['name']}]"]
    lines.append(f"model = {profile['model_path']}")
    lines.append("ngl = 999")
    lines.append(f"ctx-size = {cfg['ctx']}")
    lines.append("jinja = true")
    lines.append("parallel = 1")
    lines.append(f"flash-attn = {cfg['flash_attn']}")
    if cfg.get("split_mode") and cfg.get("tensor_split"):
        lines.append(f"split-mode = {cfg['split_mode']}")
        lines.append(f"tensor-split = {cfg['tensor_split']}")
    elif cfg.get("device"):
        lines.append(f"device = {cfg['device']}")
    if profile.get("mmproj_path"):
        lines.append(f"mmproj = {profile['mmproj_path']}")
    if cfg.get("kv_quant"):
        lines.append(f"cache-type-k = {cfg['kv_quant']}")
        lines.append(f"cache-type-v = {cfg['kv_quant']}")
    if cfg.get("mtp", "off").strip().lower() == "on":
        lines.append("spec-type = draft-mtp")
        if profile.get("mtp_path"):
            lines.append(f"spec-draft-model = {profile['mtp_path']}")
    if cfg.get("extra"):
        # build_server_args shlex.splits cfg["extra"] into discrete CLI
        # tokens; the INI format has no equivalent structured way to splice
        # in arbitrary raw flags, so store the raw string verbatim as a
        # single extra-args line instead.
        lines.append(f"extra-args = {cfg['extra']}")
    return "\n".join(lines) + "\n", ok, messages


def generate_artifacts(profile):
    name = profile["name"]
    out_dir = PROFILES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    if messages:
        print(f"Resolving runtime for '{name}':")
        for m in messages:
            print(f"  {m}")

    args = build_server_args(profile)
    # An earlier version of this line called a helper that already joined
    # `args` into one string, then .join()-ed *that string* again here --
    # str.join() on a string iterates its characters, so every argument got
    # split one letter per line in the generated run.sh. Map shlex.quote over
    # each token individually instead, same as render_llama_swap_entry.
    args_str = " \\\n  ".join(shlex.quote(str(a)) for a in args)
    env_exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in effective_env.items())

    # 1. raw run.sh -- works regardless of backend, since everything
    #    ultimately calls llama-server the same way
    run_sh = out_dir / "run.sh"
    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "# Generated by modelctl. Edit the profile JSON and re-run\n"
        "# `modelctl regen " + name + "` instead of hand-editing this file.\n"
        "set -e\n"
        + (env_exports + "\n" if env_exports else "")
        + f"{effective_bin} --port ${{PORT:-8080}} \\\n  " + args_str + "\n"
    )
    run_sh.chmod(0o755)

    # 2. llama-swap config snippet (same renderer `sync` uses for the real config)
    entry_text, _, _ = render_llama_swap_entry(profile)
    swap_yaml = out_dir / "llama-swap-entry.yaml"
    swap_yaml.write_text(entry_text)

    # 3. Ollama-style Modelfile (best-effort -- Ollama doesn't expose
    #    all of these flags, so this covers what it can)
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(
        f"FROM {profile['model_path']}\n"
        f"PARAMETER num_ctx {profile['config']['ctx']}\n"
    )

    profile["artifacts_dir"] = str(out_dir)
    save_profile(profile)
    return ok


DEFAULT_HEADER = "healthCheckTimeout: 400\nglobalTTL: 0\n"


def _file_hash(path: Path) -> str:
    """Fast content hash for dedup checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


# Global preset section applied to every model instance the router spawns.
# `metrics = true` enables each instance's Prometheus /metrics endpoint,
# which the router forwards at GET /metrics?model=<name> -- `modelctl
# router stats` reads it. Per-model sections override anything here.
ROUTER_PRESET_HEADER = (
    "version = 1\n"
    "\n"
    "[*]\n"
    "metrics = true\n"
    "\n"
)


def sync_router_preset(restart: bool = True) -> int:
    """Rebuild router.preset.ini from every saved profile. Detects
    changes before writing so it doesn't force an unnecessary router
    reload. Returns the number of enabled profiles synced.

    restart: when True (the default), automatically restart the router's
    systemd --user unit after actually writing a changed preset, since
    router mode requires a restart to pick one up. Pass restart=False if
    you're not running the router as a systemd unit, or want to batch
    multiple syncs before reloading once yourself.
    """
    ROUTER_PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_profiles = [json.loads(p.read_text()) for p in sorted(PROFILES_DIR.glob("*.json"))]
    profiles = [p for p in all_profiles if p.get("enabled", True)]
    n_disabled = len(all_profiles) - len(profiles)
    body = ROUTER_PRESET_HEADER
    any_unresolved = False
    for profile in profiles:
        entry, ok, messages = render_router_preset(profile)
        if messages:
            print(f"'{profile['name']}' (router preset):")
            for m in messages:
                print(f"  {m}")
        if not ok:
            any_unresolved = True
        body += entry + "\n"

    new_hash = hashlib.sha256(body.encode()).hexdigest()
    old_hash = _file_hash(ROUTER_PRESET_PATH)
    if new_hash == old_hash:
        print("Router preset unchanged -- skipping write.")
        return len(profiles)

    if ROUTER_PRESET_PATH.exists():
        backup = ROUTER_PRESET_PATH.with_suffix(ROUTER_PRESET_PATH.suffix + ".bak")
        shutil.copy2(ROUTER_PRESET_PATH, backup)
        print(f"(previous router preset backed up to {backup})")

    ROUTER_PRESET_PATH.write_text(body)
    disabled_note = f" ({n_disabled} disabled, skipped)" if n_disabled else ""
    print(f"Wrote router preset for {len(profiles)} profile(s){disabled_note} -> {ROUTER_PRESET_PATH}")
    if any_unresolved:
        print("\nNOTE: at least one profile above couldn't be fully resolved for router mode.")

    if restart:
        restart_router_service()
    return len(profiles)


def restart_router_service() -> bool:
    """Restart the systemd --user unit running the router-mode llama-server
    so a rewritten router.preset.ini actually takes effect. Router mode has
    no --watch-config hot-reload (unlike llama-swap) -- the process only
    reads the preset file at startup, so without this the preset silently
    goes stale until someone notices and restarts it by hand (which is
    exactly what happened here: qwen models sat in the preset for hours
    while the router kept serving its startup-time list).

    Best-effort: prints a warning and returns False rather than raising if
    systemctl or the unit isn't available, so `sync` still succeeds for
    anyone running the router some other way (or not running it locally
    at all)."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", ROUTER_SERVICE_NAME],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"WARNING: couldn't restart {ROUTER_SERVICE_NAME}: {e}. Router preset was "
              f"updated but the running router process wasn't reloaded -- restart it "
              f"yourself, e.g. `systemctl --user restart {ROUTER_SERVICE_NAME}`.", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f"WARNING: `systemctl --user restart {ROUTER_SERVICE_NAME}` failed "
              f"(exit {result.returncode}): {result.stderr.strip()}. Router preset was updated "
              f"but the running router process wasn't reloaded.", file=sys.stderr)
        return False
    print(f"Restarted {ROUTER_SERVICE_NAME} to pick up the updated preset.")
    return True


def sync_all_backends(restart_router: bool = True) -> int:
    """Regenerate every backend's config from the current profiles.
    Single place to extend if a third backend is ever added.
    Returns the number of enabled profiles synced."""
    return sync_router_preset(restart=restart_router)


def cmd_defaults(args):
    if args.show:
        current = load_defaults()
        print(json.dumps(current, indent=2))
        print(f"\nStored at: {DEFAULTS_PATH}")
        print("Env overrides (MODELCTL_DEFAULT_*): " + ", ".join(
            k for k in os.environ if k.startswith("MODELCTL_DEFAULT_")
        ) or "none")
        return
    prompt_defaults()


def cmd_sync(args):
    n = sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers(dry_run=args.hermes_dry_run)
    print(f"Synced {n} enabled profile(s) -> {ROUTER_PRESET_PATH}")


def cmd_list(args):
    if not PROFILES_DIR.exists() or not any(PROFILES_DIR.glob("*.json")):
        print("No profiles saved yet. Run `modelctl pull <repo_id>` first.")
        return
    print(f"{'NAME':<30} {'STATUS':<9} {'REPO':<45} FILE")
    for p in sorted(PROFILES_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        status = "enabled" if d.get("enabled", True) else "disabled"
        print(f"{d['name']:<30} {status:<9} {d['repo_id']:<45} {d['file']}")


def cmd_show(args):
    profile = load_profile(args.name)
    print(json.dumps(profile, indent=2))


def cmd_edit(args):
    profile = load_profile(args.name)
    print(f"Editing '{args.name}' -- current config shown as default where applicable.\n")
    profile["config"] = prompt_config(profile.get("repo_id", ""), profile.get("file", args.name),
                                      current=profile.get("config"))
    profile["env"] = capture_env_passthrough()
    warn_if_env_empty(profile["env"])
    save_profile(profile)
    generate_artifacts(profile)
    sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print("Updated, regenerated artifacts, and pushed to llama-swap config.")


def cmd_regen(args):
    profile = load_profile(args.name)
    generate_artifacts(profile)
    sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"Regenerated artifacts in {profile['artifacts_dir']} and pushed to llama-swap config.")


def cmd_verify(args):
    """Check (and optionally repair) saved profiles' model/mmproj/mtp files
    against their repo, catching the pre-fix flat-download-directory
    collision where a second repo's same-named file could silently replace
    or masquerade as an earlier profile's file.

    Without --fix: report-only, exit code 1 if any profile has a mismatch
    or missing file (0 if everything's ok or unknown/unresolvable).
    With --fix: re-download the correct file (into the new per-repo
    directory, alongside the old file rather than deleting it) for every
    mismatched/missing field, repoint the profile at it, regenerate its
    artifacts, and re-sync.
    """
    if getattr(args, "name", None):
        names = [args.name]
    else:
        if not PROFILES_DIR.exists() or not any(PROFILES_DIR.glob("*.json")):
            print("No profiles saved yet.")
            return
        names = [p.stem for p in sorted(PROFILES_DIR.glob("*.json"))]

    any_bad = False
    any_fixed = False
    for name in names:
        profile = load_profile(name)
        checks = verify_profile_files(profile)
        if not checks:
            continue

        changed = False
        for c in checks:
            if c["status"] == "ok":
                continue
            if c["status"] == "unknown":
                print(f"{name}: {c['field']} ({c['path']}) -- couldn't verify against "
                      f"{profile.get('repo_id')} (repo unreachable, or basename isn't unique "
                      f"in the repo).")
                continue

            any_bad = True
            if c["status"] == "mismatch":
                print(f"{name}: {c['field']} MISMATCH -- {c['path']} is {c['local_size']} "
                      f"bytes, but {profile['repo_id']}'s copy is {c['expected_size']} bytes. "
                      f"This is the flat-directory collision bug: this file was probably "
                      f"overwritten by a different repo's pull.")
            else:
                print(f"{name}: {c['field']} MISSING -- {c['path']} doesn't exist on disk.")

            if not args.fix:
                continue

            base_dir = Path(c["path"]).parent
            # If the file already lives under a per-repo directory (a
            # previous --fix run, or a profile pulled after the collision
            # fix), reuse the grandparent as the base so a second repair
            # pass doesn't nest <repo_id>/<repo_id>/... underneath itself.
            repo_suffix = Path(profile["repo_id"])
            if base_dir.as_posix().endswith(repo_suffix.as_posix()):
                for _ in repo_suffix.parts:
                    base_dir = base_dir.parent
            new_path = download_if_needed(profile["repo_id"], c["repo_filename"], base_dir)
            profile[c["field"]] = new_path
            changed = True
            any_fixed = True
            print(f"  -> fixed: {c['field']} now points at {new_path}")

        if changed:
            save_profile(profile)
            generate_artifacts(profile)

    if any_fixed:
        sync_all_backends(restart_router=not args.no_router_restart)
        if not args.no_hermes:
            sync_hermes_custom_providers()

    if not any_bad:
        print("All checked profiles look OK.")
    elif not args.fix:
        print("\nRe-run with --fix to re-download and repoint the affected profile(s).")
        sys.exit(1)


def _format_placement(cfg: dict) -> str:
    if cfg.get("split_mode") and cfg.get("tensor_split"):
        return f"split {cfg['tensor_split']} ({cfg['split_mode']})"
    return cfg.get("device") or "(backend default)"


def cmd_place(args):
    """Report (and with --apply, rewrite) each profile's GPU placement from
    its VRAM footprint estimate: fits the primary card -> pin to it; too
    big -> capacity-ratio split; too big entirely -> loud warning."""
    inventory = get_gpu_inventory(force_remap=args.remap)
    if not inventory:
        print("Error: couldn't read GPU inventory (is xpu-smi installed and working?).")
        sys.exit(1)
    d = load_defaults()
    primary = resolve_primary_gpu(inventory, d)

    print("GPUs: " + ", ".join(
        f"{g['device']}={_format_size(g['total_bytes'])}" for g in inventory))
    print(f"Placement budget: {d['vram_limit_pct']}% of card capacity; "
          f"primary: {primary}\n")

    if args.name:
        names = [args.name]
    else:
        names = [p.stem for p in sorted(PROFILES_DIR.glob("*.json"))]
        if not names:
            print("No profiles saved yet.")
            return

    changed = False
    print(f"{'NAME':<28} {'ESTIMATE':<18} {'CURRENT':<22} RECOMMENDED")
    for name in names:
        profile = load_profile(name)
        cfg = profile.get("config", {})
        est = estimate_vram_footprint(profile)
        if est is None:
            print(f"{name:<28} {'? (file missing)':<18} "
                  f"{_format_placement(cfg):<22} -")
            continue
        rec = modelctl_vram.recommend_placement(
            est["total"], inventory, d["vram_limit_pct"], primary)
        est_col = f"~{_format_size(est['total'])} ({est['quality']})"
        rec_col = _format_placement(rec) if rec else "?"
        if rec and not rec["fits"]:
            rec_col += "  WARNING: exceeds combined VRAM budget!"
        print(f"{name:<28} {est_col:<18} {_format_placement(cfg):<22} {rec_col}")

        if args.apply and rec:
            same = (cfg.get("device", "") == rec["device"]
                    and cfg.get("split_mode", "") == rec["split_mode"]
                    and cfg.get("tensor_split", "") == rec["tensor_split"])
            if not same:
                cfg["device"] = rec["device"]
                cfg["split_mode"] = rec["split_mode"]
                cfg["tensor_split"] = rec["tensor_split"]
                profile["config"] = cfg
                save_profile(profile)
                generate_artifacts(profile)
                changed = True
                if rec["fits"]:
                    print("  -> applied")
                else:
                    print("  -> WARNING: exceeds combined VRAM budget -- applied anyway")

    if changed:
        sync_all_backends(restart_router=not args.no_router_restart)
        if not args.no_hermes:
            sync_hermes_custom_providers()
    elif args.apply:
        print("\nNothing to change -- all placements already match.")
    else:
        print("\nRe-run with --apply to rewrite placements to the recommendations.")


def cmd_disable(args):
    profile = load_profile(args.name)
    if not profile.get("enabled", True):
        print(f"'{args.name}' is already disabled.")
        return
    profile["enabled"] = False
    save_profile(profile)
    sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"Disabled '{args.name}'. The profile and its artifacts are kept on disk, but it's "
          f"excluded from the router preset and Hermes providers until you run "
          f"`modelctl enable {args.name}`.")


def cmd_enable(args):
    profile = load_profile(args.name)
    if profile.get("enabled", True):
        print(f"'{args.name}' is already enabled.")
        return
    profile["enabled"] = True
    save_profile(profile)
    sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"Enabled '{args.name}' and pushed it back into the router preset"
          + ("." if args.no_hermes else " and Hermes providers."))


def cmd_remove(args):
    profile = load_profile(args.name)
    profile_path = PROFILES_DIR / f"{args.name}.json"
    artifacts_dir = PROFILES_DIR / args.name

    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    profile_path.unlink()
    print(f"Removed profile '{args.name}'. Model file left on disk at "
          f"{profile.get('model_path')} -- delete it yourself if you don't need it for "
          f"another profile.")

    sync_all_backends(restart_router=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _local_weights_bytes(model_path: Path) -> int:
    """Total on-disk size of a model: the file itself, or the sum of all
    sibling shards when it's the first part of a -NNNNN-of-MMMMM split
    (profiles only store the first shard's path)."""
    m = SHARD_RE.match(model_path.name)
    if not m:
        return model_path.stat().st_size
    prefix = m.group(1)
    return sum(p.stat().st_size
               for p in model_path.parent.glob(f"{prefix}-*-of-*.gguf")
               for pm in [SHARD_RE.match(p.name)]
               if pm and pm.group(1) == prefix)


def estimate_vram_footprint(profile):
    """Estimate this profile's VRAM footprint at its configured ctx/kv_quant.
    Returns the estimate dict from modelctl_vram.estimate_from_parts, or
    None when the model file is missing. Computed on demand, never stored
    -- files and ctx change, and recomputing is cheap."""
    model_path = Path(profile["model_path"])
    if not model_path.exists():
        return None
    weights = _local_weights_bytes(model_path)
    mmproj = profile.get("mmproj_path")
    mmproj_bytes = (Path(mmproj).stat().st_size
                    if mmproj and Path(mmproj).exists() else 0)
    cfg = profile.get("config", {})
    ctx = int(cfg.get("ctx") or DEFAULT_CTX)
    params = modelctl_vram.gguf_kv_params(
        modelctl_vram.read_gguf_kv_metadata(str(model_path)))
    return modelctl_vram.estimate_from_parts(
        weights, ctx, cfg.get("kv_quant") or "f16",
        gguf_params=params, mmproj_bytes=mmproj_bytes)


def get_gpu_inventory(force_remap: bool = False) -> list:
    """Live GPU inventory: [{device: 'SYCL0', name, total_bytes, free_bytes}],
    sorted biggest-first. Free bytes are read fresh from xpu-smi on every
    call; only the SYCL-name mapping is cached (GPU_MAP_PATH), since probing
    it needs a slow llama-server --list-devices run. Returns [] when xpu-smi
    is unavailable -- callers degrade to warnings, never errors."""
    xpu = modelctl_vram.xpu_devices()
    if not xpu:
        return []

    mapping = None
    if not force_remap and GPU_MAP_PATH.exists():
        try:
            mapping = json.loads(GPU_MAP_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            mapping = None
    if mapping is None:
        sycl = modelctl_vram.llama_list_devices(LLAMA_SERVER_BIN)
        mapping = modelctl_vram.match_devices(sycl, xpu)
        if mapping:
            GPU_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
            GPU_MAP_PATH.write_text(json.dumps(mapping, indent=2))
    if not mapping:
        # Last resort: assume enumeration orders agree (true on this box).
        mapping = {f"SYCL{d['xpu_id']}": d["xpu_id"] for d in xpu}

    by_id = {d["xpu_id"]: d for d in xpu}
    inventory = []
    for device, xid in mapping.items():
        d = by_id.get(xid)
        if d:
            inventory.append({"device": device, "name": d["name"],
                              "total_bytes": d["total_bytes"],
                              "free_bytes": d["free_bytes"]})
    return sorted(inventory, key=lambda d: -d["total_bytes"])


def resolve_primary_gpu(inventory, defaults=None) -> str:
    """The configured primary GPU, or the biggest card when unset."""
    d = defaults if defaults is not None else load_defaults()
    if d.get("primary_gpu"):
        return d["primary_gpu"]
    return inventory[0]["device"] if inventory else ""


def compute_pull_placement_hint(weights_bytes):
    """Placement recommendation for a not-yet-downloaded model, from its
    remote size and the default ctx/kv_quant. Heuristic quality only (no
    local GGUF header to parse yet) -- good enough to seed the pull
    prompts' defaults. Returns None when size or GPU inventory is
    unavailable."""
    if not weights_bytes:
        return None
    inventory = get_gpu_inventory()
    if not inventory:
        return None
    d = load_defaults()
    est = modelctl_vram.estimate_from_parts(
        weights_bytes, int(d["ctx"]), d["kv_quant"])
    rec = modelctl_vram.recommend_placement(
        est["total"], inventory, d["vram_limit_pct"],
        resolve_primary_gpu(inventory, d))
    if rec:
        print(f"Estimated footprint at ctx={d['ctx']}: "
              f"~{_format_size(est['total'])} (heuristic) "
              f"-> suggested placement: {_format_placement(rec)}")
        if not rec["fits"]:
            print("  WARNING: estimate exceeds combined VRAM budget -- "
                  "consider a smaller quant.")
    return rec


def print_log_tail(log_path, n=40):
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        print(f"  (no log file at {log_path})")
        return
    tail = lines[-n:]
    print(f"  --- last {len(tail)} line(s) of {log_path} ---")
    for line in tail:
        print(f"  {line}")
    print("  ---")


def cmd_test(args):
    profile = load_profile(args.name)

    print(f"Checking '{profile['name']}' before launch ...")
    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    for m in messages:
        print(f"  {m}")
    if not ok:
        print(f"\nNot launching '{profile['name']}' -- fix the error(s) above first.")
        sys.exit(1)

    port = find_free_port()
    cmd = [effective_bin, "--port", str(port)] + build_server_args(profile)
    env = dict(os.environ)
    env.update(effective_env)

    log_path = PROFILES_DIR / profile["name"] / "test.log"
    print(f"\nLaunching '{profile['name']}' on port {port} (log: {log_path}) ...")
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        except FileNotFoundError as e:
            print(f"FAIL: couldn't even start the process -- {e}")
            print(f"  command was: {' '.join(cmd[:3])} ...")
            sys.exit(1)

    try:
        deadline = time.time() + args.timeout
        healthy = False
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"FAIL: process exited early (code {proc.returncode}).")
                print_log_tail(log_path)
                return
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(2)

        if not healthy:
            print(f"FAIL: never became healthy within {args.timeout}s.")
            print_log_tail(log_path)
            return
        print("Loaded OK, sending a real prompt to confirm it actually generates ...")

        body = json.dumps({
            "messages": [{"role": "user", "content": "What is 9*13? Answer with just the number."}],
            "max_tokens": 512,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
        except Exception as e:
            print(f"FAIL: completion request errored: {e}")
            return

        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        finish = resp.get("choices", [{}])[0].get("finish_reason")
        tps = resp.get("timings", {}).get("predicted_per_second")

        print(f"finish_reason: {finish}")
        print(f"content: {content!r}")
        if tps:
            print(f"generation speed: {tps:.1f} tok/s")

        if finish == "stop" and content.strip():
            print(f"PASS: '{profile['name']}' loads and generates correctly.")
        else:
            print(f"WARN: loaded and responded, but finish_reason was '{finish}' "
                  f"or content was empty -- inspect manually before trusting this profile.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def cmd_router_status(args):
    try:
        rows = router_status()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not rows:
        print("No models registered with the router.")
        return

    print(f"{'MODEL':<32} {'STATUS':<10} {'GPU':<22} NOTES")
    for r in rows:
        notes = ""
        if r["failed"]:
            notes = f"FAILED (exit {r['exit_code']})"
        elif not r["from_preset"]:
            notes = "(not a modelctl profile)"
        print(f"{r['name']:<32} {r['status']:<10} {r['gpu']:<22} {notes}")


def cmd_router_load(args):
    profile_path = PROFILES_DIR / f"{args.name}.json"
    if getattr(args, "force", False):
        pass  # explicit override: no check
    elif profile_path.exists():
        profile = json.loads(profile_path.read_text())
        ok, msgs = check_vram_for_load(profile, evict=getattr(args, "evict", False))
        for m in msgs:
            print(m)
        if not ok:
            sys.exit(1)
    else:
        print(f"NOTE: no modelctl profile named '{args.name}' -- "
              f"skipping VRAM check.")

    start = time.time()
    ok, msg = router_load(args.name)
    print(msg)
    if not ok:
        sys.exit(1)
    outcome = _wait_for_router_model(args.name, "loaded", timeout=300)
    elapsed = time.time() - start
    if outcome == "loaded":
        print(f"'{args.name}' loaded in {elapsed:.1f}s.")
    elif outcome == "failed":
        print(f"'{args.name}' FAILED to load after {elapsed:.1f}s -- "
              f"check `modelctl router status` and the router logs.")
        sys.exit(1)
    else:
        print(f"'{args.name}' still not loaded after {elapsed:.0f}s -- "
              f"check `modelctl router status` later.")


def cmd_router_unload(args):
    ok, msg = router_unload(args.name)
    print(msg)
    if not ok:
        sys.exit(1)


class _ModelctlArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass that enforces the one cross-argument rule
    argparse can't express declaratively: 'pull' needs repo_id unless
    --tui is passed. Overriding parse_args() (rather than checking in
    main()) means this fires for any caller of parse_args(), including
    tests that never go through main()."""

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        if getattr(ns, "command", None) == "pull" and not ns.tui and ns.repo_id is None:
            self.error("repo_id is required unless --tui is passed")
        return ns


def build_arg_parser():
    parser = _ModelctlArgumentParser(prog="modelctl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="search Hugging Face for models")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--limit", type=int, default=15)
    p_search.add_argument("--tag", action="append", choices=["mtp"],
                           help="filter to repos matching this tag (repeatable); currently only 'mtp' is supported")
    p_search.add_argument("--min-gb", type=float, default=None,
                           help="only show repos with at least one quant at or above this size in GB")
    p_search.add_argument("--max-gb", type=float, default=None,
                           help="only show repos with at least one quant at or below this size in GB")
    p_search.add_argument("--sort", default="downloads", choices=["downloads", "likes", "lastModified"],
                           help="sort order for results (default: downloads)")
    p_search.set_defaults(func=cmd_search)

    p_pull = sub.add_parser("pull", help="pull a model from a HF repo and configure it")
    p_pull.add_argument("repo_id", nargs="?", default=None,
                         help="required unless --tui is passed (the wizard starts at search)")
    p_pull.add_argument("--tui", action="store_true",
                         help="use the interactive Textual wizard instead of the input()-based prompts")
    p_pull.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_pull.add_argument("--no-router-restart", action="store_true",
                         help="don't restart the router-mode systemd service after updating its preset")
    p_pull.set_defaults(func=cmd_pull)

    p_list = sub.add_parser("list", help="list saved profiles")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show a saved profile's JSON")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_show)

    p_edit = sub.add_parser("edit", help="re-prompt config for a saved profile")
    p_edit.add_argument("name")
    p_edit.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_edit.add_argument("--no-router-restart", action="store_true",
                         help="don't restart the router-mode systemd service after updating its preset")
    p_edit.set_defaults(func=cmd_edit)

    p_regen = sub.add_parser("regen", help="regenerate artifacts from a saved profile")
    p_regen.add_argument("name")
    p_regen.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_regen.add_argument("--no-router-restart", action="store_true",
                          help="don't restart the router-mode systemd service after updating its preset")
    p_regen.set_defaults(func=cmd_regen)

    p_verify = sub.add_parser("verify", help="check saved profiles' model/mmproj/mtp files against "
                                              "their repo, and optionally repair mismatches")
    p_verify.add_argument("name", nargs="?", default=None, help="check only this profile (default: all)")
    p_verify.add_argument("--fix", action="store_true",
                           help="re-download and repoint any mismatched or missing file")
    p_verify.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_verify.add_argument("--no-router-restart", action="store_true",
                           help="don't restart the router-mode systemd service after updating its preset")
    p_verify.set_defaults(func=cmd_verify)

    p_place = sub.add_parser("place", help="recommend (or apply) VRAM-fit GPU placement for profiles")
    p_place.add_argument("name", nargs="?", default=None, help="only this profile (default: all)")
    p_place.add_argument("--apply", action="store_true",
                          help="rewrite profile placement to the recommendation and re-sync")
    p_place.add_argument("--remap", action="store_true",
                          help="rebuild the cached SYCL<->xpu-smi device mapping")
    p_place.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_place.add_argument("--no-router-restart", action="store_true",
                          help="don't restart the router-mode systemd service after updating its preset")
    p_place.set_defaults(func=cmd_place)

    p_disable = sub.add_parser("disable", help="exclude a saved profile from router/Hermes sync without deleting it")
    p_disable.add_argument("name")
    p_disable.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_disable.add_argument("--no-router-restart", action="store_true",
                            help="don't restart the router-mode systemd service after updating its preset")
    p_disable.set_defaults(func=cmd_disable)

    p_enable = sub.add_parser("enable", help="re-include a previously disabled profile in router/Hermes sync")
    p_enable.add_argument("name")
    p_enable.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_enable.add_argument("--no-router-restart", action="store_true",
                           help="don't restart the router-mode systemd service after updating its preset")
    p_enable.set_defaults(func=cmd_enable)

    p_remove = sub.add_parser("remove", aliases=["rm"], help="delete a saved profile and its generated artifacts")
    p_remove.add_argument("name")
    p_remove.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_remove.add_argument("--no-router-restart", action="store_true",
                           help="don't restart the router-mode systemd service after updating its preset")
    p_remove.set_defaults(func=cmd_remove)

    p_sync = sub.add_parser("sync", help="push all profiles into the live router preset.ini")
    p_sync.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_sync.add_argument("--hermes-dry-run", action="store_true", help="show what Hermes config would change")
    p_sync.add_argument("--no-router-restart", action="store_true",
                         help="don't restart the router-mode systemd service after updating its preset")
    p_sync.set_defaults(func=cmd_sync)

    p_defaults = sub.add_parser("defaults", help="configure default runtime settings for new profiles")
    p_defaults.add_argument("--show", action="store_true", help="show current defaults and exit")
    p_defaults.set_defaults(func=cmd_defaults)

    p_test = sub.add_parser("test", help="actually launch a profile and verify it generates correctly")
    p_test.add_argument("name")
    p_test.add_argument("--timeout", type=int, default=300, help="seconds to wait for health (default 300)")
    p_test.set_defaults(func=cmd_test)

    p_router = sub.add_parser("router", help="manage the router-mode llama-server (status, load/unload)")
    router_sub = p_router.add_subparsers(dest="router_command", required=True)

    p_router_status = router_sub.add_parser("status", help="show loaded/unloaded models and GPU placement")
    p_router_status.set_defaults(func=cmd_router_status)

    p_router_load = router_sub.add_parser("load", help="load a model now instead of waiting for a request")
    p_router_load.add_argument("name")
    p_router_load.add_argument("--evict", action="store_true",
                                help="unload loaded models (largest first) until this one fits")
    p_router_load.add_argument("--force", action="store_true",
                                help="skip the VRAM fit check entirely")
    p_router_load.set_defaults(func=cmd_router_load)

    p_router_unload = router_sub.add_parser("unload", help="unload a model now to free its GPU memory")
    p_router_unload.add_argument("name")
    p_router_unload.set_defaults(func=cmd_router_unload)

    return parser


def run_pull_wizard(no_hermes: bool = False, no_router_restart: bool = False):
    """Launch the Textual pull wizard (modelctl pull --tui). Imports
    textual lazily so it never becomes a hard dependency for the rest of
    modelctl -- only --tui users need it installed."""
    try:
        from modelctl_tui import PullWizardApp
    except ImportError as e:
        name_matches = e.name == "textual" or (e.name or "").startswith("textual.")
        if not isinstance(e, ModuleNotFoundError) or not name_matches:
            raise
        print(
            "The --tui wizard requires the 'textual' package, which isn't installed.\n"
            "Install it with: pip install textual",
            file=sys.stderr,
        )
        sys.exit(1)
    PullWizardApp(no_hermes=no_hermes, no_router_restart=no_router_restart).run()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except EOFError:
        print("\nInput closed unexpectedly. Cancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
