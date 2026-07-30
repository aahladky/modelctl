#!/usr/bin/env python3
"""
modelctl - search, pull, and configure local GGUF models from Hugging Face.

Backend-agnostic: generates a run.sh (raw llama-server command) and an
Ollama-style Modelfile from one saved profile, and syncs the profile
into the live llama-swap config.yaml. Profiles are plain JSON so
they're easy to inspect, edit by hand, or regenerate later.
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
import urllib.parse
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ModuleNotFoundError as e:
    if e.name != "huggingface_hub":
        raise
    print(
        "modelctl requires the 'huggingface-hub' package. Install the pinned "
        "runtime dependencies with `uv pip install --python .venv/bin/python "
        "-r requirements.txt`, or run modelctl through the ./modelctl launcher.",
        file=sys.stderr,
    )
    sys.exit(1)

import modelctl_vram

try:
    import yaml
except ImportError:
    yaml = None
else:
    # PyYAML's default dumper folds multi-line strings (e.g. a parsed-back
    # "cmd: |" block) into an escaped single-line flow scalar, which is
    # exactly the unreadable, hard-to-hand-edit output the "profiles are
    # plain JSON/YAML so they're easy to inspect and edit by hand" design
    # goal is trying to avoid. Force block-literal style for any string
    # containing a newline, everywhere this module calls yaml.dump().
    def _yaml_str_presenter(dumper, data):
        style = "|" if "\n" in data else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)
    yaml.add_representer(str, _yaml_str_presenter)

STATE_DIR = Path(os.environ.get("MODELCTL_HOME", Path.home() / ".local/share/modelctl"))
PROFILES_DIR = STATE_DIR / "profiles"
DEFAULT_MODELS_DIR = Path(os.environ.get("MODELCTL_MODELS_DIR", Path.home() / "models"))

_resolved = os.environ.get("MODELCTL_LLAMA_SERVER") or shutil.which("llama-server")
LLAMA_SERVER_BIN = _resolved or "llama-server"
LLAMA_SERVER_RESOLVED = _resolved is not None

# The real llama-swap config.yaml that the llama-swap binary/service reads.
# Distinct from PROFILES_DIR/<name>/llama-swap-entry.yaml, which is just the
# per-profile snippet -- sync_llama_swap_config() is what actually merges
# those snippets into this file.
LLAMA_SWAP_CONFIG_PATH = Path(os.environ.get("MODELCTL_LLAMA_SWAP_CONFIG", Path.home() / "services" / "llama-swap" / "config.yaml"))
# The systemd --user unit running llama-swap. restart_llama_swap_service()
# reload-or-restarts this after writing a changed config.yaml, mirroring
# restart_router_service()/restart_openarc_service().
LLAMA_SWAP_SERVICE_NAME = os.environ.get("MODELCTL_LLAMA_SWAP_SERVICE", "llama-swap.service")
# llama-swap's own OpenAI-compatible base URL -- what sync_hermes_custom_providers()
# points its modelctl-managed GGUF provider bucket at (distinct from
# LLAMA_SWAP_CONFIG_PATH/LLAMA_SWAP_SERVICE_NAME, which are about the config
# file and process, not the HTTP API clients actually call).
LLAMA_SWAP_BASE_URL = os.environ.get("MODELCTL_LLAMA_SWAP_BASE_URL", "http://127.0.0.1:9292/v1/")

# `modelctl test --evals ...` shells out to these rather than reimplementing
# lm-evaluation-harness invocation a second time -- bench.sh/speed.py already
# carry the debugged-live logic (reasoning-model stop-sequence fix, sane
# token budgets, thinking-disabled-by-default) documented in
# ~/services/llama-swap/README.md, and fixing a bug there (e.g. the missing
# langdetect/immutabledict deps ifeval needed) fixes both call paths at
# once. Deliberately NOT reimplemented as profile-only: bench.sh/speed.py
# take any llama-swap model id, including hand-authored (non-modelctl)
# entries like fast-7b, so they stay useful standalone too.
LLAMA_SWAP_DIR = Path(os.environ.get("MODELCTL_LLAMA_SWAP_DIR", Path.home() / "services" / "llama-swap"))
BENCH_SH = Path(os.environ.get("MODELCTL_BENCH_SH", LLAMA_SWAP_DIR / "bench.sh"))
SPEED_PY = Path(os.environ.get("MODELCTL_SPEED_PY", LLAMA_SWAP_DIR / "speed.py"))
# Substring-matched, not an exact set -- covers instruct/plus/infilling
# variants (humaneval_instruct, mbpp_plus, ...) without needing to enumerate
# every lm-eval task name. These execute model-generated code unsandboxed on
# this host (see README) -- modelctl requires an explicit --confirm-unsafe-code
# flag before running any of them, same two-gate opt-in bench.sh's README
# documents (--confirm_run_unsafe_code + HF_ALLOW_CODE_EVAL=1).
CODE_EVAL_TASK_PREFIXES = ("humaneval", "mbpp")

# --- OVMS backend (OpenVINO IR models, served via Docker + ovms-proxy.py --
# see that script's docstring in ~/services/llama-swap/ for why a plain `docker run`
# isn't enough). Each profile becomes one llama-swap `models:` entry, synced
# into LLAMA_SWAP_CONFIG_PATH by sync_llama_swap_config() the exact same way
# llama-cpp-backend profiles are -- unlike the retired OpenArc backend (one
# shared process, its own JSON registry, its own load/unload REST API),
# there's no separate "backend process" to manage here: llama-swap already
# does on-demand start/stop/eviction for OVMS models the same way it does
# for GGUF ones, via `docker run`/`docker stop` under the hood.
#
# 2026-07-09: replaces the OpenArc backend (retired -- openarc.service is
# stopped, and its 3 profiles were migrated to this backend, renamed to
# match the llama-swap model IDs they were already hand-authored under:
# qwen3-6-27b-int4-ov -> big-qwen, gemma-4-31b-it-int4-ov -> big-gemma,
# qwen3.6-35b-ov -> big-qwen-moe). See unified-llama-swap-serving memory.
OVMS_MODEL_REPOSITORY = Path(os.environ.get("MODELCTL_OVMS_MODELS_DIR", Path.home() / "services" / "ovms" / "models"))
OVMS_PROXY_SCRIPT = Path(os.environ.get("MODELCTL_OVMS_PROXY_SCRIPT", Path.home() / "services" / "llama-swap" / "ovms-proxy.py"))
OVMS_DEFAULT_TASK = os.environ.get("MODELCTL_OVMS_TASK", "text_generation")
OVMS_DEFAULT_DEVICE = os.environ.get("MODELCTL_OVMS_DEVICE", "GPU.0")
OVMS_DEFAULT_TOOL_PARSER = os.environ.get("MODELCTL_OVMS_TOOL_PARSER", "hermes3")
# Blank by default: only reasoning/"thinking" models (currently just Qwen3-family
# -- OVMS only implements a "qwen3" reasoning parser as of 2026-07) need this.
# Setting tool_parser without reasoning_parser on a thinking model crashes OVMS's
# guided tool-call decoding the moment the model closes a <think> block -- see
# the unified-llama-swap-serving memory for the incident that found this.
OVMS_DEFAULT_REASONING_PARSER = os.environ.get("MODELCTL_OVMS_REASONING_PARSER", "")
OVMS_DEFAULT_TTL = int(os.environ.get("MODELCTL_OVMS_TTL", "1800"))
# optimum-cli (from optimum-intel) does the actual PyTorch -> OpenVINO IR
# conversion + NNCF quantization. Not a modelctl dependency itself -- resolve
# it the same way LLAMA_SERVER_BIN is resolved: an env var override, else
# whatever's on PATH, else OpenArc's venv as a last-resort default (this
# machine's actual convention for where optimum-intel happens to be
# installed, a leftover from OpenArc being the first thing here that needed
# it -- nothing OpenArc-specific about the tool itself).
_resolved_optimum_cli = (os.environ.get("MODELCTL_OPTIMUM_CLI") or shutil.which("optimum-cli")
                          or str(Path.home() / "OpenArc" / ".venv" / "bin" / "optimum-cli"))
OPTIMUM_CLI_BIN = _resolved_optimum_cli
OVMS_DEFAULT_WEIGHT_FORMAT = os.environ.get("MODELCTL_OVMS_WEIGHT_FORMAT", "int4")
OVMS_DEFAULT_GROUP_SIZE = int(os.environ.get("MODELCTL_OVMS_GROUP_SIZE", "128"))
OVMS_DEFAULT_RATIO = float(os.environ.get("MODELCTL_OVMS_RATIO", "1.0"))

# Hermes Agent integration: modelctl can keep Hermes' custom_providers list
# in sync with saved profiles so pulled models show up in `hermes model`.
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_CONFIG = Path(os.environ.get("MODELCTL_HERMES_CONFIG", HERMES_HOME / "config.yaml"))

# Defaults tuned for serving local GGUF models to Hermes Agent. Hermes sends
# long contexts and benefits from KV cache quantization for memory headroom.
DEFAULT_DEVICE = os.environ.get("MODELCTL_DEFAULT_DEVICE", "SYCL0")
DEFAULT_CTX = int(os.environ.get("MODELCTL_DEFAULT_CTX", "32768"))
# K and V caches can be quantized independently; both default to q8_0
# (the old single MODELCTL_DEFAULT_KV_QUANT is still honored as the
# fallback for whichever of the two isn't set explicitly).
DEFAULT_CACHE_TYPE_K = os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_K", "q8_0")
DEFAULT_CACHE_TYPE_V = os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_V", "q8_0")
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

# Integrated GPUs (UHD/Iris) report shared system RAM as their memory and
# must never be a placement target -- exclude them from the inventory by
# name. Override the pattern with MODELCTL_GPU_EXCLUDE (regex, case
# insensitive; empty string disables filtering).
GPU_EXCLUDE_PATTERN = os.environ.get("MODELCTL_GPU_EXCLUDE", r"UHD Graphics|Iris")


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

    # Legacy fallback: the old single MODELCTL_DEFAULT_KV_QUANT env var (or
    # persisted "kv_quant" key) applies to both K and V when the new,
    # independent settings aren't present. Empty-string env values are
    # treated as unset so they fall through to the persisted/hardcoded
    # defaults instead of winning as an empty override.
    legacy_kv_quant = os.environ.get("MODELCTL_DEFAULT_KV_QUANT") or persisted.get("kv_quant")

    return {
        "device": pick("device", DEFAULT_DEVICE),
        "ctx": int(pick("ctx", DEFAULT_CTX)),
        # cache_type_k/v resolution: new env var -> new persisted key ->
        # legacy kv_quant (env or persisted) -> hardcoded q8_0.
        "cache_type_k": (os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_K")
                         or persisted.get("cache_type_k")
                         or legacy_kv_quant
                         or DEFAULT_CACHE_TYPE_K),
        "cache_type_v": (os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_V")
                         or persisted.get("cache_type_v")
                         or legacy_kv_quant
                         or DEFAULT_CACHE_TYPE_V),
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
    cache_type_k = input(f"K cache quant, e.g. q8_0 [{current['cache_type_k']}]: ").strip() or current["cache_type_k"]
    cache_type_v = input(f"V cache quant, e.g. q4_0 [{current['cache_type_v']}]: ").strip() or current["cache_type_v"]
    flash_attn = input(f"Flash attention [{current['flash_attn']}]: ").strip() or current["flash_attn"]

    ttl = prompt_int("llama-swap idle TTL in seconds", current["ttl"])

    mtp = input(f"Multi-token prediction, on/off -- only if the GGUF has MTP heads [{current.get('mtp', DEFAULT_MTP)}]: ").strip() or current.get("mtp", DEFAULT_MTP)

    defaults = {
        "device": device,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "ctx": ctx,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
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


def router_status() -> list:
    """Query llama-swap's live registered/running model state (via the same
    LlamaSwapClient the web console uses) and merge in each profile's static
    GPU placement (device, or split-mode + tensor-split) so 'what's loaded,
    and on which GPU(s)' is one call. Pure data, no printing -- same
    reasoning as search_models(): keep this usable by a future TUI without
    dragging a CLI-shaped API along with it.

    Raises RuntimeError if llama-swap can't be reached at all (unlike the
    HF-lookup helpers, there's no sane empty-result fallback here -- "the
    router is down" is worth surfacing distinctly from "it's up with no
    models registered")."""
    from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError
    try:
        state = LlamaSwapClient().runtime_state(raise_on_unavailable=True)
    except ModelctlSwapError as e:
        raise RuntimeError(str(e))

    gpu_by_name = {}
    for p in sorted(PROFILES_DIR.glob("*.json")):
        profile = json.loads(p.read_text())
        cfg = profile.get("config", {})
        if cfg.get("split_mode") and cfg.get("tensor_split") or cfg.get("device"):
            gpu_by_name[profile["name"]] = _format_placement(cfg)

    # llama-swap's own state vocabulary (ready/stopped/loading/queued/
    # unloading/failed/unknown) collapsed to the loaded/unloaded language
    # the rest of modelctl (eviction, `router status`) already speaks.
    status_by_state = {"ready": "loaded", "stopped": "unloaded",
                        "unknown": "unloaded", "failed": "failed"}

    rows = []
    for name, info in sorted(state.items()):
        raw = info["state"]
        rows.append({
            "name": name,
            "status": status_by_state.get(raw, raw),
            "failed": raw == "failed",
            "exit_code": None,
            "gpu": gpu_by_name.get(name, "?"),
            # A modelctl profile, vs. something llama-swap knows about that
            # modelctl doesn't manage (a hand-authored config.yaml entry).
            "from_preset": (PROFILES_DIR / f"{name}.json").exists(),
        })
    return rows


def router_load(name: str, timeout: int = 300):
    """Load a model now (through llama-swap) instead of waiting for the
    first request to trigger it. Blocks until the load completes or
    `timeout` elapses, hence the generous default (vs. unload's, which
    should be near-instant). Same modelctl_services.runtime_service the
    web console's load button uses (modelctl_web/mutate.py) -- one
    implementation, two thin callers."""
    from modelctl_services import runtime_service
    result = runtime_service.load_model(name, timeout=timeout)
    if result.ok:
        return True, f"'{name}' loaded in {result.elapsed_s:.1f}s."
    return False, (result.messages[0] if result.messages else f"load failed for '{name}'")


def router_unload(name: str, timeout: int = 30):
    """Unload a model to free its GPU memory immediately, rather than
    waiting for llama-swap's own idle-TTL eviction to decide for you."""
    from modelctl_services import runtime_service
    result = runtime_service.unload_model(name)
    if result.ok:
        return True, (result.messages[0] if result.messages else f"unloaded '{name}'.")
    return False, (result.messages[0] if result.messages else f"unload failed for '{name}'")


def _wait_for_router_model(name, want, timeout=300, poll=2):
    """Poll router_status() until model `name` reaches `want` ('loaded' or
    'unloaded'), returns 'failed' if it fails, or None on timeout/router
    unreachable.

    The router keeps a model's failed flag from a PREVIOUS load attempt
    until a new load overwrites it, so `failed` alone can't be trusted on
    the first poll of a retry -- the desired-state checks run first, and
    the failed check is skipped on the initial poll (stale state predates
    the request we're waiting on)."""
    deadline = time.time() + timeout
    first_poll = True
    while time.time() < deadline:
        try:
            rows = router_status()
        except RuntimeError:
            return None
        row = next((r for r in rows if r["name"] == name), None)
        status = row["status"] if row else "unloaded"
        if want == "loaded" and status == "loaded":
            return "loaded"
        if want == "unloaded" and status in ("unloaded", "stopped"):
            return "unloaded"
        if row and row["failed"] and status != "loading" and not first_poll:
            return "failed"
        first_poll = False
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
    """Rebuild Hermes' provider entries from all saved modelctl profiles.

    2026-07-09: llama-router.service and openarc.service were both retired
    in favor of a single llama-swap front door (:9292). Both modelctl-backed
    model types (llama-cpp GGUF and OVMS) fold into ONE provider,
    'local-swap-managed', forced to that exact name (bypassing the
    URL-reuse matching below) so it never collides with and overwrites
    'local-swap' -- the separate, hand-authored bucket for entries modelctl
    doesn't know about (config.yaml's `matrix:` section, or any model added
    directly to config.yaml per the "experiment convention" in
    ~/services/llama-swap/README.md) -- even though both providers point at the same
    :9292 URL.

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
    # NOT get_router_base_url() (still 127.0.0.1:7071, llama-router.service,
    # retired 2026-07-09) -- every modelctl-managed profile now runs behind
    # llama-swap on the same port as everything else.
    swap_url = LLAMA_SWAP_BASE_URL

    all_profiles = [json.loads(p.read_text()) for p in sorted(PROFILES_DIR.glob("*.json"))]
    enabled = [p for p in all_profiles if p.get("name") and p.get("enabled", True)]
    managed_profiles = [p for p in enabled if p.get("backend", "llama-cpp") in ("llama-cpp", "ovms")]

    models_map = {}
    for profile in managed_profiles:
        # llama-cpp profiles know a static context_length up front; OVMS
        # profiles don't (llama-swap's /v1/models doesn't report one for
        # them either -- see the unified-llama-swap-serving memory).
        ctx = profile.get("config", {}).get("ctx")
        models_map[profile["name"]] = {"context_length": int(ctx)} if ctx else {}

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

    def upsert(url, label, models, force_name=None):
        # Reuse an existing entry's name if one already targets this exact
        # base URL (so a manually-renamed provider keeps its name). Skipped
        # when force_name is given: llama-swap fronts both modelctl-managed
        # profiles and hand-authored entries (config.yaml's matrix: section,
        # any model added directly per the README's experiment convention)
        # on the SAME port, so URL-matching alone would merge this
        # function's fully-managed model list into a provider bucket that
        # also holds hand-added models -- and the delete-what's-missing
        # cleanup below would then wipe those hand-added entries out from
        # under the user. force_name keeps modelctl's bucket separate by
        # name even though it shares a URL with another provider.
        target_name = force_name
        if not target_name:
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
        # Models no longer passed in (deleted/disabled/renamed profiles) are dropped.
        merged_models = {
            mname: {**existing_models.get(mname, {}), **mctx}
            for mname, mctx in models.items()
        }

        providers[target_name] = {
            **existing_entry,
            "name": target_name,
            "api": url,
            "models": merged_models,
        }
        return target_name

    synced = []
    if models_map:
        managed_name = upsert(swap_url, "local-swap-managed", models_map,
                               force_name="local-swap-managed")
        synced.append((managed_name, swap_url, list(models_map.keys())))

    # One-time cleanup: drop any leftover per-profile 'ovms-*' providers from
    # before that backend was replaced, 'local-openarc'/'local-router' from
    # before their respective services were retired, and 'local-swap-gguf'
    # from before OVMS profiles joined the same managed bucket as GGUF ones
    # -- none of these conventions apply to anything anymore.
    for pname in list(providers.keys()):
        if pname.startswith("ovms-") or pname in (
                "local-openarc", "local-router", "local-swap-gguf"):
            del providers[pname]

    if providers == before and not had_legacy_key:
        print("Hermes providers already up to date.")
        return True

    cfg["providers"] = providers

    if dry_run:
        print("Would write the following providers to Hermes config:")
        for pname, url, models in synced:
            print(f"  - {pname} @ {url}: models = [{', '.join(models)}]")
        return True

    write_hermes_config(cfg)
    total_models = sum(len(models) for _, _, models in synced)
    print(f"Synced {total_models} model(s) across {len(synced)} provider(s) @ {HERMES_CONFIG}")
    for pname, url, _ in synced:
        print(f"  {pname} -> {url}")
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
    # A per-profile "binary" pins the build explicitly (e.g. a fork with an
    # not-yet-upstream architecture); it wins over the global resolution and
    # survives env-less regen/sync runs.
    pinned_bin = profile.get("binary") or ""
    if pinned_bin:
        if os.access(pinned_bin, os.X_OK):
            if device and not binary_supports_device(pinned_bin, device):
                messages.append(f"WARNING: pinned binary {pinned_bin} doesn't appear to "
                                f"support device '{device}'.")
            effective_bin = pinned_bin
            messages.append(f"using pinned binary: {pinned_bin}")
        else:
            messages.append(f"ERROR: pinned binary not executable: {pinned_bin}")
            ok = False
            effective_bin = None
    else:
        effective_bin = LLAMA_SERVER_BIN
    if not pinned_bin and LLAMA_SERVER_RESOLVED and device and not binary_supports_device(LLAMA_SERVER_BIN, device):
        messages.append(f"WARNING: configured llama-server ({LLAMA_SERVER_BIN}) doesn't appear to "
                         f"support device '{device}' -- checking for an alternative build ...")
        effective_bin = None  # force the search path below

    if not pinned_bin and (not LLAMA_SERVER_RESOLVED or effective_bin is None):
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


def _env_from_scripts():
    """Populate a profile env list from the first working env script (same
    source preflight recovers from). Used by zero-config paths (pull --yes,
    place --tiers --apply) so SYCL launches don't depend on the shell modelctl
    happened to be invoked from."""
    for script in find_env_script_candidates():
        env = source_env_script(script)
        if env:
            print(f"populated env from {script}")
            return [f"{k}={v}" for k, v in env.items()]
    return []


def _auto_select_quant(groups):
    """Zero-config quant selection for `pull --yes`: delegates to the tier
    planner's recommend_quant_group with the live GPU inventory/defaults.
    Prints the choice (and a warning on fallback). Returns the planner's rec
    dict {group, fits, est_total, reason}, or None when nothing servable
    exists."""
    import modelctl_tiers
    inventory = get_gpu_inventory()
    d = load_defaults()
    if inventory:
        primary = resolve_primary_gpu(inventory, d)
        prim = next((g for g in inventory if g["device"] == primary), inventory[0])
        budget = prim["total_bytes"] * d["vram_limit_pct"] / 100.0
    else:
        primary = ""
        budget = 0
    rec = modelctl_tiers.recommend_quant_group(groups, budget, int(d["ctx"]))
    if not rec:
        print("No servable quant files found in this repo.")
        return None
    rec["primary"] = primary
    rec["budget_bytes"] = budget
    g = rec["group"]
    print(f"auto-selected: {g['label']} ({_format_size(g['total_size'])})"
          f" -- {rec['reason']}")
    if not rec["fits"]:
        print("  WARNING: estimated footprint exceeds the primary GPU -- "
              "after the pull, run `modelctl place --tiers --apply` for an "
              "offload plan.")
    return rec


def _auto_config(rec, mtp_file_chosen=False):
    """Zero-config runtime config for `pull --yes`: the user's saved defaults,
    pinned to the primary GPU when the auto-selected quant fits it (the whole
    point of the selection), defaults' split settings otherwise."""
    d = load_defaults()
    ctk, ctv = _resolve_cache_types(d)
    cfg = {
        "device": d.get("device", ""),
        "split_mode": d["split_mode"],
        "tensor_split": d["tensor_split"],
        "ctx": int(d["ctx"]),
        "cache_type_k": ctk or "q8_0",
        "cache_type_v": ctv or "q4_0",
        "flash_attn": d["flash_attn"],
        "ttl": int(d["ttl"]),
        "mtp": "on" if mtp_file_chosen else d.get("mtp", DEFAULT_MTP),
        "extra": "",
    }
    if rec and rec.get("fits") and rec.get("primary"):
        cfg["device"] = rec["primary"]
        cfg["split_mode"] = ""
        cfg["tensor_split"] = ""
        # Tier 1: let --fit size ngl/ctx to whatever actually fits at load
        # (adapts to co-resident models). cfg["ctx"] stays as the advertised
        # ceiling from auto_ctx; it's not emitted as a flag.
        cfg["fit"] = "on"
    return cfg


def pull_model(repo_id, quant_label=None, want_mtp=True, resync=True,
               progress_cb=None):
    """Zero-config pull pipeline (the `pull --yes` core, callable from the web
    service): auto-quant -> download -> auto-config -> auto-ctx -> env -> save
    -> artifacts -> sync. Returns the saved profile dict, or None on failure
    (reasons are printed). progress_cb(event, info) receives coarse events:
    ("file_start", name) / ("file_done", name) -- byte-level progress is
    polled from the target file's size by the caller."""
    print(f"Fetching file list for {repo_id} ...")
    try:
        contents = get_repo_contents(repo_id)
    except Exception as e:
        print(f"Error: couldn't list files for '{repo_id}': {e}")
        return None
    groups = contents["quant_groups"]
    mtp_files = contents["mtp_files"]
    if not groups:
        print("No selectable model/quant files found in this repo.")
        return None

    if quant_label:
        quant_rec = None
        match = [g for g in groups if g["label"] == quant_label]
        if not match:
            print(f"Error: quant '{quant_label}' not found in {repo_id}.")
            return None
        group = match[0]
        quant_rec = {"group": group, "fits": False, "primary": ""}
        # Still compute fit/budget so auto-config can pin tier 1 when possible.
        rec = _auto_select_quant(groups)
        if rec:
            quant_rec.update({"primary": rec["primary"],
                              "budget_bytes": rec["budget_bytes"]})
            if group["label"] == rec["group"]["label"]:
                quant_rec["fits"] = rec["fits"]
    else:
        quant_rec = _auto_select_quant(groups)
        if quant_rec is None:
            return None
        group = quant_rec["group"]

    mtp_chosen = None
    if mtp_files and want_mtp:
        mtp_chosen = mtp_files[0]["name"]
        print(f"auto-selected MTP draft head: {mtp_chosen}")
    if contents["mmproj_files"]:
        print(f"skipping {len(contents['mmproj_files'])} mmproj file(s) "
              "(zero-config pull is text-only; pull interactively to add vision)")

    dest_dir = DEFAULT_MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    local_mtp_path = None
    if mtp_chosen:
        if progress_cb:
            progress_cb("file_start", mtp_chosen)
        local_mtp_path = download_if_needed(repo_id, mtp_chosen, dest_dir)
        if progress_cb:
            progress_cb("file_done", mtp_chosen)

    shared_config = _auto_config(quant_rec, mtp_file_chosen=bool(local_mtp_path))
    env = capture_env_passthrough()
    if not env:
        env = _env_from_scripts()
    warn_if_env_empty(env)

    primary_path = None
    for part in group["files"]:
        if progress_cb:
            progress_cb("file_start", part)
        local_path = download_if_needed(repo_id, part, dest_dir)
        if progress_cb:
            progress_cb("file_done", part)
        if part == group["files"][0]:
            primary_path = local_path

    clean_label = strip_quant_from_label(group["label"])
    name = next_unique_profile_name(slugify(clean_label))
    print(f"profile name: {name}")

    config = dict(shared_config)
    if quant_rec.get("budget_bytes") and primary_path:
        import modelctl_tiers
        ctx_rec = modelctl_tiers.auto_ctx(
            primary_path, quant_rec["budget_bytes"],
            cache_type_k=config["cache_type_k"],
            cache_type_v=config["cache_type_v"])
        if ctx_rec:
            config["ctx"] = ctx_rec["ctx"]
            print(f"context: {ctx_rec['ctx']} "
                  f"(KV {_format_size(ctx_rec['kv_bytes'])}"
                  f"{', model max' if ctx_rec['ctx'] == ctx_rec['model_max'] else ''})"
                  f" -- {ctx_rec['note']}")

    profile = {
        "name": name,
        "repo_id": repo_id,
        "file": group["label"],
        "model_path": primary_path,
        "mmproj_path": None,
        "mtp_path": local_mtp_path,
        "config": config,
        "env": env,
        "enabled": True,
    }
    save_profile(profile)
    generate_artifacts(profile)
    print(f"-> saved profile '{name}'")

    sync_all_backends(restart_router=resync, restart_openarc=resync)
    return profile


def import_local(file_path: str, name: str | None = None, copy: bool = False,
                 resync: bool = True) -> dict | None:
    """Import a local GGUF file as a modelctl profile.

    Validates readability, detects split GGUF parts and companions
    (mmproj, MTP), and creates a profile referencing the file in place
    (or copying/moving into the managed model directory).

    Returns the saved profile dict, or None on failure.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        print(f"Error: file not found: {path}")
        return None
    if not path.is_file():
        print(f"Error: not a file: {path}")
        return None

    # Check it's a GGUF file (magic bytes).
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            print(f"Error: {path} does not appear to be a GGUF file")
            return None
    except OSError as e:
        print(f"Error: cannot read {path}: {e}")
        return None

    # Detect split parts and companions.
    parent = path.parent
    stem = path.stem
    companions = {"mmproj": [], "mtp": []}
    split_parts = [str(path)]

    for sibling in sorted(parent.glob("*.gguf")):
        if sibling == path:
            continue
        sname = sibling.name.lower()
        if sname.startswith("mmproj"):
            companions["mmproj"].append(str(sibling))
        elif "mtp" in sname or "draft" in sname:
            companions["mtp"].append(str(sibling))
        elif sibling.stem.startswith(stem.rsplit("-", 1)[0] if "-" in stem else stem):
            # Likely a split part of the same model.
            split_parts.append(str(sibling))

    # Generate profile name.
    if name:
        profile_name = name
    else:
        clean = path.stem
        # Strip common quant suffixes for the profile name.
        for suffix in ("-Q4_K_M", "-Q4_K_S", "-Q5_K_M", "-Q5_K_S",
                       "-Q8_0", "-Q6_K", "-Q3_K_M", "-Q3_K_S",
                       "-IQ4_XS", "-IQ3_XS", "-bf16", "-fp16"):
            if clean.upper().endswith(suffix.upper()):
                clean = clean[:len(clean) - len(suffix)]
                break
        profile_name = next_unique_profile_name(slugify(clean))

    # Auto-configure based on file analysis.
    config = _default_config()
    env = capture_env_passthrough()
    if not env:
        env = _env_from_scripts()

    # Determine mmproj and mtp paths.
    mmproj_path = companions["mmproj"][0] if companions["mmproj"] else None
    mtp_path = companions["mtp"][0] if companions["mtp"] else None

    # Optionally copy into managed directory.
    model_path = str(path)
    if copy:
        dest_dir = DEFAULT_MODELS_DIR / profile_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = dest_dir / path.name
        if not dest.exists():
            shutil.copy2(path, dest)
            model_path = str(dest)
        for aux_list, aux_key in [(companions["mmproj"], "mmproj"),
                                   (companions["mtp"], "mtp")]:
            if aux_list:
                aux_dest = dest_dir / Path(aux_list[0]).name
                if not aux_dest.exists():
                    shutil.copy2(aux_list[0], aux_dest)
                if aux_key == "mmproj":
                    mmproj_path = str(aux_dest)
                else:
                    mtp_path = str(aux_dest)

    profile = {
        "name": profile_name,
        "repo_id": "",
        "file": path.name,
        "model_path": model_path,
        "mmproj_path": mmproj_path,
        "mtp_path": mtp_path,
        "config": config,
        "env": env,
        "enabled": True,
    }
    save_profile(profile)
    generate_artifacts(profile)
    print(f"-> saved profile '{profile_name}'")

    if resync:
        sync_all_backends(restart_router=True, restart_openarc=True)
    return profile


def _default_config() -> dict:
    """Return a copy of the default profile config."""
    d = load_defaults()
    return {
        "device": d.get("device", ""),
        "split_mode": d.get("split_mode", ""),
        "tensor_split": d.get("tensor_split", ""),
        "ctx": int(d.get("ctx", 8192)),
        "cache_type_k": d.get("cache_type_k", "q8_0"),
        "cache_type_v": d.get("cache_type_v", "q8_0"),
        "flash_attn": d.get("flash_attn", "auto"),
        "ttl": int(d.get("ttl", 3600)),
        "mtp": d.get("mtp", "off"),
        "fit": d.get("fit", "on"),
        "extra": "",
    }


def cmd_pull(args):
    if getattr(args, "tui", False):
        run_pull_wizard(
            no_hermes=getattr(args, "no_hermes", False),
            no_router_restart=getattr(args, "no_router_restart", False),
        )
        return

    if getattr(args, "yes", False):
        profile = pull_model(args.repo_id,
                             resync=not args.no_router_restart)
        if profile is None:
            sys.exit(1)
        if not args.no_hermes:
            sync_hermes_custom_providers()
        print(f"\nDone. Profile '{profile['name']}' created.")
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

    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"\nDone. {len(chosen_groups)} profile(s) created and synced to llama-swap.")
    if args.no_router_restart:
        print("llama-swap NOT restarted (--no-router-restart) -- restart it yourself "
              "with `systemctl --user restart llama-swap.service`.")


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
    # An old profile may have only 'kv_quant' set (no cache_type_k/v), so the
    # spread above leaves d's cache_type_k/v as the *global* defaults instead
    # of that profile's own legacy value. Re-resolve from `current` to fix that.
    if current:
        ctk, ctv = _resolve_cache_types(current)
        if ctk:
            d["cache_type_k"] = ctk
            d["cache_type_v"] = ctv
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
    cache_type_k = input(f"K cache quant, e.g. q8_0 [{d['cache_type_k']}]: ").strip() or d["cache_type_k"]
    cache_type_v = input(f"V cache quant, e.g. q4_0 [{d['cache_type_v']}]: ").strip() or d["cache_type_v"]
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
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "flash_attn": flash_attn,
        "ttl": ttl,
        "mtp": mtp,
        "extra": extra,
    }


def prompt_ovms_config(repo_id: str, current: dict | None = None):
    """Prompt for an OVMS profile's runtime config -- what render_ovms_llama_swap_entry()
    turns into ovms-proxy.py flags. Mirrors prompt_config()'s shape (blank =
    keep the value shown).

    'name' doubles as the llama-swap model ID clients request by (matches
    the naming convention already used for the hand-authored entries this
    backend replaces: big-qwen, big-gemma, big-qwen-moe, fast-7b) -- keep it
    short and stable, since renaming it later means every client config that
    references the old name breaks.
    """
    print("\n--- OVMS runtime config (blank = keep the value shown / use the default) ---")
    d = current or {}
    target_device = input(
        f"Target device, e.g. GPU.0 / GPU.1 [{d.get('target_device') or OVMS_DEFAULT_DEVICE}]: "
    ).strip() or d.get("target_device") or OVMS_DEFAULT_DEVICE
    task = input(f"Task (text_generation/embeddings/rerank) [{d.get('task') or OVMS_DEFAULT_TASK}]: ").strip() \
        or d.get("task") or OVMS_DEFAULT_TASK
    cache_size_default = d.get("cache_size")
    cache_size_raw = input(
        f"KV cache size hint in GB, blank to omit [{cache_size_default or ''}]: "
    ).strip()
    cache_size = int(cache_size_raw) if cache_size_raw else cache_size_default
    tool_parser_default = d.get("tool_parser", OVMS_DEFAULT_TOOL_PARSER)
    tool_parser = input(
        f"Tool-call parser, blank to omit [{tool_parser_default or ''}]: "
    ).strip() or tool_parser_default
    reasoning_parser_default = d.get("reasoning_parser", OVMS_DEFAULT_REASONING_PARSER)
    reasoning_parser = input(
        "Reasoning/thinking parser -- required alongside tool_parser for "
        f"thinking models (e.g. 'qwen3' for Qwen3.x), blank to omit "
        f"[{reasoning_parser_default or ''}]: "
    ).strip() or reasoning_parser_default
    ttl = prompt_int("Idle TTL in seconds (0 = never unload)", d.get("ttl", OVMS_DEFAULT_TTL))
    return {
        "target_device": target_device,
        "task": task,
        "cache_size": cache_size,
        "tool_parser": tool_parser or None,
        "reasoning_parser": reasoning_parser or None,
        "ttl": ttl,
    }


def cmd_ovms_add(args):
    """Register a pre-converted OpenVINO IR HF repo (e.g.
    OpenVINO/Qwen3.6-27B-int4-ov) as an OVMS-backed profile. No download step
    here -- unlike the old OpenArc backend, OVMS pulls its own copy of the
    repo into OVMS_MODEL_REPOSITORY on first container start (--source_model),
    so registering a profile is just recording the config ovms-proxy.py needs."""
    repo_id = args.repo_id
    name_default = next_unique_profile_name(slugify(repo_id.split("/")[-1]))
    name = input(f"Profile name (this becomes the llama-swap model ID) [{name_default}]: ").strip() or name_default

    dest_dir = OVMS_MODEL_REPOSITORY / repo_id
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"'{dest_dir}' already has local weights (OVMS will use them as-is).")
    else:
        print(f"No local weights at '{dest_dir}' yet -- OVMS will download '{repo_id}' itself "
              f"the first time this model is requested.")

    config = prompt_ovms_config(repo_id)

    profile = {
        "name": name,
        "backend": "ovms",
        "repo_id": repo_id,
        "file": name,
        # Not a GGUF -- estimate_vram_footprint() only understands GGUF headers
        # and always returns None here. OVMS_MODEL_REPOSITORY / repo_id is
        # where the weights live (or will be pulled to), but there's no
        # single "model_path" file the way llama-cpp profiles have one.
        "model_path": None,
        "mmproj_path": None,
        "mtp_path": None,
        "config": config,
        "env": [],
        "enabled": True,
    }
    save_profile(profile)
    generate_artifacts(profile)
    print(f"-> saved profile '{name}'")

    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"\nDone. '{name}' is now in {LLAMA_SWAP_CONFIG_PATH}.")
    print(f"It doesn't need a separate load step -- request it through llama-swap "
          f"(model: \"{name}\") and it starts on demand. NOTE: it's not in the "
          f"`matrix:` section yet -- add it by hand (see ~/services/llama-swap/README.md) "
          f"so eviction/VRAM policy accounts for it.")


def detect_default_task(repo_id: str) -> tuple[str, bool]:
    """Best-effort guess at --task and whether repo_id is a VLM, from the
    source repo's config.json (a few KB, cheap to fetch before committing to
    a conversion that can take many minutes). Returns
    ('text-generation-with-past', False) if config.json can't be fetched or
    parsed -- optimum-cli's own auto-detection still works either way, this
    is just a friendlier default for the interactive prompt.

    The signal is the same one that actually bit this setup: a repo with a
    'vision_config' key is a VLM checkpoint even when it's used purely for
    text chat -- Qwen3.6-27B and gemma-4-31B-it both fail with a cryptic
    port-mismatch error at *inference* time (not conversion time) if
    registered as 'llm' (see openarc_production_setup memory)."""
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        config = json.loads(Path(config_path).read_text())
    except Exception:
        return "text-generation-with-past", False

    is_vlm = "vision_config" in config or "vision_config" in config.get("text_config", {})
    if not is_vlm:
        architectures = config.get("architectures") or []
        has_conditional_gen = any("ForConditionalGeneration" in a for a in architectures)
        has_image_token = bool(config.get("image_token_id") or config.get("image_token_index"))
        is_vlm = has_conditional_gen and has_image_token

    return ("image-text-to-text" if is_vlm else "text-generation-with-past"), is_vlm


def prompt_conversion_config(repo_id: str, task_default: str, is_vlm_default: bool) -> dict:
    """Prompt for optimum-cli export settings. Mirrors prompt_ovms_config()'s
    shape (blank = use the default shown). Defaults (int4, group-size 128,
    ratio 1.0, asymmetric) match what every existing OpenVINO/*-int4-ov repo
    on the machine already uses (read directly from their openvino_config.json)."""
    print(f"\n--- Conversion settings for '{repo_id}' (blank = use the default shown) ---")
    weight_format = input(
        f"Weight format (fp32/fp16/int8/int4/mxfp4/nf4/cb4) [{OVMS_DEFAULT_WEIGHT_FORMAT}]: "
    ).strip() or OVMS_DEFAULT_WEIGHT_FORMAT
    group_size = prompt_int("Group size (quantization block size)", OVMS_DEFAULT_GROUP_SIZE)
    ratio_raw = input(f"Ratio (int4 vs int8 layer mix, 1.0 = all int4) [{OVMS_DEFAULT_RATIO}]: ").strip()
    ratio = float(ratio_raw) if ratio_raw else OVMS_DEFAULT_RATIO
    sym = input("Symmetric quantization? y/N: ").strip().lower() in ("y", "yes")
    task_hint = " (this repo looks like a VLM)" if is_vlm_default else ""
    task = input(f"Task, blank to let optimum-cli auto-detect [{task_default}]{task_hint}: ").strip() \
        or task_default
    trust_remote_code = input(
        "Trust remote code (needed for some custom architectures)? y/N: "
    ).strip().lower() in ("y", "yes")
    return {
        "weight_format": weight_format,
        "group_size": group_size,
        "ratio": ratio,
        "sym": sym,
        "task": task,
        "trust_remote_code": trust_remote_code,
    }


def build_optimum_export_args(repo_id: str, output_dir, conv_cfg: dict) -> list:
    """Build the `optimum-cli export openvino` argv (excluding the binary
    itself). Pure and side-effect-free so it's testable without actually
    running a conversion."""
    args = ["export", "openvino", "-m", repo_id]
    if conv_cfg.get("task"):
        args += ["--task", conv_cfg["task"]]
    weight_format = conv_cfg.get("weight_format")
    if weight_format:
        args += ["--weight-format", weight_format]
    if weight_format in ("int4", "int8") and conv_cfg.get("group_size") is not None:
        args += ["--group-size", str(conv_cfg["group_size"])]
    if weight_format == "int4" and conv_cfg.get("ratio") is not None:
        args += ["--ratio", str(conv_cfg["ratio"])]
    if conv_cfg.get("sym"):
        args.append("--sym")
    if conv_cfg.get("trust_remote_code"):
        args.append("--trust-remote-code")
    args.append(str(output_dir))
    return args


def run_optimum_export(repo_id: str, output_dir, conv_cfg: dict) -> bool:
    """Run `optimum-cli export openvino` as a foreground subprocess, letting
    its own tqdm progress bars/output stream straight to the terminal --
    conversion (especially with calibration-based quantization) can run for
    many minutes on a large model, so hiding that behind a spinner would be
    worse, not better. Returns whether it exited 0."""
    if not (Path(OPTIMUM_CLI_BIN).exists() or shutil.which(OPTIMUM_CLI_BIN)):
        print(f"ERROR: optimum-cli not found at '{OPTIMUM_CLI_BIN}'. Set MODELCTL_OPTIMUM_CLI "
              f"to the right path -- it lives in whichever venv has optimum-intel installed "
              f"(e.g. OpenArc's .venv).", file=sys.stderr)
        return False

    cmd = [OPTIMUM_CLI_BIN] + build_optimum_export_args(repo_id, output_dir, conv_cfg)
    print(f"\nRunning: {' '.join(shlex.quote(a) for a in cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def cmd_ovms_convert(args):
    """Convert a source HF repo (original PyTorch weights, NOT a pre-exported
    OpenVINO IR repo) to OpenVINO IR via optimum-cli, then register it as an
    OVMS-backed profile -- the from-scratch counterpart to cmd_ovms_add,
    which only handles repos that are already converted upstream (optimum-cli
    can't pull a not-yet-converted repo the way cmd_ovms_add's OVMS-does-it-
    itself download does for a pre-converted one).

    Output goes straight into OVMS_MODEL_REPOSITORY / repo_id -- the same
    directory OVMS's --source_model/--model_repository_path already expect,
    so a freshly-converted model needs no copying, just a profile pointing
    at it (compare to the old OpenArc backend, which had this exact same
    conversion step but wrote into the same shared directory only
    incidentally, since OPENARC_MODEL_REPOSITORY and OVMS_MODEL_REPOSITORY
    were always the same path on disk)."""
    repo_id = args.repo_id
    name_default = next_unique_profile_name(slugify(repo_id.split("/")[-1]))
    name = input(f"Profile name (this becomes the llama-swap model ID) [{name_default}]: ").strip() or name_default

    task_default, is_vlm_default = detect_default_task(repo_id)
    conv_cfg = prompt_conversion_config(repo_id, task_default, is_vlm_default)

    dest_dir = OVMS_MODEL_REPOSITORY / repo_id
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"'{dest_dir}' already has files -- skipping conversion "
              f"(delete it first to force a re-convert).")
    else:
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        ok = run_optimum_export(repo_id, dest_dir, conv_cfg)
        if not ok:
            print("\nConversion failed -- see optimum-cli's own output above.")
            sys.exit(1)
        print(f"\nConverted '{repo_id}' -> {dest_dir}")

    ovms_cfg = prompt_ovms_config(repo_id)

    profile = {
        "name": name,
        "backend": "ovms",
        "repo_id": repo_id,
        "file": name,
        "model_path": None,
        "mmproj_path": None,
        "mtp_path": None,
        "config": ovms_cfg,
        "env": [],
        "enabled": True,
    }
    save_profile(profile)
    generate_artifacts(profile)
    print(f"-> saved profile '{name}'")

    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"\nDone. '{name}' is now in {LLAMA_SWAP_CONFIG_PATH}.")
    print(f"It doesn't need a separate load step -- request it through llama-swap "
          f"(model: \"{name}\") and it starts on demand. NOTE: it's not in the "
          f"`matrix:` section yet -- add it by hand (see ~/services/llama-swap/README.md) "
          f"so eviction/VRAM policy accounts for it.")


def save_profile(profile):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{profile['name']}.json"
    path.write_text(json.dumps(profile, indent=2))


# Canonical profile schema (v2):
#   name, repo_id, file, model_path, mmproj_path, mtp_path,
#   backend ("llama-cpp" | "ovms"), binary (optional pinned llama-server),
#   config: {device, split_mode, tensor_split, ctx, cache_type_k,
#            cache_type_v, flash_attn, ttl, mtp, fit, extra},
#   env: ["K=V", ...], enabled: bool,
#   runtime (optional): {mode: "managed", objective, pinned_plan_id,
#            allow_fallback, allow_untested, minimum_context,
#            maximum_cpu_bytes, maximum_storage_tier, disabled_plan_ids}
#   moe_cache (optional): {mode, gpu: {budgets_bytes, policy, ...},
#            ram: {...}, storage: {...}, prefill: {...}, decode: {...},
#            prefetch: {...}}
# load_profile normalizes older files into this shape on read.
# profile_version: 1 = original schema, 2 = adds moe_cache support.
# Missing profile_version is treated as 1; migration is additive and lazy.

# Single source of truth for moe_cache defaults lives in modelctl_profiles
# (imported lazily to keep module-import cost down); do not fork a second
# copy here.
def _default_moe_cache():
    from modelctl_profiles import _DEFAULT_MOE_CACHE, _deep_copy_dict as _dcd
    return _dcd(_DEFAULT_MOE_CACHE)


def normalize_profile(profile: dict) -> dict:
    """Normalize a loaded profile into the canonical schema. Idempotent,
    non-destructive: unknown fields are preserved, legacy ones are mapped
    (kv_quant -> cache_type_k/v, string ctx/ttl -> int), and moe_cache
    defaults are filled for profiles that lack them."""
    p = dict(profile)
    p.setdefault("enabled", True)
    p.setdefault("backend", "llama-cpp")
    p.setdefault("env", [])
    cfg = p.setdefault("config", {})
    legacy = cfg.get("kv_quant")
    if legacy and not cfg.get("cache_type_k"):
        cfg["cache_type_k"] = legacy
    if legacy and not cfg.get("cache_type_v"):
        cfg["cache_type_v"] = legacy
    for key in ("ctx", "ttl"):
        if key in cfg:
            try:
                cfg[key] = int(cfg[key])
            except (TypeError, ValueError):
                pass
    # moe_cache: fill defaults for profiles that predate the feature.
    # Profiles without moe_cache or profile_version are treated as v1 with
    # cache off.  Existing moe_cache blocks are left untouched so hand-edits
    # survive round-trips.
    if "moe_cache" not in p:
        p["moe_cache"] = _default_moe_cache()
    else:
        mc = p["moe_cache"]
        for section, defaults in _default_moe_cache().items():
            if section not in mc:
                mc[section] = _deep_copy_dict(defaults)
            elif isinstance(defaults, dict):
                for k, v in defaults.items():
                    mc[section].setdefault(k, v)
    return p


def _deep_copy_dict(d):
    """Minimal recursive dict copy (no import of copy module)."""
    out = {}
    for k, v in d.items():
        out[k] = _deep_copy_dict(v) if isinstance(v, dict) else v
    return out


def load_profile(name):
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        print(f"No profile named '{name}'. Run `modelctl list` to see saved profiles.")
        sys.exit(1)
    return normalize_profile(json.loads(path.read_text()))


def _resolve_cache_types(cfg: dict):
    """Resolve the effective --cache-type-k/--cache-type-v values for a
    profile's config dict, with backward compatibility for profiles saved
    before this feature existed (which only have a single 'kv_quant' field
    applied to both). New profiles set cache_type_k/cache_type_v directly;
    old ones fall back to kv_quant for whichever of the two isn't set.
    Returns (None, None) if neither the new fields nor the legacy field
    are present, matching the 'omit the flags entirely' behavior."""
    legacy = cfg.get("kv_quant")
    return cfg.get("cache_type_k") or legacy, cfg.get("cache_type_v") or legacy


def _capabilities_for_profile(profile, binary=None, probe=False):
    """Best-effort normalized capabilities for a profile's backend binary.

    Default path reads the on-disk probe cache only (no subprocess), so it
    is safe to call from preview/render paths.  probe=True runs a live
    probe (result is cached) -- use it on paths that generate real launch
    artifacts.  Returns None when nothing is known about the binary, in
    which case callers must fail closed for experimental features."""
    try:
        import modelctl_capabilities
        bin_ = binary or profile.get("binary") or LLAMA_SERVER_BIN
        if not bin_:
            return None
        if probe:
            return modelctl_capabilities.probe_backend(bin_)
        return modelctl_capabilities.get_cached_capabilities(bin_)
    except Exception:
        return None


def build_server_args(profile, capabilities=None):
    """Return a flat list of llama-server CLI tokens. Values with spaces are
    kept as single tokens so callers don't have to re-tokenize with shlex.

    capabilities: normalized schema-2 capabilities for the backend binary.
    When omitted and the profile has moe_cache enabled, the probe cache is
    consulted; if nothing is known, experimental cache flags fail closed
    (omitted)."""
    cfg = profile["config"]
    # fit="on" profiles (tier-1 zero-config pulls) let llama.cpp's --fit size
    # ngl AND context to whatever actually fits at load time. Both must be
    # left UNSET -- fit refuses to adjust user-set values ("context size set
    # by user -> no change"), which is also why the hardcoded -ngl 999 used
    # to neuter fit for every profile. cfg["ctx"] stays in the profile as the
    # advertised ceiling (capabilities), just not as a flag.
    fit_on = str(cfg.get("fit", "")).strip().lower() == "on"
    args = [
        "--model", str(profile['model_path']),
        "--flash-attn", cfg['flash_attn'],
        "--jinja",
        "--parallel", "1",
    ]
    if fit_on:
        args.extend(["--fit", "on"])
    else:
        # str() because ctx is an int in defaults/TUI-written profiles but a
        # string in older prompt-written ones -- subprocess argv needs str.
        args.extend(["-ngl", "999", "-c", str(cfg['ctx'])])
    # Prefer GPU split if configured. If not, fall back to explicit device.
    # Legacy profiles without either field will simply omit --device; the
    # backend's default is usually GPU0, which is the least surprising fallback.
    if cfg.get("split_mode") and cfg.get("tensor_split"):
        args.extend(["--split-mode", cfg['split_mode'], "--tensor-split", cfg['tensor_split']])
    elif cfg.get("device"):
        args.extend(["--device", cfg['device']])

    if profile.get("mmproj_path"):
        args.extend(["--mmproj", str(profile['mmproj_path'])])
    ctk, ctv = _resolve_cache_types(cfg)
    if ctk:
        args.extend(["--cache-type-k", ctk])
    if ctv:
        args.extend(["--cache-type-v", ctv])
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
    # MoE expert cache flags from structured moe_cache settings.  Fail
    # closed: build_moe_cache_args only emits experimental flags when the
    # capabilities say the backend supports them.
    if (capabilities is None
            and profile.get("moe_cache", {}).get("mode", "off") != "off"):
        capabilities = _capabilities_for_profile(profile)
    moe_cache_args = build_moe_cache_args(profile, capabilities=capabilities)
    args.extend(moe_cache_args)
    if cfg.get("extra"):
        # Extra is a raw string the user typed; split it safely to preserve
        # quoted values with spaces, but only if they provided it.
        extra_toks = shlex.split(cfg["extra"])
        if moe_cache_args:
            # Structured moe_cache settings take precedence over raw extra
            # flags: drop any --moe-cache-* tokens from extra so the two
            # sources never emit conflicting values (llama.cpp is last-wins,
            # which would make the effective budget order-dependent).
            extra_toks = _strip_moe_cache_flags(extra_toks)
        args.extend(extra_toks)
    return args


def _strip_moe_cache_flags(tokens):
    """Remove --moe-cache-* flags (and their values) from an argv token list."""
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--moe-cache"):
            i += 1
            # "--flag value" drops the value too; "--flag=value" is complete.
            if "=" not in tok and i < len(tokens) and not tokens[i].startswith("--"):
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def build_moe_cache_args(profile, plan=None, capabilities=None):
    """Return llama-server CLI tokens for MoE expert cache flags.

    Structured moe_cache settings take precedence over raw extra flags.
    When a plan carries cache-specific overrides (from the launch-plan
    compiler), those override the profile defaults.  The capabilities
    dict (from modelctl_capabilities.probe_backend) supplies the
    feature gates and flag names so we don't hard-code fork-specific
    names forever.

    Returns an empty list when moe_cache is off.  When the backend is
    unprobed or lacks moe_weight_transfer_cache, experimental flags fail
    closed and only the stock --metrics flag is emitted.
    """
    mc = profile.get("moe_cache", {})
    mode = mc.get("mode", "off")
    if mode == "off":
        return []

    # Plan-level overrides (from compile_launch_plans cache variants).
    plan_mc = {}
    if plan and hasattr(plan, "decision_data"):
        plan_mc = plan.decision_data.get("moe_cache", {})

    # Resolve effective settings: plan overrides > profile > defaults.
    gpu_section = {**mc.get("gpu", {}), **plan_mc.get("gpu", {})}
    prefill_section = {**mc.get("prefill", {}), **plan_mc.get("prefill", {})}
    storage_section = {**mc.get("storage", {}), **plan_mc.get("storage", {})}
    decode_section = {**mc.get("decode", {}), **plan_mc.get("decode", {})}

    features = capabilities.get("features", {}) if isinstance(capabilities, dict) else {}
    cli = capabilities.get("cli", {}) if isinstance(capabilities, dict) else {}

    # The web console scrapes moe_cache_* from the server's Prometheus
    # endpoint; without --metrics llama-server doesn't serve /metrics at
    # all (501) and the runtime cache column is permanently blank.
    # --metrics is a stock flag, not experimental, so it is emitted even
    # when the cache itself must fail closed.
    metrics_args = ["--metrics"]

    # Fail closed (roadmap §2.5): an unprobed, stale, or incapable backend
    # must never receive experimental cache flags.
    if not features.get("moe_weight_transfer_cache"):
        return metrics_args

    cache_bytes_flag = cli.get("moe_cache_bytes", "--moe-cache-bytes")
    cache_policy_flag = cli.get("moe_cache_policy", "--moe-cache-policy")
    admission_flag = cli.get("moe_cache_admission", "--moe-cache-admission-misses")
    prefill_admission_flag = cli.get("moe_cache_prefill", "--moe-cache-prefill-admission")

    args = []

    # Per-device cache budgets.  The fork's --moe-cache-bytes is a single
    # UNIFORM per-GPU budget: server.cpp copies it into one global
    # (g_moe_cache_budget_bytes) and ggml-sycl lazy-init applies that same
    # value to EVERY device's cache instance.  There is no per-device flag,
    # so per-device budgets collapse to their max -- summing would hand each
    # card MORE cache than the tier planner reserved there, letting the
    # cache collide with statically placed experts (OOM).
    budgets = gpu_section.get("budgets_bytes", {})
    if budgets:
        per_gpu_budget = max((b for b in budgets.values() if b > 0), default=0)
        if per_gpu_budget > 0:
            args.extend([cache_bytes_flag, str(per_gpu_budget)])

    policy = gpu_section.get("policy", "slru")
    if policy:
        args.extend([cache_policy_flag, policy])

    admission = gpu_section.get("admission_misses", 2)
    if admission is not None:
        args.extend([admission_flag, str(admission)])

    prefill_admit = prefill_section.get("admit_to_gpu_cache", False)
    args.extend([prefill_admission_flag, "on" if prefill_admit else "off"])

    # Hybrid mode: emit --moe-hybrid-mode only when miss_execution is "cpu"
    # AND the backend genuinely implements CPU miss execution.  The
    # capability is fail-closed in normalization, so an unprobed or
    # incapable backend never sees the flag.
    if (decode_section.get("miss_execution") == "cpu"
            and features.get("moe_hybrid_cpu_miss")):
        hybrid_flag = cli.get("moe_hybrid_mode", "--moe-hybrid-mode")
        args.extend([hybrid_flag, "on"])

    # Storage mode: the fork uses --no-mmap for fully-resident loading.
    # Only emit when explicitly set (mmap is the default).
    storage_mode = storage_section.get("mode", "mmap")
    if storage_mode == "mlock":
        args.extend(["--no-mmap"])

    args.extend(metrics_args)
    return args


def preflight_moe_cache(profile, capabilities=None):
    """Validate moe_cache settings against backend capabilities.

    Fail closed (roadmap §2.5): an unsupported, unprobed, stale, or
    contradictory backend must never receive experimental cache flags.
    Automatic mode omits the candidate (warning); manual mode blocks with
    an explicit reason (error).

    Returns a list of (level, message) tuples where level is 'ok',
    'warning', or 'error'.  Errors prevent launch; warnings are advisory.
    """
    messages = []
    mc = profile.get("moe_cache", {})
    mode = mc.get("mode", "off")
    if mode == "off":
        return messages

    def _omit_or_block(warn_msg, err_msg):
        if mode == "manual":
            messages.append(("error", err_msg))
        else:
            messages.append(("warning", warn_msg))

    if capabilities is None:
        _omit_or_block(
            "moe_cache requested but backend capabilities are unknown "
            "(unprobed); cache flags are suppressed",
            "moe_cache mode=manual requested but backend capabilities are "
            "unknown (unprobed); refusing to emit experimental cache flags")
        return messages

    features = capabilities.get("features", {})
    if not features.get("moe_weight_transfer_cache"):
        _omit_or_block(
            "moe_cache mode=auto but backend does not support "
            "moe_weight_transfer_cache; cache plans will be filtered",
            "moe_cache mode=manual requested but backend does not support "
            "moe_weight_transfer_cache")
        return messages

    messages.append(("ok", "binary supports MoE weight transfer cache"))

    if mc.get("decode", {}).get("miss_execution") == "cpu":
        if features.get("moe_hybrid_cpu_miss"):
            messages.append(("ok", "hybrid CPU miss execution available"))
        else:
            _omit_or_block(
                "cpu miss execution requested but backend lacks "
                "moe_hybrid_cpu_miss; hybrid flag omitted, decode misses "
                "use GPU weight transfer",
                "decode.miss_execution=cpu requested but backend lacks "
                "moe_hybrid_cpu_miss")

    if mc.get("prefetch", {}).get("enabled"):
        if not features.get("moe_cache_prefetch"):
            _omit_or_block(
                "prefetch requested but unsupported; prefetch plan omitted",
                "prefetch requested but backend lacks moe_cache_prefetch")

    return messages


def update_runtime_policy(name, runtime, resync=True):
    """Set or clear a profile's optional "runtime" section (managed mode and
    its policy: objective, pinned/disabled plans, fallback, limits). Passing
    mode != "managed" (or None) drops the section, restoring fixed-command
    rendering. Regenerates artifacts and optionally re-syncs backends."""
    profile = load_profile(name)
    if runtime and runtime.get("mode") == "managed":
        profile["runtime"] = runtime
    else:
        profile.pop("runtime", None)
    save_profile(profile)
    generate_artifacts(profile)
    if resync:
        sync_all_backends(restart_router=True, restart_openarc=True)
    return profile


def _render_worker_entry(profile, ok, effective_env, messages):
    """Managed-runtime llama-swap entry: launch the modelctl worker, which
    picks the launch plan at start time. Shared by llama-cpp and OVMS."""
    log_path = PROFILES_DIR / profile["name"] / "llama-swap.log"
    launcher = shutil.which("modelctl") or "modelctl"
    lines = [
        f"{profile['name']}:",
        "  cmd: |",
        f"    {launcher} _worker {profile['name']} --port ${{PORT}}",
        f'  logFile: "{log_path}"',
    ]
    if effective_env:
        lines.append("  env:")
        lines.extend(f'    - "{k}={v}"' for k, v in effective_env.items())
    # OVMS's proxy answers /v2/health/ready, not /health -- llama-swap's
    # default health check would time out waiting for the wrong endpoint.
    if profile.get("backend") == "ovms":
        lines.append("  checkEndpoint: /v2/health/ready")
    lines.append(f"  ttl: {profile['config']['ttl']}")
    ctx = int(profile['config'].get('ctx', 0))
    if ctx > 0:
        lines.append("  capabilities:")
        lines.append(f"    context: {ctx}")
        lines.append("    tools: true")
    return "\n".join(lines) + "\n", ok, messages


def render_llama_swap_entry(profile):
    # Managed runtime: llama-swap launches the modelctl worker, which picks
    # the launch plan at start time (policy in the profile's "runtime"
    # section). Fixed profiles keep their direct backend command.
    if profile.get("runtime", {}).get("mode") == "managed":
        ok, _bin, effective_env, messages = preflight(profile, auto_fix=True)
        return _render_worker_entry(profile, ok, effective_env, messages)

    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    caps = _capabilities_for_profile(profile, effective_bin, probe=True)
    cache_msgs = preflight_moe_cache(profile, capabilities=caps)
    messages = list(messages) + [f"moe_cache {lvl}: {m}" for lvl, m in cache_msgs
                                 if lvl != "ok"]
    args = build_server_args(profile, capabilities=caps)
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


def preflight_ovms(profile, auto_fix=True):
    """OVMS equivalent of preflight(): is this profile actually
    registerable/loadable right now. Same contract as preflight() -- never
    launches anything, just reports. `auto_fix` is accepted for symmetry but
    there's nothing here that's auto-fixable. Unlike preflight_openarc's
    model_path check, a missing local copy of the weights is NOT an error --
    OVMS pulls its own copy on first container start."""
    messages = []
    ok = True
    cfg = profile.get("config", {})

    if not OVMS_PROXY_SCRIPT.exists():
        messages.append(f"ERROR: {OVMS_PROXY_SCRIPT} not found -- every OVMS model's cmd "
                         f"depends on it (see ~/services/llama-swap/README.md).")
        ok = False

    if not shutil.which("docker"):
        messages.append("ERROR: 'docker' isn't on PATH -- required to run OVMS containers.")
        ok = False

    if not cfg.get("target_device"):
        messages.append("ERROR: no target_device set (e.g. GPU.0, GPU.1).")
        ok = False

    dest_dir = OVMS_MODEL_REPOSITORY / profile.get("repo_id", "")
    if not dest_dir.exists():
        messages.append(f"NOTE: no local weights at {dest_dir} yet -- OVMS will download "
                         f"'{profile.get('repo_id')}' itself the first time this model is "
                         f"requested (can take a while for a large model).")

    return ok, messages


def render_ovms_llama_swap_entry(profile):
    """OVMS equivalent of render_llama_swap_entry(): builds one llama-swap
    models: entry whose cmd runs ovms-proxy.py instead of a llama-server
    binary directly. Returns (entry_text, ok, messages), same contract."""
    ok, messages = preflight_ovms(profile, auto_fix=True)
    if profile.get("runtime", {}).get("mode") == "managed":
        return _render_worker_entry(profile, ok, {}, messages)
    cfg = profile["config"]
    name = profile["name"]

    args = [
        "python3", str(OVMS_PROXY_SCRIPT),
        "--model-id", "${MODEL_ID}",
        "--listen-port", "${PORT}",
        "--source-model", profile["repo_id"],
        "--ovms-model-name", name,
        "--task", cfg.get("task", OVMS_DEFAULT_TASK),
        "--target-device", cfg.get("target_device", OVMS_DEFAULT_DEVICE),
    ]
    if cfg.get("cache_size"):
        args += ["--cache-size", str(cfg["cache_size"])]
    if cfg.get("tool_parser"):
        args += ["--tool-parser", cfg["tool_parser"]]
    if cfg.get("reasoning_parser"):
        args += ["--reasoning-parser", cfg["reasoning_parser"]]
    args_str = " \\\n      ".join(shlex.quote(str(a)) for a in args)

    log_path = PROFILES_DIR / name / "llama-swap.log"
    lines = [
        f"{name}:",
        "  cmd: |",
        f"    {args_str}",
        '  proxy: "http://127.0.0.1:${PORT}"',
        '  checkEndpoint: "/v2/health/ready"',
        "  cmdStop: docker stop ${MODEL_ID}",
        f'  logFile: "{log_path}"',
        f"  ttl: {cfg.get('ttl', OVMS_DEFAULT_TTL)}",
    ]
    return "\n".join(lines) + "\n", ok, messages


def generate_ovms_artifacts(profile, out_dir):
    """OVMS equivalent of the llama-cpp branch of generate_artifacts(): just
    the llama-swap-entry.yaml snippet (same shape `sync` merges into the
    real config.yaml) for inspection. No run.sh/Modelfile -- those describe
    launching a llama-server binary directly, which doesn't apply to an
    ovms-proxy.py + docker model."""
    entry_text, ok, messages = render_ovms_llama_swap_entry(profile)
    if messages:
        print(f"Resolving runtime for '{profile['name']}':")
        for m in messages:
            print(f"  {m}")

    swap_yaml = out_dir / "llama-swap-entry.yaml"
    swap_yaml.write_text(entry_text)

    profile["artifacts_dir"] = str(out_dir)
    save_profile(profile)
    return ok


def generate_artifacts(profile):
    name = profile["name"]
    out_dir = PROFILES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if profile.get("backend") == "ovms":
        return generate_ovms_artifacts(profile, out_dir)

    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    if messages:
        print(f"Resolving runtime for '{name}':")
        for m in messages:
            print(f"  {m}")

    # Validate cache settings against the real binary (probe is cached);
    # unprobed/incapable backends fail closed -- no experimental flags in
    # the generated artifacts.
    caps = _capabilities_for_profile(profile, effective_bin, probe=True)
    cache_msgs = preflight_moe_cache(profile, capabilities=caps)
    for level, msg in cache_msgs:
        if level != "ok":
            print(f"  moe_cache {level}: {msg}")
    args = build_server_args(profile, capabilities=caps)
    # Map shlex.quote over each token individually -- joining an already-
    # joined string would iterate its characters and split every argument
    # one letter per line in the generated run.sh.
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


def sync_llama_swap_config(restart: bool = True) -> int:
    """Rebuild the REAL llama-swap config.yaml (LLAMA_SWAP_CONFIG_PATH) from
    every enabled llama-cpp- AND ovms-backend profile, merging by model name
    so any models or top-level keys llama-swap's config already has that
    modelctl doesn't manage (hand-added entries, the matrix: section,
    unrelated settings) survive untouched.

    This is the actual fix for the gap found the night OVMS was benchmarked:
    sync_all_backends() previously only ever called sync_router_preset() --
    the per-profile llama-swap-entry.yaml snippets generate_artifacts()
    writes were never merged into a real config.yaml anywhere in the
    codebase. That merge is what this function does.

    ovms-backend profiles were added 2026-07-09, replacing the retired
    OpenArc backend -- see render_ovms_llama_swap_entry()/preflight_ovms().
    """
    if yaml is None:
        print("WARNING: pyyaml not installed -- cannot sync llama-swap config.yaml. "
              "Install it with `pip install pyyaml` and re-run `modelctl sync`.", file=sys.stderr)
        return 0

    all_profiles = [json.loads(p.read_text()) for p in sorted(PROFILES_DIR.glob("*.json"))]
    profiles = [p for p in all_profiles
                if p.get("enabled", True) and p.get("backend", "llama-cpp") in ("llama-cpp", "ovms")]

    LLAMA_SWAP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if LLAMA_SWAP_CONFIG_PATH.exists():
        try:
            config = yaml.safe_load(LLAMA_SWAP_CONFIG_PATH.read_text()) or {}
        except yaml.YAMLError as e:
            print(f"ERROR: {LLAMA_SWAP_CONFIG_PATH} has invalid YAML ({e}) -- refusing to "
                  f"overwrite it. Fix or remove the file by hand and re-run sync.", file=sys.stderr)
            return 0

    config.setdefault("healthCheckTimeout", 400)
    config.setdefault("models", {})

    # Drop previously-managed models that no longer have a matching profile
    # (renamed/removed). Only models whose logFile lives under PROFILES_DIR
    # are recognized as "ours" -- anything else (hand-added models) is left
    # alone even if its name happens to collide with a disabled profile.
    managed_names = {p["name"] for p in profiles}
    for existing_name, existing_cfg in list(config["models"].items()):
        log_file = (existing_cfg or {}).get("logFile", "")
        if str(PROFILES_DIR) in log_file and existing_name not in managed_names:
            del config["models"][existing_name]

    any_unresolved = False
    for profile in profiles:
        if profile.get("backend") == "ovms":
            entry_text, ok, messages = render_ovms_llama_swap_entry(profile)
        else:
            entry_text, ok, messages = render_llama_swap_entry(profile)
        if messages:
            print(f"'{profile['name']}' (llama-swap config):")
            for m in messages:
                print(f"  {m}")
        if not ok:
            any_unresolved = True
        # entry_text is "<name>:\n  cmd: |\n    ...\n" -- parse it back into a
        # dict rather than re-deriving one by hand, so what ends up in the
        # real config.yaml is guaranteed identical to the standalone
        # llama-swap-entry.yaml snippet `regen` writes for inspection.
        config["models"].update(yaml.safe_load(entry_text))

    new_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    old_hash = _file_hash(LLAMA_SWAP_CONFIG_PATH)
    new_hash = hashlib.sha256(new_text.encode()).hexdigest()
    if new_hash == old_hash:
        print("llama-swap config unchanged -- skipping write.")
        return len(profiles)

    if LLAMA_SWAP_CONFIG_PATH.exists():
        backup = LLAMA_SWAP_CONFIG_PATH.with_suffix(LLAMA_SWAP_CONFIG_PATH.suffix + ".bak")
        shutil.copy2(LLAMA_SWAP_CONFIG_PATH, backup)
        print(f"(previous llama-swap config backed up to {backup})")

    LLAMA_SWAP_CONFIG_PATH.write_text(new_text)
    print(f"Wrote llama-swap config for {len(profiles)} profile(s) -> {LLAMA_SWAP_CONFIG_PATH}")
    if any_unresolved:
        print("\nNOTE: at least one profile above couldn't be fully resolved for llama-swap.")

    if restart:
        restart_llama_swap_service()
    return len(profiles)


def restart_llama_swap_service() -> bool:
    """Reload-or-restart the systemd --user unit running llama-swap so a
    rewritten config.yaml actually takes effect. Same failure-tolerance
    pattern as restart_router_service()/restart_openarc_service(). Uses
    reload-or-restart rather than restart: llama-swap has no ExecReload of
    its own, so this just falls back to a full restart, but stays correct
    if that ever changes."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "reload-or-restart", LLAMA_SWAP_SERVICE_NAME],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"WARNING: couldn't reload-or-restart {LLAMA_SWAP_SERVICE_NAME}: {e}. "
              f"llama-swap config was updated but the running process wasn't reloaded -- "
              f"restart it yourself, e.g. `systemctl --user reload-or-restart "
              f"{LLAMA_SWAP_SERVICE_NAME}`.", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f"WARNING: `systemctl --user reload-or-restart {LLAMA_SWAP_SERVICE_NAME}` failed "
              f"(exit {result.returncode}): {result.stderr.strip()}. llama-swap config was "
              f"updated but the running process wasn't reloaded.", file=sys.stderr)
        return False
    print(f"Reload-or-restarted {LLAMA_SWAP_SERVICE_NAME} to pick up the updated config.")
    return True


def _render_group_gid():
    """GID that owns /dev/dri's render nodes, needed for docker's group_add
    so the container's user can actually open them (matches the `stat -c
    "%g" /dev/dri/render*` dance in Intel's own OVMS docs). Returns None if
    /dev/dri/render* doesn't exist on this host (e.g. generating configs
    somewhere other than the GPU box) -- callers must treat that as
    'couldn't determine it', not 'no group needed'."""
    import glob as _glob
    nodes = sorted(_glob.glob("/dev/dri/render*"))
    if not nodes:
        return None
    return os.stat(nodes[0]).st_gid


def sync_all_backends(restart_router: bool = True, restart_openarc: bool = True) -> int:
    """Regenerate the real llama-swap config.yaml from the current
    llama-cpp-backend profiles and push it live via reload-or-restart.

    2026-07-09: llama-router.service and openarc.service were both retired
    in favor of a single llama-swap front door -- sync_router_preset()/
    restart_router_service() and sync_openarc_config()/restart_openarc_service()
    still exist below (unused from here, not deleted, in case either backend
    comes back) but are no longer called as part of a normal sync. The
    restart_router/restart_openarc parameters are kept (rather than renamed)
    so the ~11 existing call sites don't all need touching; restart_router is
    what actually gates the llama-swap restart now."""
    return sync_llama_swap_config(restart=restart_router)


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
    n = sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers(dry_run=args.hermes_dry_run)
    print(f"Synced {n} enabled profile(s) -> {LLAMA_SWAP_CONFIG_PATH}")


def cmd_list(args):
    if not PROFILES_DIR.exists() or not any(PROFILES_DIR.glob("*.json")):
        print("No profiles saved yet. Run `modelctl pull <repo_id>` first.")
        return
    print(f"{'NAME':<30} {'STATUS':<9} {'BACKEND':<10} {'REPO':<45} FILE")
    for p in sorted(PROFILES_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        status = "enabled" if d.get("enabled", True) else "disabled"
        backend = d.get("backend", "llama-cpp")
        print(f"{d['name']:<30} {status:<9} {backend:<10} {d['repo_id']:<45} {d['file']}")


def cmd_show(args):
    profile = load_profile(args.name)
    print(json.dumps(profile, indent=2))


def cmd_edit(args):
    profile = load_profile(args.name)
    print(f"Editing '{args.name}' -- current config shown as default where applicable.\n")
    if profile.get("backend") == "ovms":
        profile["config"] = prompt_ovms_config(profile["repo_id"], current=profile.get("config"))
    else:
        profile["config"] = prompt_config(profile.get("repo_id", ""), profile.get("file", args.name),
                                          current=profile.get("config"))
        profile["env"] = capture_env_passthrough()
        warn_if_env_empty(profile["env"])
    save_profile(profile)
    generate_artifacts(profile)
    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print("Updated, regenerated artifacts, and pushed to the live config.")


def cmd_regen(args):
    profile = load_profile(args.name)
    generate_artifacts(profile)
    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"Regenerated artifacts in {profile['artifacts_dir']} and pushed to the live config.")


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
        sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
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
    return cfg.get("device") or cfg.get("target_device") or "(backend default)"


def _print_tier_plan(name, cfg, plan):
    """Human-readable rendering of one profile's tier plan (dry-run)."""
    tier_desc = {1: "primary GPU", 2: "all GPUs", 3: "GPUs + RAM",
                 4: "GPUs + RAM + SSD streaming"}
    a = plan["analysis"]
    print(f"{name}: tier {plan['tier']} ({tier_desc[plan['tier']]})"
          f" -- {a['weights_gib']:.1f} GiB weights, {a['kv_gib']:.1f} GiB KV"
          f"{', MoE' if a['is_moe'] else ', dense'}")
    for label, gib, desc in plan["layout"]:
        print(f"  {label:<18} {gib:>7.1f} GiB  {desc}")
    new_cfg = plan["config"]
    changes = {k: (cfg.get(k, ""), new_cfg[k])
               for k in ("device", "split_mode", "tensor_split", "extra")
               if cfg.get(k, "") != new_cfg[k]}
    if changes:
        for k, (old, new) in changes.items():
            print(f"  {k}: {old or '(none)'}  ->  {new or '(none)'}")
    else:
        print("  config already matches the plan")
    for w in plan["warnings"]:
        print(f"  WARNING: {w}")
    return bool(changes)


def cmd_place_tiers(args, inventory, defaults, primary, names):
    """Tier-aware variant of cmd_place: plan each profile across the memory
    tiers (GPU1 / +GPU2 / +RAM / +SSD streaming) with modelctl_tiers."""
    import modelctl_tiers
    ram_available = modelctl_vram.system_ram_available()
    print(f"RAM available for planning: "
          f"{_format_size(ram_available)} "
          f"(budget {_format_size(max(0, ram_available - modelctl_tiers.RAM_RESERVE_BYTES))} "
          f"after reserve)\n")

    changed = False
    for name in names:
        profile = load_profile(name)
        if not profile.get("model_path"):
            print(f"{name}: skipped (no local GGUF model_path)\n")
            continue
        plan = modelctl_tiers.plan_tiers(
            profile, inventory, defaults["vram_limit_pct"], primary,
            ram_available=ram_available)
        if plan is None:
            print(f"{name}: couldn't analyze model layout "
                  f"({profile.get('model_path')})\n")
            continue
        differs = _print_tier_plan(name, profile.get("config", {}), plan)
        if args.apply and differs:
            cfg = profile.get("config", {})
            cfg.update(plan["config"])
            profile["config"] = cfg
            # A tier-2+ plan launches on SYCL devices; without the oneAPI env
            # the server aborts before load. Populate it once, from the same
            # env scripts preflight recovers from, if the profile has none.
            if not profile.get("env"):
                for script in find_env_script_candidates():
                    env = source_env_script(script)
                    if env:
                        profile["env"] = [f"{k}={v}" for k, v in env.items()]
                        print(f"  -> populated env from {script}")
                        break
                else:
                    print("  WARNING: profile has no env and no env script was "
                          "found -- SYCL launches will fail without LD_LIBRARY_PATH.")
            save_profile(profile)
            generate_artifacts(profile)
            changed = True
            print("  -> applied")
        print()

    if changed:
        sync_all_backends(restart_router=not args.no_router_restart,
                          restart_openarc=not args.no_router_restart)
        if not args.no_hermes:
            sync_hermes_custom_providers()
    elif args.apply:
        print("Nothing to change -- all placements already match the plans.")
    else:
        print("Re-run with --apply to rewrite placements to these plans.")


def cmd_place(args):
    """Report (and with --apply, rewrite) each profile's GPU placement from
    its VRAM footprint estimate: fits the primary card -> pin to it; too
    big -> capacity-ratio split; too big entirely -> loud warning."""
    inventory = get_gpu_inventory(force_remap=args.remap)
    if not inventory:
        print("Error: couldn't read GPU inventory (xpu-smi unavailable and "
              "llama-server --list-devices found no devices).")
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

    if args.tiers:
        cmd_place_tiers(args, inventory, d, primary, names)
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
        sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
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
    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
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
    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
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
    if profile.get("backend") == "ovms":
        print(f"Removed profile '{args.name}'. Its downloaded model weights, if any, are still "
              f"at {OVMS_MODEL_REPOSITORY / profile.get('repo_id', '')} -- delete them yourself "
              f"if no other profile needs them. If a container for it is currently running, "
              f"`docker stop {args.name}` first (or just wait out its ttl).")
    else:
        print(f"Removed profile '{args.name}'. Model file left on disk at "
              f"{profile.get('model_path')} -- delete it yourself if you don't need it for "
              f"another profile.")

    sync_all_backends(restart_router=not args.no_router_restart, restart_openarc=not args.no_router_restart)
    if not args.no_hermes:
        sync_hermes_custom_providers()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _local_weights_bytes(model_path: Path) -> int:
    """Total on-disk size of a model (first shard -> sum of all shards).
    Thin alias for modelctl_vram.weights_bytes_on_disk, which owns the
    logic so the detached calculator can use it too."""
    return modelctl_vram.weights_bytes_on_disk(model_path)


def estimate_vram_footprint(profile):
    """Estimate this profile's VRAM footprint at its configured ctx/kv_quant.
    Returns the estimate dict from modelctl_vram.estimate_from_parts, or
    None when the model file is missing. Computed on demand, never stored
    -- files and ctx change, and recomputing is cheap.

    ovms-backend profiles have no top-level model_path at all (repo_id +
    OVMS_MODEL_REPOSITORY describe where the weights live/will be pulled to
    instead) -- this always returns None for those, same as the "file
    missing" case. GGUF header parsing wouldn't apply to OpenVINO IR anyway;
    callers that already degrade open for a missing estimate (cmd_place,
    etc.) need no OVMS-specific code."""
    if not profile.get("model_path"):
        return None
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
    ctk, ctv = _resolve_cache_types(cfg)
    return modelctl_vram.estimate_from_parts(
        weights, ctx, ctk or "f16",
        gguf_params=params, mmproj_bytes=mmproj_bytes,
        cache_type_v=ctv or ctk or "f16")


def _gpu_inventory_from_llama() -> list:
    """Fallback GPU inventory when xpu-smi is unavailable: probe llama-server
    --list-devices (trying the known env scripts when the bare binary sees no
    devices -- SYCL needs LD_LIBRARY_PATH). SYCL builds are probed first so
    the device naming matches the profiles' backend on multi-build boxes.
    Same shape as get_gpu_inventory; [] on failure."""
    MiB = 1 << 20
    binaries = []
    for b in [LLAMA_SERVER_BIN] + find_llama_server_candidates():
        if b and b not in binaries:
            binaries.append(b)
    binaries.sort(key=lambda b: ("sycl" not in b.lower(), b))
    envs = [None]
    for script in find_env_script_candidates():
        env = source_env_script(script)
        if env:
            envs.append(env)
    for binary in binaries:
        for env in envs:
            devices = modelctl_vram.llama_list_devices(binary, env=env)
            if GPU_EXCLUDE_PATTERN:
                devices = [d for d in devices
                           if not re.search(GPU_EXCLUDE_PATTERN, d["name"],
                                            re.IGNORECASE)]
            if devices:
                return sorted(
                    ({"device": d["device"], "name": d["name"],
                      "total_bytes": d["total_mib"] * MiB,
                      "free_bytes": d["free_mib"] * MiB} for d in devices),
                    key=lambda d: -d["total_bytes"])
    return []


def get_gpu_inventory(force_remap: bool = False) -> list:
    """Live GPU inventory: [{device: 'SYCL0', name, total_bytes, free_bytes}],
    sorted biggest-first. Free bytes are read fresh from xpu-smi on every
    call; only the SYCL-name mapping is cached (GPU_MAP_PATH), since probing
    it needs a slow llama-server --list-devices run. Falls back to probing
    llama-server directly when xpu-smi is unavailable. Returns [] when all
    probes fail -- callers degrade to warnings, never errors."""
    xpu = modelctl_vram.xpu_devices()
    if GPU_EXCLUDE_PATTERN:
        xpu = [d for d in xpu
               if not re.search(GPU_EXCLUDE_PATTERN, d["name"], re.IGNORECASE)]
    if not xpu:
        return _gpu_inventory_from_llama()

    mapping = None
    if not force_remap and GPU_MAP_PATH.exists():
        try:
            mapping = json.loads(GPU_MAP_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            mapping = None
    if mapping is None:
        sycl = modelctl_vram.llama_list_devices(LLAMA_SERVER_BIN)
        if GPU_EXCLUDE_PATTERN:
            sycl = [s for s in sycl
                    if not re.search(GPU_EXCLUDE_PATTERN, s["name"], re.IGNORECASE)]
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
        weights_bytes, int(d["ctx"]), d["cache_type_k"],
        cache_type_v=d["cache_type_v"])
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


def llama_swap_model_ids(timeout=5):
    """Query llama-swap's /v1/models for the list of currently-registered
    model ids. Returns None (distinct from an empty list) on any connection
    failure, so callers can tell "llama-swap unreachable" apart from
    "reachable but this model isn't registered"."""
    url = LLAMA_SWAP_BASE_URL.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None
    return [m["id"] for m in data.get("data", [])]


def is_code_eval_task(task_name: str) -> bool:
    """True if `task_name` executes model-generated code and therefore
    needs the unsandboxed-execution opt-in -- see CODE_EVAL_TASK_PREFIXES."""
    return task_name.startswith(CODE_EVAL_TASK_PREFIXES)


def build_bench_env(limit, think, max_gen_toks, allow_code_eval):
    """Environment overlay for a bench.sh subprocess call. limit=None or 0
    means 'run the full task' (bench.sh's own LIMIT=0 convention)."""
    env = dict(os.environ)
    env["LIMIT"] = str(limit) if limit else "0"
    env["THINK"] = "1" if think else "0"
    env["MAX_GEN_TOKS"] = str(max_gen_toks)
    if allow_code_eval:
        env["HF_ALLOW_CODE_EVAL"] = "1"
    return env


def build_bench_command(model_id, tasks, allow_code_eval=False):
    """argv for a bench.sh invocation. `tasks` is comma-joined into bench.sh's
    single positional <task1,task2,...> arg."""
    cmd = [str(BENCH_SH), model_id, ",".join(tasks)]
    if allow_code_eval:
        cmd.append("--confirm_run_unsafe_code")
    return cmd


def build_speed_command(model_id, max_tokens=256, runs=3, think=False):
    """argv for a speed.py invocation. speed.py is stdlib-only (no venv
    dependencies), so it always runs under whichever python3 is running
    modelctl itself."""
    cmd = [sys.executable, str(SPEED_PY), model_id, str(max_tokens), str(runs)]
    if think:
        cmd.append("--think")
    return cmd


def run_profile_evals(profile, args):
    """The --evals path of `modelctl test`: run real lm-evaluation-harness
    tasks (and/or the 'speed' pseudo-task) against a profile through the
    live llama-swap service, by shelling out to bench.sh/speed.py -- see the
    BENCH_SH/SPEED_PY comment for why this isn't reimplemented here. Unlike
    the default quick smoke test below, this works for OVMS profiles too,
    since it never needs a local llama-server binary -- it just talks to
    whatever llama-swap already has registered."""
    model_id = profile["name"]

    ids = llama_swap_model_ids()
    if ids is None:
        print(f"Could not reach llama-swap at {LLAMA_SWAP_BASE_URL} -- is it running? "
              f"(`systemctl --user status llama-swap`)")
        sys.exit(1)
    if model_id not in ids:
        seen = ", ".join(ids) if ids else "(none)"
        print(f"'{model_id}' isn't currently registered with llama-swap (saw: {seen}).\n"
              f"Run `modelctl sync` to push this profile's config.yaml entry live, then retry.")
        sys.exit(1)

    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    speed_requested = "speed" in evals
    tasks = [e for e in evals if e != "speed"]
    if not speed_requested and not tasks:
        print("--evals didn't contain any task names (or just whitespace/commas).")
        sys.exit(1)

    code_tasks = [t for t in tasks if is_code_eval_task(t)]
    if code_tasks and not args.confirm_unsafe_code:
        print(f"Task(s) {', '.join(code_tasks)} execute model-generated code UNSANDBOXED on "
              f"this host -- no container/VM, just a subprocess with your own user's privileges "
              f"(see ~/services/llama-swap/README.md). Re-run with --confirm-unsafe-code to proceed.")
        sys.exit(1)

    if tasks and not BENCH_SH.exists():
        print(f"Expected {BENCH_SH} to exist -- has ~/services/llama-swap/bench.sh moved?")
        sys.exit(1)
    if speed_requested and not SPEED_PY.exists():
        print(f"Expected {SPEED_PY} to exist -- has ~/services/llama-swap/speed.py moved?")
        sys.exit(1)

    exit_code = 0
    if speed_requested:
        print(f"=== speed: {model_id} ===")
        sys.stdout.flush()
        result = subprocess.run(build_speed_command(model_id, think=args.think), check=False)
        exit_code = exit_code or result.returncode
        print()

    if tasks:
        print(f"=== evals: {model_id}: {', '.join(tasks)} ===")
        sys.stdout.flush()
        allow_code_eval = bool(code_tasks)
        env = build_bench_env(args.limit, args.think, args.max_gen_toks, allow_code_eval)
        result = subprocess.run(build_bench_command(model_id, tasks, allow_code_eval), env=env, check=False)
        exit_code = exit_code or result.returncode

    if exit_code:
        sys.exit(exit_code)


# Fields a web/CLI edit may touch. Config keys land in profile["config"];
# profile keys on the profile itself. Anything else is rejected with a
# message rather than silently written, so typos don't become phantom config.
EDITABLE_CONFIG_KEYS = {"device", "split_mode", "tensor_split", "ctx",
                        "cache_type_k", "cache_type_v", "flash_attn", "ttl",
                        "mtp", "fit", "extra"}
EDITABLE_PROFILE_KEYS = {"binary", "env", "enabled"}
_INT_CONFIG_KEYS = {"ctx", "ttl"}


def update_profile_config(name, updates: dict, resync=True):
    """Field-level, non-interactive profile edit (the web/EDIT core): apply
    `updates` (validated against EDITABLE_*_KEYS), save, regenerate artifacts,
    optionally re-sync backends. Returns (profile, messages) -- messages
    include ignored keys, preflight output, and anything notable."""
    profile = load_profile(name)
    cfg = profile.setdefault("config", {})
    messages = []
    for key, value in (updates or {}).items():
        if key in EDITABLE_CONFIG_KEYS:
            if key in _INT_CONFIG_KEYS:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    messages.append(f"ignored {key}: '{value}' isn't a number")
                    continue
            cfg[key] = value
        elif key in EDITABLE_PROFILE_KEYS:
            profile[key] = value
        else:
            messages.append(f"ignored unknown field: {key}")
    save_profile(profile)
    generate_artifacts(profile)
    ok, _bin, _env, pf_messages = preflight(profile)
    messages.extend(pf_messages)
    if not ok:
        messages.append("WARNING: preflight reports this profile isn't runnable "
                        "as configured -- see messages above.")
    if resync:
        sync_all_backends(restart_router=True, restart_openarc=True)
    return profile, messages


def smoke_test_profile(name, timeout=600, prompt=None, proc_register=None):
    """Launch-and-generate-once verification (the `test` core, callable from
    the web service). Returns a structured result:
    {ok, stage, messages, tok_per_s, content, finish_reason, log_path} --
    never sys.exits. OVMS profiles return ok=None with guidance."""
    profile = load_profile(name)
    if profile.get("backend") == "ovms":
        return {"ok": None, "stage": "ovms", "messages": [
            f"'{name}' is an OVMS profile -- request it through llama-swap "
            "instead, which starts it on demand."]}

    result = {"ok": False, "stage": "preflight", "messages": [],
              "tok_per_s": None, "content": None, "finish_reason": None,
              "log_path": None}
    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    result["messages"].extend(messages)
    if not ok:
        return result

    # Block the launch when cache validation fails (manual mode against an
    # incapable/unprobed backend) rather than firing an untruthful command.
    caps = _capabilities_for_profile(profile, effective_bin, probe=True)
    cache_msgs = preflight_moe_cache(profile, capabilities=caps)
    cache_errors = [m for lvl, m in cache_msgs if lvl == "error"]
    result["messages"].extend(f"moe_cache {lvl}: {m}" for lvl, m in cache_msgs
                              if lvl != "ok")
    if cache_errors:
        return result

    port = find_free_port()
    cmd = [effective_bin, "--port", str(port)] + build_server_args(
        profile, capabilities=caps)
    env = dict(os.environ)
    env.update(effective_env)

    log_path = PROFILES_DIR / profile["name"] / "test.log"
    result["log_path"] = str(log_path)
    result["stage"] = "launch"
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            if proc_register:
                proc_register(proc)
        except FileNotFoundError as e:
            result["messages"].append(f"couldn't start the process: {e}")
            return result

    try:
        deadline = time.time() + timeout
        healthy = False
        while time.time() < deadline:
            if proc.poll() is not None:
                result["stage"] = "load"
                result["messages"].append(
                    f"process exited early (code {proc.returncode})")
                return result
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(2)

        if not healthy:
            result["stage"] = "load"
            result["messages"].append(f"never became healthy within {timeout}s")
            return result

        result["stage"] = "generate"
        body = json.dumps({
            "messages": [{"role": "user", "content":
                          prompt or "What is 9*13? Answer with just the number."}],
            "max_tokens": 512,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read())
        except Exception as e:
            result["messages"].append(f"completion request errored: {e}")
            return result

        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        finish = resp.get("choices", [{}])[0].get("finish_reason")
        result["content"] = content
        result["finish_reason"] = finish
        result["tok_per_s"] = resp.get("timings", {}).get("predicted_per_second")
        result["ok"] = bool(finish == "stop" and content.strip())
        if not result["ok"]:
            result["messages"].append(
                f"loaded and responded, but finish_reason='{finish}' or empty "
                "content -- inspect manually.")
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def cmd_test(args):
    profile = load_profile(args.name)

    if getattr(args, "evals", None):
        run_profile_evals(profile, args)
        return

    if profile.get("backend") == "ovms":
        print(f"'{profile['name']}' is an OVMS profile -- there's no local llama-server binary "
              f"to launch directly. Request it through llama-swap instead, which starts it on "
              f"demand:\n"
              f"  curl http://127.0.0.1:9292/v1/chat/completions -d "
              f"'{{\"model\":\"{profile['name']}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n"
              f"...or run `modelctl test {profile['name']} --evals gsm8k,ifeval` for a real eval "
              f"through llama-swap, which works for OVMS profiles too.")
        return

    print(f"Checking '{profile['name']}' before launch ...")
    res = smoke_test_profile(args.name, timeout=args.timeout)
    for m in res["messages"]:
        print(f"  {m}")
    if res["stage"] == "preflight":
        print(f"\nNot launching '{profile['name']}' -- fix the error(s) above first.")
        sys.exit(1)
    print(f"\nLaunching '{profile['name']}' (log: {res['log_path']}) ...")
    if res["stage"] in ("launch", "load"):
        print("FAIL: " + (res["messages"][-1] if res["messages"] else "load failed"))
        print_log_tail(res["log_path"])
        return
    if res["stage"] == "generate":
        print(f"FAIL: {res['messages'][-1] if res['messages'] else 'generation failed'}")
        return

    print(f"finish_reason: {res['finish_reason']}")
    print(f"content: {res['content']!r}")
    if res["tok_per_s"]:
        print(f"generation speed: {res['tok_per_s']:.1f} tok/s")
    if res["ok"]:
        print(f"PASS: '{profile['name']}' loads and generates correctly.")
    else:
        print(f"WARN: loaded and responded, but finish_reason was '{res['finish_reason']}' "
              f"or content was empty -- inspect manually before trusting this profile.")


def cmd_web(args):
    """Run the web console in the foreground (the systemd unit runs it the
    same way). Bind/token via MODELCTL_WEB_BIND / MODELCTL_WEB_TOKEN."""
    import uvicorn
    from modelctl_web.app import app
    bind = os.environ.get("MODELCTL_WEB_BIND", "0.0.0.0:9293")
    host, _, port = bind.rpartition(":")
    print(f"modelctl-web on http://{host}:{port} "
          f"(token: ~/.local/share/modelctl/web_token)")
    uvicorn.run(app, host=host or "0.0.0.0", port=int(port or 9293),
                log_level="info")


def vram_footer_lines(inventory) -> list:
    """Human-readable per-GPU 'used/total' lines for status/stats output.
    Empty list when the inventory is unavailable, so callers just skip it."""
    lines = []
    for d in inventory:
        used = d["total_bytes"] - d["free_bytes"]
        lines.append(f"{d['device']}: {_format_size(used)} / "
                     f"{_format_size(d['total_bytes'])} VRAM used ({d['name']})")
    return lines


def cmd_router_status(args):
    try:
        rows = router_status()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not rows:
        print("No models registered with the router.")
    else:
        print(f"{'MODEL':<32} {'STATUS':<10} {'GPU':<22} {'EST':<10} NOTES")
        any_failed = False
        for r in rows:
            est_col = "?"
            profile_path = PROFILES_DIR / f"{r['name']}.json"
            if profile_path.exists():
                est = estimate_vram_footprint(json.loads(profile_path.read_text()))
                if est:
                    est_col = f"~{_format_size(est['total'])}"
            notes = ""
            if r["failed"]:
                notes = (f"FAILED (exit {r['exit_code']})" if r["exit_code"] is not None
                          else "FAILED")
                any_failed = True
            elif not r["from_preset"]:
                notes = "(not a modelctl profile)"
            print(f"{r['name']:<32} {r['status']:<10} {r['gpu']:<22} {est_col:<10} {notes}")
        if any_failed:
            print("\nFor failed models: journalctl --user -u llama-swap.service -n 100")

    footer = vram_footer_lines(get_gpu_inventory())
    if footer:
        print()
        for line in footer:
            print(line)


def parse_prometheus_text(text: str) -> dict:
    """Prometheus exposition text -> {metric_name: float}. Labels are not
    used by llama-server's exporter, so plain name-value parsing is enough."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            out[name] = float(parts[-1])
        except ValueError:
            continue
    return out


def fetch_model_metrics(name: str, timeout: int = 10):
    """GET one loaded model's own /metrics directly from its upstream
    llama-server port. llama-swap's front-door /metrics is process-wide
    (CPU/GPU gauges only) and ignores a ?model= filter, so per-model
    llamacpp:* counters have to come from the worker's own port -- same
    approach as modelctl_web/app.py's _scrape_moe_cache_metrics(). Returns
    a metrics dict, or None on any failure (the row renders as '?' -- one
    broken instance shouldn't kill the whole table)."""
    from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError
    client = LlamaSwapClient(timeout=timeout)
    try:
        running = client.running_models()
    except ModelctlSwapError:
        return None
    worker = next((m for m in running if m.get("model") == name), None)
    if not worker:
        return None
    port = LlamaSwapClient._worker_port(worker)
    if not port:
        return None
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return parse_prometheus_text(r.read().decode(errors="replace"))
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None


def stats_row_from_metrics(metrics: dict) -> dict:
    """Compute the stats table columns; '?' for anything missing.
    Metric names match this llama.cpp build's exporter
    (tools/server/server-context.cpp)."""
    m = metrics or {}

    def ratio(num_key, den_key):
        num, den = m.get(num_key), m.get(den_key)
        if num is None or not den:
            return "?"
        return f"{num / den:.1f}"

    def gauge(key):
        v = m.get(key)
        return f"{v:.1f}" if v is not None else "?"

    processing, deferred = m.get("llamacpp:requests_processing"), \
        m.get("llamacpp:requests_deferred")
    requests = (f"{processing:.0f}/{deferred:.0f}"
                if processing is not None and deferred is not None else "?")
    return {
        "gen_tps": ratio("llamacpp:tokens_predicted_total",
                         "llamacpp:tokens_predicted_seconds_total"),
        "prompt_tps": ratio("llamacpp:prompt_tokens_total",
                            "llamacpp:prompt_seconds_total"),
        "last_gen_tps": gauge("llamacpp:predicted_tokens_seconds"),
        "requests": requests,
    }


def cmd_router_stats(args):
    try:
        rows = router_status()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    loaded = [r for r in rows if r["status"] == "loaded"]
    if not loaded:
        print("No models currently loaded.")
    else:
        print(f"{'MODEL':<32} {'GEN T/S':>8} {'PROMPT T/S':>11} "
              f"{'LAST GEN':>9} {'REQ P/D':>8}")
        for r in loaded:
            stats = stats_row_from_metrics(fetch_model_metrics(r["name"]))
            print(f"{r['name']:<32} {stats['gen_tps']:>8} "
                  f"{stats['prompt_tps']:>11} {stats['last_gen_tps']:>9} "
                  f"{stats['requests']:>8}")
        print("\n(GEN/PROMPT T/S are lifetime averages; LAST GEN is the most "
              "recent request's throughput.)")

    footer = vram_footer_lines(get_gpu_inventory())
    if footer:
        print()
        for line in footer:
            print(line)


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

    # router_load() (LlamaSwapClient.warm_load) blocks until the model
    # reports ready or the timeout elapses -- no separate poll needed.
    ok, msg = router_load(args.name)
    print(msg)
    if not ok:
        sys.exit(1)


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

    # Hidden _worker subcommand for managed runtime
    import modelctl_worker
    modelctl_worker.register_subcommand(sub)

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

    p_pull = sub.add_parser("pull", help="pull a GGUF model from a HF repo and configure it as a llama.cpp profile")
    p_pull.add_argument("repo_id", nargs="?", default=None,
                         help="required unless --tui is passed (the wizard starts at search)")
    p_pull.add_argument("--tui", action="store_true",
                         help="use the interactive Textual wizard instead of the input()-based prompts")
    p_pull.add_argument("--yes", action="store_true",
                         help="zero-config mode: auto-pick the largest quant that fits the primary GPU, "
                              "accept all defaults, no prompts")
    p_pull.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_pull.add_argument("--no-router-restart", action="store_true",
                         help="don't restart llama-swap after regenerating its config")
    p_pull.set_defaults(func=cmd_pull)

    p_ovms_add = sub.add_parser(
        "ovms-add",
        help="register a native OpenVINO IR repo (e.g. OpenVINO/Qwen3.6-27B-int4-ov) as an "
             "OVMS profile (no download -- OVMS pulls its own copy on first start)",
    )
    p_ovms_add.add_argument("repo_id", help="HF repo of a pre-exported OpenVINO IR model")
    p_ovms_add.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_ovms_add.add_argument("--no-router-restart", action="store_true",
                             help="don't restart llama-swap after saving this profile")
    p_ovms_add.set_defaults(func=cmd_ovms_add)

    p_ovms_convert = sub.add_parser(
        "ovms-convert",
        help="convert a source HF repo (original weights, not pre-exported OpenVINO IR) "
             "to OpenVINO IR via optimum-cli, then register it as an OVMS profile",
    )
    p_ovms_convert.add_argument("repo_id", help="HF repo of the original (non-OpenVINO) model")
    p_ovms_convert.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_ovms_convert.add_argument("--no-router-restart", action="store_true",
                                 help="don't restart llama-swap after saving this profile")
    p_ovms_convert.set_defaults(func=cmd_ovms_convert)

    p_list = sub.add_parser("list", help="list saved profiles")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show a saved profile's JSON")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_show)

    p_edit = sub.add_parser("edit", help="re-prompt config for a saved profile")
    p_edit.add_argument("name")
    p_edit.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_edit.add_argument("--no-router-restart", action="store_true",
                         help="don't restart llama-swap after regenerating its config")
    p_edit.set_defaults(func=cmd_edit)

    p_regen = sub.add_parser("regen", help="regenerate artifacts from a saved profile")
    p_regen.add_argument("name")
    p_regen.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_regen.add_argument("--no-router-restart", action="store_true",
                          help="don't restart llama-swap after regenerating its config")
    p_regen.set_defaults(func=cmd_regen)

    p_verify = sub.add_parser("verify", help="check saved profiles' model/mmproj/mtp files against "
                                              "their repo, and optionally repair mismatches")
    p_verify.add_argument("name", nargs="?", default=None, help="check only this profile (default: all)")
    p_verify.add_argument("--fix", action="store_true",
                           help="re-download and repoint any mismatched or missing file")
    p_verify.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_verify.add_argument("--no-router-restart", action="store_true",
                           help="don't restart llama-swap after regenerating its config")
    p_verify.set_defaults(func=cmd_verify)

    p_place = sub.add_parser("place", help="recommend (or apply) VRAM-fit GPU placement for profiles")
    p_place.add_argument("name", nargs="?", default=None, help="only this profile (default: all)")
    p_place.add_argument("--tiers", action="store_true",
                          help="plan across memory tiers (GPU / +GPU2 / +RAM / +SSD streaming) "
                               "with expert-granular MoE offload instead of the simple VRAM-fit rule")
    p_place.add_argument("--apply", action="store_true",
                          help="rewrite profile placement to the recommendation and re-sync")
    p_place.add_argument("--remap", action="store_true",
                          help="rebuild the cached SYCL<->xpu-smi device mapping")
    p_place.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_place.add_argument("--no-router-restart", action="store_true",
                          help="don't restart llama-swap after regenerating its config")
    p_place.set_defaults(func=cmd_place)

    p_web = sub.add_parser("web", help="run the web console in the foreground")
    p_web.set_defaults(func=cmd_web)

    p_disable = sub.add_parser("disable", help="exclude a saved profile from router/Hermes sync without deleting it")
    p_disable.add_argument("name")
    p_disable.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_disable.add_argument("--no-router-restart", action="store_true",
                            help="don't restart llama-swap after regenerating its config")
    p_disable.set_defaults(func=cmd_disable)

    p_enable = sub.add_parser("enable", help="re-include a previously disabled profile in router/Hermes sync")
    p_enable.add_argument("name")
    p_enable.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_enable.add_argument("--no-router-restart", action="store_true",
                           help="don't restart llama-swap after regenerating its config")
    p_enable.set_defaults(func=cmd_enable)

    p_remove = sub.add_parser("remove", aliases=["rm"], help="delete a saved profile and its generated artifacts")
    p_remove.add_argument("name")
    p_remove.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_remove.add_argument("--no-router-restart", action="store_true",
                           help="don't restart llama-swap after regenerating its config")
    p_remove.set_defaults(func=cmd_remove)

    p_sync = sub.add_parser("sync", help="push all profiles into the live llama-swap config.yaml")
    p_sync.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_sync.add_argument("--hermes-dry-run", action="store_true", help="show what Hermes config would change")
    p_sync.add_argument("--no-router-restart", action="store_true",
                         help="don't restart llama-swap after regenerating its config")
    p_sync.set_defaults(func=cmd_sync)

    p_defaults = sub.add_parser("defaults", help="configure default runtime settings for new profiles")
    p_defaults.add_argument("--show", action="store_true", help="show current defaults and exit")
    p_defaults.set_defaults(func=cmd_defaults)

    p_test = sub.add_parser("test", help="launch-and-generate-once smoke test, or run real evals with --evals")
    p_test.add_argument("name")
    p_test.add_argument("--timeout", type=int, default=300, help="seconds to wait for health (default 300)")
    p_test.add_argument(
        "--evals", metavar="<task1,task2,...>",
        help="run lm-evaluation-harness tasks against this profile through llama-swap instead of "
             "the quick launch-and-generate-once smoke test -- works for OVMS profiles too, unlike "
             "the default check. Comma-separated lm-eval task names (gsm8k, ifeval, mmlu_pro, "
             "bbh_cot_fewshot_<subtask>, humaneval_instruct, mbpp_instruct, ...), plus the "
             "pseudo-task 'speed' for a raw tokens/sec throughput check. Requires the profile to "
             "already be live in llama-swap (`modelctl sync` first). See ~/services/llama-swap/README.md "
             "for which tasks work here and why (only generate_until tasks; classic loglikelihood "
             "multiple-choice tasks like mmlu/hellaswag/arc/winogrande don't).")
    p_test.add_argument("--limit", type=int, default=30,
                         help="examples per eval task, --evals only (default 30; 0 = full task)")
    p_test.add_argument("--think", action="store_true",
                         help="--evals only: leave reasoning-model 'thinking' enabled (default: "
                              "disabled for speed -- see README, confirmed ~15x slower with it on, "
                              "and can silently zero out scores if the token budget runs out mid-thought)")
    p_test.add_argument("--max-gen-toks", type=int, default=2048, dest="max_gen_toks",
                         help="--evals only: generation token budget (default 2048)")
    p_test.add_argument(
        "--confirm-unsafe-code", action="store_true", dest="confirm_unsafe_code",
        help="--evals only: required to run humaneval*/mbpp* tasks -- they execute "
             "model-generated code UNSANDBOXED on this host, no container/VM")
    p_test.set_defaults(func=cmd_test)

    p_router = sub.add_parser("router", help="manage models on the llama-swap front door (status, load/unload)")
    router_sub = p_router.add_subparsers(dest="router_command", required=True)

    p_router_status = router_sub.add_parser("status", help="show loaded/unloaded models and GPU placement")
    p_router_status.set_defaults(func=cmd_router_status)

    p_router_stats = router_sub.add_parser("stats", help="per-model throughput and VRAM stats")
    p_router_stats.set_defaults(func=cmd_router_stats)

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
