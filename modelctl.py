#!/usr/bin/env python3
"""
modelctl - search, pull, and configure local GGUF models from Hugging Face.

Backend-agnostic: generates a run.sh (raw llama-server command), a
llama-swap config snippet, and an Ollama-style Modelfile from one
saved profile. Profiles are plain JSON so they're easy to inspect,
edit by hand, or regenerate later.
"""
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
LLAMA_SWAP_CONFIG = Path(os.environ.get("MODELCTL_LLAMA_SWAP_CONFIG", Path.home() / "llama-swap" / "config.yaml"))
LLAMA_SWAP_HEADER = Path(os.environ.get("MODELCTL_LLAMA_SWAP_HEADER", Path.home() / "llama-swap" / "config.header.yaml"))
ROUTER_PRESET_PATH = Path(os.environ.get("MODELCTL_ROUTER_PRESET", Path.home() / "llama-router" / "router.preset.ini"))
ROUTER_PORT = os.environ.get("MODELCTL_ROUTER_PORT", "7071")

# Hermes Agent integration: modelctl can keep Hermes' custom_providers list
# in sync with saved profiles so pulled models show up in `hermes model`.
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_CONFIG = Path(os.environ.get("MODELCTL_HERMES_CONFIG", HERMES_HOME / "config.yaml"))
# Base URL of the llama-swap / OpenAI-compatible endpoint Hermes should talk to.
# Defaults to the one already configured in Hermes, then to a local llama-swap.
LLAMA_SWAP_BASE_URL = os.environ.get("MODELCTL_LLAMA_SWAP_BASE_URL")

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

# Persisted user defaults live next to profiles so they survive re-installs.
DEFAULTS_PATH = STATE_DIR / "defaults.json"


# Env vars worth carrying into generated configs if they're set in the
# shell modelctl is run from (e.g. after sourcing llama-sycl-env.sh).
# Override the list with MODELCTL_PASSTHROUGH_ENV="VAR1,VAR2,...".
DEFAULT_PASSTHROUGH = ["LD_LIBRARY_PATH", "SYCL_CACHE_PERSISTENT", "ONEAPI_DEVICE_SELECTOR", "GGML_SYCL_DISABLE_OPT"]
PASSTHROUGH_VARS = os.environ.get("MODELCTL_PASSTHROUGH_ENV", ",".join(DEFAULT_PASSTHROUGH)).split(",")

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

    device = input(f"GPU device (blank = use split strategy) [{current.get('device', '') or '(none)'}]: ").strip() or current.get("device", "")
    split_mode = input(f"Split mode [{current['split_mode']}]: ").strip() or current["split_mode"]
    tensor_split = input(f"Tensor split weights [{current['tensor_split']}]: ").strip() or current["tensor_split"]
    ctx_raw = input(f"Context length [{current['ctx']}]: ").strip()
    ctx = int(ctx_raw) if ctx_raw.isdigit() else current["ctx"]
    kv_quant = input(f"KV cache quant [{current['kv_quant']}]: ").strip() or current["kv_quant"]
    flash_attn = input(f"Flash attention [{current['flash_attn']}]: ").strip() or current["flash_attn"]

    ttl_raw = input(f"llama-swap idle TTL in seconds [{current['ttl']}]: ").strip()
    ttl = int(ttl_raw) if ttl_raw.isdigit() else current["ttl"]

    defaults = {
        "device": device,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "ctx": ctx,
        "kv_quant": kv_quant,
        "flash_attn": flash_attn,
        "ttl": ttl,
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


def get_llama_swap_base_url(cfg: dict = None) -> str:
    """Figure out the OpenAI-compatible base URL Hermes should use.

    Resolution order:
      1. MODELCTL_LLAMA_SWAP_BASE_URL env var
      2. The base_url already configured in Hermes (model.base_url or any custom_provider)
      3. http://127.0.0.1:7070/v1 (common llama-swap default)
    """
    if LLAMA_SWAP_BASE_URL:
        return LLAMA_SWAP_BASE_URL.rstrip("/") + "/"
    cfg = cfg if cfg is not None else read_hermes_config()
    # Hermes' top-level model block may already point at the swap endpoint.
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("base_url"):
        return model_cfg["base_url"].rstrip("/") + "/"
    for cp in cfg.get("custom_providers") or []:
        if cp.get("base_url"):
            return cp["base_url"].rstrip("/") + "/"
    return "http://127.0.0.1:7070/v1"


def sync_hermes_custom_providers(dry_run: bool = False):
    """Rebuild Hermes' custom_providers list from all saved modelctl profiles.

    Each profile becomes a selectable model in `hermes model`. The provider
    name is reused if Hermes already has a custom provider pointing at the
    same llama-swap base URL; otherwise a sensible default is created.
    """
    if yaml is None:
        print("WARNING: pyyaml not installed -- cannot sync Hermes custom_providers. "
              "Install it with `pip install pyyaml` and re-run `modelctl sync`.", file=sys.stderr)
        return False

    if not HERMES_CONFIG.exists():
        print(f"NOTE: Hermes config not found at {HERMES_CONFIG} -- skipping Hermes sync.")
        return False

    cfg = read_hermes_config()
    base_url = get_llama_swap_base_url(cfg)

    # Decide the provider name. Prefer an existing custom provider that already
    # uses this base URL, otherwise default to something descriptive.
    existing = cfg.get("custom_providers") or []
    provider_name = None
    for cp in existing:
        if cp.get("base_url", "").rstrip("/") + "/" == base_url:
            provider_name = cp.get("name")
            break
    if not provider_name:
        provider_name = "LocalLlama"

    profiles = sorted(PROFILES_DIR.glob("*.json"))
    models_map = {}
    for p in profiles:
        profile = json.loads(p.read_text())
        name = profile.get("name")
        if not name:
            continue
        # llama-swap exposes models by profile name at /v1/models.
        ctx = profile.get("config", {}).get("ctx")
        models_map[name] = {"context_length": int(ctx)} if ctx else {}

    if not models_map:
        print("No saved profiles -- leaving Hermes custom_providers unchanged.")
        return False

    new_provider = {"name": provider_name, "base_url": base_url, "models": models_map}

    # Replace just the entry/entries for this base URL, but preserve any
    # non-llama-swap custom providers that point at a different base URL
    # (e.g. Ollama on another port).
    kept = [cp for cp in existing if cp.get("base_url", "").rstrip("/") + "/" != base_url]
    final = kept + [new_provider]

    if final == existing:
        print("Hermes custom_providers already up to date.")
        return True

    cfg["custom_providers"] = final

    if dry_run:
        print("Would write the following custom_providers to Hermes config:")
        for cp in final:
            if "models" in cp:
                model_list = ", ".join(cp["models"].keys())
                print(f"  - {cp['name']} @ {cp['base_url']}: models = [{model_list}]")
            else:
                print(f"  - {cp.get('name')} @ {cp.get('base_url')}")
        return True

    write_hermes_config(cfg)
    print(f"Synced {len(models_map)} model(s) to Hermes custom_providers @ {HERMES_CONFIG}")
    print(f"  provider: {provider_name}, base_url: {base_url}")
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
    name = re.sub(r"-[IQ]?[0-9]_[A-Za-z0-9_]+(?:-\d+)?$", "", name, flags=re.IGNORECASE)
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
    effective_env = {k: v for e in (profile.get("env") or []) for k, v in [e.split("=", 1)]}
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


def cmd_search(args):
    query = " ".join(args.query)
    print(f"Searching Hugging Face for: {query}\n")
    models = api.list_models(search=query, limit=args.limit, sort="downloads")
    rows = []
    for m in models:
        tags = m.tags or []
        gguf_hint = "GGUF" if any("gguf" in t.lower() for t in tags) or "gguf" in m.id.lower() else ""
        rows.append((m.id, m.downloads or 0, m.likes or 0, gguf_hint))

    if not rows:
        print("No results.")
        return

    print(f"{'REPO':<55} {'DOWNLOADS':>10} {'LIKES':>6}  TAG")
    for repo_id, downloads, likes, hint in rows:
        print(f"{repo_id:<55} {downloads:>10,} {likes:>6,}  {hint}")
    print(f"\n{len(rows)} results. Use: modelctl pull <repo_id>")


SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def list_gguf_files(repo_id: str):
    files = api.list_repo_files(repo_id)
    return [f for f in files if f.endswith(".gguf")]


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


def download_if_needed(repo_id: str, filename: str, dest_dir: Path) -> str:
    """Skip the download entirely if the file's already sitting in dest_dir --
    avoids re-pulling a multi-GB mmproj/model file if it's already present
    from an earlier run or another profile that shares it."""
    target = dest_dir / Path(filename).name
    if target.exists() and target.stat().st_size > 0:
        print(f"  already present, skipping download: {target.name}")
        return str(target)
    print(f"  downloading {filename} -> {dest_dir} ...")
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))


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
    repo_id = args.repo_id
    print(f"Fetching file list for {repo_id} ...")
    try:
        gguf_files = list_gguf_files(repo_id)
    except Exception as e:
        print(f"Error: couldn't list files for '{repo_id}': {e}")
        sys.exit(1)

    if not gguf_files:
        print("No .gguf files found in this repo.")
        sys.exit(1)

    # Two strictly separate lists -- a file can only ever appear in one of
    # these, so there's no way to pick the same mmproj file from both menus.
    mmproj_files = [f for f in gguf_files if "mmproj" in f.lower()]
    quant_files = [f for f in gguf_files if f not in mmproj_files]
    groups = group_files(quant_files)

    print(f"\n=== Model files ({len(groups)}) ===")
    for i, g in enumerate(groups):
        tag = f"  [{len(g['files'])}-part split]" if g["sharded"] else ""
        print(f"  [{i}] {g['label']}{tag}")

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
            print(f"  [{i}] {f}")
        mi = prompt_pick("mmproj", "Pick mmproj number, or blank to skip: ")
        if mi != -1:
            if 0 <= mi < len(mmproj_files):
                mmproj_chosen = mmproj_files[mi]
            else:
                print(f"  index {mi} isn't in the mmproj list, skipping mmproj.")

    dest_dir = Path(input(f"Download directory [{DEFAULT_MODELS_DIR}]: ").strip() or DEFAULT_MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    local_mmproj_path = None
    if mmproj_chosen:
        print("Fetching mmproj (shared across all selected profiles):")
        local_mmproj_path = download_if_needed(repo_id, mmproj_chosen, dest_dir)

    if len(chosen_groups) > 1:
        print(f"\n{len(chosen_groups)} files selected -- config below is shared across all of them.")
    shared_config = prompt_config(repo_id, chosen_groups[0]["label"] if chosen_groups else "")
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
            "config": config,
            "env": env,
        }
        save_profile(profile)
        generate_artifacts(profile)
        print(f"-> saved profile '{name}'")

    sync_llama_swap_config()
    sync_router_preset()
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"\nDone. {len(chosen_groups)} profile(s) created and pushed to {LLAMA_SWAP_CONFIG}.")
    print("llama-swap is watching that file (--watch-config) so it should pick this up live -- no restart needed.")


def prompt_int(prompt: str, default: int) -> str:
    """Prompt for an integer, re-asking on garbage input (e.g. a numpad
    sending escape codes instead of digits with NumLock off) instead of
    silently accepting whatever the terminal sent."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return str(default)
        if raw.lstrip("-").isdigit():
            return raw
        print(f"  '{raw}' isn't a number -- try again, or press Enter for the default ({default}).")


def prompt_config(repo_id: str = "", label: str = ""):
    print("\n--- runtime config (blank = sensible default) ---")
    d = load_defaults()
    device = input(f"GPU device [{d.get('device') or 'blank = use split strategy'}]: ").strip() or d.get("device", "")
    split_mode = input(f"Split mode [{d['split_mode']}]: ").strip() or d["split_mode"]
    tensor_split = input(f"Tensor split weights [{d['tensor_split']}]: ").strip() or d["tensor_split"]
    ctx = prompt_int("Context length", d["ctx"])
    kv_quant = input(f"KV cache quant, e.g. q8_0 [{d['kv_quant']}]: ").strip() or d["kv_quant"]
    flash_attn = input(f"Flash attention [{d['flash_attn']}]: ").strip() or d["flash_attn"]


    ttl = prompt_int("llama-swap idle TTL in seconds", d["ttl"])
    extra = input("Any extra llama-server flags (raw string, optional): ").strip()
    return {
        "device": device,
        "split_mode": split_mode,
        "tensor_split": tensor_split,
        "ctx": ctx,
        "kv_quant": kv_quant,
        "flash_attn": flash_attn,
        "ttl": ttl,
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
        "-c", cfg['ctx'],
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
    if cfg.get("extra"):
        # Extra is a raw string the user typed; split it safely to preserve
        # quoted values with spaces, but only if they provided it.
        args.extend(shlex.split(cfg["extra"]))
    return args


def render_llama_swap_entry(profile):
    ok, effective_bin, effective_env, messages = preflight(profile, auto_fix=True)
    args = build_server_args(profile)
    # Escape values that would break YAML string syntax if used unquoted.
    def yaml_escape(token):
        token = str(token)
        if token and not re.match(r"^[A-Za-z0-9._/+:@-]+$", token):
            return json.dumps(token)
        return token
    args_str = " \\\n      ".join(yaml_escape(a) for a in args)
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
    if cfg.get("extra"):
        # build_server_args shlex.splits cfg["extra"] into discrete CLI
        # tokens; the INI format has no equivalent structured way to splice
        # in arbitrary raw flags, so store the raw string verbatim as a
        # single extra-args line instead.
        lines.append(f"extra-args = {cfg['extra']}")
    return "\n".join(lines) + "\n", ok, messages


def args_to_shell_line(args) -> str:
    """Render a flat argument list as a shell-safe command line."""
    return " ".join(shlex.quote(str(a)) for a in args)


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
    args_str = " \\\n  ".join(args_to_shell_line(args))
    env_exports = "\n".join(f'export {k}="{v}"' for k, v in effective_env.items())

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


DEFAULT_HEADER = "healthCheckTimeout: 60\nglobalTTL: 0\n"


def _file_hash(path: Path) -> str:
    """Fast content hash for dedup checks."""
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def sync_llama_swap_config():
    """Rebuild the live llama-swap config.yaml's `models:` section from
    every saved profile. Static settings (healthCheckTimeout, groups, etc.)
    live in a separate header file modelctl never touches.

    Always backs up whatever was at LLAMA_SWAP_CONFIG before overwriting it --
    sync is destructive by design, but you should never lose the old file
    with no way to get it back.
    """
    first_transition = not LLAMA_SWAP_HEADER.exists()
    if first_transition:
        LLAMA_SWAP_HEADER.parent.mkdir(parents=True, exist_ok=True)
        LLAMA_SWAP_HEADER.write_text(DEFAULT_HEADER)

    if LLAMA_SWAP_CONFIG.exists():
        backup = LLAMA_SWAP_CONFIG.with_suffix(LLAMA_SWAP_CONFIG.suffix + ".bak")
        shutil.copy2(LLAMA_SWAP_CONFIG, backup)
        if first_transition:
            print(f"NOTE: {LLAMA_SWAP_CONFIG} already existed and wasn't created by modelctl "
                  f"(no header file yet). It's about to be fully replaced.")
            print(f"Full copy of the previous file saved to: {backup}")
            print(f"If it had custom macros/groups/apiKeys/etc, move them into "
                  f"{LLAMA_SWAP_HEADER} now -- modelctl will never touch that file.")
        else:
            print(f"(previous config backed up to {backup})")

    profiles = sorted(PROFILES_DIR.glob("*.json"))
    header = LLAMA_SWAP_HEADER.read_text().rstrip() + "\n"
    body = "models:\n" if profiles else "models: {}\n"
    any_unresolved = False
    for p in profiles:
        profile = json.loads(p.read_text())
        entry, ok, messages = render_llama_swap_entry(profile)
        if messages:
            print(f"'{profile['name']}':")
            for m in messages:
                print(f"  {m}")
        if not ok:
            any_unresolved = True
        indented = "\n".join("  " + line if line else line for line in entry.splitlines())
        body += indented + "\n"

    LLAMA_SWAP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    new_content = header + "\n" + body
    # Only write if content actually changed — avoids triggering llama-swap's
    # --watch-config reload (which kills in-flight model processes).
    import hashlib
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()
    old_hash = _file_hash(LLAMA_SWAP_CONFIG)
    if new_hash == old_hash:
        print(f"Config unchanged — skipping write (no llama-swap reload triggered).")
        return
    LLAMA_SWAP_CONFIG.write_text(new_content)

    if any_unresolved:
        print("\nNOTE: at least one profile above couldn't be fully resolved -- its entry was "
              "still written to config.yaml, but llama-swap will fail to start it until that's fixed.")

    # Sync Hermes context_length_cache.yaml so Hermes knows each model's
    # actual context window without hardcoding it in config.yaml.
    _sync_hermes_context_cache(profiles)


def sync_router_preset():
    """Rebuild router.preset.ini from every saved profile. Mirrors
    sync_llama_swap_config's change-detection-before-write guard so it
    doesn't force an unnecessary router reload."""
    ROUTER_PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)

    profiles = sorted(PROFILES_DIR.glob("*.json"))
    body = ""
    any_unresolved = False
    for p in profiles:
        profile = json.loads(p.read_text())
        entry, ok, messages = render_router_preset(profile)
        if messages:
            print(f"'{profile['name']}' (router preset):")
            for m in messages:
                print(f"  {m}")
        if not ok:
            any_unresolved = True
        body += entry + "\n"

    import hashlib
    new_hash = hashlib.sha256(body.encode()).hexdigest()
    old_hash = _file_hash(ROUTER_PRESET_PATH)
    if new_hash == old_hash:
        print("Router preset unchanged -- skipping write.")
        return

    ROUTER_PRESET_PATH.write_text(body)
    print(f"Wrote router preset for {len(profiles)} profile(s) -> {ROUTER_PRESET_PATH}")
    if any_unresolved:
        print("\nNOTE: at least one profile above couldn't be fully resolved for router mode.")


def _sync_hermes_context_cache(profiles):
    """Write per-model context lengths to Hermes' context_length_cache.yaml.

    This lets Hermes resolve the correct context length for each model
    automatically when the user switches models, instead of relying on a
    single global model.context_length in config.yaml.
    """
    if yaml is None:
        return
    base_url = get_llama_swap_base_url()
    cache = {}
    for p in profiles:
        profile = json.loads(p.read_text())
        name = profile.get("name")
        ctx = profile.get("config", {}).get("ctx")
        if name and ctx:
            cache[f"{name}@{base_url}"] = int(ctx)
    if not cache:
        return
    cache_path = HERMES_HOME / "context_length_cache.yaml"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(yaml.dump({"context_lengths": cache}, default_flow_style=False))
    print(f"Synced {len(cache)} model context lengths to {cache_path}")


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
    sync_llama_swap_config()
    sync_router_preset()
    if not args.no_hermes:
        sync_hermes_custom_providers(dry_run=args.hermes_dry_run)
    n = len(list(PROFILES_DIR.glob("*.json")))
    print(f"Synced {n} profile(s) -> {LLAMA_SWAP_CONFIG}")


def cmd_list(args):
    if not PROFILES_DIR.exists() or not any(PROFILES_DIR.glob("*.json")):
        print("No profiles saved yet. Run `modelctl pull <repo_id>` first.")
        return
    print(f"{'NAME':<30} {'REPO':<45} FILE")
    for p in sorted(PROFILES_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        print(f"{d['name']:<30} {d['repo_id']:<45} {d['file']}")


def cmd_show(args):
    profile = load_profile(args.name)
    print(json.dumps(profile, indent=2))


def cmd_edit(args):
    profile = load_profile(args.name)
    print(f"Editing '{args.name}' -- current config shown as default where applicable.\n")
    profile["config"] = prompt_config(profile.get("repo_id", ""), profile.get("file", args.name))
    profile["env"] = capture_env_passthrough()
    warn_if_env_empty(profile["env"])
    save_profile(profile)
    generate_artifacts(profile)
    sync_llama_swap_config()
    sync_router_preset()
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print("Updated, regenerated artifacts, and pushed to llama-swap config.")


def cmd_regen(args):
    profile = load_profile(args.name)
    generate_artifacts(profile)
    sync_llama_swap_config()
    sync_router_preset()
    if not args.no_hermes:
        sync_hermes_custom_providers()
    print(f"Regenerated artifacts in {profile['artifacts_dir']} and pushed to llama-swap config.")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def main():
    parser = argparse.ArgumentParser(prog="modelctl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="search Hugging Face for models")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--limit", type=int, default=15)
    p_search.set_defaults(func=cmd_search)

    p_pull = sub.add_parser("pull", help="pull a model from a HF repo and configure it")
    p_pull.add_argument("repo_id")
    p_pull.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_pull.set_defaults(func=cmd_pull)

    p_list = sub.add_parser("list", help="list saved profiles")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show a saved profile's JSON")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_show)

    p_edit = sub.add_parser("edit", help="re-prompt config for a saved profile")
    p_edit.add_argument("name")
    p_edit.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_edit.set_defaults(func=cmd_edit)

    p_regen = sub.add_parser("regen", help="regenerate artifacts from a saved profile")
    p_regen.add_argument("name")
    p_regen.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_regen.set_defaults(func=cmd_regen)

    p_sync = sub.add_parser("sync", help="push all profiles into the live llama-swap config.yaml")
    p_sync.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_sync.add_argument("--hermes-dry-run", action="store_true", help="show what Hermes config would change")
    p_sync.set_defaults(func=cmd_sync)

    p_defaults = sub.add_parser("defaults", help="configure default runtime settings for new profiles")
    p_defaults.add_argument("--show", action="store_true", help="show current defaults and exit")
    p_defaults.set_defaults(func=cmd_defaults)

    p_test = sub.add_parser("test", help="actually launch a profile and verify it generates correctly")
    p_test.add_argument("name")
    p_test.add_argument("--timeout", type=int, default=300, help="seconds to wait for health (default 300)")
    p_test.set_defaults(func=cmd_test)

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
