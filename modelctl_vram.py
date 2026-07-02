"""VRAM-awareness helpers for modelctl: GGUF metadata, footprint estimation,
GPU probing, and the placement rule.

Pure module: no modelctl import, no printing, no state. Everything either
returns data or a safe empty value ({} / [] / None) on failure, so callers
degrade to warnings instead of crashing when files/tools are absent.
"""
import json
import math
import re
import struct
import subprocess

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types -> struct format. Type ids per the GGUF spec:
# 0..7 scalars, 8 string, 9 array, 10 uint64, 11 int64, 12 float64.
_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9
# Arrays longer than this are tokenizer vocab etc. -- skip their contents.
_MAX_ARRAY_KEEP = 1024


def _read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise struct.error(f"truncated GGUF: wanted {n} bytes, got {len(data)}")
    return data


def _read_string(f):
    (length,) = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _read_scalar(f, type_id):
    fmt = _SCALAR_FMT[type_id]
    (value,) = struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))
    if type_id == 7:
        return bool(value)
    return value


def _read_value(f, type_id):
    """Read one metadata value. Returns None for values we deliberately
    skip (huge arrays); the caller drops those keys."""
    if type_id in _SCALAR_FMT:
        return _read_scalar(f, type_id)
    if type_id == _TYPE_STRING:
        return _read_string(f)
    if type_id == _TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(f, 4))
        (count,) = struct.unpack("<Q", _read_exact(f, 8))
        if elem_type in _SCALAR_FMT and count <= _MAX_ARRAY_KEEP:
            return [_read_scalar(f, elem_type) for _ in range(count)]
        # Skip contents: fixed-size elements seek, strings read one by one.
        if elem_type in _SCALAR_FMT:
            f.seek(struct.calcsize(_SCALAR_FMT[elem_type]) * count, 1)
        elif elem_type == _TYPE_STRING:
            for _ in range(count):
                _read_string(f)
        else:
            raise struct.error(f"unsupported nested array type {elem_type}")
        return None
    raise struct.error(f"unknown GGUF value type {type_id}")


def read_gguf_kv_metadata(path):
    """Parse a GGUF file's metadata KV section into a plain dict.
    Returns {} on any failure (missing file, wrong magic, truncation,
    unsupported version) -- callers fall back to heuristic estimates."""
    try:
        with open(path, "rb") as f:
            if _read_exact(f, 4) != GGUF_MAGIC:
                return {}
            (version,) = struct.unpack("<I", _read_exact(f, 4))
            if version < 2:  # v1 used 32-bit counts; not worth supporting
                return {}
            _tensor_count, kv_count = struct.unpack("<QQ", _read_exact(f, 16))
            meta = {}
            for _ in range(kv_count):
                key = _read_string(f)
                (type_id,) = struct.unpack("<I", _read_exact(f, 4))
                value = _read_value(f, type_id)
                if value is not None:
                    meta[key] = value
            return meta
    except (OSError, struct.error, MemoryError, OverflowError):
        return {}


# Bytes per cached element for llama.cpp cache types (block bytes / block
# size). Unknown types fall back to f16 (conservative over-estimate).
CACHE_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}

# Fallback KV bytes/token when the GGUF header can't be parsed. Sized on a
# ~30B dense GQA model at f16 (64 layers * 8 KV heads * 256 dims * 2 bytes
# = 256 KiB/token) so the guess errs LARGE for most models -- heuristic
# estimates feed placement and guard decisions, and over-estimating only
# costs a needless split/warning, while under-estimating risks an OOM'd
# load. Old-style MHA models (no GQA) can still exceed this; those are
# rare in current GGUF releases.
HEURISTIC_KV_BYTES_PER_TOKEN = 256 * 1024


def _mean(value):
    if isinstance(value, list):
        return sum(value) / len(value) if value else None
    return value


def gguf_kv_params(meta):
    """Extract the fields KV-cache sizing needs from GGUF metadata.
    Returns None when anything essential is missing, so callers fall back
    to the heuristic instead of computing garbage."""
    arch = meta.get("general.architecture")
    if not arch:
        return None

    def g(suffix):
        return meta.get(f"{arch}.{suffix}")

    block_count = g("block_count")
    heads_raw = g("attention.head_count_kv")
    n_kv_heads = _mean(heads_raw)
    if not block_count or not n_kv_heads:
        return None

    k_dim = g("attention.key_length")
    v_dim = g("attention.value_length")
    if k_dim is None or v_dim is None:
        embed = g("embedding_length")
        head_count = _mean(g("attention.head_count"))
        if not embed or not head_count:
            return None
        head_dim = embed / head_count
        k_dim = k_dim if k_dim is not None else head_dim
        v_dim = v_dim if v_dim is not None else head_dim

    swa_pattern_raw = g("attention.sliding_window_pattern")
    if isinstance(swa_pattern_raw, list) and len(swa_pattern_raw) == block_count:
        swa_pattern = swa_pattern_raw
    else:
        swa_pattern = None

    return {"block_count": block_count, "n_kv_heads": n_kv_heads,
            "k_dim": k_dim, "v_dim": v_dim,
            "kv_heads_per_layer": heads_raw if isinstance(heads_raw, list) else None,
            "swa_window": g("attention.sliding_window"),
            "swa_pattern": swa_pattern,
            "k_dim_swa": g("attention.key_length_swa"),
            "v_dim_swa": g("attention.value_length_swa")}


def kv_cache_bytes(params, ctx, kv_quant):
    """KV cache size in bytes for a context of `ctx` tokens.

    Sliding-window-attention models (Gemma family) only cache
    `swa_window` tokens on their SWA layers -- llama.cpp allocates those
    layers at the window size, so charging full ctx per layer would
    over-count by an order of magnitude. When the GGUF provides a
    sliding_window_pattern, compute per-layer; otherwise fall back to the
    uniform full-ctx formula."""
    bpe = CACHE_TYPE_BYTES.get((kv_quant or "f16").strip().lower(), 2.0)
    pattern = params.get("swa_pattern")
    window = params.get("swa_window")
    if not pattern or not window:
        return int(params["block_count"] * ctx * params["n_kv_heads"]
                   * (params["k_dim"] + params["v_dim"]) * bpe)

    heads = params.get("kv_heads_per_layer")
    k_swa = params.get("k_dim_swa") or params["k_dim"]
    v_swa = params.get("v_dim_swa") or params["v_dim"]
    total = 0.0
    for i, is_swa in enumerate(pattern):
        h = heads[i] if heads else params["n_kv_heads"]
        if is_swa:
            tokens, dims = min(ctx, window), k_swa + v_swa
        else:
            tokens, dims = ctx, params["k_dim"] + params["v_dim"]
        total += tokens * h * dims * bpe
    return int(total)


def estimate_from_parts(weights_bytes, ctx, kv_quant, gguf_params=None,
                        mmproj_bytes=0):
    """VRAM footprint estimate: weights + KV cache + overhead.

    overhead = max(1 GiB, 10% of weights) covers compute buffers and
    allocator fragmentation -- deliberately rough; the placement rule's
    vram_limit_pct margin absorbs the remaining error.
    """
    weights = int(weights_bytes) + int(mmproj_bytes)
    if gguf_params:
        kv = kv_cache_bytes(gguf_params, ctx, kv_quant)
        quality = "exact"
    else:
        kv = int(ctx) * HEURISTIC_KV_BYTES_PER_TOKEN
        quality = "heuristic"
    overhead = max(1 << 30, int(weights * 0.10))
    return {"weights": weights, "kv_bytes": kv, "overhead": overhead,
            "total": weights + kv + overhead, "quality": quality}


# Matches "  SYCL0: Intel(R) Graphics [0xe223] (32657 MiB, 32145 MiB free)"
# from `llama-server --list-devices`.
_DEVICE_LINE_RE = re.compile(
    r"^\s*(?P<device>[A-Za-z]+\d+):\s+(?P<name>.+?)\s+\((?P<total>\d+)\s*MiB,"
    r"\s*\d+\s*MiB free\)\s*$")


def xpu_devices(timeout=15):
    """Enumerate Intel GPUs with total/free memory via xpu-smi.
    Two-phase because the device list JSON omits memory fields; the
    per-device query includes them. Returns [] on any failure."""
    try:
        listing = subprocess.run(["xpu-smi", "discovery", "-j"],
                                 capture_output=True, text=True, timeout=timeout)
        device_ids = [d["device_id"]
                      for d in json.loads(listing.stdout).get("device_list", [])]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError,
            TypeError):
        return []

    devices = []
    for did in device_ids:
        try:
            detail = subprocess.run(["xpu-smi", "discovery", "-d", str(did), "-j"],
                                    capture_output=True, text=True, timeout=timeout)
            info = json.loads(detail.stdout)
            devices.append({
                "xpu_id": did,
                "name": info.get("device_name", ""),
                "total_bytes": int(info["memory_physical_size_byte"]),
                "free_bytes": int(info["memory_free_size_byte"]),
            })
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError,
                KeyError, TypeError, ValueError):
            continue
    return devices


def llama_list_devices(binary, timeout=30):
    """Parse `llama-server --list-devices` into
    [{device: 'SYCL0', name, total_mib}]. Returns [] on any failure."""
    try:
        result = subprocess.run([binary, "--list-devices"],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices = []
    for line in (result.stdout or "").splitlines() + (result.stderr or "").splitlines():
        m = _DEVICE_LINE_RE.match(line)
        if m:
            devices.append({"device": m.group("device"), "name": m.group("name"),
                            "total_mib": int(m.group("total"))})
    return devices


def match_devices(sycl_devices, xpu_device_list):
    """Map SYCL device names to xpu-smi device ids by nearest total memory.
    SYCL and xpu-smi enumeration orders aren't guaranteed to agree, but
    total VRAM is a reliable fingerprint on a mixed-card system. Pairs
    whose sizes differ by >25% are left unmapped rather than guessed."""
    remaining = list(xpu_device_list)
    mapping = {}
    for s in sycl_devices:
        if not remaining:
            break
        s_bytes = s["total_mib"] * 1024 * 1024
        best = min(remaining, key=lambda d: abs(d["total_bytes"] - s_bytes))
        if abs(best["total_bytes"] - s_bytes) > 0.25 * best["total_bytes"]:
            continue
        mapping[s["device"]] = best["xpu_id"]
        remaining.remove(best)
    return mapping


def tensor_split_ratio(total_bytes_list):
    """Derive a llama-server --tensor-split ratio from card capacities,
    e.g. 32GB+12GB -> '8,3'. GiB-rounded then GCD-reduced for readability."""
    gib = [max(1, round(t / (1 << 30))) for t in total_bytes_list]
    divisor = math.gcd(*gib)
    return ",".join(str(g // divisor) for g in gib)


def recommend_placement(estimate_total, inventory, limit_pct, primary_device):
    """The static placement rule from the design spec:
    fits within limit_pct of the primary card alone -> pin to it;
    fits within limit_pct of all cards combined -> layer-split by capacity;
    doesn't fit at all -> same split, fits=False (caller warns loudly).

    Returns None if primary_device isn't in the inventory."""
    primary = next((d for d in inventory if d["device"] == primary_device), None)
    if primary is None:
        return None
    frac = limit_pct / 100.0
    if estimate_total <= primary["total_bytes"] * frac:
        return {"device": primary_device, "split_mode": "", "tensor_split": "",
                "fits": True}
    ordered = [primary] + [d for d in inventory if d is not primary]
    combined_budget = sum(d["total_bytes"] for d in ordered) * frac
    if len(ordered) == 1:
        # Single-GPU system: a one-way "split" is meaningless -- keep the
        # pin and let fits=False carry the over-budget warning.
        return {"device": primary_device, "split_mode": "", "tensor_split": "",
                "fits": estimate_total <= combined_budget}
    return {"device": "", "split_mode": "layer",
            "tensor_split": tensor_split_ratio([d["total_bytes"] for d in ordered]),
            "fits": estimate_total <= combined_budget}
