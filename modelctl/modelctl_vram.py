"""VRAM-awareness helpers for modelctl: GGUF metadata, footprint estimation,
GPU probing, and the placement rule.

Pure module: no modelctl import, no printing, no state in the estimator/probe
functions -- everything either returns data or a safe empty value ({} / [] /
None) on failure, so callers degrade to warnings instead of crashing when
files/tools are absent. The exception is main(), the deliberate CLI entry
point, which does print.
"""
import hashlib
import json
import math
import re
import os
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


def kv_cache_bytes(params, ctx, cache_type_k, cache_type_v=None):
    """KV cache size in bytes for a context of `ctx` tokens.

    K and V caches can be quantized independently (--cache-type-k /
    --cache-type-v); cache_type_v=None means "same as K". Computed as two
    separate sums since K and V may differ in both element type and, on
    SWA models, per-layer head dim.

    Sliding-window-attention models (Gemma family) only cache
    `swa_window` tokens on their SWA layers -- llama.cpp allocates those
    layers at the window size, so charging full ctx per layer would
    over-count by an order of magnitude. When the GGUF provides a
    sliding_window_pattern, compute per-layer; otherwise use the uniform
    full-ctx formula."""
    bpe_k = CACHE_TYPE_BYTES.get((cache_type_k or "f16").strip().lower(), 2.0)
    bpe_v = CACHE_TYPE_BYTES.get(((cache_type_v or cache_type_k) or "f16").strip().lower(), 2.0)
    pattern = params.get("swa_pattern")
    window = params.get("swa_window")

    if not pattern or not window:
        n = params["block_count"] * ctx * params["n_kv_heads"]
        return int(n * params["k_dim"] * bpe_k) + int(n * params["v_dim"] * bpe_v)

    heads = params.get("kv_heads_per_layer")
    k_swa = params.get("k_dim_swa") or params["k_dim"]
    v_swa = params.get("v_dim_swa") or params["v_dim"]
    k_elems = 0.0
    v_elems = 0.0
    for i, is_swa in enumerate(pattern):
        h = heads[i] if heads else params["n_kv_heads"]
        tokens = min(ctx, window) if is_swa else ctx
        k_elems += tokens * h * (k_swa if is_swa else params["k_dim"])
        v_elems += tokens * h * (v_swa if is_swa else params["v_dim"])
    return int(k_elems * bpe_k) + int(v_elems * bpe_v)


def estimate_from_parts(weights_bytes, ctx, kv_quant, gguf_params=None,
                        mmproj_bytes=0, cache_type_v=None):
    """VRAM footprint estimate: weights + KV cache + overhead.

    overhead = max(1 GiB, 10% of weights) covers compute buffers and
    allocator fragmentation -- deliberately rough; the placement rule's
    vram_limit_pct margin absorbs the remaining error.
    """
    weights = int(weights_bytes) + int(mmproj_bytes)
    if gguf_params:
        kv = kv_cache_bytes(gguf_params, ctx, kv_quant, cache_type_v)
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
    r"\s*(?P<free>\d+)\s*MiB free\)\s*$")


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


def llama_list_devices(binary, timeout=30, env=None):
    """Parse `llama-server --list-devices` into
    [{device: 'SYCL0', name, total_mib, free_mib}]. Returns [] on any failure.
    `env` (dict of extra vars, e.g. from source_env_script) is overlaid on the
    current process environment for the probe."""
    try:
        import os as _os
        result = subprocess.run([binary, "--list-devices"],
                                capture_output=True, text=True, timeout=timeout,
                                env={**_os.environ, **env} if env else None)
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices = []
    for line in (result.stdout or "").splitlines() + (result.stderr or "").splitlines():
        m = _DEVICE_LINE_RE.match(line)
        if m:
            devices.append({"device": m.group("device"), "name": m.group("name"),
                            "total_mib": int(m.group("total")),
                            "free_mib": int(m.group("free"))})
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


# Multi-part GGUF shard naming, e.g. model-00001-of-00003.gguf. Own copy
# (not imported from modelctl) so this module stays self-contained and
# usable as a detached calculator.
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def weights_bytes_on_disk(model_path):
    """Total on-disk size of a model: the file itself, or the sum of all
    sibling shards sharing its exact -NNNNN-of-MMMMM prefix."""
    from pathlib import Path
    model_path = Path(model_path)
    m = SHARD_RE.match(model_path.name)
    if not m:
        return model_path.stat().st_size
    prefix = m.group(1)
    return sum(p.stat().st_size
               for p in model_path.parent.glob(f"{prefix}-*-of-*.gguf")
               for pm in [SHARD_RE.match(p.name)]
               if pm and pm.group(1) == prefix)


def _shard_paths(model_path):
    """All GGUF shard paths for a model: the file itself, or every sibling
    shard sharing its -NNNNN-of-MMMMM prefix (sorted)."""
    from pathlib import Path
    model_path = Path(model_path)
    m = SHARD_RE.match(model_path.name)
    if not m:
        return [model_path]
    prefix = m.group(1)
    return sorted(p for p in model_path.parent.glob(f"{prefix}-*-of-*.gguf")
                  if (pm := SHARD_RE.match(p.name)) and pm.group(1) == prefix)


# ggml quant type traits: type id -> (block_size, block_bytes), mirrored from
# ggml/src/ggml.c type_traits[] + ggml.h's enum ggml_type. Used to compute
# exact tensor byte sizes from the GGUF tensor table. Ids not listed here
# (newer/exotic quants) are counted as unknown so callers can warn that the
# layout math is incomplete rather than silently undercounting.
GGML_TYPE_BLOCK = {
    0: (1, 4),        # F32
    1: (1, 2),        # F16
    2: (32, 18),      # Q4_0
    3: (32, 20),      # Q4_1
    6: (32, 22),      # Q5_0
    7: (32, 24),      # Q5_1
    8: (32, 34),      # Q8_0
    9: (32, 36),      # Q8_1
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    15: (256, 292),   # Q8_K
    16: (256, 66),    # IQ2_XXS
    17: (256, 74),    # IQ2_XS
    18: (256, 98),    # IQ3_XXS
    19: (256, 50),    # IQ1_S
    20: (32, 18),     # IQ4_NL
    21: (256, 110),   # IQ3_S
    22: (256, 82),    # IQ2_S
    23: (256, 136),   # IQ4_XS
    24: (1, 1),       # I8
    25: (1, 2),       # I16
    26: (1, 4),       # I32
    27: (1, 8),       # I64
    28: (1, 8),       # F64
    29: (256, 56),    # IQ1_M
    30: (1, 2),       # BF16
    34: (256, 54),    # TQ1_0
    35: (256, 66),    # TQ2_0
    39: (32, 17),     # MXFP4
    40: (16, 10),     # NVFP4
    41: (32, 17),     # Q1_0
}

# Routed MoE expert tensors (blk.N.ffn_gate_exps.weight etc.). Shared experts
# (_shexp) deliberately do NOT match -- they run every token and belong with
# the non-expert/GPU-resident set.
_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_.*_exps\.weight$")

# Repeating (per-layer) tensors of any kind, for dense per-layer byte math.
_LAYER_TENSOR_RE = re.compile(r"^blk\.(\d+)\.")


def read_gguf_tensors(path):
    """Parse one GGUF shard fully: (metadata_dict, [(name, type_id, n_elements)]).

    Same degrade-to-empty contract as read_gguf_kv_metadata: returns ({}, [])
    on any failure. Split GGUFs keep their tensor infos in each shard (a
    metadata-only first shard legitimately has zero), so callers must iterate
    all shards -- see gguf_model_layout."""
    try:
        with open(path, "rb") as f:
            if _read_exact(f, 4) != GGUF_MAGIC:
                return {}, []
            (version,) = struct.unpack("<I", _read_exact(f, 4))
            if version < 2:
                return {}, []
            tensor_count, kv_count = struct.unpack("<QQ", _read_exact(f, 16))
            meta = {}
            for _ in range(kv_count):
                key = _read_string(f)
                (type_id,) = struct.unpack("<I", _read_exact(f, 4))
                value = _read_value(f, type_id)
                if value is not None:
                    meta[key] = value
            tensors = []
            for _ in range(tensor_count):
                name = _read_string(f)
                (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
                dims = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
                (type_id,) = struct.unpack("<I", _read_exact(f, 4))
                _read_exact(f, 8)  # in-shard data offset -- not needed
                n_el = 1
                for d in dims:
                    n_el *= d
                tensors.append((name, type_id, n_el))
            return meta, tensors
    except (OSError, struct.error, MemoryError, OverflowError):
        return {}, []


def gguf_model_layout(model_path):
    """Weight layout of a (possibly sharded) GGUF for tier placement.

    Returns None on failure; otherwise a dict with:
      arch, meta, block_count, is_moe,
      weight_bytes           -- sum over all tensors
      non_expert_bytes       -- attention/norms/embeddings/shared experts
      other_bytes            -- non-expert, non-layer tensors (embd/output)
      layer_bytes            -- non-expert bytes inside blk.N.* (dense FFN etc.)
      expert_bytes_per_layer -- {layer_index: bytes} for routed experts
      unknown_type_tensors   -- count of tensors skipped (unlisted quant type)
    """
    meta = {}
    total = non_expert = other = layer_non_expert = 0
    expert = {}
    unknown = 0
    saw_tensors = False
    for shard in _shard_paths(model_path):
        shard_meta, tensors = read_gguf_tensors(shard)
        if shard_meta:
            meta = meta or shard_meta
        for name, type_id, n_el in tensors:
            saw_tensors = True
            traits = GGML_TYPE_BLOCK.get(type_id)
            if traits is None:
                unknown += 1
                continue
            block_size, block_bytes = traits
            nbytes = (n_el // block_size) * block_bytes
            total += nbytes
            em = _EXPERT_RE.match(name)
            if em:
                layer = int(em.group(1))
                expert[layer] = expert.get(layer, 0) + nbytes
            else:
                non_expert += nbytes
                if _LAYER_TENSOR_RE.match(name):
                    layer_non_expert += nbytes
                else:
                    other += nbytes
    if not saw_tensors:
        return None
    arch = meta.get("general.architecture", "")
    block_count = meta.get(f"{arch}.block_count") if arch else None
    return {"arch": arch, "meta": meta, "block_count": block_count or 0,
            "is_moe": bool(expert), "weight_bytes": total,
            "non_expert_bytes": non_expert, "other_bytes": other,
            "layer_bytes": layer_non_expert,
            "expert_bytes_per_layer": expert,
            "unknown_type_tensors": unknown}


def system_ram_available():
    """MemAvailable from /proc/meminfo in bytes; 0 when unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _fmt_gib(n):
    return f"{n / (1 << 30):.2f}GiB"


def main(argv=None):
    """Detached VRAM calculator: exact footprint math for one GGUF,
    no modelctl required. Returns an exit code (0 ok, 1 error)."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="modelctl_vram",
        description="Estimate a GGUF model's VRAM footprint (weights + KV cache + overhead).")
    parser.add_argument("model", help="path to the .gguf file (first shard if split)")
    parser.add_argument("--ctx", type=int, default=32768, help="context length (default 32768)")
    parser.add_argument("--cache-type-k", default="f16", help="K cache type (default f16)")
    parser.add_argument("--cache-type-v", default=None,
                        help="V cache type (default: same as K)")
    parser.add_argument("--mmproj", default=None, help="optional mmproj file to include")
    args = parser.parse_args(argv)

    meta = read_gguf_kv_metadata(args.model)
    params = gguf_kv_params(meta)
    if not params:
        print(f"error: couldn't read usable GGUF metadata from {args.model} -- "
              f"not a GGUF v2+ file, or missing attention fields.")
        return 1

    from pathlib import Path
    weights = weights_bytes_on_disk(args.model)
    if args.mmproj:
        try:
            mmproj_bytes = Path(args.mmproj).stat().st_size
        except OSError as exc:
            print(f"error: couldn't read mmproj file {args.mmproj} -- {exc}")
            return 1
    else:
        mmproj_bytes = 0
    ctk = args.cache_type_k
    ctv = args.cache_type_v or ctk
    k_only = kv_cache_bytes(params, args.ctx, ctk, ctk)
    est = estimate_from_parts(weights, args.ctx, ctk, gguf_params=params,
                              mmproj_bytes=mmproj_bytes, cache_type_v=ctv)

    arch = meta.get("general.architecture", "?")
    swa = params.get("swa_pattern")
    swa_note = (f", SWA {sum(1 for x in swa if x)}/{len(swa)} layers "
                f"@ window {params.get('swa_window')}" if swa else "")
    print(f"{arch}: {params['block_count']} layers, "
          f"{params['n_kv_heads']:g} KV heads (mean), "
          f"k/v dims {params['k_dim']:g}/{params['v_dim']:g}{swa_note}")
    print(f"ctx {args.ctx}, K {ctk}, V {ctv}")
    print()
    print(f"weights:  {_fmt_gib(est['weights'])}")
    print(f"kv cache: {_fmt_gib(est['kv_bytes'])}  (K {ctk} + V {ctv}; both-{ctk} would be {_fmt_gib(k_only)})")
    print(f"overhead: {_fmt_gib(est['overhead'])}")
    print(f"total:    {_fmt_gib(est['total'])}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())


_FP_CACHE = {}


def file_fingerprint(path, full_limit=512 << 20, head_tail=1 << 20):
    """Content fingerprint of a file: full sha256 up to `full_limit`;
    head+tail+size+mtime sha256 for larger files (a 183 GB GGUF can't be
    hashed whole on every plan compile). Cached by (path, size, mtime) so
    repeated compiles don't re-read. Returns 'missing' for absent paths and
    '' for None."""
    if not path:
        return ""
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    key = (path, st.st_size, int(st.st_mtime))
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    if st.st_size <= full_limit:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        h.update(str(int(st.st_mtime)).encode())
        with open(path, "rb") as f:
            h.update(f.read(head_tail))
            f.seek(max(0, st.st_size - head_tail))
            h.update(f.read(head_tail))
    fp = h.hexdigest()[:16]
    _FP_CACHE[key] = fp
    return fp
