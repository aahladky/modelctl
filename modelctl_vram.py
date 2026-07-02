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

# Fallback KV bytes/token when the GGUF header can't be parsed -- sized on
# ~30B dense models at f16 so the guess errs large for smaller models.
HEURISTIC_KV_BYTES_PER_TOKEN = 96 * 1024


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
    n_kv_heads = _mean(g("attention.head_count_kv"))
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

    return {"block_count": block_count, "n_kv_heads": n_kv_heads,
            "k_dim": k_dim, "v_dim": v_dim}


def kv_cache_bytes(params, ctx, kv_quant):
    """KV cache size in bytes for a context of `ctx` tokens."""
    bpe = CACHE_TYPE_BYTES.get((kv_quant or "f16").strip().lower(), 2.0)
    return int(params["block_count"] * ctx * params["n_kv_heads"]
               * (params["k_dim"] + params["v_dim"]) * bpe)


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
