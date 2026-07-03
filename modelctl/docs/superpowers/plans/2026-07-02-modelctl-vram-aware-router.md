# VRAM-Aware llama-router Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give modelctl VRAM-aware GPU placement, guarded explicit router loads, and router observability (throughput stats, VRAM footers, load timing), per `docs/superpowers/specs/2026-07-02-modelctl-vram-aware-router-design.md`.

**Architecture:** New pure module `modelctl_vram.py` (GGUF header parsing, VRAM estimation math, xpu-smi/llama-server device probing, placement rule — no modelctl import, everything mockable). `modelctl.py` gains profile-aware wrappers (`estimate_vram_footprint`, `get_gpu_inventory`), a `place` subcommand, a VRAM guard in `router load`, a `[*]` preset section, and `router stats`.

**Tech Stack:** Python 3 stdlib only (struct, subprocess, urllib, json, math). Tests: `unittest` + `unittest.mock`, matching the existing `test_modelctl.py` style. Test runner: `python3 -m unittest`.

**Machine facts (verified):** primary GPU Intel 0xe223 32GB = xpu-smi device 0; Arc B580 12GB = xpu-smi device 1. `xpu-smi discovery -j` lists devices; `xpu-smi discovery -d <id> -j` adds `memory_physical_size_byte` / `memory_free_size_byte`. Router metrics names (from `tools/server/server-context.cpp`): counters `llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`, `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`, `llamacpp:n_decode_total`; gauges `llamacpp:prompt_tokens_seconds`, `llamacpp:predicted_tokens_seconds`, `llamacpp:requests_processing`, `llamacpp:requests_deferred`. There is NO kv-cache metric in this build.

---

## File Structure

- **Create** `modelctl_vram.py` — pure functions: GGUF reader, KV/estimate math, device probes, placement rule.
- **Create** `test_modelctl_vram.py` — unit tests for the new module.
- **Modify** `modelctl.py` — imports `modelctl_vram`; adds `_local_weights_bytes`, `estimate_vram_footprint`, `get_gpu_inventory`, `cmd_place`, VRAM guard + timing in `cmd_router_load`, preset `[*]` header, `cmd_router_stats`, status additions, `vram_limit_pct`/`primary_gpu` defaults, parser entries.
- **Modify** `test_modelctl.py` — tests for the modelctl-side pieces; update preset-sync tests for the new header.

---

### Task 1: GGUF header reader (`read_gguf_kv_metadata`)

**Files:**
- Create: `modelctl_vram.py`
- Create: `test_modelctl_vram.py`

- [ ] **Step 1.1: Write the failing tests**

Create `test_modelctl_vram.py`:

```python
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_vram


def gguf_bytes(kvs, version=3, magic=b"GGUF"):
    """Build a minimal GGUF file: header + metadata KVs, zero tensors.

    kvs: dict of key -> (type_id, value). Supported type_ids here:
    4=uint32, 8=string, 9=array-of-uint32 (value = list of ints),
    10=uint64.
    """
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = [magic, struct.pack("<I", version), struct.pack("<QQ", 0, len(kvs))]
    for key, (type_id, value) in kvs.items():
        out.append(s(key))
        out.append(struct.pack("<I", type_id))
        if type_id == 4:
            out.append(struct.pack("<I", value))
        elif type_id == 8:
            out.append(s(value))
        elif type_id == 9:
            out.append(struct.pack("<I", 4) + struct.pack("<Q", len(value)))
            out.extend(struct.pack("<I", v) for v in value)
        elif type_id == 10:
            out.append(struct.pack("<Q", value))
        else:
            raise AssertionError(f"fixture doesn't support type {type_id}")
    return b"".join(out)


class TestReadGgufKvMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"

    def test_reads_scalar_and_string_kvs(self):
        self.path.write_bytes(gguf_bytes({
            "general.architecture": (8, "qwen3"),
            "qwen3.block_count": (4, 48),
            "qwen3.embedding_length": (4, 5120),
            "qwen3.attention.head_count": (4, 40),
            "qwen3.attention.head_count_kv": (4, 8),
        }))
        meta = modelctl_vram.read_gguf_kv_metadata(str(self.path))
        self.assertEqual(meta["general.architecture"], "qwen3")
        self.assertEqual(meta["qwen3.block_count"], 48)
        self.assertEqual(meta["qwen3.attention.head_count_kv"], 8)

    def test_reads_int_arrays(self):
        self.path.write_bytes(gguf_bytes({
            "general.architecture": (8, "x"),
            "x.attention.head_count_kv": (9, [8, 8, 4, 4]),
        }))
        meta = modelctl_vram.read_gguf_kv_metadata(str(self.path))
        self.assertEqual(meta["x.attention.head_count_kv"], [8, 8, 4, 4])

    def test_uint64_values(self):
        self.path.write_bytes(gguf_bytes({"x.block_count": (10, 32)}))
        meta = modelctl_vram.read_gguf_kv_metadata(str(self.path))
        self.assertEqual(meta["x.block_count"], 32)

    def test_wrong_magic_returns_empty(self):
        self.path.write_bytes(gguf_bytes({}, magic=b"NOPE"))
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})

    def test_truncated_file_returns_empty(self):
        full = gguf_bytes({"general.architecture": (8, "qwen3")})
        self.path.write_bytes(full[: len(full) - 3])
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})

    def test_missing_file_returns_empty(self):
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})

    def test_v1_unsupported_returns_empty(self):
        self.path.write_bytes(gguf_bytes({}, version=1))
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: `ModuleNotFoundError: No module named 'modelctl_vram'`

- [ ] **Step 1.3: Implement the reader**

Create `modelctl_vram.py`:

```python
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
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: all 7 tests PASS

- [ ] **Step 1.5: Commit**

```bash
git add modelctl_vram.py test_modelctl_vram.py
git commit -m "Add GGUF header reader for VRAM estimation"
```

---

### Task 2: KV-cache math and footprint estimation

**Files:**
- Modify: `modelctl_vram.py`
- Modify: `test_modelctl_vram.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `test_modelctl_vram.py`:

```python
class TestGgufKvParams(unittest.TestCase):
    def test_derives_head_dim_from_embedding(self):
        meta = {
            "general.architecture": "qwen3",
            "qwen3.block_count": 48,
            "qwen3.embedding_length": 5120,
            "qwen3.attention.head_count": 40,
            "qwen3.attention.head_count_kv": 8,
        }
        p = modelctl_vram.gguf_kv_params(meta)
        self.assertEqual(p, {"block_count": 48, "n_kv_heads": 8,
                             "k_dim": 128.0, "v_dim": 128.0})

    def test_explicit_key_value_length_wins(self):
        meta = {
            "general.architecture": "x",
            "x.block_count": 32,
            "x.attention.head_count": 32,
            "x.attention.head_count_kv": 8,
            "x.embedding_length": 4096,
            "x.attention.key_length": 192,
            "x.attention.value_length": 128,
        }
        p = modelctl_vram.gguf_kv_params(meta)
        self.assertEqual(p["k_dim"], 192)
        self.assertEqual(p["v_dim"], 128)

    def test_per_layer_kv_head_array_averaged(self):
        meta = {
            "general.architecture": "x",
            "x.block_count": 4,
            "x.embedding_length": 1024,
            "x.attention.head_count": 8,
            "x.attention.head_count_kv": [8, 8, 4, 4],
        }
        p = modelctl_vram.gguf_kv_params(meta)
        self.assertEqual(p["n_kv_heads"], 6.0)

    def test_missing_arch_or_fields_returns_none(self):
        self.assertIsNone(modelctl_vram.gguf_kv_params({}))
        self.assertIsNone(modelctl_vram.gguf_kv_params({
            "general.architecture": "x", "x.block_count": 32,
        }))


class TestKvCacheBytes(unittest.TestCase):
    def test_f16_math(self):
        params = {"block_count": 48, "n_kv_heads": 8, "k_dim": 128, "v_dim": 128}
        # 48 layers * 32768 ctx * 8 heads * (128+128) dims * 2 bytes
        expected = 48 * 32768 * 8 * 256 * 2
        self.assertEqual(modelctl_vram.kv_cache_bytes(params, 32768, "f16"), expected)

    def test_q8_0_block_size(self):
        params = {"block_count": 1, "n_kv_heads": 1, "k_dim": 32, "v_dim": 32}
        # q8_0: 34 bytes per 32 elements = 1.0625 bytes/element
        self.assertEqual(modelctl_vram.kv_cache_bytes(params, 1, "q8_0"),
                         int(64 * 1.0625))

    def test_unknown_quant_conservative_f16(self):
        params = {"block_count": 1, "n_kv_heads": 1, "k_dim": 32, "v_dim": 32}
        self.assertEqual(modelctl_vram.kv_cache_bytes(params, 1, "weird"), 128)


class TestEstimateFromParts(unittest.TestCase):
    def test_exact_estimate(self):
        params = {"block_count": 48, "n_kv_heads": 8, "k_dim": 128, "v_dim": 128}
        weights = 18 * (1 << 30)
        est = modelctl_vram.estimate_from_parts(weights, 32768, "q8_0",
                                                gguf_params=params)
        self.assertEqual(est["quality"], "exact")
        self.assertEqual(est["weights"], weights)
        self.assertEqual(est["kv_bytes"],
                         modelctl_vram.kv_cache_bytes(params, 32768, "q8_0"))
        self.assertEqual(est["overhead"], int(weights * 0.10))  # >1GiB weights
        self.assertEqual(est["total"],
                         est["weights"] + est["kv_bytes"] + est["overhead"])

    def test_heuristic_fallback(self):
        est = modelctl_vram.estimate_from_parts(2 * (1 << 30), 8192, "f16",
                                                gguf_params=None)
        self.assertEqual(est["quality"], "heuristic")
        self.assertEqual(est["kv_bytes"], 8192 * modelctl_vram.HEURISTIC_KV_BYTES_PER_TOKEN)
        self.assertEqual(est["overhead"], 1 << 30)  # 10% of 2GiB < 1GiB floor

    def test_mmproj_included_in_weights(self):
        est = modelctl_vram.estimate_from_parts(100, 1, "f16", mmproj_bytes=50)
        self.assertEqual(est["weights"], 150)
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: new tests ERROR with `AttributeError: ... has no attribute 'gguf_kv_params'`

- [ ] **Step 2.3: Implement**

Append to `modelctl_vram.py`:

```python
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
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: all tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add modelctl_vram.py test_modelctl_vram.py
git commit -m "Add KV-cache math and VRAM footprint estimation"
```

---

### Task 3: GPU probes (xpu-smi, llama-server --list-devices, matching)

**Files:**
- Modify: `modelctl_vram.py`
- Modify: `test_modelctl_vram.py`

- [ ] **Step 3.1: Write the failing tests**

Append to `test_modelctl_vram.py`:

```python
def _fake_run_factory(responses):
    """responses: list of (stdout, returncode) consumed in call order."""
    it = iter(responses)

    def fake_run(cmd, **kwargs):
        stdout, code = next(it)
        return mock.Mock(stdout=stdout, stderr="", returncode=code)
    return fake_run


class TestXpuDevices(unittest.TestCase):
    LIST_JSON = json.dumps({"device_list": [
        {"device_id": 0}, {"device_id": 1},
    ]})
    DEV0_JSON = json.dumps({"device_id": 0, "device_name": "Intel(R) Graphics [0xe223]",
                            "memory_physical_size_byte": "34242297856",
                            "memory_free_size_byte": "33705361408"})
    DEV1_JSON = json.dumps({"device_id": 1, "device_name": "Intel(R) Arc(TM) B580 Graphics",
                            "memory_physical_size_byte": "12809404416",
                            "memory_free_size_byte": "12000000000"})

    def test_enumerates_devices_with_memory(self):
        fake = _fake_run_factory([(self.LIST_JSON, 0), (self.DEV0_JSON, 0),
                                  (self.DEV1_JSON, 0)])
        with mock.patch.object(modelctl_vram.subprocess, "run", side_effect=fake):
            devices = modelctl_vram.xpu_devices()
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0], {"xpu_id": 0,
                                      "name": "Intel(R) Graphics [0xe223]",
                                      "total_bytes": 34242297856,
                                      "free_bytes": 33705361408})

    def test_missing_tool_returns_empty(self):
        with mock.patch.object(modelctl_vram.subprocess, "run",
                               side_effect=FileNotFoundError()):
            self.assertEqual(modelctl_vram.xpu_devices(), [])

    def test_bad_json_returns_empty(self):
        fake = _fake_run_factory([("not json", 0)])
        with mock.patch.object(modelctl_vram.subprocess, "run", side_effect=fake):
            self.assertEqual(modelctl_vram.xpu_devices(), [])


class TestLlamaListDevices(unittest.TestCase):
    OUTPUT = """\
Available devices:
  SYCL0: Intel(R) Graphics [0xe223] (32657 MiB, 32145 MiB free)
  SYCL1: Intel(R) Arc(TM) B580 Graphics (12215 MiB, 11800 MiB free)
"""

    def test_parses_device_lines(self):
        fake = _fake_run_factory([(self.OUTPUT, 0)])
        with mock.patch.object(modelctl_vram.subprocess, "run", side_effect=fake):
            devices = modelctl_vram.llama_list_devices("llama-server")
        self.assertEqual(devices, [
            {"device": "SYCL0", "name": "Intel(R) Graphics [0xe223]", "total_mib": 32657},
            {"device": "SYCL1", "name": "Intel(R) Arc(TM) B580 Graphics", "total_mib": 12215},
        ])

    def test_failure_returns_empty(self):
        with mock.patch.object(modelctl_vram.subprocess, "run",
                               side_effect=OSError()):
            self.assertEqual(modelctl_vram.llama_list_devices("llama-server"), [])


class TestMatchDevices(unittest.TestCase):
    XPU = [
        {"xpu_id": 0, "name": "big", "total_bytes": 34242297856, "free_bytes": 0},
        {"xpu_id": 1, "name": "small", "total_bytes": 12809404416, "free_bytes": 0},
    ]

    def test_matches_by_nearest_total(self):
        sycl = [{"device": "SYCL0", "name": "big", "total_mib": 32657},
                {"device": "SYCL1", "name": "small", "total_mib": 12215}]
        self.assertEqual(modelctl_vram.match_devices(sycl, self.XPU),
                         {"SYCL0": 0, "SYCL1": 1})

    def test_reversed_enumeration_still_matches(self):
        sycl = [{"device": "SYCL0", "name": "small", "total_mib": 12215},
                {"device": "SYCL1", "name": "big", "total_mib": 32657}]
        self.assertEqual(modelctl_vram.match_devices(sycl, self.XPU),
                         {"SYCL0": 1, "SYCL1": 0})

    def test_wildly_different_sizes_skipped(self):
        sycl = [{"device": "SYCL0", "name": "?", "total_mib": 999999}]
        self.assertEqual(modelctl_vram.match_devices(sycl, self.XPU), {})
```

Also add `import json` at the top of `test_modelctl_vram.py` if not already present.

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: `AttributeError: ... no attribute 'xpu_devices'`

- [ ] **Step 3.3: Implement**

Append to `modelctl_vram.py`:

```python
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
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: all tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add modelctl_vram.py test_modelctl_vram.py
git commit -m "Add xpu-smi and llama-server device probes with size-based matching"
```

---

### Task 4: Placement rule

**Files:**
- Modify: `modelctl_vram.py`
- Modify: `test_modelctl_vram.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `test_modelctl_vram.py`:

```python
class TestTensorSplitRatio(unittest.TestCase):
    def test_32_12_reduces_to_8_3(self):
        self.assertEqual(
            modelctl_vram.tensor_split_ratio([34242297856, 12809404416]), "8,3")

    def test_equal_cards(self):
        self.assertEqual(modelctl_vram.tensor_split_ratio([16 << 30, 16 << 30]), "1,1")


class TestRecommendPlacement(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
        {"device": "SYCL1", "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def test_fits_primary_pins_to_it(self):
        rec = modelctl_vram.recommend_placement(20 << 30, self.INVENTORY, 90, "SYCL0")
        self.assertEqual(rec, {"device": "SYCL0", "split_mode": "",
                               "tensor_split": "", "fits": True})

    def test_too_big_for_primary_splits(self):
        rec = modelctl_vram.recommend_placement(35 << 30, self.INVENTORY, 90, "SYCL0")
        self.assertEqual(rec["device"], "")
        self.assertEqual(rec["split_mode"], "layer")
        self.assertEqual(rec["tensor_split"], "8,3")
        self.assertTrue(rec["fits"])

    def test_too_big_for_everything_flagged(self):
        rec = modelctl_vram.recommend_placement(60 << 30, self.INVENTORY, 90, "SYCL0")
        self.assertEqual(rec["split_mode"], "layer")
        self.assertFalse(rec["fits"])

    def test_limit_pct_boundary(self):
        total = self.INVENTORY[0]["total_bytes"]
        just_under = int(total * 0.90) - 1
        just_over = int(total * 0.90) + 1
        self.assertEqual(modelctl_vram.recommend_placement(
            just_under, self.INVENTORY, 90, "SYCL0")["device"], "SYCL0")
        self.assertEqual(modelctl_vram.recommend_placement(
            just_over, self.INVENTORY, 90, "SYCL0")["device"], "")

    def test_unknown_primary_returns_none(self):
        self.assertIsNone(modelctl_vram.recommend_placement(
            1, self.INVENTORY, 90, "CUDA0"))
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: `AttributeError: ... no attribute 'tensor_split_ratio'`

- [ ] **Step 4.3: Implement**

Append to `modelctl_vram.py`:

```python
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
    return {"device": "", "split_mode": "layer",
            "tensor_split": tensor_split_ratio([d["total_bytes"] for d in ordered]),
            "fits": estimate_total <= combined_budget}
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl_vram -v`
Expected: all tests PASS

- [ ] **Step 4.5: Commit**

```bash
git add modelctl_vram.py test_modelctl_vram.py
git commit -m "Add static VRAM placement rule and tensor-split ratio derivation"
```

---

### Task 5: modelctl defaults (`vram_limit_pct`, `primary_gpu`) and profile-aware wrappers

**Files:**
- Modify: `modelctl.py` (imports; `load_defaults`; new helpers after `find_free_port`)
- Modify: `test_modelctl.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `test_modelctl.py`:

```python
class TestVramDefaults(unittest.TestCase):
    def test_vram_keys_present_with_defaults(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")):
            d = modelctl.load_defaults()
        self.assertEqual(d["vram_limit_pct"], 90)
        self.assertEqual(d["primary_gpu"], "")

    def test_env_override(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_VRAM_LIMIT_PCT": "80",
                                            "MODELCTL_DEFAULT_PRIMARY_GPU": "SYCL1"}):
            d = modelctl.load_defaults()
        self.assertEqual(d["vram_limit_pct"], 80)
        self.assertEqual(d["primary_gpu"], "SYCL1")


class TestLocalWeightsBytes(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_single_file(self):
        p = self.dir / "model.gguf"
        p.write_bytes(b"x" * 100)
        self.assertEqual(modelctl._local_weights_bytes(p), 100)

    def test_sharded_sums_all_parts(self):
        for i in (1, 2, 3):
            (self.dir / f"model-0000{i}-of-00003.gguf").write_bytes(b"x" * 10 * i)
        first = self.dir / "model-00001-of-00003.gguf"
        self.assertEqual(modelctl._local_weights_bytes(first), 60)


class TestEstimateVramFootprint(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "model.gguf"
        self.model.write_bytes(b"x" * 1000)

    def test_missing_model_returns_none(self):
        profile = {"model_path": str(Path(self.tmp.name) / "gone.gguf"),
                   "config": {"ctx": 4096, "kv_quant": "q8_0"}}
        self.assertIsNone(modelctl.estimate_vram_footprint(profile))

    def test_heuristic_when_header_unparseable(self):
        profile = {"model_path": str(self.model),
                   "config": {"ctx": 4096, "kv_quant": "q8_0"}}
        est = modelctl.estimate_vram_footprint(profile)
        self.assertEqual(est["quality"], "heuristic")
        self.assertEqual(est["weights"], 1000)

    def test_mmproj_counted(self):
        mmproj = Path(self.tmp.name) / "mmproj.gguf"
        mmproj.write_bytes(b"y" * 500)
        profile = {"model_path": str(self.model), "mmproj_path": str(mmproj),
                   "config": {"ctx": 4096, "kv_quant": "q8_0"}}
        est = modelctl.estimate_vram_footprint(profile)
        self.assertEqual(est["weights"], 1500)


class TestGetGpuInventory(unittest.TestCase):
    XPU = [
        {"xpu_id": 0, "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
        {"xpu_id": 1, "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.map_path = Path(self.tmp.name) / "gpu_map.json"

    def test_no_xpu_smi_returns_empty(self):
        with mock.patch.object(modelctl.modelctl_vram, "xpu_devices", return_value=[]):
            self.assertEqual(modelctl.get_gpu_inventory(), [])

    def test_builds_and_caches_map(self):
        sycl = [{"device": "SYCL0", "name": "big", "total_mib": 32657},
                {"device": "SYCL1", "name": "small", "total_mib": 12215}]
        with mock.patch.object(modelctl, "GPU_MAP_PATH", self.map_path), \
             mock.patch.object(modelctl.modelctl_vram, "xpu_devices",
                               return_value=self.XPU), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               return_value=sycl) as mock_list:
            inv = modelctl.get_gpu_inventory()
            inv2 = modelctl.get_gpu_inventory()  # second call: cached map
        self.assertEqual(mock_list.call_count, 1)
        self.assertEqual(inv[0]["device"], "SYCL0")  # sorted, biggest first
        self.assertEqual(inv[0]["free_bytes"], 30 << 30)
        self.assertEqual(inv, inv2)
        self.assertTrue(self.map_path.exists())

    def test_fallback_map_when_list_devices_fails(self):
        with mock.patch.object(modelctl, "GPU_MAP_PATH", self.map_path), \
             mock.patch.object(modelctl.modelctl_vram, "xpu_devices",
                               return_value=self.XPU), \
             mock.patch.object(modelctl.modelctl_vram, "llama_list_devices",
                               return_value=[]):
            inv = modelctl.get_gpu_inventory()
        self.assertEqual([d["device"] for d in inv], ["SYCL0", "SYCL1"])
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl -v 2>&1 | tail -5`
Expected: failures/errors for the new test classes (missing attributes)

- [ ] **Step 5.3: Implement**

In `modelctl.py`:

(a) Add the import after the `huggingface_hub` import:

```python
import modelctl_vram
```

(b) Add near the other module constants (after `DEFAULTS_PATH`):

```python
# Cached SYCL-name -> xpu-smi-id mapping, built by matching total VRAM
# between `llama-server --list-devices` and `xpu-smi discovery`. Rebuild
# with `modelctl place --remap` after hardware changes.
GPU_MAP_PATH = STATE_DIR / "gpu_map.json"
DEFAULT_VRAM_LIMIT_PCT = int(os.environ.get("MODELCTL_DEFAULT_VRAM_LIMIT_PCT", "90"))
DEFAULT_PRIMARY_GPU = os.environ.get("MODELCTL_DEFAULT_PRIMARY_GPU", "")
```

(c) In `load_defaults()`, add to the returned dict:

```python
        "vram_limit_pct": int(pick("vram_limit_pct", DEFAULT_VRAM_LIMIT_PCT)),
        "primary_gpu": pick("primary_gpu", DEFAULT_PRIMARY_GPU),
```

(d) Add after `find_free_port()`:

```python
def _local_weights_bytes(model_path: Path) -> int:
    """Total on-disk size of a model: the file itself, or the sum of all
    sibling shards when it's the first part of a -NNNNN-of-MMMMM split
    (profiles only store the first shard's path)."""
    m = SHARD_RE.match(model_path.name)
    if not m:
        return model_path.stat().st_size
    prefix = m.group(1)
    return sum(p.stat().st_size for p in model_path.parent.glob(f"{prefix}-*-of-*.gguf")
               if SHARD_RE.match(p.name))


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
        except json.JSONDecodeError:
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
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK

- [ ] **Step 5.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Add VRAM defaults, footprint wrapper, and cached GPU inventory"
```

---

### Task 6: `modelctl place` command

**Files:**
- Modify: `modelctl.py` (new `cmd_place` after `cmd_verify`; parser entry in `build_arg_parser`)
- Modify: `test_modelctl.py`

- [ ] **Step 6.1: Write the failing tests**

Append to `test_modelctl.py`:

```python
class TestCmdPlace(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
        {"device": "SYCL1", "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        model = Path(self.tmp.name) / "model.gguf"
        model.write_bytes(b"x" * 1000)
        (self.profiles_dir / "small-model.json").write_text(json.dumps({
            "name": "small-model", "repo_id": "r/a", "file": "f",
            "model_path": str(model),
            "config": {"ctx": 4096, "kv_quant": "q8_0", "flash_attn": "auto",
                       "split_mode": "layer", "tensor_split": "3,1",
                       "ttl": 3600, "mtp": "off", "extra": ""},
            "env": [], "enabled": True,
        }))

    def _args(self, **kw):
        defaults = {"name": None, "apply": False, "remap": False,
                    "no_hermes": True, "no_router_restart": True}
        defaults.update(kw)
        return mock.Mock(**defaults)

    def test_report_only_does_not_touch_profile(self):
        before = (self.profiles_dir / "small-model.json").read_text()
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY):
            modelctl.cmd_place(self._args())
        self.assertEqual((self.profiles_dir / "small-model.json").read_text(), before)

    def test_apply_rewrites_placement_and_syncs(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "generate_artifacts") as mock_gen, \
             mock.patch.object(modelctl, "sync_all_backends") as mock_sync:
            modelctl.cmd_place(self._args(apply=True))
        saved = json.loads((self.profiles_dir / "small-model.json").read_text())
        # tiny model fits the primary card alone -> pinned, split cleared
        self.assertEqual(saved["config"]["device"], "SYCL0")
        self.assertEqual(saved["config"]["split_mode"], "")
        self.assertEqual(saved["config"]["tensor_split"], "")
        mock_gen.assert_called_once()
        mock_sync.assert_called_once()

    def test_no_gpu_inventory_exits_nonzero(self):
        with mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir), \
             mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]):
            with self.assertRaises(SystemExit):
                modelctl.cmd_place(self._args())
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl.TestCmdPlace -v`
Expected: `AttributeError: ... no attribute 'cmd_place'`

- [ ] **Step 6.3: Implement `cmd_place`**

Add to `modelctl.py` after `cmd_verify`:

```python
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
                print(f"  -> applied")

    if changed:
        sync_all_backends(restart_router=not args.no_router_restart)
        if not args.no_hermes:
            sync_hermes_custom_providers()
    elif args.apply:
        print("\nNothing to change -- all placements already match.")
    else:
        print("\nRe-run with --apply to rewrite placements to the recommendations.")
```

- [ ] **Step 6.4: Add the parser entry**

In `build_arg_parser()`, after the `p_verify` block:

```python
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
```

- [ ] **Step 6.5: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK

- [ ] **Step 6.6: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Add 'modelctl place' command for VRAM-fit GPU placement"
```

---

### Task 7: Placement hint in `pull`

**Files:**
- Modify: `modelctl.py` (`cmd_pull` before the `prompt_config` call; `prompt_config` signature)
- Modify: `test_modelctl.py`

- [ ] **Step 7.1: Write the failing test**

Append to `test_modelctl.py`:

```python
class TestPullPlacementHint(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 30 << 30},
    ]

    def test_hint_computed_from_remote_size(self):
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY):
            hint = modelctl.compute_pull_placement_hint(18 << 30)
        self.assertEqual(hint["device"], "SYCL0")

    def test_no_inventory_returns_none(self):
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]):
            self.assertIsNone(modelctl.compute_pull_placement_hint(18 << 30))

    def test_no_size_returns_none(self):
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY):
            self.assertIsNone(modelctl.compute_pull_placement_hint(None))
```

- [ ] **Step 7.2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl.TestPullPlacementHint -v`
Expected: `AttributeError: ... no attribute 'compute_pull_placement_hint'`

- [ ] **Step 7.3: Implement**

(a) Add to `modelctl.py` (after `resolve_primary_gpu`):

```python
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
```

(b) In `prompt_config`, extend the signature and the defaults overlay:

```python
def prompt_config(repo_id: str = "", label: str = "", mtp_file_chosen: bool = False,
                  current: dict = None, placement: dict = None):
```

and directly after the existing `d = {**load_defaults(), "extra": "", **(current or {})}` line add:

```python
    if placement and not current:
        # Seed device/split defaults from the VRAM-fit recommendation for
        # new profiles; edits keep the profile's own values instead.
        d = {**d, "device": placement["device"],
             "split_mode": placement["split_mode"] or d["split_mode"],
             "tensor_split": placement["tensor_split"] or d["tensor_split"]}
```

(c) In `cmd_pull`, change the `shared_config = prompt_config(...)` call to:

```python
    placement_hint = compute_pull_placement_hint(chosen_groups[0].get("total_size"))
    shared_config = prompt_config(repo_id, chosen_groups[0]["label"] if chosen_groups else "",
                                   mtp_file_chosen=bool(local_mtp_path),
                                   placement=placement_hint)
```

- [ ] **Step 7.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK

- [ ] **Step 7.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Seed pull config prompts with VRAM-fit placement hint"
```

---

### Task 8: Preset `version = 1` header and `[*]` metrics section

**Files:**
- Modify: `modelctl.py` (`sync_router_preset`)
- Modify: `test_modelctl.py` (existing `sync_router_preset` tests around lines 555-610)

- [ ] **Step 8.1: Write the failing test**

Append to `test_modelctl.py` (inside or next to the existing sync_router_preset test class, reusing its setup pattern — adapt names to the existing class):

```python
    def test_preset_starts_with_version_and_global_metrics_section(self):
        # Reuse this class's existing profile/patching setup verbatim.
        with self._patched_env():  # or the existing with-block pattern
            modelctl.sync_router_preset(restart=False)
        text = self.preset_path.read_text()
        self.assertTrue(text.startswith("version = 1\n"))
        self.assertIn("[*]\nmetrics = true\n", text)
```

Note for the implementer: the existing tests in `test_modelctl.py` around lines 555-610 construct a profile, patch `PROFILES_DIR`, `ROUTER_PRESET_PATH`, `preflight`, and `restart_router_service`, then call `modelctl.sync_router_preset()`. Copy that exact with-block instead of `self._patched_env()` above, and use the same preset-path variable those tests use. Any existing test that asserts the preset file's full content or first line must be updated to expect the new header.

- [ ] **Step 8.2: Run tests to verify the new one fails**

Run: `python3 -m unittest test_modelctl 2>&1 | tail -3`
Expected: the new test FAILS (no `version = 1` header yet)

- [ ] **Step 8.3: Implement**

In `modelctl.py`, add above `sync_router_preset`:

```python
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
```

and in `sync_router_preset`, change `body = ""` to:

```python
    body = ROUTER_PRESET_HEADER
```

- [ ] **Step 8.4: Run full suite, fix any preset-content assertions**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK (after updating any test that asserted the old headerless content)

- [ ] **Step 8.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Emit version header and [*] metrics section in router preset"
```

---

### Task 9: VRAM guard + load timing in `router load`

**Files:**
- Modify: `modelctl.py` (`cmd_router_load`, new helpers; parser flags)
- Modify: `test_modelctl.py`

- [ ] **Step 9.1: Write the failing tests**

Append to `test_modelctl.py`:

```python
class TestCheckVramForLoad(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "big", "total_bytes": 34242297856,
         "free_bytes": 10 << 30},
        {"device": "SYCL1", "name": "small", "total_bytes": 12809404416,
         "free_bytes": 12 << 30},
    ]

    def _profile(self, device="SYCL0"):
        return {"name": "m", "model_path": "/x/model.gguf",
                "config": {"ctx": 4096, "kv_quant": "q8_0", "device": device,
                           "split_mode": "", "tensor_split": ""}}

    def test_no_inventory_degrades_open(self):
        with mock.patch.object(modelctl, "get_gpu_inventory", return_value=[]):
            ok, msgs = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)
        self.assertTrue(any("skipping VRAM check" in m for m in msgs))

    def test_no_estimate_degrades_open(self):
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=None):
            ok, _ = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)

    def test_fits_passes(self):
        est = {"total": 5 << 30, "quality": "exact"}
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est):
            ok, _ = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)

    def test_exact_over_free_blocks(self):
        est = {"total": 20 << 30, "quality": "exact"}
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est), \
             mock.patch.object(modelctl, "router_status", return_value=[]):
            ok, msgs = modelctl.check_vram_for_load(self._profile())
        self.assertFalse(ok)
        self.assertTrue(any("--evict" in m for m in msgs))

    def test_heuristic_over_free_warns_but_passes(self):
        est = {"total": 20 << 30, "quality": "heuristic"}
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est):
            ok, msgs = modelctl.check_vram_for_load(self._profile())
        self.assertTrue(ok)
        self.assertTrue(any("heuristic" in m for m in msgs))

    def test_split_profile_targets_all_gpus(self):
        est = {"total": 20 << 30, "quality": "exact"}
        profile = self._profile()
        profile["config"]["split_mode"] = "layer"
        profile["config"]["tensor_split"] = "8,3"
        # 10 + 12 GiB free combined > 20 GiB estimate -> fits
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=self.INVENTORY), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               return_value=est):
            ok, _ = modelctl.check_vram_for_load(profile)
        self.assertTrue(ok)

    def test_evict_unloads_largest_first(self):
        est = {"total": 20 << 30, "quality": "exact"}
        rows = [
            {"name": "small-loaded", "status": "loaded", "failed": False,
             "exit_code": None, "gpu": "SYCL0", "from_preset": True},
            {"name": "big-loaded", "status": "loaded", "failed": False,
             "exit_code": None, "gpu": "SYCL0", "from_preset": True},
        ]
        estimates = {"m": est,
                     "small-loaded": {"total": 2 << 30, "quality": "exact"},
                     "big-loaded": {"total": 18 << 30, "quality": "exact"}}
        # After evicting big-loaded, free jumps enough to fit.
        inventories = [self.INVENTORY,
                       [dict(self.INVENTORY[0], free_bytes=28 << 30),
                        self.INVENTORY[1]]]

        def fake_est(profile):
            return estimates[profile["name"]]

        def fake_load_profile(name):
            return {"name": name, "model_path": "/x", "config": {"device": "SYCL0"}}

        with mock.patch.object(modelctl, "get_gpu_inventory",
                               side_effect=inventories), \
             mock.patch.object(modelctl, "estimate_vram_footprint",
                               side_effect=fake_est), \
             mock.patch.object(modelctl, "load_profile",
                               side_effect=fake_load_profile), \
             mock.patch.object(modelctl, "router_status", return_value=rows), \
             mock.patch.object(modelctl, "router_unload",
                               return_value=(True, "ok")) as mock_unload, \
             mock.patch.object(modelctl, "_wait_for_router_model",
                               return_value="unloaded"):
            ok, _ = modelctl.check_vram_for_load(self._profile(), evict=True)
        self.assertTrue(ok)
        mock_unload.assert_called_once_with("big-loaded")
```

- [ ] **Step 9.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl.TestCheckVramForLoad -v`
Expected: `AttributeError: ... no attribute 'check_vram_for_load'`

- [ ] **Step 9.3: Implement the guard helpers**

Add to `modelctl.py` after `router_unload`:

```python
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
```

- [ ] **Step 9.4: Wire into `cmd_router_load` with timing**

Replace `cmd_router_load` in `modelctl.py`:

```python
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
```

And in `build_arg_parser()`, extend the `p_router_load` block:

```python
    p_router_load = router_sub.add_parser("load", help="load a model now instead of waiting for a request")
    p_router_load.add_argument("name")
    p_router_load.add_argument("--evict", action="store_true",
                                help="unload loaded models (largest first) until this one fits")
    p_router_load.add_argument("--force", action="store_true",
                                help="skip the VRAM fit check entirely")
    p_router_load.set_defaults(func=cmd_router_load)
```

- [ ] **Step 9.5: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK

- [ ] **Step 9.6: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Add VRAM guard with --evict/--force and load timing to router load"
```

---

### Task 10: `router status` VRAM footer, estimates, and failed-hint

**Files:**
- Modify: `modelctl.py` (`cmd_router_status`)
- Modify: `test_modelctl.py`

- [ ] **Step 10.1: Write the failing test**

Append to `test_modelctl.py`:

```python
class TestVramFooter(unittest.TestCase):
    def test_footer_lines(self):
        inventory = [{"device": "SYCL0", "name": "big",
                      "total_bytes": 32 << 30, "free_bytes": 10 << 30}]
        lines = modelctl.vram_footer_lines(inventory)
        self.assertEqual(len(lines), 1)
        self.assertIn("SYCL0", lines[0])
        self.assertIn("22.0GB", lines[0])   # used = total - free
        self.assertIn("32.0GB", lines[0])

    def test_empty_inventory_no_lines(self):
        self.assertEqual(modelctl.vram_footer_lines([]), [])
```

- [ ] **Step 10.2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl.TestVramFooter -v`
Expected: `AttributeError: ... no attribute 'vram_footer_lines'`

- [ ] **Step 10.3: Implement**

Add to `modelctl.py` before `cmd_router_status`:

```python
def vram_footer_lines(inventory) -> list:
    """Human-readable per-GPU 'used/total' lines for status/stats output.
    Empty list when the inventory is unavailable, so callers just skip it."""
    lines = []
    for d in inventory:
        used = d["total_bytes"] - d["free_bytes"]
        lines.append(f"{d['device']}: {_format_size(used)} / "
                     f"{_format_size(d['total_bytes'])} VRAM used ({d['name']})")
    return lines
```

Replace `cmd_router_status` with:

```python
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
                notes = f"FAILED (exit {r['exit_code']})"
                any_failed = True
            elif not r["from_preset"]:
                notes = "(not a modelctl profile)"
            print(f"{r['name']:<32} {r['status']:<10} {r['gpu']:<22} {est_col:<10} {notes}")
        if any_failed:
            print("\nFor failed models: journalctl --user -u "
                  f"{ROUTER_SERVICE_NAME} -n 100")

    footer = vram_footer_lines(get_gpu_inventory())
    if footer:
        print()
        for line in footer:
            print(line)
```

- [ ] **Step 10.4: Run tests to verify they pass**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK

- [ ] **Step 10.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Add VRAM footer, footprint column, and failed-hint to router status"
```

---

### Task 11: `modelctl router stats`

**Files:**
- Modify: `modelctl.py` (metrics fetch/parse, `cmd_router_stats`, parser)
- Modify: `test_modelctl.py`

- [ ] **Step 11.1: Write the failing tests**

Append to `test_modelctl.py`:

```python
PROM_SAMPLE = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 12000
llamacpp:prompt_seconds_total 10
llamacpp:tokens_predicted_total 4500
llamacpp:tokens_predicted_seconds_total 300
llamacpp:n_decode_total 4600
llamacpp:prompt_tokens_seconds 1180.5
llamacpp:predicted_tokens_seconds 14.9
llamacpp:requests_processing 1
llamacpp:requests_deferred 0
"""


class TestParsePrometheus(unittest.TestCase):
    def test_parses_values_skips_comments(self):
        m = modelctl.parse_prometheus_text(PROM_SAMPLE)
        self.assertEqual(m["llamacpp:prompt_tokens_total"], 12000.0)
        self.assertEqual(m["llamacpp:predicted_tokens_seconds"], 14.9)
        self.assertNotIn("# HELP llamacpp:prompt_tokens_total", m)

    def test_garbage_lines_ignored(self):
        m = modelctl.parse_prometheus_text("weird\nllamacpp:x notanumber\n")
        self.assertEqual(m, {})


class TestStatsRow(unittest.TestCase):
    def test_computed_columns(self):
        metrics = modelctl.parse_prometheus_text(PROM_SAMPLE)
        row = modelctl.stats_row_from_metrics(metrics)
        self.assertEqual(row["gen_tps"], "15.0")       # 4500/300
        self.assertEqual(row["prompt_tps"], "1200.0")  # 12000/10
        self.assertEqual(row["last_gen_tps"], "14.9")  # gauge
        self.assertEqual(row["requests"], "1/0")       # processing/deferred

    def test_missing_metrics_render_question_marks(self):
        row = modelctl.stats_row_from_metrics({})
        self.assertEqual(row["gen_tps"], "?")
        self.assertEqual(row["requests"], "?")
```

- [ ] **Step 11.2: Run tests to verify they fail**

Run: `python3 -m unittest test_modelctl.TestParsePrometheus test_modelctl.TestStatsRow -v`
Expected: `AttributeError: ... no attribute 'parse_prometheus_text'`

- [ ] **Step 11.3: Implement**

Add to `modelctl.py` after `cmd_router_status` (also add `import urllib.parse` next to the existing urllib imports):

```python
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
    """GET the router's forwarded per-instance /metrics for one loaded
    model. Returns a metrics dict, or None on any failure (the row renders
    as '?' -- one broken instance shouldn't kill the whole table)."""
    url = (router_root_url().rstrip("/") + "/metrics?model="
           + urllib.parse.quote(name, safe=""))
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
```

And in `build_arg_parser()`, after the `p_router_status` block:

```python
    p_router_stats = router_sub.add_parser("stats", help="per-model throughput and VRAM stats")
    p_router_stats.set_defaults(func=cmd_router_stats)
```

- [ ] **Step 11.4: Run full suite**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: OK

- [ ] **Step 11.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Add 'modelctl router stats' reading per-model Prometheus metrics"
```

---

### Task 12: Manual verification against the live system

**Files:** none (verification only)

- [ ] **Step 12.1: Device mapping and inventory**

Run: `python3 -c "import modelctl; print(modelctl.get_gpu_inventory(force_remap=True))"`
Expected: two entries, SYCL device names, totals ≈ 34242297856 and 12809404416, plausible free bytes. If the `--list-devices` line format differs from `_DEVICE_LINE_RE` (check with `~/workspace/llama.cpp/build-sycl/bin/llama-server --list-devices` after sourcing `/opt/intel/oneapi/setvars.sh`), fix the regex in `modelctl_vram.py` and add the real line format to `TestLlamaListDevices`.

- [ ] **Step 12.2: Placement report**

Run: `python3 modelctl.py place`
Expected: every saved profile listed with an `exact`-quality estimate (they're all downloaded GGUFs) and a sane recommendation. Spot-check one estimate against the GGUF file size + expected KV cache.

- [ ] **Step 12.3: Preset + metrics**

Run: `python3 modelctl.py sync --no-hermes` then `head -8 ~/llama-router/router.preset.ini`
Expected: `version = 1`, `[*]`, `metrics = true` header; router restarts cleanly (`systemctl --user status llama-router.service`).

- [ ] **Step 12.4: Stats and status live**

Run: `python3 modelctl.py router load <small-profile>` then `python3 modelctl.py router stats` and `python3 modelctl.py router status`
Expected: load prints elapsed seconds; stats shows a row with numbers after sending one request through Hermes or curl; status shows EST column and the VRAM footer. If any metric renders `?`, curl `http://127.0.0.1:7071/metrics?model=<name>` and reconcile names in `stats_row_from_metrics`.

- [ ] **Step 12.5: Guard behavior**

With a large model loaded, run `python3 modelctl.py router load <other-large-profile>` (no flags).
Expected: guard blocks with the shortfall message. Re-run with `--evict`: the loaded large model unloads, the new one loads. `--force` skips the check.

- [ ] **Step 12.6: Final commit (any verification fixes)**

```bash
git add -A modelctl.py modelctl_vram.py test_modelctl.py test_modelctl_vram.py
git commit -m "Adjust VRAM probes/metrics to live system behavior"
```

---

## Self-Review Notes

- **Spec coverage:** Component A → Tasks 1-7; Component B → Task 9; Component C → Tasks 8, 10, 11; error-handling table → degrade-open paths in Tasks 5, 9, 10, 11; testing section → per-task tests + Task 12 manual pass. Non-goals (watcher, unit management, models-max) intentionally absent.
- **Known deferred detail:** the `--list-devices` output format is pattern-matched from typical ggml backend output; Task 12.1 verifies it against the real binary and fixes the regex if needed — this is the one place live output can differ from the fixture.
- **Type consistency check:** inventory dicts are `{device, name, total_bytes, free_bytes}` everywhere (`modelctl_vram.xpu_devices` returns `xpu_id`-keyed precursors used only inside `get_gpu_inventory`); estimates are `{weights, kv_bytes, overhead, total, quality}` everywhere; placement recs are `{device, split_mode, tensor_split, fits}` everywhere.
