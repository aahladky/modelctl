# Split K/V Cache Quantization + Standalone Calculator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independent `--cache-type-k`/`--cache-type-v` per profile (CLI + TUI + estimator), plus a detached VRAM calculator CLI in `modelctl_vram.py`, per `docs/superpowers/specs/2026-07-01-split-kv-cache-quant-design.md` (including its 2026-07-02 addendum).

**Architecture:** `_resolve_cache_types(cfg)` in modelctl.py bridges legacy `kv_quant`-only profiles; new profiles/defaults carry `cache_type_k`/`cache_type_v`. `modelctl_vram.kv_cache_bytes` computes K and V sums separately (optional `cache_type_v=None` = same as K, so all existing callers stay valid). `modelctl_vram.py` gains `weights_bytes_on_disk` (shard-aware, modelctl-free) and an argparse `__main__`.

**Tech Stack:** Python 3 stdlib; unittest (+ existing textual test harness in test_modelctl_tui.py). Runner: `python3 -m unittest`. Baselines: test_modelctl (110), test_modelctl_vram (40), test_modelctl_tui (run `python3 -m unittest test_modelctl_tui 2>&1 | tail -2` for its count first).

---

## File Structure

- **Modify** `modelctl_vram.py` — split-type `kv_cache_bytes`/`estimate_from_parts`; `weights_bytes_on_disk`; `main(argv)` + `__main__`.
- **Modify** `modelctl.py` — `_resolve_cache_types`; emitters; defaults constants/`load_defaults`/`prompt_defaults`/`prompt_config`; `estimate_vram_footprint`; `compute_pull_placement_hint`; `_local_weights_bytes` delegates.
- **Modify** `modelctl_tui.py` — ConfigureScreen dual inputs.
- **Modify** `test_modelctl_vram.py`, `test_modelctl.py`, `test_modelctl_tui.py`.

---

### Task 1: Split-type KV math in modelctl_vram

**Files:** Modify `modelctl_vram.py`, `test_modelctl_vram.py`

- [ ] **Step 1.1: Write the failing tests** — append to `test_modelctl_vram.py`:

```python
class TestSplitCacheTypes(unittest.TestCase):
    UNIFORM = {"block_count": 2, "n_kv_heads": 4, "k_dim": 128, "v_dim": 64}

    def test_v_type_defaults_to_k_type(self):
        both = modelctl_vram.kv_cache_bytes(self.UNIFORM, 100, "q8_0")
        explicit = modelctl_vram.kv_cache_bytes(self.UNIFORM, 100, "q8_0", "q8_0")
        self.assertEqual(both, explicit)

    def test_split_types_uniform_path(self):
        # K: 2*100*4*128 elems @ 1.0625 ; V: 2*100*4*64 elems @ 0.5625
        expected = int(2 * 100 * 4 * 128 * (34 / 32)) + int(2 * 100 * 4 * 64 * (18 / 32))
        self.assertEqual(
            modelctl_vram.kv_cache_bytes(self.UNIFORM, 100, "q8_0", "q4_0"),
            expected)

    def test_split_types_swa_path(self):
        params = {"block_count": 6, "n_kv_heads": 14.0, "k_dim": 512, "v_dim": 512,
                  "kv_heads_per_layer": [16, 16, 16, 16, 16, 4],
                  "swa_window": 1024,
                  "swa_pattern": [True, True, True, True, True, False],
                  "k_dim_swa": 256, "v_dim_swa": 256}
        ctx = 64000
        k_bytes = int((5 * 1024 * 16 * 256 + 1 * 64000 * 4 * 512) * (34 / 32))
        v_bytes = int((5 * 1024 * 16 * 256 + 1 * 64000 * 4 * 512) * (18 / 32))
        self.assertEqual(
            modelctl_vram.kv_cache_bytes(params, ctx, "q8_0", "q4_0"),
            k_bytes + v_bytes)

    def test_estimate_from_parts_passes_v_type(self):
        params = dict(self.UNIFORM)
        est_split = modelctl_vram.estimate_from_parts(
            1000, 100, "q8_0", gguf_params=params, cache_type_v="q4_0")
        est_same = modelctl_vram.estimate_from_parts(
            1000, 100, "q8_0", gguf_params=params)
        self.assertLess(est_split["kv_bytes"], est_same["kv_bytes"])
```

- [ ] **Step 1.2: Run to verify failure**

Run: `python3 -m unittest test_modelctl_vram.TestSplitCacheTypes -v`
Expected: TypeError/AssertionError (extra positional arg not accepted).

- [ ] **Step 1.3: Implement** — replace `kv_cache_bytes` in `modelctl_vram.py` with:

```python
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
```

And extend `estimate_from_parts`: signature becomes
`def estimate_from_parts(weights_bytes, ctx, kv_quant, gguf_params=None, mmproj_bytes=0, cache_type_v=None):`
and its exact-path call becomes `kv = kv_cache_bytes(gguf_params, ctx, kv_quant, cache_type_v)`. (The heuristic path is unchanged — it doesn't model K/V separately.)

NOTE on rounding: the previous uniform formula did ONE int() over the combined sum; the new one rounds K and V separately, so results may differ by 1 byte from the old function. Existing tests compute expectations via the function itself or exact multiples where both roundings agree — run the full vram suite; if an existing test fails by an off-by-one, fix the EXPECTATION to the split-rounding form (mirroring test_split_types_uniform_path), and say so in your report.

- [ ] **Step 1.4: Run full vram suite**

Run: `python3 -m unittest test_modelctl_vram 2>&1 | tail -3`
Expected: 44 tests OK.

- [ ] **Step 1.5: Commit**

```bash
git add modelctl_vram.py test_modelctl_vram.py
git commit -m "Support independent K/V cache types in KV estimation"
```

---

### Task 2: weights_bytes_on_disk + calculator CLI in modelctl_vram

**Files:** Modify `modelctl_vram.py`, `test_modelctl_vram.py`, `modelctl.py`, `test_modelctl.py`

- [ ] **Step 2.1: Write the failing tests** — append to `test_modelctl_vram.py`:

```python
class TestWeightsBytesOnDisk(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_single_file(self):
        p = self.dir / "model.gguf"
        p.write_bytes(b"x" * 100)
        self.assertEqual(modelctl_vram.weights_bytes_on_disk(p), 100)

    def test_sharded_sums_exact_prefix_only(self):
        for i in (1, 2):
            (self.dir / f"model-0000{i}-of-00002.gguf").write_bytes(b"x" * 10)
            (self.dir / f"model-instruct-0000{i}-of-00002.gguf").write_bytes(b"y" * 100)
        first = self.dir / "model-00001-of-00002.gguf"
        self.assertEqual(modelctl_vram.weights_bytes_on_disk(first), 20)


class TestCalculatorMain(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"
        self.path.write_bytes(gguf_bytes({
            "general.architecture": (8, "qwen3"),
            "qwen3.block_count": (4, 48),
            "qwen3.embedding_length": (4, 5120),
            "qwen3.attention.head_count": (4, 40),
            "qwen3.attention.head_count_kv": (4, 8),
        }))

    def _run(self, argv):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = modelctl_vram.main(argv)
        return code, buf.getvalue()

    def test_breakdown_output(self):
        code, out = self._run([str(self.path), "--ctx", "32768",
                               "--cache-type-k", "q8_0", "--cache-type-v", "q4_0"])
        self.assertEqual(code, 0)
        self.assertIn("qwen3", out)
        self.assertIn("weights", out.lower())
        self.assertIn("total", out.lower())
        self.assertIn("K q8_0", out)
        self.assertIn("V q4_0", out)

    def test_unparseable_gguf_exits_1(self):
        bad = Path(self.tmp.name) / "bad.gguf"
        bad.write_bytes(b"NOPE1234")
        code, out = self._run([str(bad)])
        self.assertEqual(code, 1)
```

- [ ] **Step 2.2: Run to verify failure**

Run: `python3 -m unittest test_modelctl_vram.TestWeightsBytesOnDisk test_modelctl_vram.TestCalculatorMain -v`
Expected: AttributeError (no `weights_bytes_on_disk` / `main`).

- [ ] **Step 2.3: Implement** — append to `modelctl_vram.py`:

```python
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
    mmproj_bytes = Path(args.mmproj).stat().st_size if args.mmproj else 0
    ctk = args.cache_type_k
    ctv = args.cache_type_v or ctk
    k_only = kv_cache_bytes(params, args.ctx, ctk, ctk)
    kv = kv_cache_bytes(params, args.ctx, ctk, ctv)
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
    print(f"kv cache: {_fmt_gib(kv)}  (K {ctk} + V {ctv}; both-{ctk} would be {_fmt_gib(k_only)})")
    print(f"overhead: {_fmt_gib(est['overhead'])}")
    print(f"total:    {_fmt_gib(est['total'])}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
```

NOTE: the two `from pathlib import Path` function-local imports keep the module's top-level imports unchanged (json/math/re/struct/subprocess); that's deliberate — pathlib is only needed by these two entry points.

- [ ] **Step 2.4: Delegate from modelctl** — in `modelctl.py`, replace the body of `_local_weights_bytes` with a delegation (keep the function so existing callers/tests are untouched):

```python
def _local_weights_bytes(model_path: Path) -> int:
    """Total on-disk size of a model (first shard -> sum of all shards).
    Thin alias for modelctl_vram.weights_bytes_on_disk, which owns the
    logic so the detached calculator can use it too."""
    return modelctl_vram.weights_bytes_on_disk(model_path)
```

- [ ] **Step 2.5: Run all suites**

Run: `python3 -m unittest test_modelctl_vram test_modelctl 2>&1 | tail -3`
Expected: 48 + 110 = 158 OK (TestLocalWeightsBytes in test_modelctl must still pass via the delegation).

- [ ] **Step 2.6: Commit**

```bash
git add modelctl_vram.py test_modelctl_vram.py modelctl.py
git commit -m "Add detached VRAM calculator CLI and shared shard-aware sizing"
```

---

### Task 3: _resolve_cache_types + emitters in modelctl

**Files:** Modify `modelctl.py`, `test_modelctl.py`

- [ ] **Step 3.1: Write the failing tests** — append to `test_modelctl.py`:

```python
class TestResolveCacheTypes(unittest.TestCase):
    def test_both_new_fields(self):
        self.assertEqual(
            modelctl._resolve_cache_types({"cache_type_k": "q8_0", "cache_type_v": "q4_0"}),
            ("q8_0", "q4_0"))

    def test_legacy_only(self):
        self.assertEqual(modelctl._resolve_cache_types({"kv_quant": "q8_0"}),
                         ("q8_0", "q8_0"))

    def test_one_new_field_no_legacy(self):
        self.assertEqual(modelctl._resolve_cache_types({"cache_type_k": "q8_0"}),
                         ("q8_0", None))

    def test_neither(self):
        self.assertEqual(modelctl._resolve_cache_types({}), (None, None))


class TestSplitCacheTypeEmission(unittest.TestCase):
    def _profile(self, cfg_extra):
        cfg = {"flash_attn": "auto", "ctx": 4096, "split_mode": "",
               "tensor_split": "", "ttl": 3600, "mtp": "off", "extra": ""}
        cfg.update(cfg_extra)
        return {"name": "m", "model_path": "/x/m.gguf", "mmproj_path": None,
                "config": cfg}

    def test_build_server_args_distinct_types(self):
        args = modelctl.build_server_args(self._profile(
            {"cache_type_k": "q8_0", "cache_type_v": "q4_0"}))
        self.assertIn("q8_0", args[args.index("--cache-type-k") + 1])
        self.assertIn("q4_0", args[args.index("--cache-type-v") + 1])

    def test_build_server_args_legacy_kv_quant(self):
        args = modelctl.build_server_args(self._profile({"kv_quant": "q8_0"}))
        self.assertEqual(args[args.index("--cache-type-k") + 1], "q8_0")
        self.assertEqual(args[args.index("--cache-type-v") + 1], "q8_0")

    def test_build_server_args_k_only(self):
        args = modelctl.build_server_args(self._profile({"cache_type_k": "q8_0"}))
        self.assertIn("--cache-type-k", args)
        self.assertNotIn("--cache-type-v", args)

    def test_router_preset_distinct_types(self):
        with mock.patch.object(modelctl, "preflight",
                               return_value=(True, "llama-server", {}, [])):
            text, _, _ = modelctl.render_router_preset(self._profile(
                {"cache_type_k": "q8_0", "cache_type_v": "q4_0"}))
        self.assertIn("cache-type-k = q8_0", text)
        self.assertIn("cache-type-v = q4_0", text)

    def test_router_preset_legacy(self):
        with mock.patch.object(modelctl, "preflight",
                               return_value=(True, "llama-server", {}, [])):
            text, _, _ = modelctl.render_router_preset(self._profile(
                {"kv_quant": "q5_1"}))
        self.assertIn("cache-type-k = q5_1", text)
        self.assertIn("cache-type-v = q5_1", text)

    def test_router_preset_k_only(self):
        with mock.patch.object(modelctl, "preflight",
                               return_value=(True, "llama-server", {}, [])):
            text, _, _ = modelctl.render_router_preset(self._profile(
                {"cache_type_k": "q8_0"}))
        self.assertIn("cache-type-k = q8_0", text)
        self.assertNotIn("cache-type-v", text)
```

- [ ] **Step 3.2: Run to verify failure**

Run: `python3 -m unittest test_modelctl.TestResolveCacheTypes test_modelctl.TestSplitCacheTypeEmission -v`
Expected: AttributeError (no `_resolve_cache_types`).

- [ ] **Step 3.3: Implement.** Add near `build_server_args`:

```python
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
```

In `build_server_args`, replace:
```python
    if cfg.get("kv_quant"):
        args.extend(["--cache-type-k", cfg['kv_quant'], "--cache-type-v", cfg['kv_quant']])
```
with:
```python
    ctk, ctv = _resolve_cache_types(cfg)
    if ctk:
        args.extend(["--cache-type-k", ctk])
    if ctv:
        args.extend(["--cache-type-v", ctv])
```

In `render_router_preset`, replace:
```python
    if cfg.get("kv_quant"):
        lines.append(f"cache-type-k = {cfg['kv_quant']}")
        lines.append(f"cache-type-v = {cfg['kv_quant']}")
```
with:
```python
    ctk, ctv = _resolve_cache_types(cfg)
    if ctk:
        lines.append(f"cache-type-k = {ctk}")
    if ctv:
        lines.append(f"cache-type-v = {ctv}")
```

In `estimate_vram_footprint`, replace the final call:
```python
    return modelctl_vram.estimate_from_parts(
        weights, ctx, cfg.get("kv_quant") or "f16",
        gguf_params=params, mmproj_bytes=mmproj_bytes)
```
with:
```python
    ctk, ctv = _resolve_cache_types(cfg)
    return modelctl_vram.estimate_from_parts(
        weights, ctx, ctk or "f16",
        gguf_params=params, mmproj_bytes=mmproj_bytes,
        cache_type_v=ctv or ctk or "f16")
```

- [ ] **Step 3.4: Run all suites**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: 120 + 48 = 168 OK. (Existing TestBuildServerArgs fixtures use `kv_quant` — the legacy fallback keeps them passing unchanged; if one fails, the fallback is wrong, not the test.)

- [ ] **Step 3.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Emit independent cache-type-k/v with legacy kv_quant fallback"
```

---

### Task 4: Defaults + prompts (CLI)

**Files:** Modify `modelctl.py`, `test_modelctl.py`

- [ ] **Step 4.1: Write the failing tests** — append to `test_modelctl.py`:

```python
class TestCacheTypeDefaults(unittest.TestCase):
    def test_new_keys_default_q8_0(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": ""}):
            d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q8_0")
        self.assertEqual(d["cache_type_v"], "q8_0")

    def test_legacy_env_var_applies_to_both(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": "q5_1"}):
            d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q5_1")
        self.assertEqual(d["cache_type_v"], "q5_1")

    def test_new_env_vars_win_and_diverge(self):
        with mock.patch.object(modelctl, "DEFAULTS_PATH", Path("/nonexistent/x.json")), \
             mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": "q5_1",
                                            "MODELCTL_DEFAULT_CACHE_TYPE_V": "q4_0"}):
            d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q5_1")   # legacy still covers K
        self.assertEqual(d["cache_type_v"], "q4_0")   # new var wins for V

    def test_persisted_legacy_kv_quant_applies_to_both(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "defaults.json"
            p.write_text(json.dumps({"kv_quant": "q5_0"}))
            with mock.patch.object(modelctl, "DEFAULTS_PATH", p), \
                 mock.patch.dict("os.environ", {"MODELCTL_DEFAULT_KV_QUANT": ""}):
                d = modelctl.load_defaults()
        self.assertEqual(d["cache_type_k"], "q5_0")
        self.assertEqual(d["cache_type_v"], "q5_0")
```

- [ ] **Step 4.2: Run to verify failure**

Run: `python3 -m unittest test_modelctl.TestCacheTypeDefaults -v`
Expected: KeyError `cache_type_k`.

- [ ] **Step 4.3: Implement in `modelctl.py`.**

(a) Replace the `DEFAULT_KV_QUANT = ...` constant line with:

```python
# K and V caches can be quantized independently; both default to q8_0
# (the old single MODELCTL_DEFAULT_KV_QUANT is still honored as the
# fallback for whichever of the two isn't set explicitly).
DEFAULT_CACHE_TYPE_K = os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_K", "q8_0")
DEFAULT_CACHE_TYPE_V = os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_V", "q8_0")
```

Then grep for remaining `DEFAULT_KV_QUANT` uses (`grep -n DEFAULT_KV_QUANT modelctl.py`) and update each (the `load_defaults` entry is handled below; any other use should reference the K constant).

(b) In `load_defaults()`, replace the `"kv_quant": pick(...)` entry with:

```python
        # cache_type_k/v resolution: new env var -> new persisted key ->
        # legacy kv_quant (env or persisted) -> hardcoded q8_0.
        "cache_type_k": (os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_K")
                         or persisted.get("cache_type_k")
                         or pick("kv_quant", None)
                         or DEFAULT_CACHE_TYPE_K),
        "cache_type_v": (os.environ.get("MODELCTL_DEFAULT_CACHE_TYPE_V")
                         or persisted.get("cache_type_v")
                         or pick("kv_quant", None)
                         or DEFAULT_CACHE_TYPE_V),
```

NOTE: `pick("kv_quant", None)` returns the `MODELCTL_DEFAULT_KV_QUANT` env value or persisted `kv_quant` or None. Empty-string env values are falsy and fall through — that's what the tests rely on.

(c) `prompt_config`: replace the single `kv_quant = input(...)` line with:

```python
    cache_type_k = input(f"K cache quant, e.g. q8_0 [{d['cache_type_k']}]: ").strip() or d["cache_type_k"]
    cache_type_v = input(f"V cache quant, e.g. q4_0 [{d['cache_type_v']}]: ").strip() or d["cache_type_v"]
```

and in its returned dict replace `"kv_quant": kv_quant,` with `"cache_type_k": cache_type_k, "cache_type_v": cache_type_v,`.

IMPORTANT: `prompt_config` seeds `d = {**load_defaults(), "extra": "", **(current or {})}`. An old profile's `current` has only `kv_quant`; the overlay must map it so edits show the profile's own value. After building `d`, add:

```python
    if current and current.get("kv_quant") and not current.get("cache_type_k"):
        d["cache_type_k"] = current["kv_quant"]
        d["cache_type_v"] = current.get("cache_type_v") or current["kv_quant"]
```

(d) `prompt_defaults`: replace the single `kv_quant = input(...)` line with two analogous prompts reading/writing `current["cache_type_k"]`/`current["cache_type_v"]`, and update the saved dict to carry both new keys (drop `kv_quant`).

(e) `compute_pull_placement_hint`: replace `d["kv_quant"]` with `d["cache_type_k"]` and pass `cache_type_v=d["cache_type_v"]` to `estimate_from_parts`.

(f) Grep for any other `["kv_quant"]` / `.get("kv_quant")` reads in modelctl.py (`grep -n kv_quant modelctl.py`) — remaining ones should only be inside `_resolve_cache_types` and the `prompt_config` legacy-overlay from (c). Fix any straggler via `_resolve_cache_types`.

- [ ] **Step 4.4: Run all suites**

Run: `python3 -m unittest test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: all OK. Existing tests that build config dicts with `kv_quant` keep passing through the legacy fallback. `TestVramDefaults`/`TestPullPlacementHint` patch DEFAULTS_PATH already; if a defaults-shape assertion fails, update it to the new keys and note it.

- [ ] **Step 4.5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Split kv_quant into cache_type_k/v across defaults and prompts"
```

---

### Task 5: TUI ConfigureScreen dual inputs

**Files:** Modify `modelctl_tui.py`, `test_modelctl_tui.py`

- [ ] **Step 5.1: Locate current code.** `modelctl_tui.py:296` has `yield Input(value=d["kv_quant"], id="config-kv-quant")` and `modelctl_tui.py:316` reads it into `"kv_quant": ...`. Read the surrounding ConfigureScreen class first (labels/layout pattern) so the two new inputs match the existing widget/label idiom exactly.

- [ ] **Step 5.2: Write the failing tests.** In `test_modelctl_tui.py`, find every fixture dict containing `"kv_quant": "q8_0"` (lines ~352-445, ~1080) and update them to `"cache_type_k": "q8_0", "cache_type_v": "q8_0"`. In the ConfigureScreen test class, replace assertions on `#config-kv-quant` with assertions that BOTH `#config-cache-type-k` and `#config-cache-type-v` exist and are pre-filled from defaults, and that the submitted config dict carries both new keys. Follow the existing test idiom in that file (read it first).

- [ ] **Step 5.3: Run to verify failure**

Run: `python3 -m unittest test_modelctl_tui 2>&1 | tail -3`
Expected: failures on the missing new input IDs.

- [ ] **Step 5.4: Implement in `modelctl_tui.py`:** replace the single kv-quant Input with two (matching the surrounding label/Input pattern):

```python
        yield Label("K cache quant")
        yield Input(value=d["cache_type_k"], id="config-cache-type-k")
        yield Label("V cache quant")
        yield Input(value=d["cache_type_v"], id="config-cache-type-v")
```
(adapt Label usage to whatever the existing rows actually use — copy the neighboring row's structure), and in the submit handler replace the `"kv_quant":` entry with:

```python
            "cache_type_k": self.query_one("#config-cache-type-k", Input).value,
            "cache_type_v": self.query_one("#config-cache-type-v", Input).value,
```

- [ ] **Step 5.5: Run all three suites**

Run: `python3 -m unittest test_modelctl_tui test_modelctl test_modelctl_vram 2>&1 | tail -3`
Expected: all OK.

- [ ] **Step 5.6: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Split KV cache quant into K and V inputs in the TUI wizard"
```

---

### Task 6: Live verification (controller runs this)

- [ ] **Step 6.1:** Calculator standalone: `python3 modelctl_vram.py ~/models/<some-real-model>.gguf --ctx 131072 --cache-type-k q8_0 --cache-type-v q4_0` → sane breakdown, K/V shown separately. Also copy the file to /tmp and run it from there to prove detachment.
- [ ] **Step 6.2:** Legacy profiles unchanged: `python3 modelctl.py place` → same estimates as before (all profiles still kv_quant-only → fallback applies to both).
- [ ] **Step 6.3:** Divergence end-to-end: `modelctl show qwen3.6-moe-mtp-longctx`, hand-edit its JSON to `"cache_type_k": "q8_0", "cache_type_v": "q4_0"` (keep kv_quant or remove — either works), run `modelctl regen qwen3.6-moe-mtp-longctx --no-hermes --no-router-restart`, check the generated run.sh and preset section emit distinct flags, and `modelctl place qwen3.6-moe-mtp-longctx` shows the smaller estimate (~28.2GB → recommended SYCL0). Revert or keep per user's earlier interest (leave the edit IN — the user asked for exactly this configuration — but sync with restart at the end: `modelctl sync --no-hermes`).
- [ ] **Step 6.4:** Full suites green; commit any verification adjustments.

---

## Self-Review Notes

- Spec coverage: helper (Task 3), constants/defaults chain (Task 4), prompts CLI (Task 4), emitters (Task 3), TUI (Task 5), estimator integration + calculator (addendum → Tasks 1-2), out-of-scope items respected (no profile migration; edits/pulls naturally write new keys).
- Type consistency: `_resolve_cache_types` returns a 2-tuple of str|None used at three sites; `kv_cache_bytes(params, ctx, cache_type_k, cache_type_v=None)` signature consistent across Tasks 1, 2 (calculator), and 3 (estimate_vram_footprint call).
- Known judgment point: Task 5 gives structural instructions rather than verbatim code because the TUI's exact widget idiom must be copied from the file; the implementer is instructed to read it first and match.
