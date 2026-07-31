# Release A target-hardware matrix — 2026-07-30

> Dated evidence. Run with `modelctl_acceptance.run_matrix()`; see
> `test_release_a_matrix.py` for the harness test.

**Hardware:** SYCL0 Intel Arc Pro B70 (31.9 GiB) · SYCL1 Intel Arc B580
(11.9 GiB) · 22.6 GiB RAM available · model store on `/dev/nvme1n1p1`
(btrfs), hardware fingerprint `0f433333f639fd46`.

**Runtime:** `llama.cpp/build-sycl/bin/llama-server`, binary fingerprint
`b13dc03640c7ee2d`, capability fingerprint `592ab2863de3d131`.

Two fixtures were used, because no single model exercises the whole
matrix: a dense 0.6B for the placement cells, and a real sparse MoE
(Qwen3.5-35B-A3B, IQ4_XS, 16.3 GiB) for the MoE cells.

## Results

| cell | fixture | status | measurement |
|---|---|---|---|
| `cpu-only` | Qwen3-0.6B | **passed** | 50.3 t/s gen, load 2.0s, page-cache-served |
| `single-gpu` | Qwen3-0.6B | **passed** | 201.5 t/s gen, load 4.0s |
| `second-gpu` | Qwen3-0.6B | **passed** | 162.2 t/s gen, load 4.0s |
| `two-asymmetric-gpus` | Qwen3-0.6B | **passed** | 180.9 t/s gen, load 4.0s |
| `fits-one-gpu` | Qwen3-0.6B | **passed** | 153.1 t/s gen, load 4.0s |
| `concurrent-prompt-and-decode` | Qwen3-0.6B | **passed** | 201.7 t/s gen, `--parallel 4` |
| `cache-disabled` | Qwen3.5-35B-A3B | **passed** | 45.7 t/s gen, load 18.0s, bulk-read |
| `cache-enabled` | Qwen3.5-35B-A3B | **passed** | 44.1 t/s gen, load 12.0s |
| `combined-vram` | Qwen3.5-35B-A3B | **passed** | 43.5 t/s gen, split 3,1 across both GPUs |
| `mmap-storage-backed` | Qwen3.5-35B-A3B | **passed** | 34.5 t/s gen, experts offloaded to CPU via mmap |
| `ram-spill` | Qwen3.5-35B-A3B | *skipped* | needs 24.0 GiB free RAM, had 22.6 GiB |
| `main-plus-draft` | — | *skipped* | no draft/MTP model on the fixture profile |

**10 passed · 0 failed · 2 skipped.**

Both skips are properties of this machine and this fixture choice, not of
the product: `ram-spill` needs more free RAM than was available at run
time, and `main-plus-draft` needs a fixture with an MTP companion (the
`gemma4-26b-mtp` profile has one and would cover it).

## Observations worth keeping

**The expert transfer cache is not currently a win on this model.**
`cache-enabled` measured 44.1 t/s against `cache-disabled` at 45.7 t/s —
within noise of each other, and if anything slightly slower. The model
fits comfortably in SYCL0's VRAM, so there is nothing for the cache to
avoid transferring; this is the expected result for a model that is not
oversized, and it is exactly why an experimental plan requires a
measured improvement before it may outrank a safe one. The cache's value has
to be demonstrated on a model that does not fit, which is the
`ram-spill` / oversized case still outstanding.

**Offloading experts to CPU costs about a quarter of throughput here.**
`mmap-storage-backed` at 34.5 t/s against 45.7 t/s fully resident on GPU.
Load time was also shorter (16.4s vs 18.0s) because less went to VRAM.

**The smaller GPU is not proportionally slower.** SYCL1 (B580, 11.9 GiB)
reached 162 t/s against SYCL0's 201 t/s on the same dense model, and the
3:1 split across both landed at 181 t/s — between the two, as expected
when work is distributed by capacity.

## What the run found

`combined-vram` and `mmap-storage-backed` failed on the first pass with a
bare `preflight_failed`. Two things were wrong, one in the harness and
one in the report:

1. **Cells inherited the fixture profile's `moe_cache` configuration.**
   The `qwen3.5-35b` profile sets `mode=manual` with
   `decode.miss_execution=cpu`, and the runtime at this date did not
   implement `moe_hybrid_cpu_miss`. The fail-closed launch gate
   correctly refused to launch — a product success being reported as
   a matrix failure. Cells now declare their own cache state.

2. **The failure reason was uninformative.** `preflight_failed` is a
   class, not an explanation; the run's validation messages carry the
   actual reason and were being discarded. The runner now reports them.

Neither was a defect in the serving path, but the first one is a good
demonstration that the fail-closed gate fires on real hardware against a
profile asking for a feature that does not exist yet.

## Reproducing

```python
import modelctl, modelctl_acceptance, modelctl_hardware, modelctl_launch
p = modelctl.normalize_profile(modelctl.load_profile("qwen3.5-35b"))
snap = modelctl_hardware.capture_hardware_snapshot()
backend = modelctl_launch.resolve_backend(p)
results = modelctl_acceptance.run_matrix(p, snap, backend, include_heavy=True)
print(modelctl_acceptance.render_report(results, p, snap, backend))
```

`plan_matrix()` takes the same arguments and reports what would run
without launching anything.
