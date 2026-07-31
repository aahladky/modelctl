# Runtime promotion report — 2026-07-31 (second and third, same day)

Promotes `feature/sycl-moe-expert-cache@b9ce49c29` into the umbrella
repository, replacing the pin at `724f705c1`. Driven by the local-first
re-review (`modelctl/docs/modelctl-re-review-2026-07-31-local-first.md`),
P1.4–6 and P2 (G3).

**Addendum (same day): promotion 3 at `f6c0f5674` — hybrid execution.**
See "Promotion 3" at the end of this report.

## What is being promoted

| commit | contents |
|---|---|
| `d5e0a4d14` | Real build provenance in `--modelctl-capabilities` (commit, compiler, build type, backends, `dynamic_backends` from CMake) — closes the known-wrong `dynamic_backends` limitation carried in the previous manifest |
| `3950a6ea2` | Learned cache geometry (fused gate_up + unequal projections), per-projection tensor-origin identity (two models on one device can no longer be served each other's weights), explicit device-queue ownership with fail-closed adoption, `moe_cache_min_batch` honours the env override |
| `df28f7c7c` | Task G3 — CPU execution of cache misses over host/mmap weights (`moe_cpu_execute_misses`), host-tested; not yet wired into the op (G4–G7 pending) |
| `b9ce49c29` | Log the learned geometry at finalization |

## Gates run (this machine, 2026-07-31)

**Build forms.** CPU-only Release (gcc 16.1.1): clean, every SYCL/cache
feature reports false, provenance fields populate. SYCL Release
`GGML_BACKEND_DL=OFF` (icpx 2026.1, `build-sycl-int` scratch dir, same
flags as the promoted config): clean.

**Host-only unit tests.** 27 `test-moe-cache` + 20 `test-moe-hybrid`
cases pass under both gcc and icpx builds — includes the new naming,
learned-geometry, cross-model identity, concurrency, and G3 executor
cases.

**modelctl real-process suite.** `MODELCTL_REAL_TESTS=1
test_release_a_real`: 11/11 against a real llama-server and real GGUF.

**Correctness (tiny-moe fixture, SYCL0 = Arc Pro B70).** Token-identical
greedy output across every condition, compared per-prompt:

| condition | result |
|---|---|
| cache off, 3 identical requests | baseline `[29332, 26333, …]` |
| cache on (64 MiB), 3 identical requests | identical |
| cache on, reset before last request | identical |
| 4 distinct prompts × 2 reps, cache off | baseline set |
| 4 distinct prompts × 2 reps, cache on | identical; hits 35, promotions 48, slots_used 16, evictions 0 |
| **main + draft (two contexts, one device), cache on, 4 distinct prompts × 2 reps** | identical to the no-draft cache-off control; hits with both contexts live; clean SIGTERM exit; zero `UR_RESULT`/device-loss lines |

Learned geometry confirmed from the server log:
`moe_cache: geometry learned (gate=32768 up=32768 down=32768), 682 slots
of 98304 bytes` — sizes observed from real staged copies, not assumed
from the first tensor.

**Performance sanity.** 128-token greedy decode wall time on the fixture:
promoted 0.11 s vs candidate 0.12 s — within noise. (The tiny fixture is
a gross-regression check only; this promotion changes staging-path
bookkeeping, not kernels. The real-model cache measurements from
2026-07-30 remain the policy evidence.)

**Integration checks.** `ci/checks.sh`: all pass except the intentional
"working tree ahead of pin", which this promotion resolves.

## Known limitations carried forward

- The cache remains an experimental **transfer cache**; hybrid
  GPU-hit/CPU-miss execution is G4–G7 and `moe_hybrid_cpu_miss`
  correctly reports false. G3's executor exists but nothing calls it yet.
- Geometry learning finalizes on the first tensor-origin revisit; if the
  scheduler stages one tensor for several splits inside a single pass,
  later projections finalize absent and are simply never cached
  (fail-safe, not fail-wrong).
- `GGML_BACKEND_DL=ON` was not re-run this round (unchanged since the
  morning promotion; the provenance fix makes the binary now *report*
  its DL mode truthfully from the CMake option).

## Post-promotion state, and a repair found along the way

Attempting the in-place rebuild surfaced that **`build-sycl` had been
un-rebuildable since the repository moved** from `~/workspace/llama.cpp`
into `~/workspace/moe-serving/llama.cpp` (2026-07-30): its CMake cache
still baked the old absolute paths, so every `cmake --build` failed its
build-system check. The serving path (llama-swap → `build-sycl/bin/
llama-server`) was unaffected — only rebuilds were broken, silently.

Fix: the old tree was rotated to `build-sycl-prev-724f705c1` (kept as
the binary rollback for the previous validated pair) and `build-sycl`
was configured **fresh at the correct path** with the same flags, from
`b9ce49c29`, during a window with no model loaded in llama-swap. The
rebuilt binary's own capability report confirms the promoted commit and
build form, and it reproduces the baseline token arrays with the cache
active (35 hits, learned geometry 3×32768).

Also repaired: `llama-sycl-env.sh` exported the dead old build path into
PATH, and the home-directory agent rules still named the old location;
both now point at the moe-serving tree. `build-sycl-int` (the validation
scratch build of the same commit) is redundant and can be deleted.

---

## Promotion 3 (same day): true hybrid GPU-hit/CPU-miss execution

Promotes `f6c0f5674` (Tasks G4/G5 wired) on top of `b9ce49c29`.

### What it is

Under `--moe-hybrid-mode on`, a miss the cache declines to admit is
never transferred to the device: the scheduler hook records it in a
per-staged-tensor plan and `ggml_sycl_mul_mat_id` computes those
experts' rows on CPU over the original host/mmap weights (threaded,
ggml's own dequantizers), concurrently with the queued GPU rows, merging
through the same in-order queue. Fused kernels stay off while a plan is
pending. `/metrics` gains `moe_hybrid_*` counters.

### Correctness gates (all token-identical, zero device loss)

| condition | result |
|---|---|
| tiny-MoE, 6 distinct prompts × 16 tokens, hybrid vs cache-off | identical (programmatic diff) |
| all-CPU-tier stress (admission 100: 5,577 CPU rows, 206 skips) | identical |
| threaded CPU tier vs sequential | bit-identical (unit + fixture) |
| main + draft two-context + hybrid | identical, clean exit |
| **Qwen3.5-122B-A10B IQ1_M**, 24-token greedy, hybrid vs cache-only | identical; learned geometry 3×811008 B, 1765 slots |

### Performance verdict (the honest number)

Decode on the 122B IQ1_M target (`-ot exps=CPU`, threshold=1, 4 GiB
cache, admission 2, 96 tokens):

| condition | decode t/s |
|---|---|
| cache, GPU misses (B) | **4.33** |
| hybrid, CPU misses, single-thread tier | 1.91 |
| hybrid, CPU misses, threaded tier (C) | 2.46 |

Hybrid avoided 35.9 GB of expert H2D transfer and still lost: IQ1_M
dequantization on CPU costs more than the PCIe transfer it replaces at
this miss rate. The design doc's §2 warning ("judge G against the
op-aware threshold baseline") held. Consequently `moe_hybrid_cpu_miss`
reports **implemented** (it is), the feature is **opt-in** and the
control plane's experimental-margin guardrail will not auto-select it
against these numbers. The acceptance matrix gained a `hybrid-cpu-miss`
cell so the comparison reruns in one command; the likeliest change to
flip the verdict is a vec_dot-grade CPU tier (needs a cross-backend
path to ggml-cpu's kernels).
