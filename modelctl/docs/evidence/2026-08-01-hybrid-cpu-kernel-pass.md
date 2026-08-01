# Hybrid CPU kernel pass — miss-path profile, rewrite, 122B battery

2026-08-01. Raw numbers only. Machine: i7-14700K (20 cores / 28
threads, AVX2 + AVX_VNNI, no AVX-512), SYCL0 Arc Pro B70 32 GiB,
SYCL1 Arc B580 12 GiB, 31 GiB RAM + 64 GiB swapfile + 8 GiB zram.

Fork work is on `feature/sycl-moe-expert-cache`, on top of
`f4d390349`. Full per-condition detail (argv, env, token arrays, metric
scrapes, system snapshots) in
[2026-08-01-hybrid-cpu-kernel-pass.json](2026-08-01-hybrid-cpu-kernel-pass.json).

## 1. What the miss path was doing

`moe_cpu_expert_gemv` dequantized one weight row at a time with
ggml-base's `ggml_get_type_traits()->to_float`, then took a
double-accumulated f32 dot. `moe_cpu_execute_gemvs` split work **by
job**, so with `workers = min(n_threads, n_jobs)` a decode step that
missed four experts ran on four threads of the twenty-six available.

Instrumented first (`GGML_MOE_HYBRID_PROFILE=1` gives a per-row ns
breakdown; row/byte/thread counters are always on), then measured on
real expert weights through a new host-only microbenchmark,
`tests/bench-moe-hybrid.cpp`.

### Model geometry actually under test

`Qwen3.5-122B-A10B-UD-IQ1_M.gguf` (31.87 GiB, single file). Its 144
routed-expert tensors are **IQ2_XXS**, not IQ1_M — 48 blocks x
{gate,up} 3072x1024x256 and {down} 1024x3072x256, 811,008 B per expert
projection, `expert_used_count` 8. `hybrid-moe.md` previously
attributed the loss to "IQ1_M dequantization cost"; the file type is
the UD mix's name, the expert tensors are IQ2_XXS. Nothing about the
diagnosis changes, but the profile records the real type.

Incidental, recorded because it affects how the accuracy numbers below
read: in `blk.0.ffn_gate_exps`, 1650 of 2000 sampled rows carry a
zero fp16 block scale, and only 604 of 4096 reference outputs exceed
1e-6. Both code paths read the same bytes and agree on this, so it does
not affect any comparison here.

### Baseline profile (microbench, real weights, 4 miss experts/op)

| threads | per op | GiB/s | ns/row | dequant | matmul |
|---|---|---|---|---|---|
| 1 | 15.306 ms | 0.20 | 3737 | 2595 ns/row (69.6%) | 1132 ns/row (30.4%) |
| 26 available, 4 used | 7.370 ms | 0.41 | 1799 | 3137 ns/row (66.7%) | 1563 ns/row (33.3%) |

**Two-thirds of the tier's thread time was `to_float`, and it ran on
four threads.** Projected CPU-tier cost at 48 layers x 3 projections:
1061 ms per decode token.

## 2. Changes, each microbenched

Same op, same weights, `--experts 4`, `blk.0.ffn_gate_exps`:

| step | per op | GiB/s | ns/row | threads | dispatch |
|---|---|---|---|---|---|
| baseline (as shipped) | 7.370 ms | 0.41 | 1799 | 4 | 0.4% |
| (a) quantized `vec_dot` | 0.308 ms | 9.80 | 75 | 4 | 10.6% |
| (b)+(c) pool + row slicing | 0.099 ms | 30.64 | 24 | 26 | 2-5% |

**74x on the warm microbenchmark.** Projected CPU-tier cost per decode
token: 1061 ms -> 14.3 ms.

**(a) Quantized CPU matmul.** The activation is quantized once per
distinct activation into the weight type's `vec_dot_type` (Q8_K for
IQ2_XXS) and dotted straight against the quantized weight blocks by
ggml-cpu's `vec_dot`. Nothing is materialized as f32; `dequant_ns`
goes to zero because there is no dequantization left to time.

Reaching those kernels needed the cross-backend edge `hybrid-moe.md`
had listed as not implemented: backends link only ggml-base, so
`ggml-sycl` now also links `ggml-cpu`, guarded on `if (TARGET
ggml-cpu)`. Where it is absent the tier falls back to the old
dequantize-and-dot path, and
`moe_hybrid_cpu_kernel_rows_total` vs `moe_hybrid_cpu_weight_rows_total`
says which ran.

**Trap found on the way in:** without `ggml_cpu_init()` every quantized
`vec_dot` returns exactly 0.0, silently. The first version of this
change produced a 27x speedup on all-zero output, and the microbench's
self-consistency check passed because both passes were equally wrong.
`test-moe-hybrid`'s K-quant case is what caught it. The tier now calls
`ggml_cpu_init()` before first use, and the bench compares against an
independent dequantize-and-dot reference rather than only against
itself.

**(b) Threading.** Work is sliced by output row rather than by job, so
parallelism is bounded by `ne01 * n_jobs` (4096-12288 rows) instead of
by the miss count (4). Slices are claimed dynamically from an atomic
cursor — an expert already in page cache finishes far sooner than one
still faulting, and a static split leaves whoever drew the cold one
running alone. Every output row still has exactly one writer, so the
result is bit-identical to sequential execution; the bench asserts this
each run (`check max abs 0 ... bit-identical`).

Workers now live in a persistent pool woken by a generation counter,
instead of `std::thread` per call — at decode that was 144 spawn sets
per token, and once (a) made the work cheap, spawning was 10.6% of the
op. `moe_cpu_tier_threads()` in ggml-sycl.cpp also dropped its hard cap
of 16, which had been leaving ten threads idle.

**(c) Batching.** In the batch-1 decode branch every miss expert reads
the same activation row; that is now one device-to-host copy and one
`from_float` for the whole batch instead of one per expert.

### Accuracy cost of (a)

The kernel path dots against the activation quantized to Q8_K, so it
cannot equal the f32 reference. Measured against dequantize-and-dot
over the same weights: **relative RMS 0.0073**, max abs 0.0074 on a
reference ranging [-0.8165, 0.7998]. `test-moe-hybrid` pins this to a
derived bound (int8 quantizer step x sum|w|) that block
reinterpretation misses by three orders of magnitude.

## 3. 122B battery (three conditions + old-kernel control)

Protocol per `modelctl/docs/runtime/moe-cache-testing.md`: fixed prompt,
greedy (temperature 0, top_k 1), seed 42, `cache_prompt=false`, fresh
server per condition on port 18131, health polled at 1 Hz, 32-token
warmup, engagement verified (`moe_cache_learning==0` and
`misses_total>0`) then `POST /cache/reset`, 128-token measured decode,
metrics scraped before teardown. **Three replicates**, all against the
final binary. Commands assembled by hand to the shape of the P1 bench's
recorded argv; no saved profile, artifact or llama-swap config was
touched.

Common: `-ngl 999 -c 4096 --split-mode layer --tensor-split 8,3
--cache-type-k q8_0 --cache-type-v q4_0 --flash-attn auto --jinja
--parallel 1 --fit off --device SYCL0,SYCL1 --no-warmup --ubatch-size
128`.

| | config | env |
|---|---|---|
| C1 | static placement, no cache, default threshold | — |
| C2 | `-ot ffn_.*_exps=CPU --moe-cache-bytes SYCL0=4294967296 --moe-cache-policy slru --moe-cache-admission-misses 2 --moe-cache-prefill-admission off` | `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1` |
| C3 | C2 + `--moe-hybrid-mode on` | as C2 |
| C4 | identical to C3 | C3 env + `GGML_MOE_HYBRID_NO_VEC_DOT=1` |

C4 is the pre-change dequantize-and-dot miss path in the same binary —
a same-session control for change (a). It carries the **new** threading,
so it is not the old build.

### Measured decode

| condition | gen tok/s per run | mean | sd | prompt tok/s | load s |
|---|---|---|---|---|---|
| C1 static | 23.9317 / 24.0517 / 23.9424 | **23.9753** | 0.054 | 56.11 | 32.8 |
| C2 cache only | 5.1113 / 5.6205 / 5.4499 | **5.3939** | 0.212 | 9.17 | 18.6 |
| C3 cache + hybrid | 6.5309 / 5.7271 / 5.8411 | **6.0330** | 0.355 | 12.03 | 15.8 |
| C4 cache + hybrid, old kernel | 4.7022 / 5.5637 / 5.4162 | **5.2274** | 0.376 | 9.88 | 16.6 |

C3 vs C2 +11.9%, C3 vs C4 +15.4%. The per-condition spread is 4-18%, so
these end-to-end means are not separated by three runs on their own.

### CPU miss tier (counters cover warmup + measured)

| condition | tier wall s | weight GB | GB/s | ns/row | kernel rows | dispatch | quant act |
|---|---|---|---|---|---|---|---|
| C3 | 7.61 / 9.18 / 9.86 | 75.0 | **8.44** | **56** | 157752320 / 157752320 | 2.94% | 1.53% |
| C4 | 17.54 / 15.62 / 15.52 | 75.1 | **4.63** | **103** | 0 / 157991253 | 1.94% | — |

This is the number the end-to-end means cannot resolve, and the two sets
do not overlap: **the miss path is 1.84x faster per row in-server**, on
identical work (75 GB, ~158M rows either way).

In-server the tier reaches 8.44 GB/s against the microbenchmark's 30.6
GiB/s, because in-server it is reading a 75 GB weight stream from page
cache and storage rather than re-reading 3 MiB out of L3. **The kernel
change moved the tier from compute-bound to memory-bound**, which is
also why 74x on the bench is 1.84x here, and why C3 vs C4 end-to-end is
smaller still: at 26 threads the tier is now largely hidden behind the
GPU work it runs alongside.

### Cache counters (measured run, post-reset, mean of 3)

| condition | hit ratio | hits | misses | H2D GB | fallbacks | staging skips | H2D avoided GB | CPU rows | GPU rows |
|---|---|---|---|---|---|---|---|---|---|
| C2 | 0.5012 | 81755 | 81371 | 24.0 | 51806 | 0 | 0.0 | 0 | 0 |
| C3 | 0.5029 | 82041 | 81085 | 23.9 | 13505 | 58472 | 47.4 | 92433 | 105911 |
| C4 | 0.5015 | 81806 | 81314 | 24.0 | 13502 | 58602 | 47.5 | 92573 | 106067 |

Engagement verified in every cache run of every replicate.

## 4. Token identity

**On the deterministic tiny-MoE fixture the rail passes.**
`scripts/moe-cache-correctness/run_seq.sh`, 4 prompts x 2 repeats x 8
tokens, `-ot exps=CPU`, 4 MiB cache, slru, admission 2,
`GGML_OP_OFFLOAD_MOE_MIN_BATCH=1`: hybrid on and hybrid off produce
**identical token arrays in all 8 sequences**. Hybrid was engaged, not
bypassed — 72 staging skips, 582 CPU rows, and all 62,592 weight rows
through the quantized kernels.

**On the 122B the rail cannot be evaluated, for a reason that predates
this work.** Re-running an identical condition, same binary, same
config, seed 42, does not reproduce its own token sequence:

| condition | r1 vs r2 | r1 vs r3 |
|---|---|---|
| C1 static | identical | identical |
| C2 cache only, **hybrid off** | diverge at 2 (warmup) | diverge at 41 (measured) |
| C3 cache + hybrid | diverge at 2 | identical (measured) |
| C4 cache + hybrid, old kernel | diverge at 65 | diverge at 65 |

C2 runs no CPU-tier code at all, and static placement reproduces
perfectly, so the nondeterminism is in the transfer-cache path itself,
not in anything this pass touched. Hybrid-on did match hybrid-off
exactly in replicate 1 (128/128) and matched static exactly in
replicate 3 — the sequences move around, and which pair happens to
agree moves with them.

Consequence for acceptance: "hybrid on vs off must match at greedy seed
42" has no fixed reference on this model while C2 does not reproduce
itself. **Recorded as a failed acceptance item, and as a pre-existing
cache-path defect worth its own chain link.**

## 5. Laguna A/B rider (placement only, unrelated to the fork change)

laguna-s2.1 runs on its own pinned binary (`~/src/llama.cpp-laguna`),
untouched by this work. Both configs were launched directly on a
scratch port (18141) with commands assembled by
`modelctl_launch.launch_command_for_profile` from an in-memory profile
override; the live llama-swap service was not involved in the
measurement. Protocol: untimed 32-token warmup, then 3 runs of 256
tokens, temperature 0, thinking off — the shape of
`services/llama-swap/speed.py`, which produced the recorded anchors.

- **A — incumbent + shexp pins** (the ornith treatment, pins-only):
  saved config with `blk\.(1[0-9]|[1-9])\.ffn_.*_shexp=SYCL0,
  blk\.2[0-8]\.ffn_.*_shexp=SYCL1, ffn_.*_shexp=SYCL0` prepended to the
  existing `-ot`; `tensor_split 22,10` and the routed-expert rules
  untouched.
- **B — the planner's exported replan**, verbatim from
  `moe-review/replan-diff-laguna-s2.1/`: `tensor_split 8,3`, routed
  layers 1-17/18-22, `--no-mmap`, plus its own pins.

| config | run 1 | run 2 | run 3 | mean (server) | mean (client) | load s |
|---|---|---|---|---|---|---|
| A incumbent + shexp pins | 12.2869 | 13.9590 | 14.2375 | **13.4945** | 13.2704 | 46.1 |
| B planner replan | 10.5538 | 12.1623 | 13.0673 | **11.9278** | 11.7185 | 30.0 |

**A is faster by 13.1%.** Both climb across their three runs (page
cache warming; loadavg rose 4.5 -> 8.2 across the pair), and A's third
run at 14.2375 sits on the 2026-07-31 reference of 14.20.

Against the anchors: A **-4.97%** vs the 2026-07-31 reference (14.20) —
inside the 5% line by 0.03 pp — and **-1.57%** vs the 2026-08-01 P1
repeat (13.71). B is -16.00% / -13.00%.

### Applied

A, per "apply the faster config, with pins". Pins-only edit to
`config.extra`; `tensor_split` unchanged at 22,10. Profile saved
(backup `laguna-s2.1.json.bak.20260801-193007`), artifacts regenerated,
llama-swap config rewritten (backup
`config.yaml.bak.20260801-193007`). The derived launch argv was
asserted equal to the measured A command before saving.

The llama-swap config diff is **one line** — laguna's `-ot` rule. No
other entry changed, and laguna keeps its own pinned binary and env.
**llama-swap was not restarted** (pid 2587028, started 04:51:12,
unchanged; zero models resident throughout); the pins take effect at
its next natural reload.

Planning inputs: unchanged from the P5 backfill recorded at
2026-08-01T17:57:49 (RAM 33285996544, vram_limit_pct 90, primary SYCL0,
per-device inventory, bandwidth overrides,
`moe_cache_per_device_budgets: false`).

Admission for the applied config, from `modelctl place laguna-s2.1
--tiers` after the edit: tier 3 (GPUs + RAM), 54.7 GiB weights, 6.2 GiB
KV — 11.3 GiB fixed across both GPUs, 18.1 GiB experts layers 1-17 on
SYCL0, 5.3 GiB layers 18-22 on SYCL1, 27.2 GiB layers 23-47 in RAM.
Carried warning, pre-existing and not introduced by the pins:
"CPU-resident share exceeds the RAM budget even though the whole model
fits on paper -- treat this as tier 4 in practice."

The planner still proposes B; it is recorded as the slower of the two
and was not applied.

## 6. Acceptance

| item | result |
|---|---|
| miss-path profile before/after | recorded (sections 1-3) |
| pure-cache regression | none — C2 5.394 mean, its own path untouched |
| token identity, deterministic fixture | **pass** (8/8 sequences) |
| token identity, 122B | **cannot be evaluated** — reference nondeterministic with hybrid off |
| `ci/checks.sh` (full) | **pass, every check** — see below |
| laguna within 5% on the winning config | **pass** — A at -4.97% of the 14.20 reference (-1.57% of the 13.71 repeat) |

`ci/checks.sh` (full, run at 19:30 with the fork work still uncommitted,
so the pin and working tree agreed): submodule pin, manifest, static
checks, **1210 tests passed / 11 skipped in 44s**, console offline
build, CPU-only build, MoE cache + hybrid host-only tests, capability
truthfulness, ASan/UBSan on both host-only suites, layering — **all
checks passed**, no exceptions.

Because the token-identity item cannot be shown green, **the submodule
pin does not move and `integration-manifest.json` is unchanged.** Fork
work stays committed on `feature/sycl-moe-expert-cache`, which now sits
one commit ahead of the pin — so a `ci/checks.sh` run after this commit
reports the pin/working-tree item as a mismatch. That is the intended
state, not a regression: the pin is deliberately behind unvalidated
fork work.

## 7. Not done

- The batched-prompt branch still issues one job per (expert, row), so
  an expert's weights are re-streamed once per routed row. Decode is
  unaffected (batch 1) and the measured config runs
  `--moe-cache-prefill-admission off`, so no skips happen during
  prefill at all. Left alone.
- Optional ornith storage-bound hybrid-vs-static pair: not run.
- `llama.cpp/build-sycl-base/` is a leftover scratch build directory
  (gitignored) from an abandoned attempt to build the baseline binary
  by copying a configured CMake tree; the copy wrote some outputs back
  through absolute paths in its generated makefiles, so it was
  discarded in favour of the `GGML_MOE_HYBRID_NO_VEC_DOT` control.
