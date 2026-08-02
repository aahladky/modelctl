# Cache determinism hunt — the cache was never the cause

2026-08-01. Raw numbers only. Machine: i7-14700K (20 cores / 28 threads,
AVX2 + AVX_VNNI, no AVX-512), SYCL0 Arc Pro B70 32 GiB, SYCL1 Arc B580
12 GiB, 31 GiB RAM + 64 GiB swapfile + 8 GiB zram, oneAPI 2026.1.

Fork work on `feature/sycl-moe-expert-cache`, on top of `03ec8c277`. Full
per-run detail (argv, env, token arrays, per-step top-8 logprobs, metric
scrapes, machine-load samples) in
[2026-08-01-onednn-determinism.json](2026-08-01-onednn-determinism.json).

Model under test: `Qwen3.5-122B-A10B-UD-IQ1_M.gguf` (31.87 GiB, 48 blocks,
256 experts/layer, `expert_used_count` 8, routed-expert tensors IQ2_XXS).

## 0. Result

The chain arrived here with: at 122B, greedy, seed 42, `cache_prompt=false`,
an identical binary and config does not reproduce its own token sequence when
the MoE transfer cache is on, while static placement appeared to. The
hybrid-kernel pass could not evaluate its token-identity rail because the
reference itself moved; the P1 madvise bench saw an advise-on/advise-off pair
diverge at token 120 of 128.

**Neither the transfer cache, nor the hybrid CPU tier, nor the mmap-advise
tier is the cause. Static placement was not a reproducing control either.**
The divergence is oneDNN's GPU matmul: `DnnlGemmWrapper::gemm` built its
primitive without oneDNN's `deterministic` attribute, and oneDNN's default
permits an implementation whose reduction order varies between executions —
the same inputs are not required to return the same float. It is reached
whenever a matmul's batch exceeds `MMQ_MAX_BATCH_SIZE` (32), i.e. during
prompt processing, in **every** condition.

The fix is one attribute. Sections 1–4 are how that was established; 5–7 are
the fix and its acceptance.

## 1. Instrumentation: an execution fingerprint

`ggml/src/ggml-sycl/moe-fingerprint.{hpp,cpp}` (new; entirely inert unless
`GGML_MOE_FINGERPRINT` names a file). Per routed `MUL_MAT_ID` it records the
branch taken, the tensor shapes, the hybrid plan size, the CPU/GPU row counts
and a hash of the **routing tensor** — which experts the op was told to use.
Per staging decision it records `(layer, projection, expert)` and the
disposition: hit / promote / hybrid-skip / fallback / learning. Geometry
finalization and per-step cache counters get their own records.

Pointers are never printed raw: each distinct pointer becomes the ordinal of
its first appearance, so two runs touching the same objects in the same order
produce byte-identical fingerprints, while a different allocator address does
not by itself manufacture a diff.

`GGML_MOE_FINGERPRINT_DST=1` additionally hashes each routed op's output;
with `GGML_MOE_FINGERPRINT_DST_LAYER=<n>` it also hashes that layer's
**inputs** — the staged expert weights (over the used experts only) and the
activations. That is what lets a record say whether an op is the source of a
divergence or merely the first place anyone looked. It forces a device sync
per op and is for localization only, never for a battery.

Validation: on the deterministic tiny-MoE fixture, two runs' fingerprints are
byte-identical apart from the pid header.

## 2. The fingerprint diff that reframed the case

Three identical C2 runs (transfer cache on, hybrid off, threshold 1), 128-token
measured decode, fresh server each.

- Warmup (32 tokens): all three **identical**.
- Measured tokens: r1 vs r2 diverge at **step 41**, r1 vs r3 at **step 41**,
  r2 vs r3 at **step 65** (at step 41 the flip is 14131 vs 795).
- Fingerprints: diverge at **step 1** — inside the warmup request's prompt
  processing, layer 14 projection 0, where one run stages expert 189 and the
  other 183. Different routing.

At that point the cache has served nothing. Every staging disposition in steps
0 and 1 is `l` (learning) or `f` (fallback) — 2104 learning records and
13487 / 13505 / 13511 fallbacks across the three runs, **zero hits and zero
promotes**. The first promote lands at step 2 and the first hit at step 3.

Two runs routed differently while the cache was inert, so the cache is
downstream of the cause.

Also worth recording: the tokens first differ at step 41, but the logits
differ from **step 0**, and the largest logprob delta before the tokens
diverge is **1.35 nats** (r1 vs r2, at step 4; 1.03 for r1 vs r3). This is
not micro-deltas flipping argmax ties — it is substantial numeric drift that
the greedy argmax absorbs for 40 steps and then stops absorbing. The flip
itself happens on a top-1/top-2 gap of 0.10–0.21 nats.

## 3. The control the chain was missing

### First, a naming correction

`C1` in the P1 and hybrid benches meant **experts in VRAM** — those runs put
`-ot ffn_.*_exps=CPU` in the cache conditions only, so C1 had no tensor
override at all and reported 23.98 tok/s. That is a structurally different
graph: no per-expert split, no `copy_experts`, no host→device staging. It is
not a control for "host-resident experts, staged per token, cache off".

Every condition in this report carries `-ot ffn_.*_exps=CPU`, including C1
and C0, so placement is held fixed and only the cache and the offload
threshold vary. **`C1` here therefore means static placement with
host-resident experts and the default threshold, and its ~6.8 tok/s is not
comparable to the 23.98 tok/s C1 of the earlier benches.**

### The control

Even so, C1 is not the control for this defect. At decode it leaves the
routed MoE on the CPU backend (threshold 32, batch 1), so it never exercises
the offloaded GPU MoE path at all.

**C0** — `-ot ffn_.*_exps=CPU`, `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1`, **no cache
flags whatsoever** — is. Three runs, 128-token decode:

| pair | warmup identical | measured first divergence | max abs Δ logprob (step) |
|---|---|---|---|
| r1 vs r2 | no | **step 2** | 16.8540 (step 20) |
| r1 vs r3 | no | step 90 | 16.2333 (step 126) |
| r2 vs r3 | yes | **step 2** | 16.8766 (step 20) |

Op-level fingerprints were identical in all three; token sequences were not.
With no cache in the process. The cache is exonerated.

(The op-level fingerprint being identical here proves the op *sequence and
shapes* matched; it cannot see routing in a no-cache run, because the routing
hash was added after this measurement and the staging records only exist when
a cache does. Sections 4 onward use the routing hash.)

## 4. Localization

**In-process.** One server, one model load, the same request four times with
`cache_prompt=false`: the first step's logprobs already differ by 0.14–0.26
nats between repeats — allocator, page cache, thread pool and model bytes all
held fixed.

**Cold, single request, three fresh processes**, layer-0 input/output hashing.
The 44-token prompt is processed as three ubatches (2 / 40 / 4):

| ubatch | layer-0 routing | staged weights (used experts) | activations in | MoE output |
|---|---|---|---|---|
| ne12=2 | identical | identical | identical | identical |
| **ne12=40** | identical | **identical** | **DIFFER** | **differ** |
| ne12=4 | identical | identical | identical | identical |

Identical verdict in all three pairwise comparisons.

The staged expert weights are byte-identical, so host→device staging delivers
the right bytes and the MoE op's weights are right. The op's *input
activations* already differ, so the divergence is upstream of the first FFN.
And it is confined to the 40-token ubatch.

That is the batch-size boundary in `ggml_sycl_mul_mat`: `MMVQ_MAX_BATCH_SIZE`
is 8 and `MMQ_MAX_BATCH_SIZE` is 32 (`ggml/src/ggml-sycl/common.hpp:116,178`).
At 2 and 4 rows the dense matmuls take ggml's own `mul_mat_vec_q` kernel; at
40 they fall through both gates into `ggml_sycl_op_mul_mat_sycl`, which for
`gemm_flops >= 256^3` calls `DnnlGemmWrapper::row_gemm`
(`ggml/src/ggml-sycl/ggml-sycl.cpp:2773`). That wrapper set only
`scratchpad_mode` on its primitive attributes.

### Knob isolation

Cold triples, first-token top-16 logprobs, spread across the three runs:

| condition | max abs Δ logprob | verdict |
|---|---|---|
| C0 — threshold 1, no cache | 0.28279114 | diverges |
| C1 — static placement, default threshold | 0.18254042 | **diverges** |
| C0 + unquantized KV cache (f16/f16) | 0.26513958 | diverges |
| C0 + `GGML_SYCL_ENABLE_DNN=0` (oneDNN off) | **0.00000000** | reproduces |
| C0 + `GGML_SYCL_DETERMINISTIC=1` (the fix) | **0.00000000** | reproduces |

All five conditions produced the same first token (271) in every run — which
is exactly why token identity alone did not catch this for a month.

C1 diverging is worth stating plainly: **static placement was never
deterministic either.** The earlier finding that it reproduced was three
replicates of a token sequence whose argmax happened to survive.

`--flash-attn off` could not be measured: it is incompatible with the `q4_0` V
cache and the server refuses to create a context.

## 5. The fix

`ggml/src/ggml-sycl/gemm.hpp`: set oneDNN's `deterministic` primitive
attribute, behind a runtime knob `GGML_SYCL_DETERMINISTIC` that defaults to 1.
`GGML_SYCL_DETERMINISTIC=0` restores the previous behaviour, so the cost is
measurable rather than assumed.

Both arms of section 4's table are fixes; the attribute was chosen over
disabling oneDNN because it keeps oneDNN's kernel selection and constrains
only the reduction order.

## 6. Acceptance rail and battery

Protocol per `modelctl/docs/runtime/moe-cache-testing.md`: fixed prompt,
greedy (temperature 0, top_k 1), seed 42, `cache_prompt=false`, fresh server
per run on port 18147, health polled at 1 Hz, 32-token warmup, engagement
verified (`moe_cache_learning == 0` and `misses_total > 0`) then
`POST /cache/reset`, 128-token measured decode with `n_probs=8`, metrics
scraped before teardown, server terminated by PID and the port confirmed
free. Commands assembled to the shape of the argv recorded in the P1 and
hybrid benches; no saved profile, artifact or llama-swap config was touched.

Common: `-ngl 999 -c 4096 --split-mode layer --tensor-split 8,3
--cache-type-k q8_0 --cache-type-v q4_0 --flash-attn auto --jinja
--parallel 1 --fit off --device SYCL0,SYCL1 -ot ffn_.*_exps=CPU --no-warmup
--ubatch-size 128`. Cache conditions add `--moe-cache-bytes SYCL0=4294967296
--moe-cache-policy slru --moe-cache-admission-misses 2
--moe-cache-prefill-admission off` and `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1`.

### Rail: does each condition reproduce itself?

| condition | runs | warmup identical | measured identical | logprob identical | max abs Δlogprob |
|---|---|---|---|---|---|
| C2 cache on, hybrid off | 10 | 10/10 | 10/10 | 10/10 | 0 |
| C3 cache + hybrid | 10 | 10/10 | 10/10 | 10/10 | 0 |
| C3a cache + hybrid + mmap advise | 10 | 10/10 | 10/10 | 10/10 | 0 |
| C1 static placement, default threshold | 5 | 5/5 | 5/5 | 5/5 | 0 |

### Battery

| condition | gen tok/s mean | sd | prompt tok/s mean | load s mean | hit ratio | slots used |
|---|---|---|---|---|---|---|
| C1 static placement, default threshold | **6.2365** | 0.8087 | 11.5464 | 16.5 | — | — |
| C2 cache on, hybrid off | **5.0413** | 0.2915 | 9.4615 | 17.3 | 0.50235 | 1.0000 |
| C3 cache + hybrid | **5.1646** | 0.5526 | 9.6935 | 16.5 | 0.49916 | 1.0000 |
| C3a cache + hybrid + mmap advise | **5.1083** | 0.3968 | 10.1794 | 19.5 | 0.49916 | 1.0000 |

### Cost of determinism (same binary, attribute off)

| condition | gen tok/s mean | sd | prompt tok/s mean | load s mean | hit ratio | slots used |
|---|---|---|---|---|---|---|
| C1 static, determinism OFF | **7.3547** | 1.1833 | 12.4532 | 17.2 | — | — |
| C2 cache on, hybrid off, determinism OFF | **5.4802** | 0.2416 | 10.0298 | 18.6 | 0.49945 | 1.0000 |
| C3 cache + hybrid, determinism OFF | **6.0666** | 0.4918 | 11.2102 | 17.1 | 0.50144 | 1.0000 |

| condition | runs | warmup identical | measured identical | logprob identical | max abs Δlogprob |
|---|---|---|---|---|---|
| C1 static, determinism OFF | 5 | 2/5 | 1/5 | 1/5 | 16.8791 |
| C2 cache on, hybrid off, determinism OFF | 5 | 5/5 | 1/5 | 1/5 | 20.7038 |
| C3 cache + hybrid, determinism OFF | 5 | 5/5 | 1/5 | 1/5 | 20.7051 |

#### Paired comparison

| pair | deterministic | non-deterministic | delta |
|---|---|---|---|
| C1det vs C1nondet | 6.2365 | 7.3547 | -15.20% |
| C2det vs C2nondet | 5.0413 | 5.4802 | -8.01% |
| C3det vs C3nondet | 5.1646 | 6.0666 | -14.87% |

Concurrent machine load across the battery (442 samples, 15 s apart): loadavg(1m) 2.63–17.15, mean 8.99; MemAvailable 22.7–25.2 GiB.

### Cross-condition comparisons

### Cross-condition (run 1 of each)

| pair | warmup identical | measured identical | first divergence | max abs Δlogprob |
|---|---|---|---|---|
| C3 cache + hybrid vs C3a cache + hybrid + mmap advise | True | True | — | 0 |
| C2 cache on, hybrid off vs C3 cache + hybrid | True | False | 65 | 13.0935 |
| C2 cache on, hybrid off vs C1 static placement, default threshold | False | False | 2 | 16.6842 |
| C1 static placement, default threshold vs C3a cache + hybrid + mmap advise | False | False | 2 | 16.6675 |

Two of those rows carry the case docket.

**The advise flicker is closed by this fix.** The P1 madvise bench recorded
C4A vs C4B — two runs differing only by `GGML_MOE_CACHE_MMAP_ADVISE`, which
is semantically inert on read-only file-backed mappings — agreeing through
token 119 and diverging at 120. With the reduction order pinned, advise-on
and advise-off are **bit-identical**: same warmup, same 128 measured tokens,
max abs Δlogprob 0. Their **execution fingerprints are byte-identical too**
(md5 `4c2906e4c297346e6fdf7b22d49c3ce7` for C3det r1 and C3adet r1 and r5
alike), so advise changes neither the ops that run, the experts they route
to, nor the staging decisions taken — it is inert on the execution path and
not only on the output. It was never an advise defect; it was two runs of a
nondeterministic binary that happened to agree for a while, which is also
what "diverges at token 120 of 128" should have looked like in hindsight.

**Hybrid-on vs hybrid-off does not agree on the 122B, and now that is a
measurement rather than noise.** C2 (cache, hybrid off) and C3 (cache +
hybrid) share a warmup and diverge at measured token 65, reproducibly. That
is the CPU tier's own arithmetic, not nondeterminism: it dots the activation
quantized to Q8_K against the weight blocks while the GPU rows quantize to
Q8_1, measured in the hybrid-kernel pass at relative RMS 0.0073. Each
condition reproduces itself perfectly; they disagree with each other by a
fixed amount. The standing rail "hybrid on must match hybrid off at greedy
seed 42" therefore holds on the tiny fixture (where it passed 8/8) and does
not hold on this model — as a property of the tier's design, now separable
from the defect that used to mask it.

The remaining row, C2 vs C1, diverges at token 2 and is not a rail: those
two conditions differ in offload threshold as well as in the cache, so at
decode one runs the routed MoE on the GPU and the other on the CPU backend.
They are expected to compute different numbers; each reproduces itself.

## 7. Acceptance

| item | result |
|---|---|
| root cause identified and localized | **yes** — oneDNN matmul reduction order, §4 |
| 10/10 identical sequences, cache-on (C2) | **pass** — tokens, warmup and logprobs, max abs Δ 0 |
| 10/10 identical sequences, cache+hybrid (C3) | **pass** — same |
| 10/10 identical sequences, +advise (C3a) | **pass** — same |
| execution fingerprints byte-identical within a condition | **pass** — one md5 per condition across all runs |
| advise-on vs advise-off (the P1 C4 docket) | **pass** — bit-identical output *and* byte-identical fingerprints |
| static placement (C1) reproduces | **pass** — 5/5, having failed before the fix |
| tiny fixture, hybrid on vs off | **pass** — 8/8 sequences identical, hybrid engaged (72 staging skips, 582 CPU rows, all 62592 weight rows through the quantized kernels) |
| hybrid on vs off at 122B | **does not hold** — stable divergence at token 65, CPU-tier arithmetic, not nondeterminism (see cross-condition above) |
| `ci/checks.sh` (full) | **pass** — every check except the two expected pre-commit pin items |
| modelctl suite | **pass** — 1210 passed, 11 skipped in 79 s |

The one item that is not green was not green before either; the difference
is that it is now a reproducible measurement of a designed approximation
rather than an unevaluable rail.

## 8. Found in passing, not fixed

Both verified by reading, both out of scope for a determinism fix, both
flagged as their own work:

- `moe_expert_cache::hybrid_plan_pending()` (`moe-cache.hpp:294`,
  `moe-cache.cpp:439`) has **no production caller** — only
  `tests/test-moe-cache.cpp`. Its header documents it as how an op avoids
  fused kernels while a plan is pending; production enforces that instead
  through `hplan.empty()` in `ggml_sycl_mul_mat_id`, which only covers the
  tensor that just took the plan. The API advertises an enforcement
  mechanism nothing uses.
- `moe_expert_cache::reset()` clears slots, the layer index, miss counts,
  `m_tick` and the advice batches, but not `m_hybrid_plans`. `reset()` is
  reachable from an HTTP thread via `POST /cache/reset`; a plan recorded
  before it can be taken after it.

Also recorded, from the mechanism sweep, as dead code paths **on this
model** that go live on a Q4_K/Q5_K/Q6_K MoE: the fused MMVQ path
(`ggml_sycl_mul_mat_vec_q_id`'s type switch has no IQ2_XXS case, so it
always returns false), and with it `opt_for_reorder_id` and the whole
`note_staged_base` / `is_staged_base` apparatus — which still costs a
`std::set` insert under a mutex per expert per step here and buys nothing.

## 9. Not done

- Only the 122B was measured. ornith-397b (UD-IQ4_NL) runs the same code
  path and should behave the same way, but was not re-measured; its P1
  numbers predate this fix and were taken under the nondeterminism.
- laguna-s2.1 runs its own pinned binary (`~/src/llama.cpp-laguna`), which
  does not contain this fix and remains nondeterministic.
- The oneDNN **graph** SDPA path (`fattn-onednn.cpp`, used for flash
  attention) takes no `primitive_attr`, so `set_deterministic` does not
  reach it. It was deterministic in every run measured here, but that is an
  observation, not a guarantee — a different attention shape could select a
  different SDPA kernel. `GGML_SYCL_FA_ONEDNN=0` is the knob if it ever
  matters.
- `--flash-attn off` as a control: incompatible with the `q4_0` V cache on
  this model, so that arm could not be run.
- Whether `MMQ_MAX_BATCH_SIZE`/`MMVQ_MAX_BATCH_SIZE` are the right
  boundaries at all is untouched here; they are only how the divergence was
  localized.

## 10. Checks and the pin

`ci/checks.sh` (full), run with the fork work committed and the submodule
checked out at it but before the superproject commit that advances the pin:
submodule URL, static checks, **1210 passed / 11 skipped in 79 s**, console
offline build, CPU-only build, MoE cache and hybrid host-only tests,
CPU-only capability truthfulness, ASan/UBSan on both host-only suites,
layering — all **PASS**, with the two pin items failing as
"working tree is at 85b7e6556b6b, pinned at f4d390349ff7", which is exactly
what the commit that follows resolves. `ci/checks.sh --quick` after that
commit: pin and working tree agree, manifest agrees with the pin, all green.

The pin advances `03ec8c277` -> `85b7e6556`, so the determinism fix and the
hybrid CPU kernel work land as one validated pair, and
`integration-manifest.json` names the same commit.

**`llama.cpp/build-sycl` was deliberately not rebuilt**, per the work order.
It therefore holds `03ec8c277`'s code and is now behind the pin: anything
served from that binary still has the nondeterministic matmul. The
measurements here were all taken with a separate build of the pinned commit.
