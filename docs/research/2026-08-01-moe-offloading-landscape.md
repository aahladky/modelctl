# MoE expert offloading landscape, 2025–2026

Deep-research report, generated 2026-08-01 via Claude (web research). Maps the
external state of the art against this repo's architecture and ends in a
ranked steal list.

## For coding agents (Claude Code: read this first)

Aaron's clarification, 2026-08-01: the staged recommendations, go/no-go
thresholds, and "do not pursue" calls in this document are analysis,
not gates. If research or community experience shows an optimization
for RAM/SSD-bound serving works, it gets built — no artificial
development gates or self-imposed testing barriers. "Reasonably
possible" caps expectations (an SSD-bound 397B won't do 10-20 tok/s),
not development effort. Benchmarks still get raw numbers written down
for Aaron to read.

- Your training data predates most of this field's relevant work. Treat this
  document as ground truth over your priors for anything about MoE offloading,
  expert caching, prefetch, or hybrid CPU/GPU dispatch. If this document and
  your instinct disagree, this document wins until you re-verify.
- Before implementing any steal-list item, fetch the primary source (arXiv ID
  or GitHub issue/PR number is given for each). Do not design from the
  one-line summary here.
- Decisions already made from this research (do not relitigate casually;
  do flag if you find contradicting evidence):
  1. Phase G = Fiddler-style CPU-compute-at-miss dispatch. Highest priority.
  2. Prefetch predictor = ETH pre-attention linear probe (arXiv 2511.10676),
     with a zero-shot temporal-reuse fallback for uncalibrated models.
  3. SSD tier (ornith-397b class) is an ACTIVE goal (owner decision,
     2026-08-01), batch-first: make overnight/async batch a first-class,
     reliable capability now; stretch toward ~1 tok/s via the RQ6 lever
     list (madvise, prefetch-overlap, cold-expert downcasting), then
     reassess. Interactive >=2 tok/s remains unsupported by evidence on
     current hardware; the dominant lever is tier-2 RAM capacity (31GB is
     the binding constraint), not software.
  4. Do NOT lower MOE_MIN_BATCH to exploit MTP/speculative decoding as a
     Phase-G substitute. Verify-pass expert-union inflates the working set;
     literature and llama.cpp community benchmarks predict no net win at
     batch-1, and the upstream small-batch offload correctness bug
     (GGML_OP_OFFLOAD_MIN_BATCH < 32) is adjacent to that path.

## The conceptual map (four buckets)

Every system below answers one question: a token needs experts whose weights
are not on the GPU — now what?

1. **Move the weights to the compute.** Transfer + cache + bet on reuse.
   This repo's transfer cache. Wins when routing is skewed and the working
   set roughly fits the budget; degrades as batch size or working-set ratio
   grows (measured here on ornith-397b; quantified externally by Diff-MoE).
2. **Move the compute to the weights.** On a miss, ship the activation
   (KBs) to the CPU and run the expert there instead of dragging weights
   (100s of MBs) across PCIe. Fiddler, KTransformers, HybriMoE. This repo's
   hybrid CPU-miss scaffolding is this bucket, currently unreferenced.
3. **Know the future.** Predict expert selection, prefetch to overlap
   transfer with compute. ProMoE, ExFlow/FATE, ETH pre-attention, SpecPrefetch.
   This repo's `moe_cache_prefetch` capability, currently unimplemented.
4. **Change what you need.** Mixed-precision expert tiers (HOBBIT),
   cache-aware routing, pruning/merging. Not currently in scope.

---

# Full report

## TL;DR
- **The niche is empty and defensible.** No system — not Intel's own IPEX-LLM
  FlashMoE, not KTransformers' XPU path, not any llama.cpp SYCL PR, not
  OpenVINO GenAI — implements a *persistent, eviction-based, hit/miss-aware
  expert weight cache on Intel Arc/SYCL*. Every Intel-GPU MoE system uses
  static placement + per-pass streaming. This fork's transfer cache is, as
  far as the public record shows, the only one of its kind on Level
  Zero/oneAPI.
- **The single highest-value technique to steal is Fiddler-style
  CPU-compute-at-miss on the decode path ("Phase G" + the hybrid
  scaffolding), not prefetch.** It converts the inert-during-decode cache
  into a working hybrid engine and has an order-of-magnitude precedent
  (Fiddler averages 8.2–10.1x over prior offloading methods). Second is a
  near-zero-shot pre-attention linear expert predictor (ETH Zurich, arXiv
  2511.10676; 93.0–97.6% accuracy, heterogeneous-model-friendly) to feed
  prefetch. The SSD tier for ornith-397b stays **active** by owner
  decision, batch-first — a >=2 tok/s interactive floor is not achievable
  NVMe-bound on this hardware (see the RQ6 lever list for what can move
  the number).
- **Stay a fork, but publicly answer llama.cpp issue #20757.** Upstream
  maintainer appetite for hit-aware dispatch complexity in the ggml
  scheduler is unproven; #20757 is closed with no PR and no C++ work
  started. This repo already contains the C++ implementation the issue's
  author explicitly requested — engaging there is high-leverage, low-risk
  visibility.

## Key findings

### RQ1 — The Intel Arc / SYCL expert-caching niche is confirmed empty
- **IPEX-LLM FlashMoE** is a command-line wrapper on top of llama.cpp
  (`flash-moe` wrapping `llama-cli`/`llama-server`). It runs DeepSeek-V3/R1
  671B and Qwen3-MoE 235B on 1–2 Arc A770/B580 via **static placement +
  on-demand streaming** — no eviction policy, no hit/miss tracking, no
  persistent slot buffer. The repo was archived read-only on Jan 28 2026.
  Intel publishes **no decode tok/s** for FlashMoE on Arc — only a demo GIF.
- **KTransformers XPU path** (`doc/en/xpu.md`) uses static YAML expert
  placement (`--cpu_infer`, `optimize_rules/xpu/*.yaml`) and sets
  `SYCL_CACHE_PERSISTENT=1` (the oneAPI JIT kernel cache — *not* an
  expert-weight cache). Its Expert Deferral and "CPU-GPU Expert Scheduling"
  (Jan 22 2026 changelog) are CUDA-Graph + x86 AMX features, not ported to
  SYCL/Level Zero. The XPU doc states serving is not supported on Intel GPU
  for now. No Arc tok/s published; all decode benchmarks (e.g. 2.42–4.09x
  over Fiddler) are NVIDIA+Xeon.
- **llama.cpp SYCL backend** has MoE work only in kernel correctness/quant
  (reordered Q4_K/Q5_K/Q6_K `MUL_MAT_ID`, fused top-k gating dispatch via
  `GGML_SYCL_ENABLE_FUSION`) — no expert cache, prefetch, or hit-aware
  dispatch. Issue #20757's design is generic-ggml/NVIDIA-PoC, not
  SYCL-targeted.
- **OpenVINO 2026.0** added GPT-OSS-20B CPU/GPU support and int4 3D-MatMul
  weight compression for MoE — model enablement, not dynamic offloading with
  a VRAM cache.

**Verdict: definitively empty.** "Caching" in every Intel-GPU MoE system
means static placement, not persistent-with-eviction. Note that
KTransformers *does* run on Arc via `--device xpu` (since May 14 2025) and
FlashMoE targets Arc directly — the *platform* is served for basic MoE
inference; it is specifically the *caching/hit-aware-dispatch layer* that no
one has built for Arc.

### RQ2 — Upstream llama.cpp landscape since March 2026
- **Issue #20757** (opened Mar 19 2026 by e1n00r) is **closed with no
  assignee, no labels, no linked branches or PRs, and no C++ implementation
  started.** The author is a self-described Python developer explicitly
  requesting a C++ contributor familiar with
  `ggml_backend_sched_compute_splits()` to implement the slot buffer and
  two-tier cache. Their tinyserve PoC (HuggingFace/Python, RTX PRO 2000 8GB,
  GPT-OSS-120B) reports steady-state 12–14 tok/s at ~98–100% hit rate vs
  0.5–1 tok/s uncached. The proposed design mirrors this fork closely (SLRU
  20/80, admission-on-second-miss, `--moe-expert-cache-size N`, pinned
  Tier-2 RAM, `POSIX_MADV_WILLNEED/DONTNEED`, slot-index remap in
  `MUL_MAT_ID`).
- **The `--cpu-moe`/`--n-cpu-moe`/`-ot` ecosystem is mature and is the
  baseline competitor.** `--n-cpu-moe` counts from the highest-numbered
  layers; `-ot` regex gives per-tensor placement. The op-offload threshold
  is confirmed at 32 upstream (`GGML_OP_OFFLOAD_MIN_BATCH`), while
  **ik_llama.cpp uses `32 x total_experts / active_experts`** — directly
  relevant to the fork's MOE_MIN_BATCH knob.
- **CUDA-graph work on the CPU-MoE path is fragile:** ik_llama.cpp has a
  known regression where Gemma4 26B-A4B CPU-expert token-gen is ~4.4x
  *slower* than upstream (issue #1765), and disabling CUDA graphs didn't
  help — evidence the decode-path hybrid is delicate even for expert
  maintainers.
- **Discussion #22183** (MoE offload to a second slower GPU) confirmed
  auto-fit assigns layers proportionally, which is suboptimal when GPU0 >>
  GPU1 in speed; the practical answer is manual `-ot` regex per device.
  Corroborates this repo's finding that the scheduler's offload pass
  selects the first capable backend, so only GPU0's cache engages.
- **MTP/speculative decoding landed upstream** (PR #22673, merged May 16
  2026, feature-flagged `--spec-type mtp`, `--spec-draft-n-max`). Community
  benchmark: Qwen3.6-27B decode 38->65 tok/s (1.71x) on a single RTX 5090.
  Critically: multiple independent benchmarks show **MoE at batch-1 on a
  single consumer GPU sees no net speedup** from MTP because the verifier
  pass activates the union of experts.

### RQ3 — Decode-path dispatch policies worth stealing
- **Fiddler (ICLR'25, arXiv 2402.07033)**: on a miss, copy the *activation*
  (batch x hidden — tiny) to CPU and compute the expert there, instead of
  copying the *weight* to GPU; ~50 ms per-expert weight-copy latency avoided
  (Quadro RTX 6000 / L4 measurements). Runs uncompressed Mixtral-8x7B
  (>90GB) at >3 tok/s on a single 24GB GPU; average 8.2–10.1x over
  DeepSpeed-MII / Mixtral-Offloading. Placement is popularity-based offline
  profiling. Code public (efeslab/fiddler). This *is* the hybrid CPU-miss
  scaffolding realized — the policy is proven.
- **KTransformers Expert Deferral (SOSP'25)**: defers some experts to
  overlap CPU/GPU compute (residual connections tolerate the delay), CPU
  utilization <75% -> ~100%, up to 1.45x on top of its kernels, <0.5%
  accuracy drop. CUDA-Graph scheduling cuts GPU launch overhead from >20%
  to near zero. Decode: 2.42–4.09x over Fiddler, 1.25–1.76x over llama.cpp
  full-precision, 1.77–1.93x quantized. The deferral *idea* is portable to
  a static graph; the CUDA-graph *mechanism* is not portable to ggml/SYCL.
- **HybriMoE (DAC'25, arXiv 2504.05897)**, built on KTransformers: dynamic
  intra-layer CPU/GPU scheduling, impact-driven inter-layer prefetch,
  score-based caching for activation instability. 1.33x prefill / 1.70x
  decode over KTransformers. Code public (PKU-SEC-Lab/HybriMoE).
- **MoE-SpeQ (arXiv 2511.14102)** verification-phase computation
  reordering: reorder the k-token batch so all tokens for the same expert
  compute contiguously, maximizing L1/L2 reuse — cheap, portable to the
  prefill/large-batch path where this cache already engages.

### RQ4 — Speculative decode x offload: the win is conditional, thin at batch-1
- **MoE-SpeQ (arXiv 2511.14102, SJTU)**: on-device quantized draft model
  predicts future experts; >96% cache hit across memory budgets; up to
  2.34x over SOTA offloading on Phi-MoE.
- **SP-MoE (arXiv 2510.10302)**: first SD-aware expert offloading;
  draft/target structural correspondence + cutoff-layer policy bounding
  per-layer prefetch depth. Draft/target attention-output cosine similarity
  up to ~94.6% (DeepSeek-Lite), ~59.8%/56.6% (Mixtral/Phi-3.5-MoE); top-1
  expert-prediction accuracy ~88.9% (DeepSeek-Lite), ~88% (Mixtral,
  Phi-3.5-MoE). Key insight: intra-request (SD) execution activates far
  fewer unique experts than inter-request batching.
- **DraftExpert (arXiv 2607.24434)** names the trap: verifying a
  multi-token block activates the *union* of target experts — increasing
  the draft set improves accuracy but triggers extra expert loading.
- **Direct answer to the MOE_MIN_BATCH question:** the literature does
  **not** support a meaningful win from the existing transfer cache alone
  by lowering the MoE offload threshold to <=K to route draft batches
  through the op-offload path. Batch-K verify inflates the working set,
  pushing up the batch-size-vs-miss-rate curve exactly where caching
  degrades. SP-MoE/MoE-SpeQ wins come from SD-aware prefetch and
  reordering, not from feeding a wider batch to a transfer cache.

### RQ5 — Expert prefetch menu, ranked for a solo C++ maintainer

| Predictor | Accuracy / recall | Training needed? | Zero-shot on arbitrary GGUF? |
|---|---|---|---|
| Pre-attention linear (ETH, 2511.10676) | 93.03% DSV2-Lite, 94.69% Qwen3-30B, 97.62% Phi-mini | 2 tiny linear layers, ranking loss | Near-zero-shot; light per-model calibration, architecture-agnostic |
| ProMoE MLP (2410.22134) | 2.20x avg prefill / 2.07x avg decode (up to 5.02x) speedup | Yes (MLP predictor) | No — per-model training |
| HOBBIT cross-layer (2411.01433) | 91% (up to 4 layers ahead) | Model-specific | No |
| AdapMoE / DAOP | 86% / 84% recall | Yes | No |
| OD-MoE SEP (2512.03927) | 95.67–99.94% recall (NF4->FP16) | Model-specific | No |
| SpecPrefetch shared adapter (2607.24787) | transfer-only; errors don't affect output | Light adapter | Partial |
| Popularity/temporal LRU priors | 29–77% hit (LRU, cache-size dependent) | None | Yes (fully zero-shot) |

- **LayerScope (ICS 2026) and SliceMoE (2512.12990) caveat**: routing
  regularities are layer-group-dependent, and modern MoE families
  intentionally weaken locality via load-balancing/entropy losses. Early
  layers show wide expert usage; deep layers sharper distributions.
- **Best value-per-LOC: the ETH pre-attention linear predictor** (Zhu,
  Bohl, Oester & Alonso, ETH Zurich, arXiv 2511.10676, 10 Nov 2025).
  Key insight: softmax and layer-norm are ranking-preserving, so expert
  selection can be approximated by simple linear functions computed
  *before* the attention block — buying prefetch lead time and covering
  layer 0. ~15pp absolute accuracy over FATE. The only high-accuracy
  near-zero-shot option for a heterogeneous GGUF fleet. Pair with a fully
  zero-shot temporal-reuse fallback for uncalibrated models.

### RQ6 — SSD/NVMe tier: park it for interactive use
- **colibri (JustVugg, July 11 2026)** streams GLM-5.2 (744B MoE, ~370GB
  int4 NVMe checkpoint, 9.9GB dense resident) on ~25GB RAM. Honest cold
  speed: **0.05–0.1 tok/s** on a 12-core/25GB laptop (each cold token
  triggers ~11GB of reads). Framework 13 with learned cache ~0.37 tok/s;
  Ryzen 9 9950X + PCIe 5.0 NVMe ~0.28 tok/s; Apple M5 Max ~1.06 tok/s;
  only the CUDA GPU path (RTX 4090, experts streamed from RAM not disk)
  reaches 20–25 tok/s. Disk-bound ceiling on a 6x5090 box: 6.84 tok/s.
  Uses an LRU learning cache and MLA KV persistence (576 floats/token).
- **MoE-Infinity (arXiv 2401.14361)**: sparsity-aware cache with activation
  tracing, 3.1–16.7x per-token latency improvement over
  vLLM/Ollama/DeepSpeed/BrainStorm at batch-1.
- **gpu_ext (arXiv 2512.12615)**: eBPF-based expert prefetching for UVM on
  GPT-OSS-120B (RTX 5090, 1.84x oversubscription) — 4.8x decode over
  framework offloading; Linux-UVM/CUDA-specific.
- **Direct answer:** ornith-397b at 0.37 tok/s is squarely in colibri's
  regime, and colibri's own numbers show a >=2 tok/s floor is not
  achievable NVMe-bound on consumer hardware for a model this size — only
  reached with a GPU compute path where experts stream from RAM, which is
  impossible at 182.6 GiB against ~44GB Arc VRAM + 31GB RAM.
- **Decision (owner, 2026-08-01): the SSD tier stays active** — batch-first
  framing (async "ticket" UX), not held to the interactive bar. Levers that
  can move 0.37 tok/s, in order of expected impact:
  1. **Tier-2 RAM capacity.** 31GB is the binding constraint; every GB
     added moves ~1GB of hot experts off NVMe permanently, and no software
     change matches that. At 96–128GB (if the board takes it) the on-disk
     tail shrinks to roughly 25–60GB — and with routing skew, most of that
     tail is cold.
  2. **madvise WILLNEED after each step / DONTNEED on eviction**
     (steal-list item 8, ~60 LOC per #20757) plus prefetch to overlap NVMe
     latency with compute.
  3. **HOBBIT-style cold-expert downcasting** — halve the bytes streamed
     for cold experts, keep hot experts at full quant (quality impact TBD,
     gate on the k-quant determinism harness).
  Honest ceiling check: colibri's best disk-bound numbers are ~1 tok/s
  (M5 Max, unified memory) and 6.84 tok/s on a 6x5090 box. Treat ~1 tok/s
  as the stretch target on current hardware, then reassess.

### RQ7 — Cache policy refinements with evidence
- **Diff-MoE (SC'25)** quantifies the batch-size-vs-miss-rate curve: with
  5% of experts cached, miss rate rises from **6.91% at batch-1 to 68.84%
  at batch-16**; communication grows 6.53x from batch 1->16 while compute
  grows only 1.55x; at batch-16, communication is 97.19% of per-iteration
  decode time in Pre-gated MoE. This is the mechanism behind static pinning
  beating the transfer cache when working set >> budget.
- **Mixture of Cache-Conditional Experts (arXiv 2412.00099)**: cache-aware
  routing halves miss rate; granular MoEs (Qwen-MoE, DeepSeek-MoE)
  tolerate approximate routing far better than Mixtral (0.5% vs 2.9%
  perplexity penalty) — relevant because this fleet skews granular
  (Qwen3.5/3.6, GLM-class).
- **SLRU + admission-after-second-miss corroborated** as the right default:
  #20757 PoC reports 8–15pp steady-state hit-rate gain over plain LRU on
  mixed traffic. **SpecMD (arXiv 2602.03921)** cautions MoE expert access
  is not consistent with temporal-locality assumptions (LRU/LFU) —
  frequency/admission gating matters more than recency alone.
- **SliceMoE (2512.12990)**: one unified cache across layers (early layers
  wide usage, deep layers sharp); warm up over the first ~10 decode steps.
- **Projection granularity / shared experts**: shared experts (`shexp` in
  GPT-OSS/GLM-5) are always active — pin to GPU; dense early layers
  (blk.0–2 in GLM-5/DeepSeek) always stay in VRAM.

### RQ8 — Adjacent capability watch
- **HOBBIT (arXiv 2411.01433)**: mixed-precision expert offloading, keeps
  high-precision hot experts, evicts far-layer experts first; up to 9.93x
  decode over SOTA; ~8,000 LOC on llama.cpp — a template for on-the-fly
  quantization tiers.
- Current research frontier (mid-2026): CoX-MoE (2605.17889), CommitMoE
  (AAAI 2026), FineMoE (EuroSys'26), OD-MoE (2512.03927, cacheless, SEP
  predictor 95.7–99.9% recall), SpecMD (2602.03921, MoE access violates
  LRU/LFU locality), OrderMoE (2607.17154, expert-similarity grouping),
  SliceMoE (bit-sliced caching).
- Practitioner scale: MTP is the dominant recent llama.cpp development;
  the `-ot`/`--n-cpu-moe` tuning guides (Doctor-Shotgun on HuggingFace) are
  community best practice.

## Transferability table

| System | HW target | Public + maintained code? | Transferable to this fork |
|---|---|---|---|
| Fiddler | CUDA | Yes (efeslab/fiddler) | **Policy** — CPU-compute-at-miss (activation copy, not weight). Maps directly onto the hybrid scaffolding + Phase G |
| KTransformers | CUDA+AMX (XPU beta) | Yes (kvcache-ai) | *Idea* of Expert Deferral; AMX/CUDA-graph mechanics not portable to SYCL |
| HybriMoE | CUDA (on KT) | Yes (PKU-SEC-Lab) | Score-based cache + impact-driven prefetch algorithm (readable, portable) |
| ProMoE | CUDA | Yes | Proactive prefetch loop structure; predictor needs training |
| ETH pre-attention | model-agnostic | Paper only | **Predictor math** — 2 linear layers, ranking loss; best zero-shot fit |
| MoE-Infinity | CUDA/CPU | Yes (TorchMoE) | Activation-tracing sparsity-aware cache; heavier |
| colibri | CPU/CUDA | Yes (JustVugg) | SSD streaming patterns, MLA KV persistence; confirms SSD ceiling |
| #20757 tinyserve | NVIDIA/Python | Yes (e1n00r) | Design validation; this repo already has the C++ they asked for |

## On the op-offload threshold and the MOE_MIN_BATCH knob
The upstream default of 32 (`GGML_OP_OFFLOAD_MIN_BATCH`) and ik_llama.cpp's
scaled `32 x total_experts/active_experts` are both about prompt-processing
GPU offload, not decode. The cache being inert during single-token decode is
structural: the cross-backend weight-copy path only fires above the
threshold. Lowering it to enable decode caching is exactly what the known
upstream correctness bug penalizes, and the MTP/batch-K literature says the
wider batch inflates the working set. The clean fix is Phase G (hit-aware
dispatch decoupled from the offload threshold), not threshold-lowering.

## Staged recommendations (with go/no-go thresholds)

**Stage 1 — Ship the decode-path hybrid (Phase G + Fiddler policy).
Highest priority.** Cached experts compute on GPU; missed experts compute
on CPU via activation-copy, not weight transfer, during batch-1 decode.
- *Benchmark that changes the plan:* on laguna-s2.1 (54.7 GiB sweet spot),
  if this beats the static split's 14.2 tok/s by >=1.3x the hybrid is
  validated; if it does not beat static, CPU kernels (not the cache) are
  the bottleneck — pivot to CPU-kernel optimization (AMX-style tiling à la
  KTransformers/CoX-MoE) before anything else.

**Stage 2 — ETH pre-attention linear predictor driving
`moe_cache_prefetch`.** Start zero-shot with a temporal-reuse fallback.
- *Threshold:* >=90% top-k recall on Qwen3.5-MoE and GPT-OSS-class. If
  recall <80% on a family, disable prefetch for that model and fall back to
  reactive caching — do not let a bad predictor pollute the cache.

**Stage 3 — Fix multi-GPU cache engagement.** Explicit `-ot`-style
per-device assignment (per discussion #22183) or extend per-device budget
maps to force B580 engagement. Modest upside; unlocks ~44GB combined budget
for the 50–70 GiB class.

**Stage 4 — Engage upstream, stay a fork.** Post implementation status on
issue #20757 (this repo is the C++ implementer the author asked for). Do
not refactor for upstream acceptance until a maintainer signals appetite —
the scheduler-complexity bar is high, and ik_llama.cpp's CPU-MoE TG
regression (#1765) shows the decode hybrid is fragile even for expert
maintainers.

**SSD tier (owner decision, 2026-08-01): active, batch-first.** Make
overnight/async batch first-class now; pull steal-list item 8 into scope;
stretch toward ~1 tok/s via the RQ6 lever list, then reassess. **Do not**
pursue naive MOE_MIN_BATCH-lowering to exploit MTP without Phase G.

## Ranked steal list (value per implementation effort)

1. **Fiddler CPU-compute-at-miss on the decode path** [RQ3] — activation
   copy, not weight copy, on miss. Evidence: >3 tok/s uncompressed 90GB
   Mixtral on one 24GB GPU; avg 8.2–10.1x over prior offloading (arXiv
   2402.07033). This is Phase G + the hybrid scaffolding realized.
2. **ETH pre-attention linear expert predictor** [RQ5] — 93.03–97.62%
   accuracy, near-zero-shot, 2 linear layers, pre-attention lead time
   (arXiv 2511.10676). Best predictor value-per-LOC.
3. **Keep SLRU + admission-after-second-miss; add unified cross-layer
   cache + ~10-step warm-up** [RQ7] — #20757 PoC (8–15pp over LRU),
   SliceMoE (2512.12990), SpecMD (2602.03921). Low effort; SLRU exists.
4. **MoE-SpeQ verification-phase computation reordering** [RQ4] —
   same-expert tokens compute contiguously for L1/L2 reuse (arXiv
   2511.14102). Cheap win on the existing prefill/large-batch path.
5. **Shared-expert / dense-layer static pinning as a hard rule** [RQ7] —
   pin `shexp` + dense early layers to GPU; only routed experts flow
   through the cache. Trivial effort.
6. **HybriMoE score-based caching + impact-driven prefetch** [RQ3] —
   1.70x decode over KTransformers (arXiv 2504.05897, code public).
   Consider after Stages 1–2 prove out.
7. **HOBBIT-style mixed-precision expert tiers** [RQ8] — up to 9.93x
   (arXiv 2411.01433). Larger undertaking; Phase-H candidate.
8. **madvise WILLNEED/DONTNEED per-expert SSD management** [RQ6] — ~60 LOC
   per #20757; DONTNEED needs adding to llama-mmap.cpp. In scope — the SSD
   tier is an active goal (see RQ6 decision).

## Caveats
- Intel publishes no FlashMoE decode tok/s on Arc, and KTransformers none
  for XPU; the "empty niche" verdict rests on documented *architecture*
  (static vs persistent-eviction), not head-to-head speed.
- #20757's exact close reason (stale-bot vs rejected) was not visible;
  treat "no maintainer appetite" as inferred, not stated.
- Predictor accuracies are model-dependent (ETH figures are
  DSV2-Lite/Qwen3-30B/Phi-mini); Ornith-397B and GLM-class are untested,
  and LayerScope/SliceMoE warn locality is weakening in newer families.
  Validate per-family before trusting prefetch. SP-MoE's cosine-similarity
  figures and top-1 accuracy figures are distinct metrics.
- Several 2026 arXiv IDs are recent preprints (SpecPrefetch 2607.24787,
  DraftExpert 2607.24434, OD-MoE 2512.03927, SpecMD 2602.03921) — treat
  headline numbers as author-reported.
- colibri third-party promotional tok/s claims conflict with the
  developer's own honest 0.05–1 tok/s cold numbers; weight the repo
  benchmarks.
- The MTP-for-MoE batch-1 caution rests partly on community benchmarks;
  the mechanism (expert-union in verify) is sound and corroborated by
  DraftExpert, but a rigorous head-to-head on these exact models does not
  exist.
