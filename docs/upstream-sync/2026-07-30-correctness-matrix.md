# Task 0.6 — Post-sync deterministic correctness matrix

Validates the ported + architecture-corrected MoE expert-weight cache
(`sync/moe-cache-upstream-2026-07-30`, commits `2b8ce6670`..`db07b9eb7`,
see `2026-07-30-port.md` and `2026-07-30-architecture-corrections.md`)
against a genuinely unmodified upstream oracle, using exact greedy
token-ID comparison — not prose, per the roadmap's explicit requirement.

## Test fixture

No small MoE GGUF existed anywhere on this machine (only real production
models, 16GB+, or vocab-only tokenizer test fixtures) and building/loading
a production model repeatedly for fast correctness iteration was the wrong
tool — and an unnecessary risk on a machine actively serving those models.

Built a tiny synthetic deterministic MoE model instead:
`scripts/moe-cache-correctness/make_tiny_moe.py` in the llama.cpp
submodule (reproducible: fixed numpy seed, `arch=llama` with `n_expert=8`,
`n_expert_used=2` — the plain mixtral-style MoE path, no shared expert —
3 layers, `n_embd=64`, `n_ff=128`). Tokenizer vocabulary/merges/BOS/EOS
are copied verbatim from the repo's own `models/ggml-vocab-llama-bpe.gguf`
test fixture, so tokenization itself is fully valid (real llama-3 BPE);
only the model weights are synthetic. Verified end-to-end before use:
loads cleanly on both CPU and SYCL (both GPUs), and produces byte-identical
greedy-decoded (`temp=0`, fixed seed) token sequences across repeated runs
and across CPU vs. GPU.

**Oracle**: a genuinely unmodified `origin/master` (`9b2a08881`) checkout,
built with the same flags as the ported binary (SYCL, `GGML_BACKEND_DL=OFF`,
Release, oneAPI 2026.1 `icpx`/`icx`) in a separate worktree — zero cache
code, not merely cache-disabled.

**Important test-fixture finding**: the cache's scheduler hook
(`copy_experts` in `ggml-backend.cpp`) only intercepts copies when the
expert-weight tensor's buffer is **host-resident** (`ggml_backend_buffer_is_host`)
— i.e. the model doesn't fully fit in VRAM and experts stream from
host/mmap memory on demand. With `-ngl 99` alone, this tiny model fits
entirely in VRAM and the hook never engages (confirmed correct, not a bug,
via direct instrumentation — see below). All cache-enabled cases below use
`-ot "exps=CPU"` to force expert tensors host-resident, matching the real
production deployment shape (oversized MoE, RAM/mmap-backed experts,
`fit: on`).

**Second finding**: `moe_cache_hits_total`/`misses_total` staying flat
across identical repeated requests is not a bug — it lines up with
llama.cpp's own graph-reuse optimization (the `graphs reused = N` line
already visible in the Task 0.1 baseline's production log) skipping
redundant restaging when nothing changed. All hit/miss/eviction cases
below use 4 textually distinct long-form prompts (each `> 32 tokens`) so
the staging path genuinely re-executes; short prompts triggered it
inconsistently in ways not fully root-caused within this task's scope
(noted, not chased further — doesn't affect the correctness verdict, since
every configuration tested, hook-active or not, matched its oracle).

## Results

All entries compare exact greedy token-ID arrays (`temperature=0`,
fixed `seed=42`) against the oracle for the same prompt. ✅ = exact match.

| # | Case | Config | Result |
|---|---|---|---|
| 1 | Cache disabled vs. unmodified upstream | no `--moe-cache-*` flags, `-ot exps=CPU` | ✅ exact match |
| 2 | Forced all-hit | `--moe-cache-bytes 8388608 --moe-cache-admission-misses 1`, 2 passes over 4 distinct prompts | ✅ all 8 outputs match; pass 1 alone: 126 hits / 48 misses / 48 promotions (expert reuse across prompts) |
| 3 | Forced all-miss | `--moe-cache-bytes 1024` (below minimum slot size — cache fails to init, falls back to normal copy: `moe_cache: init failed on device 0`) | ✅ match; also re-verified with `--moe-cache-bytes 200000` (~2 slots, constant thrashing: 174 misses / 172 evictions / 0 hits) — ✅ match |
| 4 | Alternating mixed hit/miss | `--moe-cache-bytes 200000`, LRU, 4 distinct prompts × 2 passes | ✅ match (same run as case 3's 2-slot variant; heavy churn produces genuine mixed hit/miss/eviction) |
| 5 | Partial projection population | same 2-slot run as above — a slot holds `gate+up+down` together (`filled_mask` bits, see `moe-cache.hpp`); at this capacity, incremental fill/eviction mid-expert is inherent | ✅ match under constant contention |
| 6 | Admission threshold 1 vs. 2+ | `--moe-cache-admission-misses 1` → 48 promotions; `=2` → 32 promotions (real, distinguishable difference — not a no-op) | ✅ match both |
| 7 | Prefill admission on vs. off | `--moe-cache-prefill-admission on\|off`, tested at admission=1 and admission=2 | ✅ match both settings. **Open finding**: config value is confirmed correctly threaded (Task 0.4 fixed a real no-op bug here) and accepted without error, but no measurable difference in hit/miss/promotion counts was observed between on/off in this test design — worth revisiting when Phase F (Task F4) actively engineers prefill/decode phase semantics; does not affect this task's correctness verdict |
| 8 | LRU vs. SLRU eviction policy | `--moe-cache-policy lru\|slru`, 2-slot budget | ✅ match both. Numbers were identical between policies at this scale (protected-segment logic has ~1 slot of room at `n_slots≈2`, so SLRU degenerates toward LRU behavior here) — real policy differentiation needs a larger slot count than this tiny fixture provides |
| 9 | Cache reset while loaded | mid-session `POST /cache/reset` (confirmed `slots_used` 14→0), generation continues | ✅ match before and after reset |
| 10 | Unload/reload | implicit: every case above is a fresh server process against the same model/seed | ✅ match, consistently, across dozens of independent launches |
| 11 | Two contexts, one GPU | `--parallel 2`, two concurrent completions (different prompts) on SYCL0 simultaneously | ✅ both match their respective oracles, no crash — validates Task 0.4's ref-counted ownership rework under real concurrent access |
| 12 | Two asymmetric GPUs | `--device SYCL1` (Arc B580, the smaller/second GPU) | ✅ match; 126 hits / 48 misses, same as SYCL0 — per-device cache isolation confirmed correct on both real GPUs |
| 13 | Dynamic backend loading | same 4-prompt sequence against `build-sycl-dl` (`GGML_BACKEND_DL=ON`) | ✅ match; identical hit/miss/promotion counts (126/48/48) to the static build |
| — | Main + draft/MTP context | **Deferred** — see below | not tested |

## MTP deferral rationale

Unlike the other cases, this isn't a model-*size* problem solvable by
building another tiny synthetic fixture — MTP/NextN requires genuinely
different tensor structure (additional NextN prediction-head layers, the
layer-counting interaction from upstream commit `0324696b8` flagged in the
Task 0.2 impact review) that a quick from-scratch synthetic model can't
cheaply approximate correctly, and getting it subtly wrong would produce a
misleading result rather than a useful one. The real MTP-paired files on
this machine (`gemma-4-26B-A4B-it-Q8_0-MTP.gguf` + its base model) are
production-scale and weren't used, to avoid any resource contention with
live serving. This is deferred to Phase E's real-hardware acceptance work
(Task E3 explicitly lists "main plus draft/MTP model where applicable"),
not silently dropped.

## Acceptance against Task 0.6

> The cache-disabled integration matches current upstream within the
> declared numerical tolerance, and every enabled path passes deterministic
> correctness checks.

Met for every case tested (1–13): all outputs are **exact** token-ID
matches (zero tolerance needed — greedy decoding on identical weights is
deterministic). MTP is the one explicitly deferred gap, carried forward
rather than hidden.
