# Does the MoE transfer cache help batch-1 decode? — 2026-07-30

**On the model this stack exists for, it nearly pays for itself: -12%.**
On a smaller MoE it does not come close: -42%. The difference tracks the
active expert set, and it points at a cheap fix that was not obvious
before measuring.

## Why this was worth measuring

Phase E's hardware matrix found `cache-enabled` ≈ `cache-disabled`, and
attempts to observe the cache's counters found no cache instance at all.
The cause is documented in
`../../docs/upstream-sync/2026-07-30-cache-inactivity-rootcause.md`: the
scheduler hook only fires for cross-backend weight copies, gated on

```cpp
get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size   // default 32
```

Decode is batch 1, so on the default configuration the cache **never
activates during generation** — exactly the case Phase G names as the
requirement ("useful interactive decoding, not only large prompt
batches"). `GGML_OP_OFFLOAD_MIN_BATCH` overrides the threshold, making
the question answerable directly.

## Setup

Runtime `build-sycl-f` at fork commit `9d175c28a` (Phase F fixes in).
One warmup plus two measured generations of 128 tokens, temperature 0.
SLRU, admission 2, prefill admission on.

Two models, because the first result turned out not to generalise:

- **Qwen3.5-35B-A3B** (UD-IQ4_XS, 16.3 GiB), experts 10+ on CPU, SYCL0
  only, ctx 4096, 2 GiB cache.
- **Qwen3.5-122B-A10B** (UD-IQ1_M, 31.9 GiB) — the actual target — using
  the saved profile's own placement: experts 29–47 on CPU, layer split
  4:1 across both GPUs, ctx 8192, 4 GiB cache.

## Results

### Qwen3.5-122B-A10B — the real use case

| | min-batch | cache | decode | cache active |
|---|---|---|---|---|
| **A** | 32 (default) | on | **25.50 t/s** | **no** |
| **B** | 1 | off | 15.16 t/s | no |
| **C** | 1 | on | 22.41 t/s | yes |

Threshold cost **-41%**, cache gain **+48%**, net **-12%**.
Break-even needed ×1.68; delivered ×1.48.

| metric | value |
|---|---|
| hits / misses | 155,307 / 26,658 |
| hit ratio | **85.3%** |
| promotions / evictions | 12,477 / 2,394 |
| slots used | 1,765 / 1,765 |
| H2D bytes | 10.1 GB |
| host-weight copy fallbacks | 14,181 |

### Qwen3.5-35B-A3B — for contrast

| | min-batch | cache | decode | cache active |
|---|---|---|---|---|
| **A** | 32 (default) | on | **42.68 t/s** | **no** |
| **B** | 1 | off | 20.52 t/s | no |
| **C** | 1 | on | 24.88 t/s | yes |

Threshold cost **-52%**, cache gain **+21%**, net **-42%**.
Hit ratio 71.6%, fallbacks 124,564.

## Reading

**The cache works, and it works better the bigger the active set.** A3B
activates ~3B parameters per token, A10B ~10B — roughly three times the
expert bytes moved per token, so three times as much for a hit to save.
The measured gain moves accordingly: +21% → +48%, hit ratio 71.6% →
85.3%, and host-weight fallbacks collapse from 124,564 to 14,181. On the
122B the cache is genuinely doing its job.

**What it cannot pay for is the way it has to be switched on.**
`op_offload_min_batch_size` is a **global** setting. Lowering it to 1 to
reach the MoE path also forces every other small-batch op through a
cross-backend copy, and that blanket cost is what the cache is fighting:
-41% before it does anything. It recovers 48% of a 41% hole and lands
12% short.

**That framing suggests the fix, and it is not Phase G.** The threshold
is a property of the *offload decision*, not of the cache. Nothing
requires it to be uniform across op types. If routed MoE expert ops were
exempted from the minimum — or given their own, lower one — the cache
would engage at batch 1 without dragging every unrelated op through PCIe
with it. On these numbers that turns a 12% loss into a substantial win,
because the +48% would be measured against the default 25.50 t/s rather
than against a self-inflicted 15.16.

That is a small, local change to `ggml_backend_sycl_device_offload_op`
and worth trying before any of Phase G.

## Revised conclusions

1. **The claim "the transfer cache cannot serve interactive decode" is
   too strong.** It was drawn from the 35B result alone. On the actual
   target model it is within 12% of break-even while fighting a
   deliberately handicapped baseline.
2. **The binding constraint is the global offload threshold, not the
   cache.** Per-op-type offload policy is the next thing to test, and it
   is far cheaper than implementing Phase G.
3. **Phase G is not invalidated, but it is also not the next step.** Its
   design (execute miss rows on CPU rather than copying weights over
   PCIe) still avoids this cost class entirely. But if an op-aware
   threshold makes the existing cache a win, G's cost/benefit changes
   substantially and should be re-derived from that baseline.
4. **Any future cache A/B must set `GGML_OP_OFFLOAD_MIN_BATCH` or verify
   activation**, or the cache is silently inert and the comparison
   reports "no difference" for the wrong reason. This is what happened in
   Phase E, where the result was attributed to the model fitting in VRAM.

## Incidental validation

First observation of the cache counters at all, which exercised two
Phase F fixes end to end: the renamed metrics appear correctly as
`moe_cache_served_projections_total` and
`moe_cache_host_weight_copy_fallbacks_total` with `# HELP`/`# TYPE`
emitted once per family, and 12,477 promotions at `admission_misses=2`
means the per-projection admission path from F3 ran. Neither substitutes
for `tests/test-moe-cache.cpp`, which still does not build.

## Reproducing

```bash
cd ~/workspace/moe-serving/llama.cpp
source ./llama-sycl-env.sh
GGML_OP_OFFLOAD_MIN_BATCH=1 ./build-sycl-f/bin/llama-server \
  --model ~/models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-IQ1_M.gguf \
  --device SYCL0,SYCL1 --split-mode layer --tensor-split 4,1 \
  -ngl 999 -c 8192 --metrics \
  -ot 'blk\.(2[9]|3[0-9]|4[0-7])\.ffn_.*_exps=CPU' \
  --moe-cache-bytes 4294967296 --moe-cache-policy slru \
  --moe-cache-admission-misses 2 --moe-cache-prefill-admission on --port 45913
```

Generate, then `curl localhost:45913/metrics | grep moe_cache`. Drop the
`--moe-cache-*` flags for the off case; drop the env var for the default.
Confirm `moe_cache: initialized` appears in the server log — without it
the cache is not running and the comparison is meaningless.

---

## Addendum: Q4_K_M (71.3 GiB) — partial, and why it was abandoned

The two runs above used quants that fit in VRAM or close to it. Q4_K_M is
the quant actually served for quality work: 71.3 GiB across 3 shards,
larger than VRAM (42.1) and RAM (26.9) **combined**, so a large slice of
the experts is reachable only through mmap from NVMe.

Two of three conditions completed before the run was stopped:

| | min-batch | cache | decode | cache active |
|---|---|---|---|---|
| **A** | 32 (default) | on | 4.56 t/s | no |
| **B** | 1 | off | **5.82 t/s** | no |
| **C** | 1 | on | *not measured* | — |

**B is faster than A — the sign is inverted.** On both smaller quants,
lowering the offload threshold cost 41–52%. Here it *gained* 28%. That is
consistent with a storage-bound regime: when weights are arriving from
NVMe, moving more work onto the GPU is not the expensive part, so the
blanket cost that dominated the earlier runs does not apply. It also
means the "lowering the threshold is a 41% hole" framing above is
specific to models that are not storage-bound.

### Why this run is not trustworthy

`--tensor-split 4,1` was copied from the `qwen3-5-122b-a10b-ud` profile,
where it was tuned for IQ1_M at 31.9 GiB. On Q4_K_M it pinned SYCL0 at
31.2/31.9 GiB while leaving **~10 GiB of SYCL1 unused**, pushing far more
onto storage than the hardware requires. The GPUs are a 2.7:1 capacity
ratio; the split should be about `8,3`.

The A/B comparison above is internally consistent — both conditions share
the placement — but the absolute numbers understate the hardware, and an
over-storage-bound baseline would have flattered C. Rather than publish a
cache result measured against a self-inflicted handicap, the run was
stopped.

### To redo it properly

Re-run all three conditions with `--tensor-split 8,3`, confirm both GPUs
are meaningfully used before trusting anything, and follow
`moe-cache-testing-methodology.md`. The open question this would answer
is the interesting one: **in a genuinely storage-bound regime, where a
cache miss can cost a disk read rather than a PCIe copy, does the cache
finally pay for itself outright?** Neither smaller quant could answer
that, and the inverted sign in row B suggests the economics there are
different enough that it might.
