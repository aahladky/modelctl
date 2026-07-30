# Does the MoE transfer cache help batch-1 decode? — 2026-07-30

**Short answer: no.** The cache works, and it helps by 21% *within* the
regime where it can run — but reaching that regime costs 52% first. Net
against doing nothing: **-42%**.

## Why this was worth measuring

Phase E's hardware matrix found `cache-enabled` ≈ `cache-disabled`, and
attempts to observe the cache's own counters found no cache instance at
all. The cause is documented in
`../../docs/upstream-sync/2026-07-30-cache-inactivity-rootcause.md`: the
scheduler hook only fires for cross-backend weight copies, and whether
such a copy happens is gated on

```cpp
get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size   // default 32
```

Decode is batch 1. So on the default configuration the cache **never
activates during generation**, which is precisely the case Phase G names
as the requirement: "useful interactive decoding, not only large prompt
batches."

`GGML_OP_OFFLOAD_MIN_BATCH` overrides that threshold, which makes the
question answerable in an afternoon rather than after implementing
Phase G on top of an untested assumption.

## Setup

Qwen3.5-35B-A3B (UD-IQ4_XS, 16.3 GiB) on SYCL0 (Arc Pro B70), experts
from layer 10 up pinned to CPU so there are genuine host-resident weights
to copy. Runtime: `build-sycl-f` at fork commit `9d175c28a` (Phase F
fixes included). Context 4096, one warmup plus two measured generations of
128 tokens each, temperature 0. Cache: 2 GiB, SLRU, admission 2.

## Results

| | min-batch | cache | decode | cache active |
|---|---|---|---|---|
| **A** | 32 (default) | on | **42.68 t/s** | **no** |
| **B** | 1 | off | 20.52 t/s | no |
| **C** | 1 | on | 24.88 t/s | yes |

Cache counters from run C:

| metric | value |
|---|---|
| hits | 137,288 |
| misses | 54,512 |
| hit ratio | 71.6% |
| promotions | 25,848 |
| evictions | 11,336 |
| slots used | 1,588 / 1,588 |
| H2D bytes | 11.6 GB |
| host-weight copy fallbacks | 124,564 |

## Reading

**The cache is not broken.** Run C shows it doing exactly what it was
designed to do: a 71.6% hit ratio, all 1,588 slots filled, SLRU
admission and eviction both working. Against the same threshold it is
worth **+21%** (24.88 vs 20.52).

**The regime is the problem.** Lowering `op_offload_min_batch_size` to 1
costs **52%** on its own (42.68 → 20.52), because forcing every
small-batch `MUL_MAT_ID` through a cross-backend copy is far more
expensive than letting the CPU compute it in place. That threshold exists
for a good reason. The cache recovers less than half of what enabling it
costs, so the whole path lands 42% below simply leaving the default
alone.

For the cache to pay off at batch 1, it would have to more than double
throughput in the lowered-threshold regime, not improve it by a fifth.
Nothing about the counters suggests headroom of that size: at a 71.6% hit
ratio the remaining misses are not where the time is going. The cost is
structural — the copy round-trip itself — not a cache-efficiency problem.

**The 124,564 host-weight fallbacks are worth noting.** Nearly half of
all projection requests bypassed the cache entirely (admission threshold
not yet met, or no free slot), and each one paid a full host-to-device
copy. That is inherent to a fixed-size cache over a working set larger
than it, not a defect.

## What this means for Phase G

Phase G proposes true GPU-hit / CPU-miss hybrid execution: classify rows
into hits and misses, run hits on cached GPU weights, run misses on
CPU-accessible weights, and merge. That design does **not** depend on the
cross-backend copy path this experiment measured — it would execute miss
rows on the CPU directly rather than dragging weights across PCIe, which
is exactly the cost that sank runs B and C.

So this result does not condemn Phase G. What it does establish:

1. **The current transfer cache cannot serve interactive decode**, on any
   setting, on this hardware. Its value is confined to prefill and other
   batch >= 32 work.
2. **Phase G cannot be built on top of the existing interception point.**
   Lowering the batch threshold to reach it is a 52% loss before any
   hybrid logic runs. G1's design work has to start from the routed-op
   partition, not from the copy hook.
3. **Any future claim that the cache helps decode needs this measurement
   repeated**, because the default configuration makes the cache silently
   inert and a naive A/B will show "no difference" for the wrong reason.

## Incidental validation

This run was the first time the cache's counters have been observed at
all, which exercised two Phase F fixes end to end:

- The renamed metrics (F7) appear correctly as
  `moe_cache_served_projections_total` and
  `moe_cache_host_weight_copy_fallbacks_total`, with `# HELP`/`# TYPE`
  emitted once per family.
- 25,848 promotions at `admission_misses=2` means the per-projection
  admission path from F3 ran, though this does not substitute for the
  unit tests in `tests/test-moe-cache.cpp`, which still do not build.

## Reproducing

```bash
cd ~/workspace/moe-serving/llama.cpp
source ./llama-sycl-env.sh
GGML_OP_OFFLOAD_MIN_BATCH=1 ./build-sycl-f/bin/llama-server \
  --model ~/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf \
  --device SYCL0 -ngl 999 -c 4096 --metrics \
  -ot 'blk\.(1[0-9]|[2-9][0-9])\.ffn_.*_exps\.=CPU' \
  --moe-cache-bytes 2147483648 --moe-cache-policy slru \
  --moe-cache-admission-misses 2 --port 45903
```

Then generate and read `curl localhost:45903/metrics | grep moe_cache`.
Drop `--moe-cache-*` for the off case; drop the env var for the default.
