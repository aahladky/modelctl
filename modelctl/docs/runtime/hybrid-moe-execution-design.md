# Hybrid MoE Execution Design

**Status:** Design + control-plane implementation  
**Target:** Release B — Interactive hybrid sparse-MoE serving

## Goal

Execute GPU-cache hits on persistent cached expert tensors and
CPU/RAM/mmap misses on CPU-resident expert weights in the same
batch, merging outputs correctly. A miss must NOT automatically
transfer full expert weights to GPU (that's the transfer-cache
behavior from Release A).

## Architecture Overview

```
Router selects (token, expert) pairs
        │
        ▼
┌─────────────────────────────┐
│   Partition Builder (GPU)   │
│  - Classify hit vs miss     │
│  - Build row index sets     │
│  - Preserve deterministic   │
│    row ordering for merge   │
└──────┬──────────┬───────────┘
       │          │
   ┌───▼───┐  ┌──▼────────┐
   │ GPU   │  │ CPU miss  │
   │ hit   │  │ execution │
   │ path  │  │ (existing │
   │       │  │  kernels) │
   └───┬───┘  └────┬──────┘
       │           │
       └─────┬─────┘
             ▼
    ┌─────────────────┐
    │  Output Merge   │
    │  (per-token     │
    │   weighted sum) │
    └─────────────────┘
```

## Key Questions Resolved

### Q1: Where are router-selected token/expert pairs available?

In the MoE FFN layer, after the router/gate projection produces
top-k expert indices and weights. The `mul_mat_id` operation
dispatches to selected experts. This is where we partition.

### Q2: Can CPU and SYCL mul_mat_id operate on disjoint row subsets?

Yes. Both CPU and SYCL `mul_mat_id` implementations accept a set
of `(row, expert)` pairs. We can split the set into GPU-resident
(cache hit) and CPU-resident (cache miss) subsets and execute them
on their respective backends.

### Q3: What is the minimum synchronization boundary?

Per-layer. All expert outputs for one layer must be available before
the next layer's attention. Within a layer, GPU and CPU work can
overlap if we use events/queues rather than global sync.

### Q4: How are shared experts handled?

Shared experts (present in DeepSeek-V2/V3/R1) are always GPU-resident
when possible. They don't participate in the cache hit/miss partition
because they're not routed — they apply to every token.

### Q5: How are quantized expert tensors represented?

GPU cache stores dequantized (f16/bf16) expert projections for direct
use in SYCL GEMM. CPU miss path uses the original quantized GGUF
tensors with existing CPU dequant+GEMM kernels.

### Q6: Can one activation select different paths per expert?

Yes. The partition builder operates per (token, expert) pair. Token 0
may have experts [3, 7] where 3 is a cache hit (GPU) and 7 is a miss
(CPU). Token 1 may have all hits or all misses.

### Q7: What batch-one path bypasses the GPU split hook?

The current SYCL MoE hook only activates for batch sizes >=
`moe_cache_min_batch` (default 32). For batch-one decode, the
default path is all-GPU (or all-CPU for -cmoe). The hybrid path
must handle batch-one by always entering the partition logic.

## Implementation Strategy

### Phase 7.3 — Partition Builder

Extract a compact representation of selected `(token_row, expert)`
work before expert computation:

```
struct moe_partition {
    // Hit rows: indices into the activation matrix where the
    // selected expert is in the GPU cache.
    int *hit_rows;      // [n_hits]
    int *hit_experts;   // [n_hits]  expert IDs for hit rows
    int  n_hits;

    // Miss rows: indices where the expert must be computed on CPU.
    int *miss_rows;     // [n_misses]
    int *miss_experts;  // [n_misses] expert IDs for miss rows
    int  n_misses;

    // Merge info: maps output rows back to original positions.
    int *original_rows; // [n_hits + n_misses]
};
```

### Phase 7.4 — CPU Miss Execution

Execute miss rows against CPU-resident/mmap-backed expert weights.

- Reuse existing quantized CPU `mul_mat_id` kernels.
- Avoid materializing complete expert copies.
- Track CPU execution time and page faults.
- Do NOT force synchronous GPU weight copy.

### Phase 7.5 — GPU Hit Execution

Use persistent cached expert tensors directly.

- Submit work on SYCL queue with events.
- Do not globally synchronize after every expert.
- Account for selected-expert reuse within same batch.

### Phase 7.6 — Output Merge

Allocate output that supports disjoint CPU/GPU contributions:

```
// For each output row:
// output[row] = sum over selected experts:
//     expert_weight * expert_output[expert][row]
//
// GPU hits contribute from cache tensors.
// CPU misses contribute from CPU execution results.
// Merge preserves accumulation order for numerical stability.
```

### Phase 7.7 — Async Admission

Record misses without blocking current-token completion.
Promote eligible experts after miss result is no longer dependent
on staging buffers. Rate-limit promotion to avoid saturating H2D.

## Safety

- Hybrid mode is OFF by default.
- Requires explicit `moe_hybrid_cpu_miss: true` in capabilities.
- Requires recognized expert geometry.
- Falls back to transfer-cache or static placement on failure.
- All-hit and all-miss cases must work without special paths.
