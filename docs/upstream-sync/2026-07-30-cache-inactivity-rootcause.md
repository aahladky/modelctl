# Root cause: why the MoE cache showed zero activity on the real qwen35moe model

Follow-up to `2026-07-30-real-model-cache-activation.md` (Task 0.7b), which
found zero `moe_cache_*` metrics on the real `Qwen3.5-122B-A10B-Q4_K_M`
model even with `-ot exps=CPU` forcing genuine host residency, and
hypothesized the cause was something structural about `qwen35moe`'s hybrid
SSM/shared-expert architecture. **That hypothesis is wrong.** The real
cause is much simpler, fully explained, and not a bug in the cache at all.

## The mechanism, with direct evidence

The cache's scheduler hook (`copy_experts` in
`ggml/src/ggml-backend.cpp:1643`) only fires when a weight tensor is
**copied across backends** — specifically when `ggml_backend_sched_compute_splits`
schedules a host-resident weight to be copied into a split running on a
different backend (see the trigger condition at
`ggml/src/ggml-backend.cpp:1590-1596`, gated on
`ggml_backend_buffer_get_usage(...) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`,
`ggml_backend_buffer_is_host(...)`, and `node->op == GGML_OP_MUL_MAT_ID`).

Whether that cross-backend copy happens at all is decided earlier, by
`ggml_backend_sched_backend_id_from_cur` (`ggml/src/ggml-backend.cpp:906+`,
untouched by the upstream commit `dee2a846b` flagged during Task 0.2/0.3 —
that commit only added a `FLASH_ATTN_EXT` exception, confirmed by reading
its diff; it does not affect `MUL_MAT_ID` scheduling). For a host-resident
weight, this function asks the compute backend whether it wants to
"offload" the op via `ggml_backend_offload_op`. The SYCL implementation
(`ggml/src/ggml-sycl/ggml-sycl.cpp:6410-6412`):

```cpp
static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ...
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}
```

`op_offload_min_batch_size` defaults to **32**
(`ggml/src/ggml-sycl/ggml-sycl.cpp:6812`, overridable via
`GGML_OP_OFFLOAD_MIN_BATCH` env var) — this is exactly where the
`moe_cache_min_batch: 32` value in the capability response comes from. For
`GGML_OP_MUL_MAT_ID`, `get_op_batch_size` returns `op->ne[2]`, i.e. the
number of tokens being routed through the expert layer in that single
graph evaluation (`ggml/src/ggml-sycl/ggml-sycl.cpp:6396-6406`).

**If fewer than 32 tokens are being processed in a single pass, the op is
assigned to run directly on CPU** (same backend the weight already lives
on) — no cross-backend copy is scheduled, so the hook is never reached.
This is completely independent of model architecture; it's a general
ggml scheduler heuristic (avoid GPU-offload overhead for batches too small
to be worth it) that predates this feature and applies to every model.

Confirmed directly with `GGML_SCHED_DEBUG=2` tracing on the real
`build-sycl-sync` binary against the real Q4_K_M model:

- **Short prompt (~20 tokens) + `n_predict=2`**: scheduler trace shows
  `SPLIT #2: CPU` containing `ffn_moe_gate`/`ffn_moe_up`/`ffn_moe_down`
  (all three routed-expert `MUL_MAT_ID` nodes) running on **CPU**. Every
  token in the request — prompt and decode alike — stayed under the
  32-token batch threshold, so the whole thing computed on CPU with no
  cache activity, matching Task 0.7b's zero-metrics finding exactly.
- **Long prompt (105 tokens) + `n_predict=4`**: same binary, same model,
  same `-ot exps=CPU`, same cache flags. `/metrics` after the request:

  ```
  moe_cache_misses_total{device="SYCL0"}     10402
  moe_cache_evictions_total{device="SYCL0"}   4392
  moe_cache_promotions_total{device="SYCL0"}  5201
  moe_cache_slots_used{device="SYCL0"}         809
  moe_cache_h2d_bytes_total{device="SYCL0"} 9203023872
  ```

  Real, substantial cache activity — 9.2 GB of host-to-device traffic —
  once prompt processing had enough tokens in a single batch to cross the
  offload threshold.

## Conclusion: not a bug, not architecture-specific, was a test-design gap

**The cache works correctly on the real `qwen35moe` production model.**
Task 0.7b's "zero activity" result was an artifact of using short test
prompts that never crossed the 32-token GPU-offload threshold — the same
threshold that (correctly) also explains why single-token decode alone
essentially never activates the cache: autoregressive decode is batch=1
by nature, always below 32, and will always run expert compute on CPU
directly regardless of this feature. The cache is only reachable when a
single graph evaluation processes ≥32 tokens through the routed-expert
layer at once — realistically, prompt processing (any prompt with a
few dozen+ tokens) or continuous batching pooling enough concurrent
requests together. Task 0.6's synthetic-model correctness matrix worked
precisely because it deliberately used "4 distinct long-form prompts each
> 32 tokens" (per its own methodology notes) — that wasn't incidental, it
was the necessary condition, and Task 0.7b's methodology just didn't carry
that same care over to the real-model test.

No code change was made. `ggml_backend_sched_backend_id_from_cur` and
`ggml_backend_sycl_device_offload_op` are general-purpose scheduler
heuristics shared by every op and every model, not something specific to
this feature, and changing the default threshold or adding a bypass for
the MoE cache specifically would be a real design decision (does the cache
want to force-offload small batches despite the overhead? is 32 still the
right default given the cache changes the cost/benefit tradeoff of
offloading?) that shouldn't be made unilaterally here — flagging for
whoever owns Phase F if it's worth revisiting, not fixing now.

## Worth carrying forward

- **Operational note, not a defect**: for typical single-user interactive
  chat (one request, short-to-medium prompts, batch-1 decode), this cache
  will mostly sit idle during decode by design — it earns its keep during
  prompt processing and under continuous batching with enough concurrent
  load to cross the batch threshold, not during lone interactive decode.
  Worth stating plainly in any user-facing documentation of the feature so
  expectations are set correctly (Task C3's "cache budget and eligibility
  constraints" plan-card field is the natural place for this).
- **Test methodology note**: any future real-model validation of this
  cache (Phase E) must use prompts long enough to cross
  `op_offload_min_batch_size` (32 by default) during prompt processing, or
  it will silently and misleadingly look inactive regardless of whether
  the feature itself works.
