# Task 0.7b — Cache activation on a real oversized MoE model

Supplementary to Task 0.7, which found the MoE cache never activates for the
flagship production profile (`qwen3.5-122b-iq1m`, IQ1_M, 34.2GB) because it
fits entirely in combined VRAM. This test used a bigger quant of the same
model family specifically to force host-resident experts and get real
activation data.

## Setup

- Model: `Qwen3.5-122B-A10B-Q4_K_M` (3 shards, ~76.6GB total — bigger than
  this machine's combined VRAM+RAM, ~68.6GB), read-only.
- Binary: `build-sycl-sync/bin/llama-server`, commit `2dbf94801` (ported +
  architecture-corrected).
- Launch: `--split-mode layer --tensor-split 4,1 -ot "exps=CPU" --cache-type-k
  q8_0 --cache-type-v q8_0 --parallel 1 --metrics --moe-cache-bytes
  4294967296,2147483648 --moe-cache-policy slru --moe-cache-admission-misses 2`
- Standalone test on port 18500, no modelctl/llama-swap involvement, no real
  profile touched.

## RAM safety trace

| Event | Available RAM |
|---|---|
| Pre-launch | 27.0 GiB |
| +10s (loading) | 29.2 GiB (buff/cache still filling) |
| +20s (loading) | 29.0 GiB |
| +30s (model loaded, 38s total) | 28.9 GiB |
| After prompt 1 | 29.0 GiB |
| After prompt 2 | 28.7 GiB |
| After prompt 3 | 28.3 GiB |
| Post-shutdown | 27.0+ GiB (process cleanly killed, RSS was ~9.3GB) |

No safety threshold was approached at any point — available RAM never
dropped below 27 GiB. `/proc/<pid>/io` `read_bytes` climbed steadily during
load (65GB → 88GB → 109GB across three 10s samples), confirming genuine
large-file mmap streaming, not a stall. Model loaded in 38s — actually
*faster* than the original 34GB IQ1_M baseline's ~32s load given the size
difference, likely because most of the ~50GB+27GB expert-tensor shards
never needed to be read at load time (mmap defers actual page-in to first
use, and `-ot exps=CPU` doesn't force-fault them upfront).

Process was cleanly terminated with `SIGTERM` and verified gone via `ps -p`
(not just `kill -0`, per the lesson from the earlier false-negative).

## Result: model loads and runs correctly, but the cache still shows zero activity

Three completions (of five attempted; two were cut off by a 3-minute shell
timeout on my end, not a server problem — the server was confirmed alive and
responsive throughout) ran successfully at real, honest throughput:

- Prompt processing: 0.95–1.6 tok/s (vs. IQ1_M's 28.96–33.86 tok/s)
- Generation: 1.10–1.26 tok/s (vs. IQ1_M's 25.97 tok/s)

This ~20-25x slowdown is expected and correct — real CPU-resident expert
compute + real H2D staging, not a regression (there is no faster path
without cache acceleration actually kicking in).

**But `/metrics` never emitted a single `moe_cache_*` line — not even a
zero-valued one** (compare: Task 0.6/0.8's tiny synthetic model reliably
showed real non-zero counters under the same `-ot exps=CPU` flag). The
server log confirms `MoE expert cache enabled: 4294967296 bytes per GPU,
policy=slru, admission=2` at startup, same as always — the subsystem
believes it's active, but produced no observable hit/miss/eviction/H2D
evidence across three real generations against a genuinely host-resident,
mmap-backed multi-shard model.

## This is not the same finding as Task 0.7, and looks more structural

Task 0.7's gap was "cache inactive because nothing is host-resident" — an
explainable, expected consequence of the profile's placement. Here,
`-ot exps=CPU` **was** used and load-time evidence (slow ~1-1.6 tok/s
throughput, real page-in behavior) confirms experts really are host-resident
and being computed on. The cache should have had things to intercept and
didn't.

Checked the obvious theory — fused gate/up tensors (Task F6 explicitly
flags "handle fused gate/up tensors ... fail closed on unknown layouts" as
an open gap) — and **ruled it out**: extracted real tensor names directly
from the model file (`blk.0.ffn_gate_exps.weight`, `blk.0.ffn_up_exps.weight`,
`blk.0.ffn_down_exps.weight`), and they match the standard split naming the
cache's hook already parses (`ggml-sycl.cpp`'s own comment: `// Parse layer
index from tensor name like "blk.5.ffn_gate_exps"`), not a fused name.

What the tensor names *do* reveal: `general.architecture = qwen35moe` is a
**hybrid SSM/Mamba + MoE architecture** — layer 0 has `ssm_a`, `ssm_alpha`,
`ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out` tensors alongside the MoE FFN
tensors, plus a **shared-expert** path (`ffn_gate_shexp`/`ffn_up_shexp`/
`ffn_down_shexp`/`ffn_gate_inp_shexp`, always active per token) *in addition
to* the routed experts (`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`,
top-k routed) that the cache targets. This is structurally different from
Task 0.6's tiny synthetic fixture, which was a plain mixtral-style MoE with
no SSM layers and no shared expert.

**I did not go further to find the exact mechanism inside the scheduler/hook
code** (out of scope for this task) — but the naming-convention theory is
ruled out with direct evidence, and the hybrid SSM+shared-expert graph
structure is the most likely remaining explanation: something about this
architecture's graph shape (possibly how the shared-expert path or SSM
layers interact with `ggml_backend_sched_compute_splits`'s split/copy
boundaries) may prevent the `copy_experts` hook from being reached, or
reach it with a tensor-name pattern this codebase hasn't been exercised
against for real. This is a real, unresolved gap — worth a dedicated
investigation before Phase E, since `qwen35moe` is literally the user's
production model family.

## Bottom line

No RAM safety issue this time (stayed at 27-29 GiB available throughout,
comfortable margin). Model loads and generates correctly at real (slow, as
expected) host-compute-bound throughput. But the cache — which should have
real work to do here — shows **zero observable activity**, a more
significant and structural finding than Task 0.7's "doesn't fit the
placement" result. This should block any claim that the cache works on
`qwen35moe`-family models until root-caused.
