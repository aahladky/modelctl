# Promotion report, 2026-07-31 (3): the remediation pair

Promotes the pair produced by the 2026-07-31 code-review remediation:

- modelctl `b5a0c44` (eight merged remediation branches)
- llama.cpp `9accaf4fb` (`feature/sycl-moe-expert-cache`, fork PR #1)

Previous validated pair: `1bf57c7` + `b05cc90dd`. It stays the rollback
target until this one is superseded.

## Why this promotion exists

A review of the stack found ~60 issues, two of which corrupted output on
this hardware; several others silently degraded the control plane. The
fixes span both repos, so the pair moves together.

## Verified on this machine

Hardware: 2x Intel Arc (SYCL0 = B70, SYCL1 = B580), oneAPI 2026.1,
`build-sycl` rebuilt from `9accaf4fb` with icx/icpx.

| Check | Result |
|---|---|
| `test-moe-cache`, `test-moe-hybrid` (host) | pass |
| Both under ASan/UBSan (clang) | pass |
| CPU-only build capability truthfulness | every SYCL/cache feature false |
| Capability probe, SYCL build | schema 3; cache/hybrid/per-device true; SYCL0+SYCL1 enumerated |
| `run_kquant.sh` (Q4_K, host-resident experts, `MOE_MIN_BATCH=1`) | cache output **token-identical to baseline**; hybrid deterministic |
| Per-device budgets, real model (OLMoE-1B-7B Q4_K_M) | `dev0=2147483648, dev1=1073741824 (unnamed devices: none)`; coherent generation |
| modelctl suite | 1051 passed, 11 skipped |
| `modelctl capabilities <sycl binary>` | status ok, schema 3 |

## The defect this pin exists to fix

`run_kquant.sh`'s first hardware run failed on the PREVIOUS pin in a way
that had gone unnoticed: with host-resident Q4_K experts at
`GGML_OP_OFFLOAD_MOE_MIN_BATCH=1`, **all three legs -- including the
no-cache baseline -- produced identical degenerate output** (one token,
repeated).

Cause: scheduler input copies are refilled from the host on every graph
run, but the one-time in-place K-quant reorder sets a sticky flag, so
from the second run onward fused kernels read raw GGUF-layout bytes as
if they were reordered. It applies with or without the expert cache,
which is exactly why it never looked like a cache regression and never
surfaced in cache-focused testing.

Also fixed here: the hybrid CPU tier's H2D result copies were issued
from stack buffers that died before the copies completed (silent logit
corruption the F32 fixture could not see); hybrid staging skips were not
gated on admission eligibility, so a cold prefill degenerated into
per-row scalar CPU GEMVs; and an abandoned graph could leave a hybrid
plan for a later tensor at a reused device address to consume.

## Not covered

- The full acceptance matrix has not been re-run. This covers
  correctness and the new per-device budget path, not the throughput
  comparison set.
- Release A baselines taken before today were produced with a broken
  `-ot` regex (it pinned the LOWER half of expert layers on models with
  >= 20 blocks). They are superseded, not reinterpreted.
- `moe_cache_prefetch` remains unimplemented; normalization still forces
  it false.
