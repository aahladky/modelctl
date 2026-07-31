# Per-op-type offload thresholds on a storage-bound MoE — 2026-07-30

**On the quant this stack actually serves, decode is 2.5× faster when routed
MoE ops are allowed to offload at batch 1.** The transfer cache accounts for
about half of that. Both findings invert what was measured on the smaller
quants, and neither generalises without checking.

## Setup

Runtime `build-sycl-f` at fork commit `724f705c1`. Model
**Qwen3.5-122B-A10B Q4_K_M**, 71.3 GiB across 3 shards — larger than VRAM
(43.8) and RAM (31) combined, so a large slice of the experts is reachable
only through mmap from NVMe. This is the quant served for quality work.

Placement was **derived for this quant with the cache budget reserved**,
not inherited:

| device | weights | cache | total | of free |
|---|---|---|---|---|
| SYCL0 | 23.1 GiB | 4.0 | 27.1 | 30.3 (89%) |
| SYCL1 | 6.1 GiB | 2.0 | 8.1 | 11.9 (68%) |
| CPU/mmap | 43.5 GiB | — | — | — |

Experts 0–12 on SYCL0, 13–15 on SYCL1, 16–47 through mmap. ctx 8192.
Measured through `modelctl_acceptance.run_matrix()`: one warmup pass, two
measured runs of 128 tokens, four distinct rotating prompts, temperature 0.

## Results

| | global min-batch | MoE min-batch | cache | decode | disk read | major faults |
|---|---|---|---|---|---|---|
| **A** | 32 | 32 | on | **3.77 t/s** | 395.9 GB | 773,822 |
| **B** | 1 | 1 | on | **9.38 t/s** | 144.9 GB | 69,860 |
| **D** | 32 | 1 | on | **9.36 t/s** | 151.8 GB | 80,249 |
| **E** | 32 | 1 | off | **5.81 t/s** | 160.1 GB | 81,207 |

Decomposed:

- **E vs A: +54%.** Moving routed MoE ops to the GPU at batch 1, with no
  cache at all.
- **D vs E: +61%.** The cache, on top of that. The largest cache gain
  measured in this project — against +21% on 35B-A3B and +48% on 122B
  IQ1_M.
- **D vs A: +148%.** Combined.
- **B vs D: +0.2%.** Nothing. See below.

## Why the default configuration is so slow

The fault counters carry the explanation, not the timings. At the default
threshold `MUL_MAT_ID` stays on CPU at batch 1, so every generated token
touches mmap'd expert weights across a 43.5 GiB working set against ~26 GiB
of available RAM. Condition A took **773,822 major faults and read 395.9 GB
from disk** — ten times the faults and 2.5× the reads of any other
condition. The machine is thrashing, and the throughput figure is measuring
that rather than compute.

Moving the expert work to the GPU stops the thrashing; the cache then stops
the experts being re-fetched.

## Contamination check

A model 2.3× larger than RAM warms the page cache monotonically, and the
page cache survives the process restart between conditions, so a later
condition can look faster for reasons unrelated to what it varies.

A failed on the first attempt, having run first against a cold cache, and
was **re-run last** — in the warmest position available. It still read 2.5×
more from disk and took 10× the faults. Its disadvantage is structural, not
positional. Among B, D and E the reads *increase* with run order (144.9 →
151.8 → 160.1 GB), the opposite of what warming would produce.

The comparison is clean.

## B vs D: the one null result, and why it matters anyway

The MoE-specific threshold bought nothing over the global one here. That is
not an argument against it:

| model | global threshold's cost | verdict |
|---|---|---|
| 122B-A10B IQ1_M (31.9 GiB, fits VRAM) | **−41%** | collateral damage to unrelated ops |
| 122B-A10B Q4_K_M (71.3 GiB, storage-bound) | **+54%** | no collateral damage measurable |

The MoE-only threshold is equivalent where the global one works and avoids
the 41% hole where it does not. It is also inert unless
`GGML_OP_OFFLOAD_MOE_MIN_BATCH` is set, so it cannot regress anything that
does not opt in.

## What this changes

**Phase G's premise no longer holds as stated.** G was justified by the
transfer cache being unable to serve interactive decode. On the storage-bound
target it serves it at +148% over the default. G's CPU-miss execution avoids
the PCIe round trip entirely and may still go further, but its cost/benefit
has to be re-derived against this baseline rather than against the default
configuration.

**The earlier "-12% net" result was specific to a quant that fits in VRAM.**
Where the model fits, lowering the threshold costs more than the cache
returns. Where it does not, both the threshold change and the cache pay
off substantially. Active expert bytes per token is not the only variable;
whether a miss costs a PCIe copy or a disk read matters more.

## Limits

One model, one quant, one placement. The sign of this effect has already
flipped once across quants — do not generalise it any further than the
IQ1_M result should have been generalised.

One measurement per condition (two requests averaged within each), so the
run-to-run spread is unknown.

The cache's hit ratio was **not** recorded: the harness captured storage
counters but not `moe_cache` counters at the time these ran. The +61% is an
A/B attribution — D and E differ in exactly one setting and nothing else
moved — not a hit-ratio measurement. Counter capture has since been added,
so a repeat would record it.

The cache initialised on **device 0 only** in every condition. Experts on
SYCL1 (layers 13–15) were served uncached. Why the device-1 cache never
initialised is unresolved; if it should have, the +61% understates the
cache.

## Reproducing

```bash
cd ~/workspace/moe-serving/modelctl
source /opt/intel/oneapi/setvars.sh --force
.venv/bin/python /home/aaron/tmp/moe-minbatch-correctness/run_offload_sweep.py
```

Placement is derived in that script with the cache reserved. Deriving for
the model alone and adding the cache afterwards overcommits SYCL0 by ~1 GiB
and loses the device mid-decode with `UR_RESULT_ERROR_DEVICE_LOST` — that
is what happened on the first attempt.
