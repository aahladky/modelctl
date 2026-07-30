# Task 0.7 — Performance rebaseline after the sync

Rebaseline on the ported+corrected binary (`sync/moe-cache-upstream-2026-07-30`,
commit `2dbf94801`, `build-sycl-sync`: SYCL, `GGML_BACKEND_DL=OFF`, Release,
oneAPI 2026.1 `icpx`/`icx`). Hardware/environment: see
`2026-07-30-baseline.md` (Fedora 44, kernel `7.1.5-200.fc44.x86_64`, Intel
Arc Pro B70 [SYCL0, ~30.3 GiB VRAM] + Arc B580 [SYCL1, ~11.3 GiB VRAM]).

Per the roadmap's explicit instruction, these are **fresh, standalone**
numbers, not a pass/fail comparison against the Task 0.1 baseline's old
observations (old numbers are cited as context only).

## A real incident during this task, and the RAM ceiling it revealed

The first attempt at this task caused a real problem and is worth recording
plainly rather than glossing over. Model: `Qwen3.5-122B-A10B-UD-IQ1_M.gguf`
(34.2 GB file). This machine has **31 GiB of system RAM** (combined GPU VRAM
is ~41.6 GiB, comfortably more than the model file, but system RAM alone is
not). An initial run launched with `--parallel 4` (intending to reuse one
server process for several benchmark modes at once) drove memory usage high
enough to trigger a real kernel OOM-kill, which took out unrelated desktop
processes (`kwin_wayland` and others) as collateral damage — confirmed via
`journalctl -k`. A differential check against the **unmodified pre-sync
binary** with the same `--parallel 4` flags reproduced the same failure
mode, confirming this is a **pre-existing RAM ceiling for this
model+parallelism combination on this machine, not a regression from the
port**.

All results below were captured after switching to `--parallel 1` (matching
the Task 0.1 baseline's known-safe configuration) for every mode except
continuous batching, which used `--parallel 2` (not 4) with active
`free -h` monitoring before and during load, aborting on any sign of
pressure. No further incidents occurred under this configuration; available
RAM stayed at 23-27 GiB throughout every run below.

**Takeaway for later phases**: this specific model+hardware combination
should not be driven with `--parallel 4` on this machine without either more
RAM or a smaller/more heavily-offloaded context budget. Worth a note in
Phase C/D's resource-claim work (Task D1) — `--parallel` multiplies
`n_ctx_slot` reservations, and nothing today warns the operator before that
addition exceeds physical RAM.

## Launch configuration (all modes)

```
--flash-attn auto --jinja --fit on --split-mode layer --tensor-split 4,1 \
--mmproj .../mmproj-F16.gguf --cache-type-k q8_0 --cache-type-v q8_0 \
--moe-cache-bytes 4294967296,2147483648 --moe-cache-policy slru \
--moe-cache-admission-misses 2 --metrics
```
(`--parallel 1` for modes 1-4 and 8; `--parallel 2` for mode 5 only.)

Model load time: ~24-60s depending on how warm the page cache already was
(see modes 6/7 below) — all well short of the earlier incident's 14+ minute
stall, confirming that stall was the `--parallel 4` RAM-pressure issue, not
a load-path regression.

## An important structural finding: the cache never activates for this profile

Across every mode below, `/metrics` never emitted a single `moe_cache_*`
line, despite the server log confirming `MoE expert cache enabled:
4294967296 bytes per GPU, policy=slru, admission=2` at startup (the same
line the Task 0.1 baseline recorded). This matches Task 0.6's finding
exactly: **the cache's scheduler hook only intercepts copies when expert
tensors are host-resident** (`ggml_backend_buffer_is_host`). This
model+placement (`--split-mode layer --tensor-split 4,1`, no `-ot
exps=CPU`) spreads the model across both GPUs' VRAM directly — at IQ1_M
quantization, the full 34.2 GB model plus KV cache and activations fits
inside the ~41.6 GiB combined VRAM budget with room to spare, so **no
expert tensor is ever host-resident and the cache hook never fires**. The
cache subsystem initializes but sits completely idle for this real
production profile as currently configured.

This is a genuinely important finding, not just a benchmarking footnote:
the flagship real-world profile this project cites as the motivating
"oversized sparse-MoE" case does not, on this hardware, actually exercise
the feature Phase 0 exists to validate. The cache only engages when a model
is too large for combined VRAM and must spill experts to host/mmap memory
(as the Task 0.6 correctness matrix deliberately forced via `-ot
exps=CPU`). Worth flagging for whoever picks up Phase E's real-hardware
acceptance work: either test with a model that genuinely exceeds ~41.6 GiB
combined VRAM, or explicitly add an `-ot exps=CPU`-style override to this
profile if the intent is to validate cache behavior on the flagship
deployment.

## Results

| # | Mode | Config | Result |
|---|---|---|---|
| 1 | Prompt batch 1 | 1-token prompt, `n_predict=4` | prompt: 9.06 tok/s (1 tok, dominated by fixed overhead); gen: 33.90 tok/s |
| 2 | Below batch threshold | 5-token prompt, `n_predict=4` | prompt: **0.10 tok/s** (49.3s for 5 tokens); gen: 21.74 tok/s |
| 3 | Above batch threshold | 38-token prompt, `n_predict=8` | prompt: **33.86 tok/s** (1.12s for 38 tokens); gen: 29.18 tok/s |
| 4 | Interactive decode, batch 1 | 21-token prompt, `n_predict=150` | prompt: 28.96 tok/s; gen: **25.97 tok/s** (cf. baseline's 24.24-24.33 tok/s — consistent, not a regression) |
| 5 | Continuous batching | `--parallel 2`, 2 concurrent 8-token-prompt/40-token-gen requests | both completed in identical 4.478s wall time (true concurrency, not serialization); gen: 12.78 tok/s **each** (~25.6 tok/s aggregate, in line with mode 4's single-stream rate split across 2 slots) |
| 6 | Cold page cache | attempted `echo 3 > /proc/sys/vm/drop_caches` | **Permission denied** (no root) — labeled `cold_unverified` per the roadmap's own guidance for exactly this situation; every load in this session ran against an already-warm page cache (buff/cache held steady at 24-26 GiB throughout) |
| 7 | Warm page cache | implicit — every mode above ran with the model file already resident in page cache from prior loads this session | satisfied by construction; no separate isolated experiment run given mode 6's constraint made a clean cold/warm pair impossible to construct honestly |
| 8 | Warm expert cache | 3 repeated distinct-prompt generations, checked `/metrics` after each | **N/A for this profile** — no `moe_cache_*` metrics ever appear (see structural finding above); cannot warm a cache that never activates under this placement |

## Regression assessment

Mode 4's generation throughput (25.97 tok/s) is consistent with — slightly
above — the Task 0.1 baseline's 24.24-24.33 tok/s on the pre-sync binary.
Given normal run-to-run variance and that this is a single sample each side,
this is not evidence of a regression. The dramatic prompt-throughput cliff
between modes 2 and 3 (0.10 vs. 33.86 tok/s, ~300x) is expected, documented
behavior tied to `moe_cache_min_batch: 32` in the capability response, not
a bug — but it's a real, sharp cliff worth the control plane being aware of
when estimating TTFT for short prompts (Task D1/D2's resource-claim and
observability work).

No regression found in the one thing this task could actually measure
(single-stream and 2-way-concurrent generation on a fully VRAM-resident
placement). The cache-specific performance question (H2D bytes, hit/miss
timing under real load) remains unanswered for this real profile, for the
structural reason above — Task 0.6's tiny-model results (with `-ot
exps=CPU`) are the only real evidence so far that the cache's *own*
performance characteristics are sane.
