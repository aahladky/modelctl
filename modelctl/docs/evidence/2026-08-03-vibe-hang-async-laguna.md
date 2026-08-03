# Vibe session 2026-08-02/03 — hang root cause, async fills A/B, laguna constant-VRAM swap

Raw numbers. Fork work on `vibe/async-fills` (three commits:
e1957ebed async fills + stale-plan purge, e5185d1a0 pool race fix,
then the default flip). Binary `build-sycl/bin/llama-server` rebuilt per
commit. Machine: plasma-discover consumed ~2 cores until ~22:50, killed
by Aaron; per-run load traces below mark which numbers it poisoned.
Environment verified on the box: oneAPI 2026.1.1-325 ALREADY installed
(open-items line saying the update was pending is stale),
compute-runtime 26.22.38646.6, kernel 7.1.5-200.fc44, Mesa 26.1.5,
UR L0 adapter libs v1 AND v2 both present.

## 1. The C3 hang, root-caused live

My async-fill build hung at the maiden protocol point (35B C3, task
launched after /cache/reset, server at 47% CPU for 3+ min). gdb on the
hung process, fork commit e1957ebed:

- compute thread: parked in `moe_cpu_execute_gemvs` -> `m_done.wait`
- all 25 pool workers: parked in `worker_main` -> `m_wake.wait`
- pool state at 0x7f8bb99a8710: `m_pending = 0xffffffff (-1)`,
  `m_generation = 6963`, `m_helpers = 25`, `m_body` non-null

Mechanism: a worker preempted between adopting the generation
(`seen = gen`) and reaching the body-read lock finds a LATER batch
published there, executes it under the stale generation, then adopts
the new generation and executes it again -- two decrements for one
counted slot, the counter goes negative, and `run()` waits forever on
`== 0`. Load-dependent (needs an ill-timed preemption), which matches
the 1-in-5 maiden intermittency and the never-on-fixture behaviour.

The chip-fdfdd66d hypothesis (stale `m_hybrid_plans` surviving reset)
was NOT the hang. Plan leakage across graphs is real (record-never-take
merges into reused staged bases via `operator[]`) and is fixed
separately (step-end purge, counted in `hybrid_stale_plans_purged`),
but the hang was the pool race.

Fix (e5185d1a0): `m_batch_gen` published with the body; workers only
take a body whose generation matches the one they adopted. Defense in
depth: caller predicate `<= 0` plus a one-shot stderr warning if the
count ever goes negative again. 8 subsequent C3 runs (2 contaminated +
6 quiet), zero hangs. Host suites + ASan/UBSan green; fixture rail
8/8 token-identical across hybrid-off / sync-fill / async-fill.

## 2. Async admission fills: paired A/B says 31% SLOWER

Design v1: promote only reserves a slot; the copy runs on a second
in-order queue at step end, one `ext_oneapi_submit_barrier` per fill on
the compute queue for ordering; slot invisible until its event
completes.

Paired A/B, maiden 35B argv verbatim, arms differ only in
`GGML_MOE_CACHE_ASYNC_FILL`, arm order alternating within pairs,
quiet machine (loadavg ~1.1-2.5), 3 pairs per condition:

| cond | pair | ran first | async tok/s | sync tok/s | delta |
|---|---|---|---|---|---|
| C2 | 0 | async | 14.3356 | 20.9057 | -6.5701 (-31.4%) |
| C2 | 1 | sync  | 14.3273 | 20.9660 | -6.6387 (-31.7%) |
| C2 | 2 | async | 14.3188 | 20.8928 | -6.5741 (-31.5%) |
| C3 | 0 | async | 13.3712 | 19.7800 | -6.4088 (-32.4%) |
| C3 | 1 | sync  | 13.1725 | 19.6301 | -6.4576 (-32.9%) |
| C3 | 2 | async | 13.4027 | 19.6112 | -6.2086 (-31.7%) |

0/6 pairs positive. Async hit ratio 0.6666/0.6642 vs sync
0.6904/0.6860 (fills land a step late). Async h2d_bytes on the compute
stream: 0 (mechanism worked); sync: 6.55/6.67 GB per run. Async
throughput was IDENTICAL under loadavg 5 and loadavg 1 -- a fixed
self-inflicted per-step cost (per-fill barriers, pageable-source copies
on the second queue), not contention. Default flipped to sync;
`GGML_MOE_CACHE_ASYNC_FILL=1` re-enables for future work.

Reading: the in-order compute queue was already pipelining sync fills;
"fills serialize in front of compute" is disproven as the C2/C3
bottleneck on this stack.

Note: sync C2 20.9 / C3 19.7 tonight vs 24.82/23.71 maiden anchors --
same protocol, quieter-but-not-identical machine plus three new commits
in the binary; not differenced against the anchors here.

## 3. Laguna-S-2.1 at constant VRAM: static residency vs cache budget

First-ever laguna-on-fork cache run. Fork binary throughout. Canary:
fork loads laguna with the live argv (44.2 s vs pinned 50.0 s) at
11.90 vs 11.65 tok/s; TOKENS DIVERGE between binaries (pinned
04b2b72cb predates the oneDNN determinism pin among months of drift) --
arms below are fork-vs-fork and internally comparable; nothing here
compares the fork to the live anchor tokens.

Per-block routed-expert bytes from the GGUF (gguf_placement.py,
cross-checked type-table vs offset-delta): 1,145,044,992 B for blocks
1-45, more for 46-47; 4,472,832 B per expert slot (gate 1,351,680 +
up 1,351,680 + down 1,769,472), 256 experts, 10 used, 48 blocks.

Arms (ctx 64000, kv q8/q8, tensor-split 22,10, shexp pins unchanged):

- L1: live -ot verbatim (exps 1-19 SYCL0, 20-28 SYCL1, rest CPU)
- L2: exps 1-13 SYCL0, 20-26 SYCL1; freed blocks -> cache budgets
  SYCL0=6,870,269,952 (blk 14-19), SYCL1=2,290,089,984 (blk 27-28);
  slru, admission 2, prefill admission off,
  GGML_OP_OFFLOAD_MOE_MIN_BATCH=1 (global floor untouched at 32)
- L3: L2 + --moe-hybrid-mode on

Warmup 32, engagement verified, one /cache/reset, then 3 x 256-token
greedy back-to-back with NO resets (run 0 = warming, runs 1-2 =
steady). Loads: L1 44.6 s, L2 49.7 s, L3 47.4 s.

| arm | run 0 | run 1 | run 2 | hit ratio (end) | h2d/run | staging avoided/run |
|---|---|---|---|---|---|---|
| L1 static | 12.6609 | 13.1961 | 13.6483 | — | — | — |
| L2 cache | 5.7178 | 6.6408 | 5.9880 | 0.6963 | ~40 GB | 0 |
| L3 cache+hybrid | 6.4323 | 7.1663 | 7.3414 | 0.6977 | ~40 GB (admissions) | ~70 GB |

L1 climbs run-to-run (page cache warming, matches the 08-01 pattern
toward the 13.49/14.20 anchors). L3 load traces include the hybrid CPU
tier's own threads (loadavg to 10.2 during runs).

Multi-GPU confirmation: only `moe_cache: created on device 0` appears.
SYCL1's budget parsed but its cache is unreachable -- the scheduler's
offload pass sends every CPU-resident expert op to the first capable
backend, so ALL 27 CPU blocks' expert traffic (~28.8 GiB working set)
funneled through SYCL0's 6.87 GB pool (24% coverage -> 0.70 hit; the
concentration is real). SYCL1's 2.1 GiB budget sat unallocated, so
L2/L3 ran with ~2.1 GiB less VRAM in use than L1; noted, direction is
against the cache arms but small against a 1.9x gap.

Steady-state verdict at equal (nominal) VRAM on the model class this
stack exists for: static 13.6 vs cache+hybrid 7.3 -- the cache arm
loses ~1.9x, with 0.70 hit ratio and ~156 MB/token of admission churn
over PCIe. Churn removal alone cannot close it: 40 GB/run at the
measured effective rate is ~3-4 s of a ~36 s run; the remaining gap is
the per-op offload machinery itself (~1,300 expert-projection staging
decisions + D2D hit copies per token vs zero for static residency).

## 4. Session state

- Fork: `vibe/async-fills` at 3 commits ahead of the pin; pin NOT
  moved; superproject untouched except this evidence file.
- checks.sh full: every check green except the deliberate
  pin-vs-worktree mismatch; suite 1954 passed / 11 skipped, ASan/UBSan
  green.
- Not run tonight: Vulkan-vs-SYCL A/B (build was compiling at session
  end), prefill-path measurements (cache prefill value untested here),
  35B C1 re-anchor on the new binary.
