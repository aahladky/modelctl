# Maiden night-lane runs — 2026-08-02

Raw numbers only. Run by explicit instruction, ahead of the quiet window
these were queued for.

Two things went wrong that are part of the record rather than footnotes
to it: the 122B weights were deleted from `~/models` mid-battery, and one
C3 run hung. Both are below.

## Machine

- Rebooted **01:10:49** (unexplained; prior uptime 2 d 12 h). The first
  runs therefore started on a cold page cache.
- `llama-swap` held **zero models** throughout (`{"running":[]}`), was
  not restarted, not reconfigured. Console `:9293` untouched.
- Binary: `llama.cpp/build-sycl/bin/llama-server`, fork commit
  `85b7e6556`, verified same morning to report that commit,
  `moe_cache_mmap_advise: true`, `GGML_SYCL_DETERMINISTIC` default 1.
- Every run: fresh server on scratch port 18147, torn down by PID, port
  confirmed free afterwards. No saved profile, artifact or llama-swap
  config was written.
- **Load traces are per run**, sampled every 5 s in a thread beside the
  decode — the gap that voided the 2026-08-01 figures.

## Protocol

Verbatim from the 2026-08-01 determinism record's own `argv` fields and
`protocol.prompt`: fixed prompt, greedy (`temperature 0`, `top_k 1`),
`seed 42`, `cache_prompt=false`, 32-token warmup, then for cache
conditions engagement verified (`moe_cache_learning == 0` AND
`misses_total > 0`) and `POST /cache/reset`, then a 128-token measured
decode with `n_probs=8`.

```
-ngl 999 -c 4096 --split-mode layer --tensor-split 8,3 --cache-type-k q8_0
--cache-type-v q4_0 --flash-attn auto --jinja --parallel 1 --fit off
--device SYCL0,SYCL1 -ot ffn_.*_exps=CPU --no-warmup --ubatch-size 128
```

| cond | adds | env |
|---|---|---|
| C1 | — | — |
| C2 | `--moe-cache-bytes SYCL0=4294967296 --moe-cache-policy slru --moe-cache-admission-misses 2 --moe-cache-prefill-admission off` | `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1` |
| C3 | C2 + `--moe-hybrid-mode on` | same |

`GGML_OP_OFFLOAD_MIN_BATCH` (the global floor) was never set by any arm;
the harness refuses to launch if it is below 32.

Paired jobs alternate arm order per pair and difference **within** each
pair. `delta = det-off − det-on`; a negative delta means the det-off arm
measured fewer tok/s.

## The model file

`Qwen3.5-122B-A10B-UD-IQ1_M.gguf` (31.87 GiB) was **deleted from
`~/models` at 01:32**, twenty runs in. `/home` is healthy (857 G free, no
kernel I/O errors); the containing directory's mtime is 01:32 and the
server's error is `gguf_init_from_file: failed to open ... (No such file
or directory)`. Nothing in this harness or in modelctl deletes model
files. The only journal entry nearby is `systemd-tmpfiles-clean` at
01:25:55. Cause not established.

`~/models/gguf/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-IQ2_M.gguf` is
a truncated 3.97 GB partial of the same model and is not usable.

Everything was re-run on **`Qwen3.5-35B-A3B-UD-IQ4_XS`** (17.5 GB, MoE,
40 blocks, same Qwen3.5 family). The 122B results taken before the
deletion are real measurements and are kept below; they are **not**
comparable with the 35B ones.

## 4a — determinism cost, paired

### On the 122B, before the deletion

**C1 static placement — 5/5 pairs**

| pair | ran first | det-on | det-off | delta | rel | sign | load(1m) det-on | load(1m) det-off |
|---|---|---|---|---|---|---|---|---|
| 0 | det-on | 16.0659 | 13.7519 | −2.3140 | −14.40% | − | 1.31–2.28 (1.58) | 2.28–3.92 (3.18) |
| 1 | det-off | 15.5207 | 14.3075 | −1.2132 | −7.82% | − | 4.07–5.73 (4.75) | 3.55–4.55 (3.97) |
| 2 | det-on | 14.8769 | 11.6102 | −3.2667 | −21.96% | − | 5.16–6.56 (5.77) | 5.42–6.56 (6.08) |
| 3 | det-off | 13.6385 | 14.7361 | +1.0976 | +8.05% | + | 5.20–6.06 (5.61) | 5.19–6.55 (5.66) |
| 4 | det-on | 14.8069 | 14.5072 | −0.2997 | −2.02% | − | 4.78–6.06 (5.29) | 4.21–5.58 (4.86) |

median delta **−1.2132 tok/s (−7.82%)**; sign test n=5, +1/−4/0 ties,
two-sided exact **p = 0.3750**.

**C2 transfer cache — 4/5 pairs; pair 4 lost to the deletion**

| pair | ran first | det-on | det-off | delta | rel | sign | load(1m) det-on | load(1m) det-off |
|---|---|---|---|---|---|---|---|---|
| 0 | det-on | 9.3900 | 9.4198 | +0.0298 | +0.32% | + | 3.60–5.19 (4.38) | 2.61–3.39 (2.93) |
| 1 | det-off | 8.9954 | 9.5344 | +0.5390 | +5.99% | + | 2.07–3.08 (2.52) | 2.48–4.52 (3.51) |
| 2 | det-on | 8.7821 | 9.7786 | +0.9965 | +11.35% | + | 1.61–1.99 (1.77) | 1.35–1.62 (1.49) |
| 3 | det-off | 8.8403 | 10.4697 | +1.6294 | +18.43% | + | 1.25–2.51 (2.12) | 1.27–1.45 (1.36) |
| 4 | det-on | — | — | — | — | · | model file gone (`server exited rc=1`) | |

median delta **+0.7677 tok/s (+8.67%)**; sign test n=4, +4/−0/0 ties,
two-sided exact **p = 0.1250**.

### On the 35B

**C1 static placement — 5/5 pairs**

| pair | ran first | det-on | det-off | delta | rel | sign | load(1m) det-on | load(1m) det-off |
|---|---|---|---|---|---|---|---|---|
| 0 | det-on | 38.5646 | 37.7777 | −0.7869 | −2.04% | − | 0.95–1.52 (1.14) | 1.44–1.52 (1.48) |
| 1 | det-off | 39.2925 | 38.9195 | −0.3730 | −0.95% | − | 2.05–2.53 (2.21) | 1.41–1.45 (1.43) |
| 2 | det-on | 37.9859 | 38.6867 | +0.7008 | +1.84% | + | 2.29–2.53 (2.41) | 2.09–2.29 (2.19) |
| 3 | det-off | 38.0086 | 37.7842 | −0.2244 | −0.59% | − | 3.07–3.36 (3.23) | 2.96–3.36 (3.15) |
| 4 | det-on | 38.0114 | 38.9424 | +0.9310 | +2.45% | + | 2.75–3.07 (2.91) | 3.00–3.40 (3.19) |

median delta **−0.2244 tok/s (−0.59%)**; sign test n=5, +2/−3/0 ties,
two-sided exact **p = 1.0000**.

**C2 transfer cache — 5/5 pairs**

| pair | ran first | det-on | det-off | delta | rel | sign | load(1m) det-on | load(1m) det-off |
|---|---|---|---|---|---|---|---|---|
| 0 | det-on | 24.6034 | 24.7822 | +0.1788 | +0.73% | + | 3.03–3.40 (3.21) | 2.58–2.87 (2.72) |
| 1 | det-off | 24.6967 | 24.7746 | +0.0780 | +0.32% | + | 1.96–2.13 (2.04) | 2.23–2.45 (2.34) |
| 2 | det-on | 24.8321 | 24.7311 | −0.1010 | −0.41% | − | 1.67–1.88 (1.76) | 1.57–1.67 (1.62) |
| 3 | det-off | 24.7614 | 24.6899 | −0.0715 | −0.29% | − | 1.34–1.40 (1.37) | 1.44–1.52 (1.48) |
| 4 | det-on | 24.7494 | 24.7916 | +0.0422 | +0.17% | + | 1.27–1.31 (1.29) | 1.24–1.54 (1.43) |

median delta **+0.0422 tok/s (+0.17%)**; sign test n=5, +3/−2/0 ties,
two-sided exact **p = 1.0000**.

### Did the knob take effect?

Within one arm every run is the same configuration, so det-on should
reproduce its token sequence across pairs and det-off should not. This
is a check on the manipulation, not on the throughput result.

| job | det-on identical | det-off identical | earliest det-off divergence |
|---|---|---|---|
| C1, 122B | **5/5** | 3/5 | token 2 |
| C2, 122B | **4/4** | 2/4 | token 41 |
| C1, 35B | **5/5** | 3/5 | token 23 |
| C2, 35B | **5/5** | 4/5 | token 78 |

`GGML_SYCL_DETERMINISTIC=0` did take effect in all four jobs, and the
det-off arm does not reproduce its own output — so each of its five
values is one sample from a distribution, and the spread inside that arm
is part of the record.

## 4b — re-anchor battery, 35B

| cond | run | gen tok/s | prompt tok/s | load s | hit ratio | slots used | effective budget B | load(1m) |
|---|---|---|---|---|---|---|---|---|
| C1 | 0 | 37.4254 | 63.1018 | 5.0 | — | — | — | 1.46–2.02 (1.66) |
| C1 | 1 | 38.8046 | 62.9643 | 5.0 | — | — | — | 1.94–2.43 (2.13) |
| C1 | 2 | 39.6072 | 63.6969 | 5.0 | — | — | — | 2.21–2.43 (2.32) |
| C1 | 3 | 39.5380 | 64.3470 | 5.0 | — | — | — | 2.54–2.97 (2.73) |
| C1 | 4 | 39.5406 | 64.6345 | 5.0 | — | — | — | 2.82–3.23 (3.01) |
| C2 | 0 | 24.7005 | 57.3578 | 5.0 | 0.69036 | 2945 | 4,294,328,320 | 2.89–3.23 (3.06) |
| C2 | 1 | 24.8625 | 56.9954 | 5.0 | 0.69036 | 2945 | 4,294,328,320 | 2.47–2.74 (2.60) |
| C2 | 2 | 24.8611 | 57.2212 | 5.0 | 0.69036 | 2945 | 4,294,328,320 | 2.14–2.35 (2.24) |
| C2 | 3 | 24.8087 | 57.4659 | 5.0 | 0.69036 | 2945 | 4,294,328,320 | 2.11–2.13 (2.12) |
| C2 | 4 | 24.8765 | 57.6238 | 5.0 | 0.69036 | 2945 | 4,294,328,320 | 1.86–2.02 (1.94) |
| C3 | 0 | 23.8438 | 58.1934 | 5.0 | 0.68602 | 2945 | 4,294,328,320 | 1.73–3.67 (2.40) |
| C3 | 1 | 23.8788 | 58.1459 | 5.0 | 0.68602 | 2945 | 4,294,328,320 | 3.08–3.46 (3.27) |
| C3 | 2 | **HUNG** — see below | | | | | | 0.00–4.62 (0.23) |
| C3 | 3 | 23.5026 | 57.9014 | 5.0 | 0.68602 | 2945 | 4,294,328,320 | 0.00–0.15 (0.08) |
| C3 | 4 | 23.5984 | 56.9356 | 5.0 | 0.68602 | 2945 | 4,294,328,320 | 2.04–2.22 (2.13) |

| cond | n | mean tok/s | sd | min | max |
|---|---|---|---|---|---|
| C1 | 5 | 38.9832 | 0.9309 | 37.4254 | 39.6072 |
| C2 | 5 | 24.8218 | 0.0726 | 24.7005 | 24.8765 |
| C3 | 4 | 23.7059 | 0.1842 | 23.5026 | 23.8788 |

### The effective cache budget

Requested `--moe-cache-bytes SYCL0=4294967296`. The runtime's own
geometry line reports **2945 slots × 1,458,176 B = 4,294,328,320 B**
(gate 450,560 + up 450,560 + down 557,056), a shortfall of **638,976 B**,
identical on every cache run. The 2026-08-01 battery recorded only the
requested figure, so this quantity has not been in evidence before.

Hit ratio is bit-stable within a condition across all runs (C2 0.69036,
C3 0.68602) and slots_used is 2945/2945 in every case.

## The C3 hang

C3 run 2 launched 01:43:39, loaded in 5 s, completed its 32-token warmup,
had `POST /cache/reset` applied, launched the measured request, printed
`n_decoded = 100, tg = 23.11 t/s` at 13 s — and then stopped. The
process sat at 6.2% CPU with system loadavg at 0.00 for **34 minutes**
until killed. `SIGTERM` did not end it; `SIGKILL` did.

C1 and C2 never did this. C3 runs 0, 1, 3 and 4 completed normally, so it
is intermittent.

`--moe-hybrid-mode on` is the only thing C3 adds to C2. The 2026-08-01
determinism record already flagged, under "found in passing, not fixed":

> `moe_expert_cache::reset()` clears slots, the layer index, miss counts,
> `m_tick` and the advice batches, but not `m_hybrid_plans`. `reset()` is
> reachable from an HTTP thread via `POST /cache/reset`; a plan recorded
> before it can be taken after it.

This protocol calls `POST /cache/reset` between the warmup and the
measured decode, on both C2 and C3 — but only C3 has hybrid plans to go
stale. That is consistent with the hang and is where a bisect should
start. **It is a hypothesis, not a diagnosis**: no bisect was run and no
fix is proposed here.

## 4c — SDPA reproducibility

`run_reproducibility.py --shape sdpa-heavy` from fork branch
`agent/sdpa-probe` (`e73d47680`), three fresh servers, greedy, seed 42,
`cache_prompt=false`, `GGML_SYCL_DETERMINISTIC=1`, same server flags as
C1. The default shape was run afterwards as a control.

| shape | prompt chars | runs | token-identical | logprob-identical | max abs Δlogprob |
|---|---|---|---|---|---|
| `sdpa-heavy` | 9195 (~2.3k tokens) | 3 | **3/3** | **3/3** | **0.00000000** |
| `default` (control) | 114 (~20 tokens) | 3 | 3/3 | 3/3 | 0.00000000 |

Token arrays, all three runs each:

```
sdpa-heavy : [733, 2424, 220, 17, 23, 279, 6326, 6326]
default    : [271, 248068, 271, 248069, 271, 49338, 264, 23268]
```

The pre-registered pass condition was max abs Δlogprob **exactly 0.0** at
the logit level, not token identity. It is met.

Scope of what that shows: on this model, at this attention shape, with
flash attention on (`--flash-attn auto`) and `GGML_SYCL_FA_ONEDNN` at its
default of 1, three fresh processes agreed bit for bit. It does not show
that the oneDNN graph SDPA path is deterministic in general — that path
still takes no `primitive_attr`, so `set_deterministic` does not reach
it, and this is one more shape observed rather than a guarantee obtained.
The 122B, which is where the original nondeterminism was found, could not
be tested: its weights were deleted.

## Harness defects found by running it

- **`--model` was a constant.** A harness that hardcodes a path cannot be
  repointed when the path stops existing. Now an argument, and `one_run`
  checks the file exists before launching: "the file is gone" and "the
  server crashed" both surface as `rc=1`, and conflating them cost
  fifteen runs before anything noticed.
- **The completion timeout was 3600 s.** A hung server was
  indistinguishable from a slow one for a full hour. Now 600 s.
- **No profile named `qwen122b-a10b` exists.** The night-lane
  registrations — the four maiden jobs and the two RPC pairs from the
  previous session — all name it, and `modelctl_nightlane.default_measure`
  routes through `test_launch_plan(profile_name, ...)`, so none of them
  could have run as registered. The 122B has never had a saved profile;
  every measurement of it in this tree was assembled as argv, which is
  what this harness does.

## Checks

`ci/checks.sh` (full) after the runs: **every check PASS**, 77 s wall;
suite **1435 passed, 11 skipped in 44 s**.

One reporting bug fixed in passing: pytest's ANSI codes land between
"N passed" and ", N skipped", so the summary grep matched only the first
half and the PASS line under-reported every run. `--color=no` makes it
whole again — a check whose label misstates its own measurement is the
same class of defect as everything else in this record.
