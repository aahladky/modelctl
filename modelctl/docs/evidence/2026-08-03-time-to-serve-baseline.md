# Time-to-serve baseline — console wizard end to end — 2026-08-03

First complete real-model chain through the console path (search ->
source -> inspect -> download -> analyze -> plans -> test -> register ->
load -> first token). Driven through the live web API (:9293) by a
scratchpad script, one call per SPA action, no code changes. Raw
per-step log: 2026-08-03-time-to-serve-steps.jsonl (this directory).

## Config

- repo: unsloth/Qwen3.5-122B-A10B-MTP-GGUF, quant Qwen3.5-122B-A10B-UD-Q4_K_M
  (78.3 GB, 3 shards), profile qwen3-5-122b-a10b-ud, wizard a0a4ed5eb08f
- plan tested/registered: tier 4 (d03d749cdf53ecf5): split_mode layer,
  tensor_split 8,3, `--fit off --device SYCL0,SYCL1`, shexp pins +
  experts blk 0-14 SYCL0 / blk 15-19 SYCL1 / rest CPU via mmap
- binary: llama.cpp/build-sycl/bin/llama-server (llama-swap managed)
- machine: llama-swap idle at start ({"running":[]}), modelctl-web the
  only console load, loadavg 1.86/2.15/2.37 at preflight (per-step
  values in the jsonl)

## Per-step wall times

Driver column from the jsonl; jobs-db columns from read-only queries of
~/.local/share/modelctl/web_jobs.db (created/started/finished fields);
"Measured at test/serve" figures from the wizard's `measured` dict
(plan-test job result), not the jsonl.

| step | driver s | jobs-db run s | queue wait s |
|------|----------|---------------|--------------|
| search (HF) | 2.6 | - | - |
| wizard create+source+inspect | 0.2 | - | - |
| download 78.3 GB | 356.1 | 354.6 | 0.09 |
| analyze | 0.3 | - | - |
| plans view | 4.4 | - | - |
| plan test (tier 4, cold) | 220.8 | 218.8 | 0.0 |
| register (view+submit) | 21.0 | - | - |
| load | 60.4 | 58.6 | 0.01 |

- happy-path sum (had the fitting plan been picked first): 665.7 s =
  11 m 06 s from search click to model ready; +2.0 s to first token.
- actual wall clock this run: 22:01:01 -> 22:15:23 = 14 m 22 s,
  including the failed first plan test (2.4 s) and ~3 min operator gap
  diagnosing it.
- download rate ~220 MB/s (~1.8 Gbps of a 2.3 Gbps link).

## Measured at test/serve

- plan test (cold): generation 4.84 tok/s, prompt 5.82 tok/s,
  load 54.31 s, cache_state cold (wizard `measured`)
- warm probe via llama-swap chat/completions (stream): first delta at
  1.99 s; 200 deltas in 40.1 s ~= 5.2 tok/s; all 200 were
  reasoning_content (thinking model) -- a 16-token probe shows an empty
  reply because content never starts before the cap. The jsonl's
  `first_token` row (ttft -1, total 11.1 s, empty reply) IS that capped
  16-token probe; the 1.99 s figure is the separate 200-token re-probe

## Failure found on the default path

First test used the first-listed enabled plan "current profile"
(9f8f950e74918644): claim SYCL0 64.46 + SYCL1 16.11 GiB vs budgets
28.17 + 11.76 GiB -- unfittable by construction, yet listed first,
enabled, unflagged (category "baseline"). acquire_reservation's
byte-budget rejection returns the same None as a live-worker collision,
so test_launch_plan (modelctl_tune.py:589) reported "reservation
conflict -- another worker holds the resources" with zero other workers
alive; failure_class recorded as reservation_conflict. The stale
q4km-offload-sweep reservation row (dead PID 2454417) is excluded from
admission and was NOT the cause; it remains visible in
/api/reservations and does not affect admission.
