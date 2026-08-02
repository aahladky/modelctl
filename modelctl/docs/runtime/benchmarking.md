# Benchmark infrastructure: pairs, anchors, and the night lane

Companion to [moe-cache-testing.md](moe-cache-testing.md), which owns the
per-run protocol (fixed prompt, greedy, seed 42, `cache_prompt=false`,
fresh server, warmup, engagement check, cache reset, measured decode).
This document covers what sits above one run: how two conditions get
compared, when a stored number may be reused, and what is allowed to run
a benchmark while nobody is watching.

All three exist because of one defect. The 2026-08-01 determinism record
reported a "cost of determinism" of −15.20% / −8.01% / −14.87%. Those are
differences between two *blocks* of runs, taken across a battery whose
1-minute load average ranged 2.63 to 17.15 with a mean of 8.99. The
numbers are real; what they measure is the attribute **and** the machine.
They are void, and the maiden night-lane jobs replace them.

## Paired comparisons — `modelctl_paired.py`

A paired comparison runs A and B back-to-back inside one pair and
compares the delta *within* the pair. Whatever the machine is doing
during pair 3, it does to both arms of pair 3.

Pair order alternates — (A,B), (B,A), (A,B), … — because "back to back"
is not symmetric: the second slot inherits the first's thermal and
page-cache state. A fixed order folds that asymmetry into every delta
with the same sign; alternating cancels it across pairs. `schedule()` is
its own function so the alternation can be read without running anything.

What comes out:

- every raw run, with its own load trace
- per-pair delta and sign, with the convention stated (`delta = b − a`)
- the median of the deltas — the statistic that goes with a sign test,
  and the one a single stalled run cannot move
- a two-sided exact sign test over the signs, ties excluded from `n`
- per-pair load comparability, checked at pair level because that is the
  level at which pairing makes its claim

There is no threshold anywhere in the module and no function returns a
winner. The sign test says how consistently the deltas agreed in
direction and how surprising that consistency would be from coin flips.
Reading it is Aaron's job.

A run that raises does not abort the comparison: its pair is recorded
incomplete with the exception text and excluded from the sign test.
Aborting would discard the pairs already taken, and one arm failing is
itself part of the record.

## Anchors — `modelctl_anchors.py`, `anchors.json`

An anchor is a reference measurement plus the fingerprint of what
produced it: build commit, profile hash, environment, driver. A battery
re-runs an anchor exactly when that fingerprint no longer matches.

Fields are compared and reported individually, not as one digest — "the
driver moved" and "the binary moved" call for different work. An
unrecorded field never matches: an anchor that cannot prove its driver is
the same driver has to be re-taken.

Two things a fingerprint cannot express:

| flag | meaning |
|---|---|
| `void` | the measurement itself was bad, whatever the conditions were. Re-runs regardless of fingerprint, with the reason attached. The old value is **kept**, not deleted — the record of a number that should not be trusted is still evidence. |
| `always_run` | laguna-s2.1's canary. Not a value to compare against; it exists to notice the machine moving under everything else, which is exactly when its fingerprint still matches. |

`env_hash` covers a whitelist (`FINGERPRINTED_ENV`): the determinism
knob, both offload thresholds, the advise knob, device selection and the
library path. A whitelist because hashing the whole environment stales
every anchor on an unrelated shell variable, and hashing nothing lets an
ambient `GGML_OP_OFFLOAD_MOE_MIN_BATCH` change the runtime silently.

The registry lives in the repo next to `night-lane.json`: an anchor that
can change without a diff is not a reference.

## The night lane — `modelctl_nightlane.py`, `night-lane.json`

A pre-registration is a comparison declared, with its criterion, before
it runs. `enabled` means **queued**, not scheduled: a person released it
to run the next time the machine is quiet, and enabling is a diff.

### Conditions are recorded, never obeyed

There was a gate here: `window_state()` opened only when llama-swap held
no models *and* loadavg(1m) was at or below 1.5, and it failed closed on
either reading being unavailable — an unread loadavg "cannot be shown
low". On a rig that runs lanes all day that is close to a permanent
veto, and the 2026-08-02 night pair nearly did not happen because of it.
It was removed on 2026-08-02. **Benchmarks are never gated.**

What stands in its place:

1. `observe()` takes the same readings — llama-swap's resident models,
   loadavg, MemAvailable — and returns them with no verdict attached.
   The type has no `open` field and no `reasons`, so there is nothing
   for a future caller to gate on. A reading that could not be taken is
   `None` and is named in `unreadable`, never substituted with a zero.
2. `cleanup_pass()` runs before each job: it sweeps orphaned lane build
   scratch (`modelctl lane sweep --orphans`), reaps llama-servers whose
   launcher has died, and records MemAvailable and loadavg on both sides
   of itself. It frees **junk only** — never the page cache, which is
   part of the measurement rather than something in its way. Nothing in
   it can stop a job: a cleanup that fails records why and the run
   continues.
3. The job runs, and its conditions live on in the per-run load trace.
   Recording is what lets a paired design survive a noisy machine;
   refusing only produces silence.

The one thing that still makes a job wait is the **GPU lock**, because
two benchmarks on one GPU is garbage rather than noise. It waits up to
six hours rather than refusing after a minute, and a wait that does run
out is filed as a **failure naming the lock's holder** — never a silent
skip.

### Dispatch

`dispatch_due()` submits to the **benchmark lane** of the existing job
store. One worker, so two night jobs can never contend for the GPUs, and
the console's jobs page renders that lane already. Jobs it declines come
back with reasons rather than being filtered away: a lane that quietly
runs three of five pre-registrations and reports three results looks
like it ran everything.

A job whose arms need a fleet node the machine cannot reach is blocked,
never run local — the local-fallback plan is byte-identical to a
fleet-free launch, so it would answer a different question with a
perfectly valid number.

Importing the module starts nothing. A caller hands it a job manager and
asks.

### The offload floor

`GGML_OP_OFFLOAD_MIN_BATCH` below 32 is a known correctness bug on this
hardware. `arm_violations()` refuses any arm that sets it lower, and
`run_job()` refuses the whole job before measuring anything. Enforced at
the lane rather than left to review because the lane runs with nobody
watching, and a floor violation at 03:00 still produces numbers that
look fine. `GGML_OP_OFFLOAD_MOE_MIN_BATCH` is a different knob and is
not covered by the floor.

### Evidence

Each run writes `docs/evidence/<date>-nightlane-<job-id>.json` — the full
record, including the job's criterion verbatim, because a result read
months later without the rule it was judged by is a number looking for a
story — and appends one line to `docs/evidence/night-lane-log.md`. The
summary line says what ran, under what load, and where the numbers are.
It carries no reading of the result, and an unrecorded load is reported
as unrecorded, never as 0.00.

## Load traces — `modelctl_load.py`

Sampled from `/proc` only, cheap enough to run in a thread beside a
decode loop; a recorder that perturbs what it measures is worse than
none. One sample is taken immediately at start, so a run shorter than
the interval still gets a load record — those are the runs most easily
perturbed and least likely to be noticed.

The property that matters most: **what could not be read is recorded as
unreadable, never as zero.** A summary reporting 0.0 for a loadavg it
could not read would make the worst run of a battery look like the
quietest, which is the failure this whole layer exists to prevent.
