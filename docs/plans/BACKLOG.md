# BACKLOG — autonomous work queue

Worked top-to-bottom by unattended sessions per the Autonomous work
protocol in /CLAUDE.md. Agents update Status and Log lines in place; do
not reorder items. Status values: ready, in_progress, blocked (memo in
decisions/), gated (dependency or bench window), parked-evidence (missed
threshold, findings written), done, needs-hands (owner-only).

Before designing any item, read
docs/research/2026-08-01-moe-offloading-landscape.md and WebFetch the
item's sources.

---

## P1 — madvise WILLNEED/DONTNEED per-expert SSD-tier management
Status: gated (bench window)
Autonomy: auto-merge to staging on gate
Spec: after each decode step on the mmap/SSD tier, POSIX_MADV_WILLNEED
the byte ranges of the experts just used; on tier eviction,
POSIX_MADV_DONTNEED the evicted range. DONTNEED does not exist in
llama-mmap.cpp yet and must be added. Scope strictly to the fork's
SSD/mmap path; no behavior change when the model is RAM/VRAM-resident.
Sources: llama.cpp issue #20757 (design sketch, ~60 LOC estimate);
src/llama-mmap.cpp WILLNEED pattern.
Gate: test-moe-cache + modelctl suite green; ornith-397b batch bench
(three-condition protocol, bench window) shows >= +10% over the 0.37
tok/s static baseline, else findings doc.
Log:
- 2026-08-01 (daytime session): implemented, fork commit f4d390349 on
  agent/P1. Advise bridge lives in llama-mmap.cpp (registry of live
  mapped fragments; DONTNEED added there per spec), cache batches miss
  ranges (WILLNEED) and evicted origins (DONTNEED) per step, flushed
  via a new scheduler step-end hook. Opt-in GGML_MOE_CACHE_MMAP_ADVISE=1;
  capability moe_cache_mmap_advise. Suite legs GREEN: test-moe-cache
  (+7 advise cases), new test-mmap-advise (/proc RSS oracle,
  anon-memory safety), test-moe-hybrid — all pass on SYCL, CPU-only,
  and ASan/UBSan builds; CPU-only probe truthfully false; modelctl
  suite Ran 1104, OK (11 skipped). Remaining: ornith-397b batch bench
  (bench window) -> evidence/2026-08-01-p1-madvise-ssd-tier.md for the
  bench plan. No staging merge until then.

## P2 — routing-trace dump flag in the fork
Status: ready
Autonomy: auto-merge to staging on gate
Spec: add `--moe-route-trace FILE` to llama-server: stream (token_idx,
layer, selected expert ids) per step, negligible overhead when off.
This is the data source for P3 and later prefetch work.
Gate: suite green; trace produced for laguna-s2.1 and one Qwen3.5-MoE
run; overhead when disabled unmeasurable (<1%).
Log:

## P3 — predictor recall study (offline, Python; no C++)
Status: gated (depends P2)
Autonomy: auto-merge study + report; adjudicates Stage 2 per thresholds
Spec: using P2 traces plus hidden-state taps, implement the ETH
pre-attention linear probe (arXiv 2511.10676) offline; train per-layer
probes per model; measure top-k recall vs the real router. Compare
against a zero-shot temporal-reuse baseline. Models: laguna-s2.1,
Qwen3.5/3.6-MoE, gpt-oss-class if present in the model dir.
Gate: recall table in evidence/. >=90% on a family -> that family is
prefetch-eligible (unlocks future C++ prefetch item); <80% -> family
marked prefetch-disabled, per the landscape doc's pollution rule.
Log:

## P4 — MoE-SpeQ same-expert batch reordering on the prefill path
Status: ready
Autonomy: auto-merge to staging on gate
Spec: where the cache already engages (prefill/large batch), reorder
token-expert work so same-expert tokens compute contiguously (arXiv
2511.14102) for L1/L2 and slot reuse. Fork-side only.
Gate: bit-identical outputs vs unreordered (correctness matrix subset);
laguna prefill throughput >= +5%, else findings.
Log:

## P5 — planner rule: pin shared experts and dense early layers
Status: ready
Autonomy: auto-merge to staging on gate
Spec: modelctl place emits placement that always pins `shexp` tensors
and dense early layers (e.g. blk.0-2 on GLM/DeepSeek-pattern models) to
GPU; only routed experts flow through cache/offload tiers.
Gate: modelctl suite + a placement snapshot test per affected profile.
Log:

## P6 — llama.cpp issue #20757 engagement draft
Status: ready
Autonomy: draft only; posting is needs-hands
Spec: draft a comment presenting this fork as the C++ implementation the
issue requested: architecture summary, what matches their design, what
differs (transfer cache vs slot remap), measured results, link to the
GitHub mirror. Tone: offering, not selling.
Gate: draft at docs/plans/evidence/20757-comment-draft.md.
Log:

## P7a — Phase G design memo
Status: ready
Autonomy: write and commit the design doc; no owner approval required
Spec: design hit/miss-aware decode dispatch: cached experts compute on
GPU; missed experts compute on CPU via activation copy (Fiddler, arXiv
2402.07033), decoupled from the op-offload threshold. The doc must
address: (1) engaging at batch-1 without touching the global threshold;
(2) the correctness battery plan (matrix + k-quant + ASan/UBSan); (3)
interaction with first-capable-backend selection; (4) clean fallback to
current behavior behind a flag; (5) rollback story. Read HybriMoE
(2504.05897) scheduling and ik_llama.cpp issue #1765 (regression
cautionary tale) first.
Gate: doc committed covering all five points. Unlocks P7b.
Log:

## P7b — Phase G implementation
Status: gated (depends P7a)
Autonomy: merge to staging only on full gate; regressions -> rollback +
findings
Spec: implement per P7a behind `--moe-hybrid-mode`, default off.
Gate: full correctness battery green (matrix, k-quant, ASan/UBSan build);
laguna-s2.1 decode >= 1.3x over the 14.2 tok/s static baseline (>=18.5
tok/s) -> merge. 1.0-1.3x -> findings + decision memo (CPU kernels may be
the bottleneck; AMX-style tiling is the named pivot). <1.0x ->
parked-evidence.
Log:

## P8 — multi-GPU cache engagement (Stage 3)
Status: gated (depends P7b)
Autonomy: auto-merge on gate
Spec: make SYCL1 (B580) cache budgets real: explicit per-device
assignment (discussion #22183 pattern) or scheduler-selection fix.
Alternatively, if evidence says descope, write the descope findings and
strip dead config — either resolution is a valid gate.
Gate: suite green + either measured B580 cache engagement or a committed
descope with config removed.
Log:

## P9 — SSD-tier batch "ticket" mode in modelctl
Status: ready
Autonomy: auto-merge on gate
Spec: first-class async batch UX per RQ6 decision: submit a prompt file
as a job, it runs in the bench window on the SSD-tier model, artifact +
stats land in a results dir. Reuse the existing job runner; colibri's
ticket framing is the reference UX.
Gate: modelctl suite + one end-to-end ornith-397b job producing output
and a stats file.
Log:

## P10 — RAM upgrade memo (research only)
Status: ready
Autonomy: memo only; purchase is owner's call via decisions/
Spec: read board identity from /sys/devices/virtual/dmi/id/ (no sudo),
determine max supported DDR5 capacity/speeds, price 96GB and 128GB kits,
compute resulting on-disk expert tail for ornith-397b per RQ6 lever 1.
Gate: memo at docs/plans/decisions/010-ram-upgrade.md with a
recommendation.
Log:

---

## needs-hands (owner)
- NH1: enable the nightly session timer: `systemctl --user enable --now
  claude-backlog.timer` after reviewing systemd/claude-backlog.*.
- NH2: confirm push policy (agent/* + staging to Gitea) and provision
  credentials, or leave agent commits local-only.
- NH3: post the P6 draft to llama.cpp #20757 when satisfied.
- NH4: answer decision memos as they appear; distill answers into Owner
  precedents in /CLAUDE.md.
