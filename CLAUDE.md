# CLAUDE.md — moe-serving

Combined repo: **modelctl** control plane + a **llama.cpp fork** (submodule,
branch `feature/sycl-moe-expert-cache`) implementing a per-GPU MoE expert
weight transfer cache on Intel Arc / SYCL. Start with [README.md](README.md);
modelctl specifics in [modelctl/README.md](modelctl/README.md) and
[modelctl/docs/](modelctl/docs/); runtime feature docs in
[modelctl/docs/runtime/](modelctl/docs/runtime/); promotion/acceptance
history in [docs/upstream-sync/](docs/upstream-sync/). The validated
modelctl+llama.cpp commit pair (and rollback target) is
[integration-manifest.json](integration-manifest.json).

## Knowledge boundaries — read before designing anything

Your training data predates most of the relevant MoE-offloading literature
and the current llama.cpp upstream state. Do not answer from priors about
what exists, what is state of the art, or what upstream supports.

1. Before any design work on the expert cache, dispatch (Phase G), prefetch,
   hybrid CPU-miss execution, or the SSD tier, read
   [docs/research/2026-08-01-moe-offloading-landscape.md](docs/research/2026-08-01-moe-offloading-landscape.md).
   It is post-cutoff ground truth (researched 2026-08-01), contains the
   decided plan of record, and ends in a ranked steal list with primary
   sources.
2. Before implementing any steal-list item, WebFetch its primary source
   (arXiv ID / GitHub issue number given in the doc).
3. For any other claim about external state of the art or upstream llama.cpp
   behavior, WebSearch first or say you are unsure — do not guess.

## Hard-won local facts (traps)

- The transfer cache hooks cross-backend weight copies and is **inert during
  batch-1 decode** (SYCL op-offload threshold 32). "Enabled" in logs does not
  mean "engaged" — see modelctl/docs/runtime/moe-cache.md.
- Never benchmark the cache without the three-condition protocol in
  modelctl/docs/runtime/moe-cache-testing.md.
- Do not force `GGML_OP_OFFLOAD_MIN_BATCH` below 32: known pre-existing
  upstream correctness bug on this architecture (incorrect output). The
  scoped `GGML_OP_OFFLOAD_MOE_MIN_BATCH` knob exists in the fork; use with
  the k-quant determinism harness.
- The ggml scheduler's offload pass selects the first capable backend, so
  only SYCL0 (Arc Pro B70) engages a cache today; a SYCL1 budget is dead
  weight until Stage 3 of the plan of record.
- Hardware: SYCL0 = Arc Pro B70 32GB, SYCL1 = Arc B580 12GB, 31GB RAM,
  oneAPI 2026.1, Ubuntu 26.04. Gitea is the source of truth; GitHub repos
  are review mirrors.

## Autonomous work protocol (evidence-gated, not approval-gated)

Sessions are expected to run without the owner present. Do not ask
questions mid-session. Adjudicate against the rules below; when a genuine
fork exceeds your authority, write a decision memo and take the next item.
Never wait.

Session loop:
1. Read docs/plans/BACKLOG.md. Pick the top item with Status: ready whose
   dependencies are met.
2. Work it in a worktree on branch `agent/<item-id>` off `staging`.
3. Drive it to its stated Gate. Update the item's Status and Log lines in
   BACKLOG.md as you go (in_progress -> done / parked-evidence / blocked).
4. Done requires ALL of: the item's Gate is green; evidence written to
   docs/plans/evidence/ (docs/upstream-sync/ style); modelctl unittest
   suite passes; relevant fork tests pass.
5. Blocked on a decision above your authority: write
   docs/plans/decisions/NNN-slug.md (context, options, your
   recommendation, what happens next under each option), set the item
   blocked, move to the next item.

Decision rules (pre-committed; do not re-ask the owner):
- Correctness gates are absolute: test-moe-cache, the modelctl suite,
  and — for anything touching dispatch or the scheduler — the correctness
  matrix and k-quant determinism harness. No performance result excuses a
  correctness failure.
- Performance thresholds adjudicate themselves: meet the gate -> merge to
  `staging` and proceed; miss it -> write findings, mark parked-evidence,
  move on. Missing a threshold is a result, not a failure.
- Rollback is cheap (integration-manifest.json names the target). Prefer
  attempting over asking.

Hard guardrails:
- Never touch ~/services/, systemd system state, or restart llama-swap,
  OVMS, or any live service. Production serving is out of bounds.
- All work in worktrees. Never commit to master. Pushes limited to
  `agent/*` and `staging` branches.
- Benchmarks that contend for SYCL0 (the B70) run only in the bench
  window (02:00-06:00 local) or when serving is verifiably idle. CPU,
  SYCL1, and offline work run anytime.
- No destructive operations outside the worktree. No credential handling.

Owner interface: the owner's only jobs are (a) answering
docs/plans/decisions/ memos when the folder is nonempty — each answer
gets distilled into an Owner precedent below; (b) a weekly promotion
sitting over `staging`; (c) items tagged needs-hands in BACKLOG.

## Owner precedents
- 2026-08-01: SSD tier stays active, batch-first, not held to the
  interactive bar (landscape doc RQ6).
