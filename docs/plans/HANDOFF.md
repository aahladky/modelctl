# HANDOFF — operating the moe-serving autonomous work system

For the owner, and for any fresh Claude session (Cowork, Code tab, or
headless) told to "read docs/plans/HANDOFF.md". Written 2026-08-01.

## What this is

Evidence-gated delegation. The repo carries the governance; sessions
carry the labor. The owner's judgment is pre-committed in files, so
work proceeds without live approval — with one carve-out: performance
benchmark results are ALWAYS owner-adjudicated (agents measure, file
bench-review, and stop).

## The pieces

- /CLAUDE.md — the constitution: session loop, decision rules,
  guardrails, execution discipline, Owner precedents. Every session
  loads it automatically.
- docs/plans/BACKLOG.md — the live board (MAIN checkout copy is
  canonical). Items carry Spec, Gate, Autonomy, Status, Log.
- docs/plans/decisions/ — the owner's inbox. Blocked sessions file
  memos here; answers become precedents in CLAUDE.md.
- docs/plans/evidence/ — item-level evidence and bench reports.
- docs/plans/console-overhaul-brief.md — console overhaul mandate.
- docs/research/2026-08-01-moe-offloading-landscape.md — post-cutoff
  ground truth on the field; read before cache/dispatch/prefetch work.
- integration-manifest.json — the validated pair; rollback target.
- systemd/claude-backlog.{service,timer} — nightly sweeper session
  (02:30): runs pending benches to bench-review, then top ready item.
- scripts/agent-tail.sh — live human-readable tail of the newest
  session transcript.

## Dispatching work (Cowork or any session on this folder)

Paste this, filling the item id:

  Follow the Autonomous work protocol in CLAUDE.md including Execution
  discipline. Live dispatched session: work BACKLOG item <ID> ONLY, to
  its Gate, then stop. Do not pick up other items. Update only <ID>'s
  Status/Log lines, in the MAIN checkout BACKLOG.

Worked example — the pending P1 hardware bench:

  Follow the Autonomous work protocol in CLAUDE.md. Run the pending
  hardware bench for BACKLOG item P1: use the build in
  llama.cpp/.claude/worktrees/agent-P1, follow the three-condition
  protocol in modelctl/docs/runtime/moe-cache-testing.md, capture raw
  artifacts plus concurrent-load context into docs/plans/evidence/,
  set P1 to bench-review. Do NOT adjudicate the result.

Dispatch one item per session; parallel sessions are fine when items
touch disjoint subsystems. Fork C++ items (P2/P4/P7b) share build dirs
— serialize those with each other.

## Checking status

- Board: docs/plans/BACKLOG.md (main checkout).
- Fleet: `systemctl --user list-units 'claude-*'`
- Live play-by-play: `scripts/agent-tail.sh` (newest session; for a
  specific one, `ls -t ~/.claude/projects/-home-aaron-workspace-moe-serving/`).
- Branches: `git branch --list 'agent/*'`; work merges to `staging`;
  master is owner-only.

## Bench policy (owner ruling 2026-08-01)

Benchmarks run anytime — no window. Sessions must record concurrent
GPU/host load in the report (contention skews numbers). Sessions never
declare a performance gate met: raw artifacts + report -> Status:
bench-review -> owner reads the report and either merges or files
findings. Correctness gates (tests, sanitizers, determinism) remain
self-adjudicating.

## Owner duties (the whole job)

1. Adjudicate bench-review items: read the report in evidence/, decide
   merge vs parked-evidence, note the ruling in the item Log.
2. Answer decisions/ memos; distill each into an Owner precedent.
3. Weekly sitting over `staging` before promoting to the validated pair.
4. needs-hands items in BACKLOG (currently: NH2 Gitea push credentials,
   NH3 post the #20757 draft when satisfied).

## State snapshot, 2026-08-01 morning

- P1 code-complete on agent/P1 (madvise + test-mmap-advise green incl.
  sanitizer); bench pending, dispatchable anytime, owner adjudicates.
- Session A working P2 (routing-trace flag) on agent/P2 — owns the
  fork build dirs. P4 queued behind it.
- Live sessions dispatched for P5 (planner pinning), P11a (console
  teardown + mockups), P6->P10 (#20757 draft, RAM memo).
- Nothing merged to staging yet. Nightly timer active (02:30 sweeper).
- Nothing pushed to Gitea (NH2 pending); all work is local branches.
