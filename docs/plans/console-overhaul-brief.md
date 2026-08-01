# Console overhaul brief

Input for BACKLOG P11a+. Owner taste is encoded here once so overhaul
work does not require live owner involvement.

## Current state (recon 2026-08-01)

FastAPI + Jinja2 + vendored htmx 2.0.4, no build step. ~5,000 lines in
modelctl_web/ total: app.py (118 defs — routes, helpers, everything),
jobs.py (JobRunner/JobStore), mutate.py, swap.py, wizard.py, 22
templates (12 pages + 9 wizard steps + fragments), 197-line style.css.
Sound bones worth keeping: reads are direct concurrent calls; every
write serializes through the single JobRunner; one shared token as
Bearer/cookie, never in URLs. Recurring audit pain concentrates in:
wizard end-to-end fragility, job cancel semantics, error visibility,
and app.py's monolith coupling.

## Owner defaults (inferred from project philosophy; strike to amend)

- Stack is an open design variable (owner ruling 2026-08-01): new
  stacks may be proposed, including ones with a build step. The judge
  is the homelab philosophy — single service, low maintenance
  overhead, still buildable in a year, no churn for churn's sake —
  applied as evaluation criteria, not as a veto.
- Single service, single token, one CSS file (can be a good one).
- Every page must render useful degraded states when subsystems are
  down (llama-swap absent, fork not built, GPU missing) — error
  surfaces show the actual cause/stderr, never a bare 500.
- Jobs are an explicit state machine; cancel and progress are
  first-class UI, not afterthoughts.
- Data-dense over decorative. Dark theme default. Operator-of-one
  audience: Aaron and, at a glance, his agents' output.
- No churn for its own sake: keep JobRunner write-serialization and
  the auth model unless the teardown finds concrete rot.

## Owner inputs (answered 2026-08-01)

- Q1 biggest pain: ALL OF IT — flows, IA, and visual are each real
  pain. Do not sequence one and quietly defer the rest; the mandate is
  comprehensive, and the mockups carry real weight.
- Q2 posture: OPEN TO A NEW STACK. Present at least one direction on
  the current stack (evolve or clean rebuild) and at least one
  new-stack direction, each scored against the philosophy criteria
  above (maintenance burden, single service, year-later
  buildability). Recommendation goes in the decision memo; owner
  picks.
- Q3 weekly-review surface: YES, CORE GOAL. The console is both the
  runtime console and the owner's weekly review surface: BACKLOG
  status, evidence browser, decisions inbox, staging/promotion
  summary. Candidate feature for P11a to design: answering decision
  memos inline from the console (writes the answer into the memo file
  and appends an Owner precedent to CLAUDE.md), collapsing the weekly
  sitting into one page.

## Success criteria for the overhaul (all phases)

- A scripted route-walk + wizard e2e + job-cancel harness against a
  stubbed backend exists BEFORE restructuring, and stays green after.
- Every audit-class defect from 2026-07-31/08-01 has a regression test.
- The console never breaks for more than one session during migration
  (phased plan required).
- Page inventory and IA documented in modelctl/docs/.

## P11a deliverables

Teardown memo (keep/kill/restructure per module), proposed IA/page
map covering both domains (runtime ops + weekly review), 2–3 static
HTML mockup directions in docs/plans/evidence/console-mockups/ (shell
+ one data-dense page + one wizard step each; openable directly in a
browser) — at least one direction on the current stack and one
assuming a new stack — phased migration plan per direction, decision
memo with a recommendation.
