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

- No node toolchain, no build step, no SPA framework. Server-rendered
  + htmx stays.
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

## Owner inputs (TBD — if still TBD when P11a runs, adopt the stated
default and say so in the memo)

- Q1 biggest pain: TBD. Default: fragile flows (wizard/jobs) first,
  then IA, then visual.
- Q2 posture: TBD. Default: evolve in place on the current stack
  (routers per domain + view-model layer); clean rebuild only if the
  teardown finds the foundation unsalvageable.
- Q3 Sunday-review surface (backlog status, evidence links, decision
  inbox, staging summary as a console section): TBD. Default:
  yes, core goal — the console becomes the weekly review surface as
  well as the runtime console.

## Success criteria for the overhaul (all phases)

- A scripted route-walk + wizard e2e + job-cancel harness against a
  stubbed backend exists BEFORE restructuring, and stays green after.
- Every audit-class defect from 2026-07-31/08-01 has a regression test.
- The console never breaks for more than one session during migration
  (phased plan required).
- Page inventory and IA documented in modelctl/docs/.

## P11a deliverables

Teardown memo (keep/kill/restructure per module), proposed IA/page
map, 2–3 static HTML mockup directions in
docs/plans/evidence/console-mockups/ (shell + one data-dense page +
one wizard step each; openable directly in a browser), phased
migration plan, decision memo with a recommendation.
