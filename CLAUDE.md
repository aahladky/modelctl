# CLAUDE.md

This is Aaron's homelab inference stack: modelctl (Python control plane +
web console) and a llama.cpp fork with a SYCL MoE expert cache, running
on 2x Intel Arc. One person owns this. Keep things simple and don't
invent process.

## The mission (Aaron's words, 2026-08-01)

"It exists to automate what I hate most about local AI as much as
possible — file management, downloads, launch configs, config files in
general, Linux after 1 failed command, keyboards & CLIs, etc; while
still allowing granular control from an interface that doesn't raise my
blood pressure. And it should do this using my existing hardware to the
fullest extent reasonably possible."

In practice: the web console is the primary interface — the CLI exists
for bootstrap and agents, not for Aaron. A failure should end in a
diagnosis and an offered fix on screen, never in a terminal archaeology
session. "Reasonably possible" caps expectations, not effort: don't
expect 10-20 tok/s from an SSD-bound model, but if research or
community experience shows an optimization for RAM/SSD-bound serving
works, build it — no artificial development gates, no self-imposed
testing barriers. Tests verify what got built; they don't decide
whether to build it.

## Things that prevent expensive repeat mistakes

- Your training predates most relevant MoE-offloading work. Before
  designing anything about the cache, dispatch, prefetch, or the SSD
  tier, read docs/research/2026-08-01-moe-offloading-landscape.md and
  fetch the papers it cites. Don't answer from priors about what
  upstream llama.cpp supports — check.
- The expert cache does nothing during normal single-token generation
  (it only engages at batch >= 32). "Enabled" in the logs does not mean
  engaged. See modelctl/docs/runtime/moe-cache.md.
- Never benchmark the cache without following
  modelctl/docs/runtime/moe-cache-testing.md.
- Don't set GGML_OP_OFFLOAD_MIN_BATCH below 32 — known upstream
  correctness bug on this hardware (wrong output). The fork's
  GGML_OP_OFFLOAD_MOE_MIN_BATCH is the scoped knob.
- Hardware: SYCL0 = Arc Pro B70 32GB, SYCL1 = Arc B580 12GB, 31GB RAM,
  oneAPI 2026.1, Ubuntu 26.04. Gitea is the source of truth; the GitHub
  repos are read-only mirrors.

## Working rules

- The serving stack is live. Don't touch ~/services/, systemd, or
  docker, and don't restart llama-swap or OVMS.
- Aaron hates git-as-workflow. Git here is an invisible journal for
  history and backup, nothing more; what exists on disk in the main
  checkout IS the project. Use a worktree/branch only to avoid
  colliding with a parallel session, and the moment your work passes
  its tests, merge it into the main checkout and delete the branch.
  Never leave finished work stranded on a branch; never ask Aaron to
  look at branches, diffs, or merges. Commit messages: one plain
  sentence. Push stays blocked.
- You'll usually run unattended. Don't stop to ask permission for
  normal work. If something genuinely needs Aaron's input, write the
  question under "Questions for Aaron" in docs/plans/BACKLOG.md and
  move on to something else.
- While iterating, run only the narrow tests for what you changed
  (a specific ctest -R, a single unittest module). Run the full suite
  and sanitizer build once at the end, not after every edit. Don't
  re-run tests when nothing they cover changed.
- When you launch anything long-running, verify within ~15 seconds that
  it's alive and producing output. A process that should have printed
  something and hasn't is a problem to diagnose now, not in 10 minutes.
- Benchmarks: run them whenever. Record the raw numbers, the exact
  command and config, and what else was using the machine at the time.
  Do NOT declare whether the result is good or merge anything based on
  it — Aaron reads the numbers and decides.
- When you finish a piece of work, write a short plain-prose note of
  what you did and measured (the docs/upstream-sync reports are the
  house style) and update its entry in docs/plans/BACKLOG.md.

## Notes from Aaron

- The SSD tier (running huge models off NVMe) stays a live interest —
  overnight/batch use; don't hold it to interactive-speed standards.
- Console overhaul: flows, layout, and visuals are all fair game; the
  stack choice is open but justify the maintenance cost; he wants a
  page where he can see agent work, the todo list, and reports. Details
  in docs/plans/console-overhaul-brief.md.
