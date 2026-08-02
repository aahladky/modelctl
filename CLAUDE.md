# CLAUDE.md

Aaron's homelab inference stack. Task context arrives inside each
dispatched work order; this file is only the safety floor for ad-hoc
sessions.

- The serving stack is live: don't touch ~/services/, systemd, or
  docker, and never restart llama-swap or OVMS.
- Git is an invisible journal: use a worktree only if working in
  parallel, merge into the main tree the moment tests pass, delete the
  branch. What's on disk in the main checkout is the project.
- Benchmarks: record raw numbers, exact config, and concurrent machine
  load. Never judge the result — Aaron reads the numbers.
- Don't set GGML_OP_OFFLOAD_MIN_BATCH below 32: known correctness bug
  on this hardware.
- One pass: do the scoped task, print a plain summary of what you did
  and measured as your final output, and stop. No questions mid-run.
- Outside the llama.cpp submodule, every session ends with all work
  committed to master and any scratch branches or worktrees removed —
  nothing is ever parked. The fork keeps normal branch discipline.
- Agent-written persistence is facts or nothing: one-line entries in
  moe-review/open-items.md, orders, evidence files. Never prose about
  process, style, or how to behave — that class of doc regrows; delete
  it on sight.
