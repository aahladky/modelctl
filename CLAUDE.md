# CLAUDE.md

Aaron's homelab inference stack. Task context arrives inside each
dispatched work order; this file is only the safety floor for ad-hoc
sessions.

- The serving stack is live: don't touch ~/services/, systemd, or
  docker, and never restart llama-swap or OVMS.
- Workflow: the ECC rules (~/.claude/rules/ecc/) govern how work gets
  done — plan first, tests before implementation, code review before
  any commit, conventional commit messages. Where an older doc here
  describes a "one pass, no questions" style, ECC wins.
- Benchmarks: record raw numbers, exact config, and concurrent machine
  load. Never judge the result — Aaron reads the numbers.
- Don't set GGML_OP_OFFLOAD_MIN_BATCH below 32: known correctness bug
  on this hardware.
- Branch hygiene: don't park work. Land the review gate, merge to
  master, and remove scratch branches/worktrees before the session
  ends. The llama.cpp fork keeps normal branch discipline.
- Agent-written persistence is facts or nothing: one-line entries in
  moe-review/open-items.md, orders, evidence files. Never prose about
  process, style, or how to behave — that class of doc regrows; delete
  it on sight.
