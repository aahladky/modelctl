---
description: Implement a roadmap task by number, e.g. /task 1.3
---

Read task $ARGUMENTS in `docs/modelctl-task-by-task-roadmap-2026-07-29.md` (plus the phase intro and cross-cutting test plan in section 10).

Then:

1. Restate the task's goal, file list, tests, and exit criteria before touching code.
2. Write or extend tests first (or alongside), in the modules the task names.
3. Implement only in the files the task lists; follow AGENTS.md conventions.
4. Respect roadmap invariants: experimental features fail closed, one canonical launch path, cold/warm results never conflated, control plane stays in Python, tensor execution stays in the fork.
5. Run the full suite: `.venv/bin/python -m unittest test_modelctl test_modelctl_vram test_modelctl_tiers test_modelctl_tui test_modelctl_web test_modelctl_capabilities -v`
6. Do not commit. Report changes against each exit criterion.
