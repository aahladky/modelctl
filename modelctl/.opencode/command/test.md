---
description: Run the full modelctl unittest suite.
---

Run the full test suite from the repo root:

```bash
.venv/bin/python -m unittest discover -p "test_*.py"
```

(`discover` picks up every `test_*.py` file -- don't hand-list modules,
that list has gone stale before and silently skipped most of the suite.)

Report the total count and any failures. Do not modify code to make tests pass unless explicitly asked. $ARGUMENTS
