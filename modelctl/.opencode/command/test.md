---
description: Run the full modelctl unittest suite.
---

Run the full test suite from the repo root:

```bash
.venv/bin/python -m unittest test_modelctl test_modelctl_vram test_modelctl_tiers test_modelctl_tui test_modelctl_web test_modelctl_capabilities -v
```

Report the total count and any failures. Do not modify code to make tests pass unless explicitly asked. $ARGUMENTS
