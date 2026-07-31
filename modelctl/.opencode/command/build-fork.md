---
description: Rebuild the SYCL llama-server fork target and report warnings.
---

Build the fork server:

```bash
cmake --build ../llama.cpp/build-sycl -j --target llama-server
```

Do not reconfigure cmake or change build flags unless asked. Report any warnings or errors in `moe-cache`, `ggml-backend.cpp`, or other fork-modified files separately from pre-existing upstream warnings. $ARGUMENTS
