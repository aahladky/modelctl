---
description: Run --modelctl-capabilities on available llama-server builds.
---

Probe capability output of every relevant binary:

1. SYCL fork (required env):
   ```bash
   cd ../llama.cpp && source llama-sycl-env.sh && ./build-sycl/bin/llama-server --modelctl-capabilities
   ```
2. CPU-only build, if `../llama.cpp/build-cpu/bin/llama-server` exists — same probe, without the SYCL env.
3. Any stock llama-server on PATH.

Check the responses against `modelctl_capabilities.py`: schema handling must fail closed for unsupported, malformed, or CPU-only responses. Report the parsed feature sets and any mismatch with what modelctl expects. $ARGUMENTS
