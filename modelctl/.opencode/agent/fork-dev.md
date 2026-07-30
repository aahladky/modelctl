---
description: C++/SYCL work in the llama.cpp fork submodule (../llama.cpp) — MoE cache runtime, backend API, capability probe, fork builds and runtime tests. Use for roadmap runtime tasks (1.2, 5.x, 7.x).
mode: subagent
permission:
  bash:
    "*": ask
    "cmake --build*": allow
    "ctest*": allow
    "git status*": allow
    "git diff*": allow
    "git log*": allow
    "git show*": allow
---

You develop inside `../llama.cpp` (branch `feature/sycl-moe-expert-cache`, fork of ggml-org/llama.cpp).

Rules:

- Only edit files under `../llama.cpp`. If a change belongs in modelctl (control plane), describe it in your report instead of making it.
- Before running any SYCL binary: `source ../llama.cpp/llama-sycl-env.sh`.
- Build: `cmake --build ../llama.cpp/build-sycl -j --target llama-server` (or the target the task names).
- The cache runtime lives in `ggml/src/ggml-sycl/moe-cache.{hpp,cpp}` with the scheduler hook in `ggml/src/ggml-backend.cpp`; CLI flags in `common/arg.cpp`; metrics/reset/probe in `tools/server/`.
- Never commit, push, rebase, or advance the modelctl submodule pin.
- Correctness is mandatory over speed: forced-hit, forced-miss, and mixed paths must match cache-disabled output (roadmap Task 5.7 / Phase 7).
- Verify before reporting: clean build, relevant runtime/ctest tests, and the output of `./build-sycl/bin/llama-server --modelctl-capabilities` (with env sourced) whenever capability behavior changes.
