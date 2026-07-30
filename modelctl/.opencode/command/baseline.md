---
description: Capture the Phase 0 hardware/software baseline (roadmap Task 0.1).
---

Perform roadmap Task 0.1 from `docs/modelctl-task-by-task-roadmap-2026-07-29.md`:

1. Record `git rev-parse HEAD` and `git submodule status`.
2. Record versions: `.venv/bin/python -V`, relevant `pip freeze` entries (textual, fastapi, uvicorn, huggingface-hub), llama-swap version, `uname -r`, `icpx --version`, GPU driver if queryable.
3. Run the full test suite and record pass/fail counts.
4. If no CPU-only build exists, create one: `cmake -B ../llama.cpp/build-cpu -S ../llama.cpp -DGGML_SYCL=OFF` then `cmake --build ../llama.cpp/build-cpu -j --target llama-server`, and run `--modelctl-capabilities` on it.
5. Run the SYCL probe (source `llama-sycl-env.sh` first).
6. Save one known-good fixed-profile command and one managed-worker command.
7. Write everything to `docs/BASELINE-2026-07-hardware-serving.md`, distinguishing environment failures from product failures. Do not commit.
