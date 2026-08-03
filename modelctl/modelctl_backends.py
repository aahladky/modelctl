"""Backend adapters (Milestone: managed runtime beyond llama.cpp).

The managed worker used to assume one binary, --port, /health, and
llama.cpp-style arguments. Adapters own those per-backend details:
command creation, environment, readiness checks, failure classification.

llama.cpp is the only backend since the OVMS teardown (2026-08-03,
tag ovms-final); the adapter seam stays so a future backend is an
adapter away, not a special case.
"""
import os
import re


class BackendError(Exception):
    pass


class LlamaCppAdapter:
    name = "llama-cpp"

    def build_command(self, profile, plan, port, binary=None):
        from modelctl_worker import _build_command
        return _build_command(profile, plan, port, binary=binary)

    # NOTE: there is deliberately no environment method here. The launch
    # environment is assembled once, inside
    # modelctl_launch.build_launch_command(), and is part of the command's
    # fingerprinted identity -- an adapter-level env would be a second
    # place to construct it.

    def readiness_url(self, profile, port):
        return f"http://127.0.0.1:{port}/health"

    def classify_failure(self, exit_code, log_tail):
        """Map exit state + log tail to a failure class."""
        tail = (log_tail or "").lower()
        if "out_of_device_memory" in tail or "out of memory" in tail and "vram" in tail:
            return "out_of_vram"
        if "out of memory" in tail or "oom" in tail:
            return "out_of_ram"
        if "unknown model architecture" in tail:
            return "unsupported_architecture"
        if "unknown buffer type" in tail or "invalid value" in tail:
            return "invalid_argument"
        if "no such file" in tail and "llama-server" in tail:
            return "missing_binary"
        if "error while loading shared libraries" in tail:
            return "missing_library"
        if "no device of requested type" in tail:
            return "device_unavailable"
        if exit_code is None:
            return "health_timeout"
        return "backend_crash"


BACKENDS = {
    "llama-cpp": LlamaCppAdapter(),
}


def get_backend(name):
    adapter = BACKENDS.get(name or "llama-cpp")
    if adapter is None:
        raise BackendError(f"no backend adapter for '{name}'")
    return adapter
