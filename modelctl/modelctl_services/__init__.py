"""Application services for modelctl.

Service functions receive typed inputs and return typed results.
They do not print, call sys.exit(), or touch the UI directly.
CLI handlers and HTTP routes both call these.

Service functions that are long-running accept a `ctx` parameter
(JobContext from the web layer) for logging, cancellation, and
subprocess registration. When called from the CLI, ctx can be a
simple CLIContext that prints to stderr.

Every service operation returns a `ServiceResult` (or a subclass adding
its own payload fields): ok, messages, warnings, data, changed_resources,
job_id, rollback_status. See result.py.

    acquisition_service  pull from Hugging Face, import + verify local GGUFs
    benchmark_service    run a labelled measurement
    cache_service        MoE cache metrics, storage calibration primitives
    hardware_service     snapshots, device settings, calibration
    plan_service         compile, rank, select, test launch plans
    profile_service      profile lifecycle mutations
    routing_service      llama-swap config sync and matrix apply/rollback
    runtime_service      load/unload/restart through llama-swap
    settings_service     defaults, diagnostics, support bundle
"""
from .result import (ROLLBACK_DONE, ROLLBACK_FAILED, ROLLBACK_NONE,
                     ROLLBACK_PARTIAL, ServiceResult)

__all__ = ["ServiceResult", "ROLLBACK_NONE", "ROLLBACK_DONE",
           "ROLLBACK_PARTIAL", "ROLLBACK_FAILED"]
