"""Benchmark service: run a labelled measurement and record it.

cache_service already owned metric scraping and
calibration; what was missing was the operation that runs a benchmark,
labels it with the mode that was actually achieved, and records it.

The labelling is the part that matters. `modelctl_benchmark` defines the
modes and their preconditions; a mode whose preconditions were not met
must not be recorded under that mode's name, or later ranking compares a
"page-cache-warm" number against one that was nothing of the kind.
"""
import modelctl

from .result import ServiceResult


def describe_modes() -> ServiceResult:
    """Every benchmark mode with its description, for the UI to offer."""
    import modelctl_benchmark
    modes = []
    for mode in modelctl_benchmark.BenchmarkMode:
        modes.append({
            "value": mode.value,
            "description": modelctl_benchmark.describe_mode(mode),
        })
    return ServiceResult(data={"modes": modes})


def validate_mode(mode, context=None) -> ServiceResult:
    """Can this mode's claims actually be honoured right now?"""
    import modelctl_benchmark
    if isinstance(mode, str):
        try:
            mode = modelctl_benchmark.BenchmarkMode(mode)
        except ValueError:
            return ServiceResult.failure(f"unknown benchmark mode: {mode}")
    ctx = context or modelctl_benchmark.BenchmarkContext()
    problems = modelctl_benchmark.validate_mode(mode, ctx)
    result = ServiceResult(data={"mode": mode.value})
    for p in problems:
        result.add_warning(p)
    result.data["label"] = modelctl_benchmark.label_for_result(
        mode, validated=not problems)
    return result


def context_for_profile(profile_name, consent_storage_cold=False):
    """Build the BenchmarkContext describing what this profile can achieve.

    Whether a mode is achievable depends on the actual model files, the
    filesystem they sit on, and the backend's capabilities -- so the
    context is derived, never assumed.
    """
    import modelctl_benchmark
    profile = modelctl.normalize_profile(modelctl.load_profile(profile_name))

    has_metrics = False
    try:
        import modelctl_capabilities
        import modelctl_launch
        backend = modelctl_launch.resolve_backend(profile)
        has_metrics = modelctl_capabilities.supports_metrics(backend.capabilities)
    except Exception:
        pass

    storage_info = None
    try:
        import modelctl_storage
        if profile.get("model_path"):
            storage_info = modelctl_storage.probe_storage(profile["model_path"])
    except Exception:
        pass

    companions = tuple(p for p in (profile.get("mmproj_path"),
                                   profile.get("mtp_path")) if p)
    return modelctl_benchmark.BenchmarkContext(
        has_cache_metrics=has_metrics,
        can_drop_caches=True,  # posix_fadvise is unprivileged and scoped
        model_path=profile.get("model_path", ""),
        storage_info=storage_info,
        consent_storage_cold=consent_storage_cold,
        companion_paths=companions,
    )


def prepare_mode(profile_name, mode, consent_storage_cold=False,
                 ctx=None) -> ServiceResult:
    """Put the machine into the state `mode` claims, and report the result.

    A precondition that could not be met does not abort the run; it
    changes the label the measurement is stored under, so a warm run can
    never later be read as a cold one.
    """
    import modelctl_benchmark
    if isinstance(mode, str):
        try:
            mode = modelctl_benchmark.BenchmarkMode(mode)
        except ValueError:
            return ServiceResult.failure(f"unknown benchmark mode: {mode}")

    bench_ctx = context_for_profile(profile_name, consent_storage_cold)
    problems = modelctl_benchmark.validate_mode(mode, bench_ctx)
    prep = modelctl_benchmark.prepare_mode(mode, bench_ctx)

    if ctx:
        for m in prep.messages:
            ctx.log(m)

    result = ServiceResult(
        ok=not prep.refused,
        messages=list(prep.messages),
        warnings=list(problems),
        data={"mode": mode.value, "label": prep.label or mode.value,
              "achieved": prep.achieved, "refused": prep.refused,
              "evidence": prep.evidence})
    if not prep.achieved and not prep.refused:
        result.add_warning(
            f"recorded as '{result.data['label']}': the {mode.value} "
            f"precondition was not met")
    return result


def run_benchmark(profile_name, mode=None, max_tokens=256, runs=3,
                  ctx=None) -> ServiceResult:
    """Run a benchmark and return the measurement with its honest label.

    A mode whose preconditions were not met is recorded under the label
    `label_for_result` gives it, never under the requested mode's name.
    """
    import modelctl_benchmark

    requested = mode or modelctl_benchmark.BenchmarkMode.NATURAL
    if isinstance(requested, str):
        try:
            requested = modelctl_benchmark.BenchmarkMode(requested)
        except ValueError:
            return ServiceResult.failure(f"unknown benchmark mode: {requested}")

    check = validate_mode(requested)
    if not check.ok:
        return check

    import os
    import subprocess

    cmd = modelctl.build_speed_command(profile_name, max_tokens=max_tokens,
                                       runs=runs)
    if ctx:
        ctx.log(f"benchmarking '{profile_name}' in {requested.value} mode")
        ctx.log(" ".join(cmd))
        ctx.raise_if_cancelled()

    output = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None)
        if ctx and hasattr(ctx, "register_process"):
            ctx.register_process(proc)
        for line in proc.stdout:
            line = line.rstrip("\n")
            output.append(line)
            if ctx:
                ctx.log(line)
                # Cancellation must reach the loop, not just the caller:
                # a benchmark can run for many minutes.
                ctx.raise_if_cancelled()
        proc.wait()
    except Exception as e:
        return ServiceResult.failure(f"benchmark failed: {e}",
                                     data={"mode": requested.value})
    if proc.returncode != 0:
        return ServiceResult.failure(
            f"benchmark exited {proc.returncode}",
            data={"mode": requested.value, "output_tail": output[-5:]})

    result = ServiceResult(
        warnings=list(check.warnings),
        data={"mode": requested.value,
              "label": check.data.get("label", requested.value),
              "output_tail": output[-5:]})
    if check.warnings:
        result.add_message(
            f"recorded as '{result.data['label']}' rather than "
            f"'{requested.value}': its preconditions were not met")
    return result.changed(f"observations:{profile_name}")
