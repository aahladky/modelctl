"""Canonical resolved-backend and launch-command types for modelctl.

Every launch path (CLI preview, managed worker, plan test, llama-swap
entry, artifact generation) must derive from the same ResolvedBackend
and LaunchCommand objects.  No path may reconstruct the command
independently after validation.

Public API:
    ResolvedBackend dataclass
    LaunchCommand dataclass
    resolve_backend(profile, binary_override=None) -> ResolvedBackend
    build_launch_command(profile, plan, backend, port=None) -> LaunchCommand
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

import modelctl
import modelctl_capabilities
import modelctl_vram


@dataclass(frozen=True)
class ResolvedBackend:
    """A concrete backend binary with its environment and probed capabilities."""
    name: str                    # "llama-cpp", "ovms"
    binary: str                  # resolved absolute path
    binary_fingerprint: str      # content hash
    environment: dict[str, str]  # effective env vars for launch
    environment_fingerprint: str # hash of sorted env
    capabilities: dict           # normalized schema-2 capabilities


@dataclass(frozen=True)
class LaunchCommand:
    """The single authoritative command for launching a model.

    Preview, worker launch, plan test, and llama-swap entries all
    derive from this object.  The command_fingerprint ties together
    runtime history, observations, and decision traces.
    """
    argv: tuple[str, ...]        # tokenized command (never a shell string)
    environment: dict[str, str]  # effective environment
    backend: ResolvedBackend
    profile_name: str
    plan_id: str
    port: int | None             # None = not yet assigned
    warnings: tuple[str, ...]
    validation: tuple            # tuple of ValidationMessage (or legacy tuples)
    command_fingerprint: str     # hash of normalized command identity


def _env_fingerprint(env: dict[str, str]) -> str:
    """Stable hash of the launch environment."""
    text = json.dumps(sorted(env.items()), separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _command_fingerprint(argv: tuple, env_fp: str, backend_fp: str) -> str:
    """Hash of the port-independent command identity.

    Port is excluded so the same logical command across restarts shares
    an identity for observation correlation.
    """
    # Strip --port from argv for identity.
    identity_args = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--port", "-p"):
            skip_next = True
            continue
        identity_args.append(arg)
    text = json.dumps({
        "argv": identity_args,
        "env": env_fp,
        "backend": backend_fp,
    }, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def resolve_backend(profile: dict, binary_override: str | None = None) -> ResolvedBackend:
    """Resolve a profile's backend to a concrete binary, environment, and capabilities.

    This is the single place where binary selection, oneAPI environment
    resolution, and capability probing happen.  All launch paths must
    call this instead of doing their own resolution.
    """
    backend_name = profile.get("backend", "llama-cpp")
    binary = binary_override or profile.get("binary") or modelctl.LLAMA_SERVER_BIN

    # Resolve oneAPI environment for SYCL builds.
    env = dict(os.environ)
    profile_env = profile.get("env") or []
    for entry in profile_env:
        if "=" in entry:
            k, v = entry.split("=", 1)
            env[k] = v

    # If the binary is a SYCL build, it may need oneAPI env scripts.
    # The capability probe handles this internally via _probe_raw's
    # retry logic, but we also need the env for the actual launch.
    if backend_name == "llama-cpp" and not binary_override:
        try:
            for script in modelctl.find_env_script_candidates():
                script_env = modelctl.source_env_script(script)
                if script_env:
                    env.update(script_env)
                    break
        except (ImportError, AttributeError):
            pass

    binary_fp = modelctl_vram.file_fingerprint(binary)
    env_fp = _env_fingerprint(env)

    # Probe capabilities.
    caps = modelctl_capabilities.probe_backend(binary)

    return ResolvedBackend(
        name=backend_name,
        binary=binary,
        binary_fingerprint=binary_fp,
        environment=env,
        environment_fingerprint=env_fp,
        capabilities=caps,
    )


def build_launch_command(
    profile: dict,
    plan,
    backend: ResolvedBackend | None = None,
    port: int | None = None,
) -> LaunchCommand:
    """Build the canonical LaunchCommand for a profile+plan combination.

    If backend is None, resolve_backend() is called.  This ensures
    every path gets the same binary, environment, and capabilities.
    """
    if backend is None:
        backend = resolve_backend(profile)

    from modelctl_errors import ValidationMessage, from_preflight_tuples

    warnings = []
    validation = []

    # Run preflight checks.
    ok, effective_bin, effective_env, preflight_msgs = modelctl.preflight(
        profile, auto_fix=True)

    # Validate cache configuration against capabilities.
    cache_msgs = modelctl.preflight_moe_cache(
        profile, capabilities=backend.capabilities)
    validation.extend(from_preflight_tuples(cache_msgs))

    # Convert legacy preflight messages.
    for severity, summary in preflight_msgs:
        if severity == "warning":
            warnings.append(summary)
        elif severity == "error":
            validation.append(ValidationMessage(
                code="backend_feature_missing",
                severity="error",
                summary=summary,
            ))

    # Build the command argv.
    from modelctl_backends import get_backend
    adapter = get_backend(backend.name)
    argv = tuple(adapter.build_command(profile, plan, port, binary=backend.binary))

    cmd_fp = _command_fingerprint(argv, backend.environment_fingerprint,
                                  backend.binary_fingerprint)

    return LaunchCommand(
        argv=argv,
        environment=backend.environment,
        backend=backend,
        profile_name=profile.get("name", ""),
        plan_id=plan.id if hasattr(plan, "id") else "",
        port=port,
        warnings=tuple(warnings),
        validation=tuple(validation),
        command_fingerprint=cmd_fp,
    )
