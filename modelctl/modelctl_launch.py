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
    launch_command_for_profile(profile, port=None, ...) -> LaunchCommand
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

import modelctl
import modelctl_capabilities
import modelctl_vram


class LaunchValidationError(RuntimeError):
    """A LaunchCommand that failed validation was asked to launch."""

    def __init__(self, command):
        self.command = command
        summaries = "; ".join(
            getattr(v, "summary", str(v)) for v in command.errors)
        super().__init__(
            f"launch blocked for profile '{command.profile_name}': {summaries}")


@dataclass(frozen=True)
class ResolvedBackend:
    """A concrete backend binary with its environment and probed capabilities."""
    name: str                    # "llama-cpp", "ovms"
    binary: str                  # resolved absolute path
    binary_fingerprint: str      # content hash
    environment: dict[str, str]  # full launch env (os.environ + overrides)
    environment_fingerprint: str # hash of sorted env
    capabilities: dict           # normalized schema-2 capabilities
    # Digest of `capabilities`. Recorded with every observation so a
    # rebuilt runtime that reports different features stales measurements
    # taken under the old contract (roadmap Task B3).
    capability_fingerprint: str = ""
    # Just the vars this profile adds on top of the ambient environment.
    # Rendered artifacts (run.sh exports, llama-swap `env:`) must use this,
    # never `environment` -- writing the whole inherited process env into a
    # checked-in config file leaks unrelated (and possibly secret) vars.
    environment_overrides: dict[str, str] = field(default_factory=dict)
    # Result of the single preflight() call resolve_backend() makes.
    preflight_ok: bool = True
    preflight_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchCommand:
    """The single authoritative command for launching a model.

    Preview, worker launch, plan test, and llama-swap entries all
    derive from this object.  The command_fingerprint ties together
    runtime history, observations, and decision traces.
    """
    argv: tuple[str, ...]        # tokenized command (never a shell string)
    # The COMPLETE launch environment: backend resolution + plan-specific
    # overrides + library-path repair, in that order. Callers launch with
    # exactly this dict and never mutate it -- a caller-side env change
    # after this object exists would not be part of the fingerprints below.
    environment: dict[str, str]
    backend: ResolvedBackend
    profile_name: str
    plan_id: str
    port: int | None             # None = not yet assigned
    warnings: tuple[str, ...]
    validation: tuple            # tuple of ValidationMessage (or legacy tuples)
    command_fingerprint: str     # hash of normalized command identity
    # Fingerprint of identity_environment(environment) -- the final env
    # this command launches under, not the pre-plan backend one. Recorded
    # with every run and observation.
    environment_fingerprint: str = ""

    @property
    def errors(self) -> tuple:
        """Blocking validation messages, if any."""
        return tuple(v for v in self.validation
                     if getattr(v, "severity", None) == "error")

    @property
    def is_valid(self) -> bool:
        """False when this command must not be launched or written out.

        Every path that starts a process, writes an artifact, or syncs a
        llama-swap entry has to consult this (roadmap Task B2).  A command
        that failed validation is still returned rather than raised on, so
        the browser and CLI can show the user exactly what was rejected and
        why -- but it is not launchable.
        """
        return not self.errors

    def raise_for_errors(self):
        """Raise LaunchValidationError if this command must not be launched.

        For callers with no way to surface a message list (the managed
        worker's start path, plan tests) -- the process must die loudly
        rather than run a command validation rejected.
        """
        if self.errors:
            raise LaunchValidationError(self)
        return self


# The environment variables that change what the launched process actually
# does, and therefore belong to launch identity. Everything else in the
# process environment (SSH vars, locale, terminal state) is noise: hashing
# it would mass-stale observations on every unrelated shell change.
#
# Hardware measurements motivated this list: GGML_OP_OFFLOAD_MOE_MIN_BATCH
# alone can reverse the performance ranking between storage-bound and
# VRAM-resident placements, so two commands differing only in it must not
# share an identity, reuse each other's observations, or show one preview.
_IDENTITY_ENV_PREFIXES = (
    "GGML_",      # ggml scheduler/offload behavior, incl. MoE thresholds
    "LLAMA_",     # llama.cpp runtime switches
    "SYCL_",      # SYCL runtime selection and caching
    "ZE_",        # Level Zero (device visibility, affinity)
    "ZES_",       # Level Zero sysman
    "UR_",        # unified runtime adapter selection
    "ONEAPI_",    # oneAPI device selector and roots
)
_IDENTITY_ENV_VARS = (
    "LD_LIBRARY_PATH",       # selects which runtime libraries actually load
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)


def identity_environment(env: dict[str, str]) -> dict[str, str]:
    """The normalized behavior-affecting subset of a launch environment.

    This is the whitelist that goes into environment and command
    fingerprints. Empty values are dropped: "unset" and "set to nothing"
    must not produce two identities."""
    out = {}
    for k, v in env.items():
        if not v:
            continue
        if k in _IDENTITY_ENV_VARS or k.startswith(_IDENTITY_ENV_PREFIXES):
            out[k] = v
    return dict(sorted(out.items()))


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

    The pipeline runs in one fixed order and exactly once (roadmap Task
    B4)::

        resolve candidate binary + effective environment (preflight)
        -> fingerprint binary and environment
        -> probe capabilities in that environment
        -> return an immutable ResolvedBackend

    build_launch_command() consumes the preflight result recorded here
    instead of running preflight a second time.  The two calls used to
    disagree: preflight's auto-fix could land on a different binary than
    the one the command was built around, and resolve_backend sourced an
    oneAPI env script for *every* llama-cpp profile while preflight only
    sourced one for SYCL profiles missing LD_LIBRARY_PATH -- so a worker
    launched with an environment the generated run.sh never exported.
    """
    backend_name = profile.get("backend", "llama-cpp")

    preflight_ok = True
    preflight_messages: tuple[str, ...] = ()
    overrides: dict[str, str] = {}

    if backend_name == "llama-cpp":
        # modelctl.preflight() is the real binary-resolution logic: it
        # honors a pinned profile["binary"], and otherwise auto-fix
        # searches known build locations for an alternative that actually
        # supports this profile's device when the configured default
        # (MODELCTL_LLAMA_SERVER, or "llama-server" on PATH) doesn't exist
        # or doesn't support it.  It also owns environment resolution:
        # profile env entries plus, for SYCL profiles with no saved
        # LD_LIBRARY_PATH, the first oneAPI script that provides one.
        ok, effective_bin, effective_env, messages = modelctl.preflight(
            profile, auto_fix=True)
        preflight_ok = ok
        preflight_messages = tuple(messages)
        overrides = dict(effective_env)
        binary = binary_override or effective_bin or modelctl.LLAMA_SERVER_BIN
    else:
        binary = binary_override or profile.get("binary") or modelctl.LLAMA_SERVER_BIN
        for entry in profile.get("env") or []:
            if "=" in entry:
                k, v = entry.split("=", 1)
                overrides[k] = v

    if backend_name == "llama-cpp" and binary and os.path.exists(binary):
        # A binary_override (autotune sweeping alternative builds) is one
        # preflight never saw, so its RUNPATH still has to reach
        # LD_LIBRARY_PATH.  Idempotent for the non-override case.
        modelctl.ensure_binary_ld_library_path(overrides, binary)

    env = dict(os.environ)
    env.update(overrides)

    binary_fp = modelctl_vram.file_fingerprint(binary)
    # Fingerprint the behavior-affecting subset of the EFFECTIVE
    # environment -- profile-declared vars plus ambient whitelisted ones
    # (offload thresholds, device visibility, library paths). It used to
    # hash only profile-declared vars, so an ambient
    # GGML_OP_OFFLOAD_MOE_MIN_BATCH changed runtime behavior without
    # changing identity and measurements taken under it were offered as
    # evidence for launches without it. Unrelated shell noise (SSH vars,
    # locale) stays excluded by the whitelist.
    env_fp = _env_fingerprint(identity_environment(env))

    # Probe capabilities.  Only llama-cpp binaries speak
    # --modelctl-capabilities; other backends get an empty (fail-closed)
    # capability set.  The probe runs under the same environment the
    # launch will use -- a SYCL binary probed under a different oneAPI
    # environment than it launches with can report different devices, or
    # nothing at all.
    if backend_name == "llama-cpp":
        caps = modelctl_capabilities.probe_backend(binary, env=env)
    else:
        caps = {"schema": 0, "backend": backend_name, "build": "",
                "features": {}, "cli": {}, "_probe_status": "unsupported"}

    return ResolvedBackend(
        name=backend_name,
        binary=binary,
        binary_fingerprint=binary_fp,
        environment=env,
        environment_fingerprint=env_fp,
        capabilities=caps,
        capability_fingerprint=modelctl_capabilities.capability_fingerprint(caps),
        environment_overrides=overrides,
        preflight_ok=preflight_ok,
        preflight_messages=preflight_messages,
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

    if backend.name == "llama-cpp":
        # Validate cache configuration against the capabilities of the
        # binary resolve_backend() actually picked.
        cache_msgs = modelctl.preflight_moe_cache(
            profile, capabilities=backend.capabilities)
        validation.extend(from_preflight_tuples(cache_msgs))

        # preflight() already ran once, inside resolve_backend(); reuse its
        # messages rather than running it again (roadmap Task B4).  It
        # returns plain message strings with the severity as a text prefix
        # ("ERROR: ..."/"WARNING: ...", or no prefix at all for
        # informational auto-fix notes) -- NOT (severity, summary) tuples
        # like preflight_moe_cache() above.  Every test mocking preflight()
        # passes an empty message list, so unpacking these as tuples here
        # crashed on every real profile that has anything to report (which
        # is nearly all of them).
        for msg in backend.preflight_messages:
            if msg.startswith("ERROR:"):
                validation.append(ValidationMessage(
                    code="backend_feature_missing",
                    severity="error",
                    summary=msg[len("ERROR:"):].strip(),
                ))
            elif msg.startswith("WARNING:"):
                warnings.append(msg[len("WARNING:"):].strip())

    # Build the command argv.
    from modelctl_backends import get_backend
    adapter = get_backend(backend.name)
    argv = tuple(adapter.build_command(profile, plan, port, binary=backend.binary))

    # The final launch environment, assembled HERE and nowhere else:
    # backend resolution already applied os.environ + preflight overrides;
    # plan-specific env goes on top, then the library path is repaired in
    # case a plan var clobbered the LD_LIBRARY_PATH preflight built.
    # Workers and plan tests used to re-apply plan.env themselves after
    # this object existed, which meant the previewed environment, the
    # probed environment, and the launched one could all differ.
    env = dict(backend.environment)
    env.update(getattr(plan, "env", None) or {})
    if backend.name == "llama-cpp" and backend.binary:
        modelctl.ensure_binary_ld_library_path(env, backend.binary)

    env_fp = _env_fingerprint(identity_environment(env))
    cmd_fp = _command_fingerprint(argv, env_fp, backend.binary_fingerprint)

    return LaunchCommand(
        argv=argv,
        environment=env,
        backend=backend,
        profile_name=profile.get("name", ""),
        plan_id=plan.id if hasattr(plan, "id") else "",
        port=port,
        warnings=tuple(warnings),
        validation=tuple(validation),
        command_fingerprint=cmd_fp,
        environment_fingerprint=env_fp,
    )


def launch_command_for_profile(
    profile: dict,
    port: int | None = None,
    plan=None,
    backend: ResolvedBackend | None = None,
) -> LaunchCommand:
    """The canonical LaunchCommand for a profile exactly as it is saved.

    Surfaces that render or launch "what this profile means right now" --
    the browser command preview, the generated run.sh, the llama-swap
    entry, and the smoke test -- have no plan of their own.  They used to
    each call preflight() plus build_server_args() directly, which meant
    four independent reconstructions of the command after validation, all
    free to drift from what the worker actually launches (roadmap Task
    B1).  They call this instead.

    Pass port=None for rendering: the argv then carries no --port at all,
    leaving the caller to substitute its own placeholder (``${PORT}``).
    Port is excluded from command_fingerprint either way, so a rendered
    command and a launched one still compare equal.
    """
    if backend is None:
        backend = resolve_backend(profile)
    if plan is None:
        import modelctl_plans
        plan = modelctl_plans.current_profile_plan(
            profile, capabilities=backend.capabilities)
    return build_launch_command(profile, plan, backend=backend, port=port)
