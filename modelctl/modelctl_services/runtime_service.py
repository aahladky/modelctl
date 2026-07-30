"""Runtime operations service.

Load, unload, restart, and status queries through llama-swap. The single
implementation behind both `modelctl router load/unload` (CLI) and the web
console's runtime page (modelctl_web/mutate.py) -- neither one duplicates
llama-swap calls of its own.

Returns structured results, never prints or sys.exit()s. `ctx`, when given,
is a modelctl_web.jobs.JobContext (or anything duck-typing .log()/
.raise_if_cancelled()) for progress logging and cooperative cancellation;
CLI callers simply omit it. raise_if_cancelled() is deliberately NOT caught
here -- modelctl_web.jobs.JobRunner._run() handles CancelledError as a
distinct outcome from an ordinary failed RuntimeResult, so it must
propagate uncaught.
"""
from dataclasses import dataclass, field


@dataclass
class RuntimeResult:
    """Result of a runtime operation."""
    ok: bool
    loaded: bool = False
    response_ok: bool = False
    elapsed_s: float = 0.0
    models: list = field(default_factory=list)
    messages: list = field(default_factory=list)


def _swap_client():
    from modelctl_web.swap import LlamaSwapClient
    return LlamaSwapClient()


def load_model(name: str, ctx=None, timeout=1800) -> RuntimeResult:
    """Warm-load a model through llama-swap."""
    from modelctl_web.swap import ModelctlSwapError
    client = _swap_client()
    if ctx:
        ctx.log(f"warm-loading '{name}' through llama-swap ...")
        ctx.raise_if_cancelled()
    try:
        res = client.warm_load(name, timeout=timeout)
    except ModelctlSwapError as e:
        return RuntimeResult(ok=False, messages=[f"{e.code}: {e.message}"])
    if ctx:
        ctx.log(f"loaded={res.get('loaded')} response_ok={res.get('response_ok')} "
                f"in {res.get('elapsed_s')}s")
    if not res.get("loaded"):
        return RuntimeResult(
            ok=False, loaded=False, response_ok=res.get("response_ok", False),
            elapsed_s=res.get("elapsed_s", 0),
            messages=["model did not appear in /running after warm load"],
        )
    return RuntimeResult(
        ok=True, loaded=True, response_ok=res.get("response_ok", False),
        elapsed_s=res.get("elapsed_s", 0),
        messages=[f"loaded={res.get('loaded')} in {res.get('elapsed_s', 0):.1f}s"],
    )


def unload_model(name: str, ctx=None) -> RuntimeResult:
    """Unload a model from llama-swap."""
    from modelctl_web.swap import ModelctlSwapError
    client = _swap_client()
    try:
        client.unload(name)
    except ModelctlSwapError as e:
        return RuntimeResult(ok=False, messages=[f"{e.code}: {e.message}"])
    if ctx:
        ctx.log(f"unloaded '{name}'")
    return RuntimeResult(ok=True, messages=[f"unloaded '{name}'"])


def restart_model(name: str, ctx=None) -> RuntimeResult:
    """Unload (if running) then reload a model."""
    from modelctl_web.swap import ModelctlSwapError
    client = _swap_client()
    try:
        if name in client.running_model_ids():
            if ctx:
                ctx.log(f"unloading '{name}' ...")
            client.unload(name)
        if ctx:
            ctx.log(f"warm-loading '{name}' ...")
            ctx.raise_if_cancelled()
        res = client.warm_load(name)
    except ModelctlSwapError as e:
        return RuntimeResult(ok=False, messages=[f"{e.code}: {e.message}"])
    if ctx:
        ctx.log(f"loaded={res.get('loaded')} in {res.get('elapsed_s')}s")
    if not res.get("loaded"):
        return RuntimeResult(
            ok=False, elapsed_s=res.get("elapsed_s", 0),
            messages=["model did not come back after restart"],
        )
    return RuntimeResult(
        ok=True, loaded=True, response_ok=res.get("response_ok", False),
        elapsed_s=res.get("elapsed_s", 0),
        messages=[f"restarted '{name}' in {res.get('elapsed_s', 0):.1f}s"],
    )


def unload_all(ctx=None) -> RuntimeResult:
    """Unload all running models."""
    from modelctl_web.swap import ModelctlSwapError
    client = _swap_client()
    try:
        running = client.running_model_ids()
        client.unload_all()
    except ModelctlSwapError as e:
        return RuntimeResult(ok=False, messages=[f"{e.code}: {e.message}"])
    if ctx:
        ctx.log(f"unloaded all ({len(running)} model(s))")
    return RuntimeResult(ok=True, models=sorted(running),
                          messages=[f"unloaded {len(running)} model(s)"])


def list_running() -> RuntimeResult:
    """List currently loaded models."""
    from modelctl_web.swap import ModelctlSwapError
    try:
        client = _swap_client()
        running = client.running_model_ids()
        return RuntimeResult(ok=True, models=sorted(running))
    except ModelctlSwapError as e:
        return RuntimeResult(ok=False, messages=[f"{e.code}: {e.message}"])


def get_state() -> dict:
    """Get the full runtime state (all models, loaded/registered status).

    Returns a dict keyed by model name with state info.
    """
    import modelctl
    from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError

    state = {}
    # Registered profiles
    for p in sorted(modelctl.PROFILES_DIR.glob("*.json")):
        try:
            profile = modelctl.load_profile(p.stem)
            name = profile["name"]
            state[name] = {
                "name": name,
                "backend": profile.get("backend", "llama-cpp"),
                "registered": True,
                "loaded": False,
                "runtime_mode": profile.get("runtime", {}).get("mode", "fixed"),
            }
        except Exception:
            pass

    # Running models from llama-swap
    try:
        client = LlamaSwapClient()
        for mid in client.running_model_ids():
            if mid in state:
                state[mid]["loaded"] = True
            else:
                state[mid] = {
                    "name": mid, "backend": "unknown",
                    "registered": False, "loaded": True,
                    "runtime_mode": "unknown",
                }
    except ModelctlSwapError:
        pass

    return state
