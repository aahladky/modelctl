"""Runtime operations service.

Load, unload, restart, and status queries through llama-swap.
Returns structured results, never prints or sys.exit.
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


def load_model(name: str) -> RuntimeResult:
    """Warm-load a model through llama-swap."""
    try:
        client = _swap_client()
        res = client.warm_load(name)
        return RuntimeResult(
            ok=res.get("loaded", False),
            loaded=res.get("loaded", False),
            response_ok=res.get("response_ok", False),
            elapsed_s=res.get("elapsed_s", 0),
            messages=[f"loaded={res.get('loaded')} in {res.get('elapsed_s', 0):.1f}s"],
        )
    except Exception as e:
        return RuntimeResult(ok=False, messages=[str(e)])


def unload_model(name: str) -> RuntimeResult:
    """Unload a model from llama-swap."""
    try:
        client = _swap_client()
        client.unload(name)
        return RuntimeResult(ok=True, messages=[f"unloaded '{name}'"])
    except Exception as e:
        return RuntimeResult(ok=False, messages=[str(e)])


def restart_model(name: str) -> RuntimeResult:
    """Unload then reload a model."""
    try:
        client = _swap_client()
        if name in client.running_model_ids():
            client.unload(name)
        res = client.warm_load(name)
        return RuntimeResult(
            ok=res.get("loaded", False),
            loaded=res.get("loaded", False),
            elapsed_s=res.get("elapsed_s", 0),
            messages=[f"restarted '{name}' in {res.get('elapsed_s', 0):.1f}s"],
        )
    except Exception as e:
        return RuntimeResult(ok=False, messages=[str(e)])


def unload_all() -> RuntimeResult:
    """Unload all running models."""
    try:
        client = _swap_client()
        running = client.running_model_ids()
        client.unload_all()
        return RuntimeResult(
            ok=True, models=sorted(running),
            messages=[f"unloaded {len(running)} model(s)"],
        )
    except Exception as e:
        return RuntimeResult(ok=False, messages=[str(e)])


def list_running() -> RuntimeResult:
    """List currently loaded models."""
    try:
        client = _swap_client()
        running = client.running_model_ids()
        return RuntimeResult(ok=True, models=sorted(running))
    except Exception as e:
        return RuntimeResult(ok=False, messages=[str(e)])


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
    except Exception:
        pass

    return state
