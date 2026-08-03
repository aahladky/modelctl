"""llama-swap routing service: config sync, matrix apply, rollback.

The matrix apply was the clearest example of what this service layer
is for: a multi-step mutation (back up, rewrite config, restart a
service, health-check, roll back on failure) coordinated inline in a web
route closure, so the CLI had no equivalent and nothing could report what
the rollback actually did.

`rollback_status` on the result is the point: "the apply failed" and "the
apply failed and we could not put the old config back" are very different
situations for whoever is looking at the screen.
"""
import shutil
import subprocess
import time
from pathlib import Path

import modelctl

from .result import (ROLLBACK_DONE, ROLLBACK_FAILED, ROLLBACK_NONE,
                     ServiceResult)

HEALTH_ATTEMPTS = 15
HEALTH_INTERVAL_S = 2
RESTART_TIMEOUT_S = 60


def _restart_swap(timeout=RESTART_TIMEOUT_S):
    return subprocess.run(
        ["systemctl", "--user", "restart", modelctl.LLAMA_SWAP_SERVICE_NAME],
        capture_output=True, text=True, timeout=timeout)


def _wait_healthy(log=None):
    from modelctl_web.swap import LlamaSwapClient, ModelctlSwapError
    for _ in range(HEALTH_ATTEMPTS):
        try:
            LlamaSwapClient().health()
            return True
        except ModelctlSwapError:
            time.sleep(HEALTH_INTERVAL_S)
    return False


def sync_config(restart: bool = True, ctx=None) -> ServiceResult:
    """Regenerate the llama-swap config from all saved profiles."""
    try:
        count = modelctl.sync_all_backends(restart_router=restart,
                                           restart_openarc=restart)
    except Exception as e:
        return ServiceResult.failure(f"sync failed: {e}")
    result = ServiceResult(messages=[f"synced {count} profile{'s' if count != 1 else ''}"],
                           data={"profiles": count})
    return result.changed("llama-swap:config")


def apply_matrix(ctx=None) -> ServiceResult:
    """Generate and apply the managed routing matrix, rolling back on failure.

    Every exit path reports rollback_status, so a caller never has to
    guess whether the on-disk config is the old one or a broken new one.
    """
    def log(msg):
        if ctx:
            ctx.log(msg)

    import modelctl_matrix
    cfg_path = Path(modelctl.LLAMA_SWAP_CONFIG_PATH)
    try:
        config = modelctl.yaml.safe_load(cfg_path.read_text()) or {}
    except OSError as e:
        return ServiceResult.failure(f"cannot read {cfg_path}: {e}")

    generated = modelctl_matrix.generate_matrix()
    merged = modelctl_matrix.merge_matrix(config.get("matrix"), generated)

    backup = cfg_path.with_suffix(f".yaml.bak-matrix-{int(time.time())}")
    try:
        shutil.copy2(cfg_path, backup)
    except OSError as e:
        return ServiceResult.failure(f"cannot write backup {backup}: {e}")
    log(f"backup: {backup}")

    new_config = dict(config)
    new_config["matrix"] = merged
    tmp = cfg_path.with_suffix(".yaml.tmp")
    try:
        tmp.write_text(modelctl.yaml.safe_dump(new_config, sort_keys=False))
        # fsync before the rename: a crash between write and rename must
        # leave the old config intact, not a half-written new one.
        with open(tmp) as f:
            import os
            os.fsync(f.fileno())
        tmp.replace(cfg_path)
    except OSError as e:
        return ServiceResult.failure(f"cannot write {cfg_path}: {e}",
                                     rollback_status=ROLLBACK_NONE)
    log("config.yaml written, restarting llama-swap")

    proc = _restart_swap()
    log((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0:
        return _rollback(cfg_path, backup, log,
                         "llama-swap restart failed after applying the matrix")

    if not _wait_healthy():
        return _rollback(cfg_path, backup, log,
                         "llama-swap did not become healthy after applying "
                         "the matrix")

    log("llama-swap healthy after apply")
    result = ServiceResult(
        messages=[f"applied {len(merged['sets'])} routing set(s)"],
        data={"sets": len(merged["sets"]), "backup": str(backup)})
    return result.changed("llama-swap:config", "llama-swap:service")


def _rollback(cfg_path, backup, log, why) -> ServiceResult:
    """Restore the backup and report honestly whether it worked."""
    log(f"{why} — rolling back to {backup}")
    try:
        shutil.copy2(backup, cfg_path)
    except OSError as e:
        return ServiceResult(
            ok=False,
            messages=[why, f"ROLLBACK FAILED: could not restore {backup}: {e}"],
            data={"backup": str(backup)},
            changed_resources=["llama-swap:config"],
            rollback_status=ROLLBACK_FAILED)
    proc = _restart_swap()
    if proc.returncode != 0:
        return ServiceResult(
            ok=False,
            messages=[why,
                      "config restored from backup, but llama-swap did not "
                      "restart — start it manually"],
            data={"backup": str(backup)},
            changed_resources=["llama-swap:config"],
            rollback_status=ROLLBACK_FAILED)
    return ServiceResult(
        ok=False,
        messages=[why, f"rolled back to {backup}"],
        data={"backup": str(backup)},
        rollback_status=ROLLBACK_DONE)
