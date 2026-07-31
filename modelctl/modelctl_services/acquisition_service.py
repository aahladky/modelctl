"""Model acquisition service: pull from Hugging Face, import from disk.

The web layer previously coordinated acquisition inline
in mutate.py, so the wizard's pre-flight checks lived in route code and
the CLI path had none of them.

Acquisition is the one workflow where the expensive, hard-to-undo step
(downloading tens of gigabytes) comes *before* anything is validated, so
the checks here run first and refuse rather than warn.
"""
import os
import shutil
from pathlib import Path

import modelctl

from .result import ServiceResult

# Refuse a download when the destination has less than the model's size
# plus this much headroom; a filesystem at 100% takes the whole desktop
# down with it, not just this job.
FREE_SPACE_HEADROOM = 2 << 30

GGUF_MAGIC = b"GGUF"


def verify_local_gguf(path) -> ServiceResult:
    """Check a local file really is a usable GGUF before it becomes a profile.

    The wizard verifies before profile creation. Doing it
    here rather than in the wizard means the CLI import path gets the same
    checks.
    """
    result = ServiceResult(data={"path": str(path)})
    p = Path(path)

    if not p.exists():
        return ServiceResult.failure(f"no such file: {p}", data={"path": str(p)})
    if p.is_dir():
        return ServiceResult.failure(f"{p} is a directory, not a GGUF file",
                                     data={"path": str(p)})
    if not os.access(p, os.R_OK):
        return ServiceResult.failure(f"{p} is not readable",
                                     data={"path": str(p)})

    size = p.stat().st_size
    result.data["size_bytes"] = size
    if size < 1024:
        return ServiceResult.failure(
            f"{p} is only {size} bytes — truncated or not a model",
            data={"path": str(p), "size_bytes": size})

    try:
        with open(p, "rb") as f:
            magic = f.read(4)
    except OSError as e:
        return ServiceResult.failure(f"cannot read {p}: {e}",
                                     data={"path": str(p)})
    if magic != GGUF_MAGIC:
        return ServiceResult.failure(
            f"{p} is not a GGUF file (magic {magic!r}, expected {GGUF_MAGIC!r})",
            data={"path": str(p), "size_bytes": size})

    # A sharded model referenced by its first shard is only usable if the
    # rest are present. Importing shard 1 of 5 produces a profile that
    # fails at load time with a much more confusing error.
    shards = _missing_shards(p)
    if shards:
        return ServiceResult.failure(
            f"{p.name} is part of a split model and "
            f"{len(shards)} shard(s) are missing: {', '.join(shards)}",
            data={"path": str(p), "missing_shards": shards})

    try:
        layout = _model_layout(p)
        if layout:
            result.data["weight_bytes"] = layout.get("weight_bytes")
    except Exception as e:
        result.add_warning(f"could not read the GGUF tensor layout: {e}")

    duplicate = _existing_profile_for(p)
    if duplicate:
        result.add_warning(
            f"profile '{duplicate}' already points at this file")
        result.data["duplicate_profile"] = duplicate

    result.add_message(f"{p.name}: valid GGUF, {modelctl._format_size(size)}")
    return result


def _missing_shards(path: Path):
    """Shard filenames referenced by a `-00001-of-000NN` name that are absent."""
    import re
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if not m:
        return []
    total = int(m.group(2))
    missing = []
    for i in range(1, total + 1):
        sibling = path.with_name(
            path.name[:m.start()] + f"-{i:05d}-of-{total:05d}.gguf")
        if not sibling.exists():
            missing.append(sibling.name)
    return missing


def _model_layout(path: Path):
    import modelctl_vram
    return modelctl_vram.gguf_model_layout(str(path))


def _existing_profile_for(path: Path):
    """Name of a saved profile already pointing at this exact file."""
    try:
        for f in Path(modelctl.PROFILES_DIR).glob("*.json"):
            import json
            try:
                data = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if data.get("model_path") and Path(data["model_path"]) == path:
                return data.get("name") or f.stem
    except OSError:
        pass
    return None


def check_destination_space(needed_bytes: int, destination=None) -> ServiceResult:
    """Refuse an acquisition that would not fit, before it starts."""
    dest = Path(destination or modelctl.DEFAULT_MODELS_DIR)
    result = ServiceResult(data={"destination": str(dest),
                                 "needed_bytes": needed_bytes})
    if not dest.is_dir():
        return ServiceResult.failure(f"destination {dest} does not exist",
                                     data=result.data)
    free = shutil.disk_usage(dest).free
    result.data["free_bytes"] = free
    required = (needed_bytes or 0) + FREE_SPACE_HEADROOM
    if needed_bytes and free < required:
        return ServiceResult.failure(
            f"{dest} has {modelctl._format_size(free)} free but this needs "
            f"{modelctl._format_size(needed_bytes)} plus "
            f"{modelctl._format_size(FREE_SPACE_HEADROOM)} headroom",
            data=result.data)
    result.add_message(f"{modelctl._format_size(free)} free at {dest}")
    return result


def import_local(file_path, name=None, copy=False, resync=True,
                 ctx=None) -> ServiceResult:
    """Verify then import a local GGUF as a profile."""
    check = verify_local_gguf(file_path)
    if not check.ok:
        return check

    if copy:
        space = check_destination_space(check.data.get("size_bytes", 0))
        if not space.ok:
            return space
        check.warnings.extend(space.warnings)

    if ctx:
        ctx.log(f"importing {file_path}")
        ctx.raise_if_cancelled()

    try:
        profile = modelctl.import_local(file_path, name=name, copy=copy,
                                        resync=resync)
    except Exception as e:
        return ServiceResult.failure(f"import failed: {e}")
    if profile is None:
        return ServiceResult.failure(f"import of {file_path} failed")

    result = ServiceResult(
        messages=check.messages + [f"profile '{profile['name']}' saved"],
        warnings=list(check.warnings),
        data={"profile": profile["name"], "config": profile.get("config", {})})
    result.changed(f"profile:{profile['name']}")
    if resync:
        result.changed("llama-swap:config")
    return result


def pull(repo_id, quant_label=None, want_mtp=True, progress_cb=None,
         ctx=None) -> ServiceResult:
    """Pull a model from Hugging Face and save it as a profile."""
    space = check_destination_space(0)
    if not space.ok:
        return space
    if ctx:
        ctx.raise_if_cancelled()
    try:
        profile = modelctl.pull_model(repo_id, quant_label=quant_label,
                                      want_mtp=want_mtp,
                                      progress_cb=progress_cb)
    except Exception as e:
        return ServiceResult.failure(f"pull of {repo_id} failed: {e}")
    if profile is None:
        return ServiceResult.failure(f"pull of {repo_id} failed")

    result = ServiceResult(
        messages=[f"profile '{profile['name']}' saved and synced"],
        data={"profile": profile["name"], "config": profile.get("config", {})})
    return result.changed(f"profile:{profile['name']}", "llama-swap:config")
