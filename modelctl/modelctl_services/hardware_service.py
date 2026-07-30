"""Hardware settings service.

Hardware snapshot capture, settings management, and GPU inventory.
Returns structured results, never prints or sys.exit.
"""
import time
from dataclasses import dataclass, field

import modelctl
import modelctl_hardware


@dataclass
class HardwareResult:
    """Result of a hardware operation."""
    ok: bool
    snapshot: object = None
    settings: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


def capture_snapshot() -> HardwareResult:
    """Capture the current hardware snapshot."""
    try:
        snap = modelctl_hardware.capture_hardware_snapshot()
        return HardwareResult(ok=True, snapshot=snap)
    except Exception as e:
        return HardwareResult(ok=False, messages=[str(e)])


def load_settings() -> dict:
    """Load hardware settings overrides."""
    return modelctl_hardware.load_settings()


def save_settings(settings: dict) -> HardwareResult:
    """Save hardware settings overrides."""
    try:
        modelctl_hardware.save_settings(settings)
        return HardwareResult(ok=True)
    except Exception as e:
        return HardwareResult(ok=False, messages=[str(e)])


def update_device_settings(device: str, enabled: bool | None = None,
                           role: str | None = None,
                           reserve_bytes: int | None = None,
                           bandwidth_gbs: int | None = None) -> HardwareResult:
    """Update settings for a single device."""
    settings = modelctl_hardware.load_settings()
    dev = settings.setdefault("devices", {}).setdefault(device, {})

    if enabled is not None:
        dev["enabled"] = enabled
    if role is not None:
        dev["role"] = role
    if reserve_bytes is not None:
        dev["reserve_bytes"] = reserve_bytes
    if bandwidth_gbs is not None:
        dev["bandwidth_gbs"] = bandwidth_gbs

    modelctl_hardware.save_settings(settings)
    return HardwareResult(ok=True, settings=settings)


def get_gpu_inventory() -> list:
    """Get the current GPU inventory."""
    return modelctl.get_gpu_inventory()


def pick_calibration_file() -> str | None:
    """Find a representative file under DEFAULT_MODELS_DIR to calibrate
    storage read speed against -- the largest .gguf, so the measurement
    reflects the same I/O pattern as actually loading a model. Returns
    None if no model directory or no .gguf files exist yet."""
    model_dir = modelctl.DEFAULT_MODELS_DIR
    if not model_dir.is_dir():
        return None
    largest = None
    largest_size = -1
    for p in model_dir.rglob("*.gguf"):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > largest_size:
            largest, largest_size = p, size
    return str(largest) if largest else None


def calibrate_storage(file_path: str) -> HardwareResult:
    """Run a bounded sequential-read calibration against a real file and
    persist the measurement (keyed by the file's mount point) so future
    hardware snapshots report it via StorageSnapshot.measured_*_bps.
    Non-destructive: calibrate_storage_sequential only reads.
    """
    import modelctl_storage
    from modelctl_services.cache_service import calibrate_storage_sequential

    result = calibrate_storage_sequential(file_path)
    if not result.ok:
        return HardwareResult(ok=False, messages=result.messages)

    info = modelctl_storage.probe_storage(file_path)
    settings = modelctl_hardware.load_settings()
    entry = settings.setdefault("storage", {}).setdefault(info.mount_point, {})
    entry["measured_sequential_read_bps"] = result.sequential_read_bps
    entry["measured_at"] = time.time()
    entry["calibration_method"] = result.method
    entry["calibration_file"] = result.file_path
    modelctl_hardware.save_settings(settings)

    mib_s = result.sequential_read_bps / (1 << 20)
    return HardwareResult(
        ok=True, settings=settings,
        messages=[f"measured {mib_s:.0f} MiB/s sequential read on "
                 f"{info.mount_point} ({result.method})"])
