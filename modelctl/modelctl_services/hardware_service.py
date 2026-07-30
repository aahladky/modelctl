"""Hardware settings service.

Hardware snapshot capture, settings management, and GPU inventory.
Returns structured results, never prints or sys.exit.
"""
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
