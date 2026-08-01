"""Hardware settings, snapshots, and fingerprinting for modelctl.

Provides persistent hardware configuration (reserves, overrides, device
roles) via hardware.json and live hardware snapshots that merge probed
data with user settings.

The snapshot captures current free memory, storage info, and a
fingerprint that changes when the hardware topology or backend
binaries change.  Launch plans and benchmark results reference the
fingerprint so stale results can be detected.
"""
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import modelctl_tiers
import modelctl_vram

from modelctl_paths import STATE_DIR
SETTINGS_PATH = STATE_DIR / "hardware.json"

_BANDWIDTH_GBS = {
    "B770": 560, "A770": 560, "A750": 512, "A580": 456,
    "B580": 456, "A380": 186, "B570": 384,
}


@dataclass(frozen=True)
class GpuSnapshot:
    device: str
    name: str
    total_bytes: int
    free_bytes: int
    reserve_bytes: int
    enabled: bool
    role: str
    bandwidth_gbs: float | None
    pci_address: str = ""
    pcie_width: int | None = None


@dataclass(frozen=True)
class StorageSnapshot:
    path: str
    kind: str
    allow_mmap: bool
    mount_point: str = ""
    filesystem: str = ""
    block_devices: tuple = ()
    transport: str = "unknown"
    rotational: bool | None = None
    raid_level: str | None = None
    total_bytes: int = 0
    free_bytes: int = 0
    measured_sequential_read_bps: int | None = None
    measured_random_read_bps: int | None = None
    measurement_age_seconds: float | None = None


@dataclass(frozen=True)
class HardwareSnapshot:
    captured_at: float
    fingerprint: str
    gpus: tuple
    ram_total_bytes: int
    ram_available_bytes: int
    ram_reserve_bytes: int
    storage: tuple
    backend_fingerprints: dict

    def gpu_by_device(self, device):
        for g in self.gpus:
            if g.device == device:
                return g
        return None

    def total_vram(self):
        return sum(g.total_bytes for g in self.gpus if g.enabled)

    def total_free_vram(self):
        return sum(max(0, g.free_bytes - g.reserve_bytes) for g in self.gpus if g.enabled)


def _system_ram():
    try:
        text = Path("/proc/meminfo").read_text()
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _pci_address(device_id):
    try:
        result = subprocess.run(
            ["xpu-smi", "discovery", "-d", str(device_id), "-j"],
            capture_output=True, text=True, timeout=10)
        info = json.loads(result.stdout)
        return info.get("pci_bdf_address", "")
    except Exception:
        return ""


def _backend_fingerprints():
    """Backend identity as version string PLUS binary content hash where a
    real file is involved -- a silent binary swap must stale observations."""
    fps = {}
    try:
        import modelctl as _mc
        candidates = [_mc.LLAMA_SERVER_BIN] + _mc.find_llama_server_candidates()
        binary = next((b for b in candidates if b and os.access(b, os.X_OK)), None)
    except Exception:
        binary = None
    if binary:
        version = "unknown"
        try:
            import modelctl as _mc_probe
            result = _mc_probe.run_llama_probe(binary, ["--version"], timeout=5)
            # Non-zero exit means the probe printed crash noise, not a
            # version (SYCL builds abort without the oneAPI env) -- keep
            # "unknown" rather than fingerprinting the abort banner.
            if result is not None and result.returncode == 0:
                version = (result.stdout or result.stderr).strip().split("\n")[0]
        except Exception:
            pass
        fps["llama-cpp"] = f"{version}#{modelctl_vram.file_fingerprint(binary)}"
    else:
        fps["llama-cpp"] = "unavailable"
    try:
        result = subprocess.run(
            ["docker", "images", "--digests", "--format", "{{.Digest}}", "ovms"],
            capture_output=True, text=True, timeout=10)
        fps["ovms"] = result.stdout.strip() or "unknown"
    except Exception:
        fps["ovms"] = "unavailable"
    return fps


def _fingerprint(gpus, backend_fps):
    import platform
    parts = [f"kernel:{platform.release()}"]
    for g in gpus:
        parts.append(
            f"{g.device}:{g.name}:{g.total_bytes}:{g.enabled}:{g.role}"
            f":{g.reserve_bytes}:{getattr(g, 'pci_address', None)}")
    for k, v in sorted(backend_fps.items()):
        parts.append(f"{k}:{v}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def probe_gpus():
    """Probe GPU inventory using existing modelctl infrastructure.

    Returns list of dicts with device, name, total_bytes, free_bytes.
    """
    import modelctl
    try:
        return modelctl.get_gpu_inventory()
    except Exception:
        return modelctl_vram.xpu_devices()


def capture_hardware_snapshot(settings=None):
    """Capture a live hardware snapshot, merging probed data with settings.

    If settings is None, loads from disk.
    """
    if settings is None:
        settings = load_settings()

    probed = probe_gpus()
    gpus = []
    for d in probed:
        dev = d.get("device", f"SYCL{d.get('xpu_id', 0)}")
        name = d.get("name", "unknown")
        gpus.append(GpuSnapshot(
            device=dev,
            name=name,
            total_bytes=d.get("total_bytes", 0),
            free_bytes=d.get("free_bytes", 0),
            reserve_bytes=0,
            enabled=True,
            role="",
            bandwidth_gbs=modelctl_tiers.gpu_bandwidth_gbs(name),
            pci_address=_pci_address(d.get("xpu_id")) if d.get("xpu_id") is not None else "",
        ))

    ram_total = _system_ram()
    ram_avail = modelctl_vram.system_ram_available()
    backend_fps = _backend_fingerprints()

    # Always probe the real model directory's topology (mount, filesystem,
    # transport, rotational/raid, capacity) -- settings["storage"], keyed by
    # mount_point, only overlays calibration measurements and an optional
    # allow_mmap override on top of that, rather than replacing the probe.
    # A settings-only snapshot (no real probe) would silently lose transport/
    # capacity detail the moment a user calibrates or overrides anything.
    storage_settings = settings.get("storage", {})
    storage_snapshots = []
    try:
        import modelctl_storage
        import modelctl as _mc
        model_dir = str(_mc.DEFAULT_MODELS_DIR)
        if os.path.isdir(model_dir):
            info = modelctl_storage.probe_storage(model_dir)
            override = storage_settings.get(info.mount_point, {})
            measured_at = override.get("measured_at")
            storage_snapshots.append(StorageSnapshot(
                path=info.path, kind=info.filesystem,
                allow_mmap=override.get("allow_mmap", info.allow_mmap),
                mount_point=info.mount_point,
                filesystem=info.filesystem,
                block_devices=info.block_devices,
                transport=info.transport,
                rotational=info.rotational,
                raid_level=info.raid_level,
                total_bytes=info.total_bytes,
                free_bytes=info.free_bytes,
                measured_sequential_read_bps=override.get("measured_sequential_read_bps"),
                measured_random_read_bps=override.get("measured_random_read_bps"),
                measurement_age_seconds=(time.time() - measured_at) if measured_at else None,
            ))
    except Exception:
        pass
    storage = tuple(storage_snapshots)

    snap = HardwareSnapshot(
        captured_at=time.time(),
        fingerprint="",
        gpus=tuple(gpus),
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        ram_reserve_bytes=settings.get("ram", {}).get("reserve_bytes", 0),
        storage=storage,
        backend_fingerprints=backend_fps,
    )
    snap = apply_settings(snap, settings)
    # Fingerprint AFTER policy is applied: reserves/roles/enabled/bandwidth
    # overrides must invalidate observations, or a settings change silently
    # reuses measurements taken under a different resource model.
    from dataclasses import replace as _dc_replace
    return _dc_replace(snap, fingerprint=_fingerprint(snap.gpus, backend_fps))


def load_settings():
    """Load hardware settings from hardware.json, returning defaults if missing."""
    defaults = {"version": 1, "devices": {}, "ram": {"reserve_bytes": 0}, "storage": {}}
    try:
        data = json.loads(SETTINGS_PATH.read_text())
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except (OSError, json.JSONDecodeError):
        return defaults


def save_settings(settings):
    """Save hardware settings to hardware.json atomically."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n")
    tmp.replace(SETTINGS_PATH)


def apply_settings(snapshot, settings):
    """Apply user settings (reserves, roles, enabled) to a snapshot."""
    dev_settings = settings.get("devices", {})
    new_gpus = []
    for g in snapshot.gpus:
        ds = dev_settings.get(g.device, {})
        new_gpus.append(GpuSnapshot(
            device=g.device,
            name=g.name,
            total_bytes=g.total_bytes,
            free_bytes=g.free_bytes,
            reserve_bytes=ds.get("reserve_bytes", g.reserve_bytes),
            enabled=ds.get("enabled", g.enabled),
            role=ds.get("role", g.role),
            bandwidth_gbs=ds.get("memory_bandwidth_gbs_override", g.bandwidth_gbs),
            pci_address=g.pci_address,
            pcie_width=ds.get("pcie_width_override", g.pcie_width),
        ))
    return HardwareSnapshot(
        captured_at=snapshot.captured_at,
        fingerprint=snapshot.fingerprint,
        gpus=tuple(new_gpus),
        ram_total_bytes=snapshot.ram_total_bytes,
        ram_available_bytes=snapshot.ram_available_bytes,
        ram_reserve_bytes=settings.get("ram", {}).get("reserve_bytes",
                                                       snapshot.ram_reserve_bytes),
        storage=snapshot.storage,
        backend_fingerprints=snapshot.backend_fingerprints,
    )


def enabled_gpus(snapshot):
    """Canonical ordered list of usable GPUs: enabled devices only, primary
    role first, then by total memory. Every planning, claim, feasibility,
    command, and matrix path must use this -- never snapshot.gpus directly,
    so disabled devices can't leak into budgets or commands."""
    gpus = [g for g in snapshot.gpus if g.enabled]
    gpus.sort(key=lambda g: (g.role != "primary", -(g.total_bytes or 0)))
    return gpus
