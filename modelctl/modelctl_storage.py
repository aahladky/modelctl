"""Storage topology probing for modelctl.

Resolves model paths to storage devices, detects filesystem and
transport characteristics, and provides storage information for
planning and benchmarking.

Public API:
    probe_storage(path: str) -> StorageInfo
    probe_all_mounts() -> list[StorageInfo]
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageInfo:
    """Expanded storage information for a model path."""
    path: str
    mount_point: str
    filesystem: str
    block_devices: tuple[str, ...]
    transport: str          # "nvme", "sata", "usb", "virtio", "unknown"
    rotational: bool | None # True=hdd, False=ssd, None=unknown
    raid_level: str | None  # "raid0", "raid1", "md", None
    total_bytes: int
    free_bytes: int
    allow_mmap: bool
    # Mount options as /proc/mounts reports them. Task D4: `noatime`,
    # `discard`, a `compress=` setting or a read-only mount all change what
    # a storage measurement means, so calibration records them alongside
    # the numbers.
    mount_options: str = ""


def _find_mount_point(path: str) -> str:
    """Find the mount point for a given path."""
    p = Path(path).resolve()
    best = "/"
    best_len = 0
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mount = parts[1]
                    if str(p).startswith(mount) and len(mount) > best_len:
                        best = mount
                        best_len = len(mount)
    except OSError:
        # Fallback: walk up until we find a mount point change.
        current = p if p.is_dir() else p.parent
        while current != current.parent:
            try:
                s1 = os.statvfs(str(current))
                s2 = os.statvfs(str(current.parent))
                if s1.f_fsid != s2.f_fsid:
                    return str(current)
            except OSError:
                pass
            current = current.parent
    return best


def _find_filesystem(mount_point: str) -> str:
    """Detect the filesystem type at a mount point."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == mount_point:
                    return parts[2]
    except OSError:
        pass
    return "unknown"


def _find_mount_options(mount_point: str) -> str:
    """Mount options for a mount point, as a comma-separated string."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4 and parts[1] == mount_point:
                    return parts[3]
    except OSError:
        pass
    return ""


def _find_block_devices(mount_point: str) -> tuple[str, ...]:
    """Find the block devices backing a mount point."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == mount_point:
                    dev = parts[0]
                    # Resolve mapper devices.
                    if "/dev/mapper/" in dev or "/dev/dm-" in dev:
                        return _resolve_dm_devices(dev)
                    return (dev,)
    except OSError:
        pass
    return ()


def _resolve_dm_devices(dm_path: str) -> tuple[str, ...]:
    """Resolve device-mapper or mdraid to underlying block devices."""
    # Try lsblk for dm/raid resolution.
    try:
        result = subprocess.run(
            ["lsblk", "-n", "-o", "NAME,TYPE", dm_path],
            capture_output=True, text=True, timeout=5)
        devices = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("disk", "part"):
                devices.append(f"/dev/{parts[0]}")
        if devices:
            return tuple(devices)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return (dm_path,)


def _detect_transport(block_device: str) -> str:
    """Detect the storage transport type."""
    # Extract base device name (e.g., "nvme0n1" from "/dev/nvme0n1p1").
    base = Path(block_device).name
    # Strip partition suffix.
    if "nvme" in base:
        base = base.rsplit("p", 1)[0] if "p" in base and base[-1].isdigit() else base
    elif base[-1].isdigit():
        # sda1 -> sda
        while base and base[-1].isdigit():
            base = base[:-1]

    # Check rotational and transport via sysfs.
    sys_path = f"/sys/block/{base}"
    try:
        rotational = Path(f"{sys_path}/queue/rotational").read_text().strip()
        if rotational == "0":
            # Check if NVMe.
            if "nvme" in base:
                return "nvme"
            # Check transport via sysfs.
            try:
                real = os.readlink(f"{sys_path}/device")
                if "nvme" in real:
                    return "nvme"
                if "usb" in real:
                    return "usb"
                if "ata" in real:
                    return "sata"
            except OSError:
                pass
            return "ssd"
        return "hdd"
    except OSError:
        pass
    return "unknown"


def _detect_rotational(block_device: str) -> bool | None:
    """Detect if a device is rotational (HDD)."""
    base = Path(block_device).name
    if base[-1].isdigit():
        while base and base[-1].isdigit():
            base = base[:-1]
    try:
        val = Path(f"/sys/block/{base}/queue/rotational").read_text().strip()
        return val == "1"
    except OSError:
        return None


def _detect_raid(mount_point: str) -> str | None:
    """Detect if the mount is on a RAID/md device."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == mount_point:
                    dev = parts[0]
                    if "/dev/md" in dev:
                        # Check RAID level from /proc/mdstat.
                        try:
                            mdstat = Path("/proc/mdstat").read_text()
                            md_name = Path(dev).name
                            for mdline in mdstat.splitlines():
                                if md_name in mdline and "raid" in mdline:
                                    for level in ("raid0", "raid1", "raid5", "raid6", "raid10"):
                                        if level in mdline:
                                            return level
                            return "md"
                        except OSError:
                            return "md"
                    if "/dev/mapper/" in dev:
                        return "lvm"
    except OSError:
        pass
    return None


def _disk_usage(mount_point: str) -> tuple[int, int]:
    """Get total and free bytes at a mount point."""
    try:
        st = os.statvfs(mount_point)
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        return total, free
    except OSError:
        return 0, 0


def probe_storage(path: str) -> StorageInfo:
    """Probe storage characteristics for a given path.

    Returns a StorageInfo with filesystem, transport, and capacity details.
    """
    resolved = str(Path(path).resolve())
    mount = _find_mount_point(resolved)
    fs = _find_filesystem(mount)
    block_devs = _find_block_devices(mount)
    transport = _detect_transport(block_devs[0]) if block_devs else "unknown"
    rotational = _detect_rotational(block_devs[0]) if block_devs else None
    raid = _detect_raid(mount)
    total, free = _disk_usage(mount)

    # mmap is safe on all Linux filesystems, but warn on tmpfs/nfs.
    allow_mmap = fs not in ("tmpfs", "nfs", "cifs", "smbfs")

    return StorageInfo(
        path=resolved,
        mount_point=mount,
        filesystem=fs,
        block_devices=block_devs,
        transport=transport,
        rotational=rotational,
        raid_level=raid,
        total_bytes=total,
        free_bytes=free,
        allow_mmap=allow_mmap,
        mount_options=_find_mount_options(mount),
    )


def probe_all_mounts() -> list[StorageInfo]:
    """Probe all mounted filesystems listed in /proc/mounts.

    Returns one StorageInfo per unique mount point (skipping virtual/fs).
    """
    skip_fs = {"proc", "sysfs", "devtmpfs", "tmpfs", "devpts", "cgroup",
               "cgroup2", "pstore", "securityfs", "debugfs", "tracefs",
               "hugetlbfs", "mqueue", "fusectl", "configfs", "binfmt_misc",
               "rpc_pipefs", "nfsd", "autofs"}
    seen = set()
    results = []
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mount, fs = parts[1], parts[2]
                    if mount in seen or fs in skip_fs:
                        continue
                    seen.add(mount)
                    try:
                        info = probe_storage(mount)
                        if info.total_bytes > 0:
                            results.append(info)
                    except Exception:
                        pass
    except OSError:
        pass
    return results
