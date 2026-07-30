"""Cache and benchmark operations service.

MoE cache metrics, benchmark execution, and storage calibration.
Returns structured results, never prints or sys.exit.
"""
from dataclasses import dataclass, field
import time

import modelctl


@dataclass
class CacheMetrics:
    """MoE cache metrics from a running model."""
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    promotions: int = 0
    evictions: int = 0
    hit_ratio: float = 0.0
    cache_bytes_allocated: int = 0
    cache_bytes_used: int = 0
    devices: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    # Hybrid-specific metrics (Phase 7.8)
    hit_rows: int = 0
    miss_rows: int = 0
    cpu_miss_time_ms: float = 0.0
    gpu_hit_time_ms: float = 0.0
    merge_time_ms: float = 0.0
    promotion_bytes: int = 0
    h2d_bytes_avoided: int = 0
    host_weight_copy_fallbacks: int = 0


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    ok: bool
    mode: str = ""
    load_seconds: float = 0.0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_vram: dict = field(default_factory=dict)
    peak_ram_bytes: int = 0
    cache_metrics: CacheMetrics | None = None
    messages: list = field(default_factory=list)
    label: str = ""


def scrape_cache_metrics(port: int) -> CacheMetrics | None:
    """Scrape MoE cache metrics from a running model's /metrics endpoint.

    Returns None if the model doesn't expose cache metrics.
    """
    import re
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as r:
            text = r.read().decode()
    except Exception:
        return None

    stats = {}
    devices = set()
    for line in text.splitlines():
        if not line.startswith("llamacpp:moe_cache_"):
            continue
        m = re.match(r"llamacpp:moe_cache_(\w+)(\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        key, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            stats[key] = float(val)
        except ValueError:
            continue
        dev_match = re.search(r'device="([^"]*)"', labels)
        if dev_match:
            devices.add(dev_match.group(1))

    if not stats:
        return None

    lookups = int(stats.get("lookups", 0))
    hits = int(stats.get("hits", 0))
    return CacheMetrics(
        lookups=lookups,
        hits=hits,
        misses=int(stats.get("misses", 0)),
        promotions=int(stats.get("promotions", 0)),
        evictions=int(stats.get("evictions", 0)),
        hit_ratio=hits / lookups if lookups > 0 else 0.0,
        cache_bytes_allocated=int(stats.get("bytes_allocated", 0)),
        cache_bytes_used=int(stats.get("bytes_used", 0)),
        devices=sorted(devices),
        raw=stats,
        # Hybrid-specific metrics
        hit_rows=int(stats.get("hit_rows", 0)),
        miss_rows=int(stats.get("miss_rows", 0)),
        cpu_miss_time_ms=stats.get("cpu_miss_time_ms", 0.0),
        gpu_hit_time_ms=stats.get("gpu_hit_time_ms", 0.0),
        merge_time_ms=stats.get("merge_time_ms", 0.0),
        promotion_bytes=int(stats.get("promotion_bytes", 0)),
        h2d_bytes_avoided=int(stats.get("h2d_bytes_avoided", 0)),
        host_weight_copy_fallbacks=int(stats.get("host_weight_copy_fallbacks", 0)),
    )


@dataclass
class CalibrationResult:
    """Result of a storage calibration run (Task D4)."""
    ok: bool
    sequential_read_bps: int = 0
    random_read_bps: int = 0
    random_read_iops: int = 0
    random_read_latency_us_p50: float = 0.0
    random_read_latency_us_p99: float = 0.0
    file_path: str = ""
    file_size_bytes: int = 0
    elapsed_seconds: float = 0.0
    method: str = ""
    # Storage identity, so a calibration is attributable to a device.
    filesystem: str = ""
    mount_point: str = ""
    mount_options: str = ""
    block_devices: tuple = ()
    transport: str = ""
    rotational: object = None
    raid_level: object = None
    # When the measurement was taken, so the UI can age it out.
    measured_at: float = 0.0
    # True when the read was served from RAM rather than the device. A
    # contaminated number is page-cache throughput wearing a disk's name,
    # and it is typically an order of magnitude too fast.
    page_cache_contaminated: bool = False
    residency_before: float | None = None
    messages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# Read enough that readahead and any small cache cannot dominate the
# result. 1 GiB takes a couple of seconds on NVMe and is still honest on
# a spinning disk.
_SEQUENTIAL_SAMPLE_BYTES = 1 << 30
_RANDOM_READ_SAMPLES = 512
_RANDOM_READ_SIZE = 4096

# Above this, the sequential figure is RAM speed, not device speed.
_CONTAMINATION_THRESHOLD = 0.5

# Fastest plausible 4 KiB read from real storage. Optane is ~10us; NVMe is
# 50-100us. Anything quicker came out of RAM.
_MIN_PLAUSIBLE_DEVICE_LATENCY_US = 10.0


def _storage_identity(path):
    try:
        import modelctl_storage
        return modelctl_storage.probe_storage(str(path))
    except Exception:
        return None


def calibrate_storage_sequential(file_path: str,
                                 read_bytes: int = _SEQUENTIAL_SAMPLE_BYTES,
                                 evict_first: bool = True) -> CalibrationResult:
    """Measure real storage read performance from a model file.

    Non-destructive: reads only, never writes.

    The file's pages are evicted first (scoped to this file, unprivileged)
    so the number describes the device rather than the page cache. When
    eviction does not take -- the model is loaded, or the filesystem is
    RAM-backed -- the result is still returned but flagged
    `page_cache_contaminated`, because a calibration that silently
    measures RAM makes every storage estimate built on it wrong.
    """
    import os
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return CalibrationResult(ok=False, messages=[f"file not found: {file_path}"])

    file_size = path.stat().st_size
    read_bytes = min(read_bytes, file_size)
    if read_bytes <= 0:
        return CalibrationResult(ok=False, messages=["file is empty"])

    result = CalibrationResult(ok=False, file_path=str(path),
                               file_size_bytes=file_size,
                               measured_at=time.time())

    info = _storage_identity(path)
    if info is not None:
        result.filesystem = info.filesystem
        result.mount_point = info.mount_point
        result.mount_options = getattr(info, "mount_options", "")
        result.block_devices = tuple(info.block_devices or ())
        result.transport = info.transport
        result.rotational = info.rotational
        result.raid_level = info.raid_level

    try:
        import modelctl_benchmark
        if evict_first:
            modelctl_benchmark.evict_from_page_cache(path)
        residency = modelctl_benchmark.page_cache_residency(path)
        if residency is not None:
            result.residency_before = round(residency.fraction, 4)
    except Exception:
        residency = None

    if file_size < _SEQUENTIAL_SAMPLE_BYTES:
        result.warnings.append(
            f"file is only {modelctl._format_size(file_size)}; a sample this "
            f"small can be dominated by readahead and cache effects")

    try:
        t0 = time.time()
        total = 0
        with open(path, "rb") as f:
            while total < read_bytes:
                chunk = f.read(min(1 << 20, read_bytes - total))
                if not chunk:
                    break
                total += len(chunk)
        elapsed = time.time() - t0
        result.sequential_read_bps = int(total / elapsed) if elapsed > 0 else 0
        result.elapsed_seconds = round(elapsed, 3)
        result.method = f"sequential-read-{total // (1024 * 1024)}MiB"
    except OSError as e:
        result.messages.append(str(e))
        return result

    if result.residency_before is not None and \
            result.residency_before > _CONTAMINATION_THRESHOLD:
        result.page_cache_contaminated = True
        result.warnings.append(
            f"{result.residency_before * 100:.0f}% of the file was still in "
            f"the page cache when this ran — the throughput below reflects "
            f"RAM, not "
            f"{result.block_devices[0] if result.block_devices else 'the device'}")
    if result.filesystem in ("tmpfs", "ramfs"):
        result.page_cache_contaminated = True
        result.warnings.append(
            f"{result.filesystem} is RAM-backed; there is no storage device "
            f"to calibrate")

    random_stats = _measure_random_reads(path, file_size)
    result.random_read_bps = random_stats.get("bps", 0)
    result.random_read_iops = random_stats.get("iops", 0)
    result.random_read_latency_us_p50 = random_stats.get("p50_us", 0.0)
    result.random_read_latency_us_p99 = random_stats.get("p99_us", 0.0)

    # No storage device answers a 4 KiB read in single-digit microseconds;
    # that is a memory copy. Per-region eviction does not always take
    # (compressed extents, delayed allocation), so the median has to be
    # sanity-checked rather than trusted.
    if 0 < result.random_read_latency_us_p50 < _MIN_PLAUSIBLE_DEVICE_LATENCY_US:
        result.warnings.append(
            f"random-read p50 of {result.random_read_latency_us_p50:.0f}us is "
            f"faster than any storage device — those reads were served from "
            f"RAM; treat the random figures as a floor, not the device's "
            f"real latency (p99 {result.random_read_latency_us_p99:.0f}us is "
            f"the more honest number)")

    result.ok = True
    result.messages.append(
        f"sequential {modelctl._format_size(result.sequential_read_bps)}/s, "
        f"random {result.random_read_iops} IOPS "
        f"(p50 {result.random_read_latency_us_p50:.0f}us)")
    return result


def _measure_random_reads(path, file_size, samples=_RANDOM_READ_SAMPLES):
    """Random-read latency and IOPS: what expert paging actually costs.

    Sequential throughput says nothing about the cost of pulling one
    expert from the middle of a 200 GiB file, which is the access pattern
    an offloaded MoE actually has.
    """
    import os
    import random

    if file_size <= _RANDOM_READ_SIZE:
        return {}
    latencies = []
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return {}
    try:
        rng = random.Random(0xC0FFEE)  # deterministic offsets across runs
        max_offset = file_size - _RANDOM_READ_SIZE
        for _ in range(samples):
            offset = rng.randrange(0, max_offset)
            # Evicting each region first keeps this a device measurement;
            # without it the second pass would read from RAM.
            try:
                os.posix_fadvise(fd, offset, _RANDOM_READ_SIZE,
                                 os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
            t0 = time.perf_counter()
            os.pread(fd, _RANDOM_READ_SIZE, offset)
            latencies.append((time.perf_counter() - t0) * 1e6)
    except OSError:
        return {}
    finally:
        os.close(fd)

    if not latencies:
        return {}
    latencies.sort()
    total_s = sum(latencies) / 1e6
    return {
        "p50_us": latencies[len(latencies) // 2],
        "p99_us": latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))],
        "iops": int(len(latencies) / total_s) if total_s > 0 else 0,
        "bps": int(len(latencies) * _RANDOM_READ_SIZE / total_s) if total_s > 0 else 0,
    }


# A calibration older than this is reported as stale: disks get replaced,
# filesystems fill up, and a year-old number should not silently steer a
# placement decision.
CALIBRATION_STALE_SECONDS = 30 * 86400


def calibration_age(measured_at):
    """(age_seconds, is_stale) for a calibration timestamp."""
    if not measured_at:
        return (None, True)
    age = max(0.0, time.time() - measured_at)
    return (age, age > CALIBRATION_STALE_SECONDS)
