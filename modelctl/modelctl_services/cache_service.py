"""Cache and benchmark operations service.

MoE cache metrics, benchmark execution, and storage calibration.
Returns structured results, never prints or sys.exit.
"""
from dataclasses import dataclass, field
import time


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
    """Result of a storage calibration run."""
    ok: bool
    sequential_read_bps: int = 0
    random_read_bps: int = 0
    file_path: str = ""
    file_size_bytes: int = 0
    elapsed_seconds: float = 0.0
    method: str = ""
    messages: list = field(default_factory=list)


def calibrate_storage_sequential(file_path: str,
                                 read_bytes: int = 100 * 1024 * 1024) -> CalibrationResult:
    """Calibrate sequential read speed using a model file.

    Reads `read_bytes` (default 100 MiB) from the file and measures throughput.
    Non-destructive: only reads, never writes.
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
        bps = int(total / elapsed) if elapsed > 0 else 0

        return CalibrationResult(
            ok=True,
            sequential_read_bps=bps,
            file_path=str(path),
            file_size_bytes=file_size,
            elapsed_seconds=round(elapsed, 3),
            method=f"sequential-read-{total // (1024*1024)}MiB",
        )
    except OSError as e:
        return CalibrationResult(ok=False, messages=[str(e)])
