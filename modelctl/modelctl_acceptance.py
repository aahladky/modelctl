"""Release A target-hardware matrix (roadmap Task E3).

Defines the hardware/placement combinations Release A has to pass, works
out which of them this machine can actually run, executes them against
real processes, and writes an acceptance report.

The cell definitions are the point. Each one states its preconditions, so
a cell that cannot run on the current machine is reported as `skipped`
with a reason rather than quietly passing -- "the matrix went green" must
never mean "most of the matrix did not run".

Every cell also states its own moe_cache state. Inheriting the fixture
profile's cache configuration made cells fail for a reason that had
nothing to do with what they were testing: a profile requesting
decode.miss_execution=cpu is correctly refused by the fail-closed launch
gate, which is a Phase B success, not a matrix result.

Public API:
    CELLS
    MatrixCell / CellResult dataclasses
    plan_matrix(...) -> list[CellResult]
    run_matrix(...) -> list[CellResult]
    render_report(results, ...) -> str
"""
import json
import time
from dataclasses import dataclass, field

import modelctl


@dataclass(frozen=True)
class MatrixCell:
    """One combination Release A must demonstrate."""
    name: str
    description: str
    # Config overrides applied to the fixture profile.
    config: dict = field(default_factory=dict)
    moe_cache: dict | None = None
    # Preconditions, checked against the live snapshot before running.
    min_gpus: int = 0
    needs_cache_capable: bool = False
    needs_moe_model: bool = False
    needs_draft_model: bool = False
    min_free_vram_bytes: int = 0
    min_free_ram_bytes: int = 0
    concurrent_requests: int = 1
    # Heavy cells need a large model and several minutes each.
    heavy: bool = False


CELLS = (
    MatrixCell(
        name="cpu-only",
        description="CPU-only execution: no layers offloaded",
        config={"extra": "-ngl 0", "device": "", "fit": "off"}),
    MatrixCell(
        name="single-gpu",
        description="One Intel SYCL GPU",
        config={"device": "SYCL0", "extra": "", "fit": "off"},
        min_gpus=1),
    MatrixCell(
        name="second-gpu",
        description="The smaller of the two asymmetric GPUs on its own",
        config={"device": "SYCL1", "extra": "", "fit": "off"},
        min_gpus=2),
    MatrixCell(
        name="two-asymmetric-gpus",
        description="Both asymmetric GPUs, split by capacity",
        config={"device": "SYCL0,SYCL1", "split_mode": "layer",
                "tensor_split": "3,1", "extra": "", "fit": "off"},
        min_gpus=2),
    MatrixCell(
        name="fits-one-gpu",
        description="Model sized to fit entirely in one GPU's VRAM",
        config={"device": "SYCL0", "extra": "", "fit": "on"},
        min_gpus=1),
    MatrixCell(
        name="concurrent-prompt-and-decode",
        description="Concurrent requests: prompt processing overlapping decode",
        config={"device": "SYCL0", "extra": "--parallel 4", "fit": "off"},
        min_gpus=1, concurrent_requests=4),
    MatrixCell(
        name="cache-disabled",
        description="Cache-capable runtime with the transfer cache off",
        config={"device": "SYCL0", "extra": "", "fit": "off"},
        moe_cache={"mode": "off"},
        min_gpus=1, needs_cache_capable=True),
    MatrixCell(
        name="cache-enabled",
        description="Transfer cache enabled on a cache-capable runtime",
        config={"device": "SYCL0", "extra": "", "fit": "off"},
        moe_cache={"mode": "manual",
                   "gpu": {"budgets_bytes": {"SYCL0": 2 << 30},
                           "policy": "slru", "admission_misses": 2},
                   "decode": {"admit_to_gpu_cache": True,
                              "miss_execution": "gpu"}},
        min_gpus=1, needs_cache_capable=True, needs_moe_model=True),
    MatrixCell(
        name="combined-vram",
        description="Model that only fits across both GPUs' combined VRAM",
        config={"device": "SYCL0,SYCL1", "split_mode": "layer",
                "tensor_split": "3,1", "extra": "", "fit": "off"},
        moe_cache={"mode": "off"},
        min_gpus=2, needs_moe_model=True, min_free_vram_bytes=30 << 30,
        heavy=True),
    MatrixCell(
        name="ram-spill",
        description="Model requiring spill into RAM (--no-mmap)",
        config={"device": "SYCL0", "fit": "off",
                "extra": r"--no-mmap -ot blk\.(1[0-9]|[2-9][0-9])\.ffn_.*_exps\.=CPU"},
        moe_cache={"mode": "off"},
        needs_moe_model=True, min_gpus=1, min_free_ram_bytes=24 << 30,
        heavy=True),
    MatrixCell(
        name="mmap-storage-backed",
        description="Model reached through mmap/storage rather than RAM",
        config={"device": "SYCL0", "fit": "off",
                "extra": r"-ot blk\.(1[0-9]|[2-9][0-9])\.ffn_.*_exps\.=CPU"},
        moe_cache={"mode": "off"},
        needs_moe_model=True, min_gpus=1, heavy=True),
    MatrixCell(
        name="main-plus-draft",
        description="Main model plus a draft/MTP model",
        config={"device": "SYCL0", "mtp": "on", "extra": "", "fit": "off"},
        moe_cache={"mode": "off"},
        min_gpus=1, needs_draft_model=True, heavy=True),
)


@dataclass
class CellResult:
    """What happened to one matrix cell."""
    cell: MatrixCell
    status: str = "pending"      # passed | failed | skipped | pending
    reason: str = ""             # why skipped, or how it failed
    measurement: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def name(self):
        return self.cell.name


def _precondition_failure(cell, snapshot, caps, profile, include_heavy):
    """Why this cell cannot run here, or None."""
    gpus = [g for g in getattr(snapshot, "gpus", ()) if g.enabled]
    if cell.min_gpus and len(gpus) < cell.min_gpus:
        return f"needs {cell.min_gpus} enabled GPU(s), found {len(gpus)}"
    if cell.needs_cache_capable:
        try:
            import modelctl_capabilities
            if not modelctl_capabilities.is_cache_capable(caps or {}):
                return "runtime does not report moe_weight_transfer_cache"
        except Exception:
            return "capability probe unavailable"
    if cell.needs_moe_model:
        try:
            import modelctl_vram
            layout = modelctl_vram.gguf_model_layout(profile.get("model_path", ""))
            if not (layout or {}).get("is_moe"):
                return "the fixture model is not a sparse MoE"
        except Exception:
            return "could not read the model layout"
    if cell.needs_draft_model and not profile.get("mtp_path"):
        return "no draft/MTP model configured on the fixture profile"
    if cell.min_free_vram_bytes:
        free = sum(max(0, g.free_bytes - g.reserve_bytes) for g in gpus)
        if free < cell.min_free_vram_bytes:
            return (f"needs {modelctl._format_size(cell.min_free_vram_bytes)} "
                    f"free VRAM, have {modelctl._format_size(free)}")
    if cell.min_free_ram_bytes:
        free = getattr(snapshot, "ram_available_bytes", 0)
        if free < cell.min_free_ram_bytes:
            return (f"needs {modelctl._format_size(cell.min_free_ram_bytes)} "
                    f"free RAM, have {modelctl._format_size(free)}")
    if cell.heavy and not include_heavy:
        return "heavy cell: needs a large model and several minutes; opt in explicitly"
    return None


def plan_matrix(profile, snapshot=None, backend=None, include_heavy=False):
    """Decide which cells this machine can run, without running any.

    Returned in the same shape as run_matrix() so a report can be produced
    for a dry run.
    """
    if snapshot is None:
        import modelctl_hardware
        snapshot = modelctl_hardware.capture_hardware_snapshot()
    caps = getattr(backend, "capabilities", None) or {}
    results = []
    for cell in CELLS:
        why = _precondition_failure(cell, snapshot, caps, profile, include_heavy)
        results.append(CellResult(cell=cell,
                                  status="skipped" if why else "pending",
                                  reason=why or ""))
    return results


def run_matrix(profile, snapshot=None, backend=None, include_heavy=False,
               only=None, max_tokens=32, log=print):
    """Execute the runnable cells against real processes.

    Each cell is measured through the same plan-test path a user's browser
    drives, so a passing matrix says the product works, not that a bespoke
    test harness works.
    """
    import modelctl_launch
    import modelctl_plans
    import modelctl_tune

    profile = modelctl.normalize_profile(dict(profile))
    if snapshot is None:
        import modelctl_hardware
        snapshot = modelctl_hardware.capture_hardware_snapshot()
    if backend is None:
        backend = modelctl_launch.resolve_backend(profile)

    results = plan_matrix(profile, snapshot, backend, include_heavy)
    for result in results:
        cell = result.cell
        if only and cell.name not in only:
            result.status = "skipped"
            result.reason = "not selected"
            continue
        if result.status == "skipped":
            log(f"skip {cell.name}: {result.reason}")
            continue

        cell_profile = dict(profile)
        cell_profile["config"] = {**profile.get("config", {}), **cell.config}
        if cell.moe_cache is not None:
            cell_profile["moe_cache"] = cell.moe_cache

        started = time.time()
        try:
            plan = modelctl_plans.current_profile_plan(
                cell_profile, snapshot, capabilities=backend.capabilities)
            log(f"run  {cell.name}: {cell.description}")
            run = modelctl_tune.test_launch_plan(
                cell_profile["name"], plan.id, log=lambda *a: None,
                max_tokens=max_tokens, runs=1,
                binary=backend.binary, profile_override=cell_profile)
            result.measurement = {
                "generation_tps": run.get("generation_tps"),
                "prompt_tps": run.get("prompt_tps"),
                "load_seconds": run.get("load_seconds"),
                "ttft_seconds": run.get("ttft_seconds"),
                "peak_vram_bytes": run.get("peak_vram_bytes"),
                "peak_ram_bytes": run.get("peak_ram_bytes"),
                "storage_activity": run.get("storage_activity"),
                "cache_state": run.get("cache_state"),
                "command_fingerprint": run.get("command_fingerprint"),
            }
            if run.get("success"):
                result.status = "passed"
            else:
                result.status = "failed"
                # A bare failure class ("preflight_failed") says nothing
                # about what was wrong; the validation messages do.
                detail = (run.get("details") or {}).get("validation") or []
                result.reason = "; ".join(
                    [run.get("failure_class") or "run did not succeed"]
                    + list(detail))
        except Exception as e:
            result.status = "failed"
            result.reason = f"{type(e).__name__}: {e}"
        result.elapsed_seconds = round(time.time() - started, 1)
        log(f"  -> {result.status}"
            + (f" ({result.reason})" if result.reason else "")
            + f" in {result.elapsed_seconds}s")
    return results


def render_report(results, profile=None, snapshot=None, backend=None) -> str:
    """A markdown acceptance report.

    Skipped cells are listed with their reasons and counted separately
    from passes, so the report cannot be read as more coverage than it is.
    """
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]

    lines = [
        "# Release A target-hardware matrix",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if profile:
        lines.append(f"**Fixture profile:** `{profile.get('name')}` — "
                     f"`{profile.get('model_path')}`")
    if backend is not None:
        lines.append(f"**Runtime:** `{getattr(backend, 'binary', '')}` "
                     f"(binary `{getattr(backend, 'binary_fingerprint', '')}`, "
                     f"capabilities `{getattr(backend, 'capability_fingerprint', '')}`)")
    if snapshot is not None:
        gpus = ", ".join(f"{g.device} {g.name}"
                         for g in getattr(snapshot, "gpus", ()) if g.enabled)
        lines.append(f"**Hardware:** {gpus} "
                     f"(fingerprint `{getattr(snapshot, 'fingerprint', '')}`)")
    lines += [
        "",
        f"**{len(passed)} passed · {len(failed)} failed · "
        f"{len(skipped)} skipped** of {len(results)} cells.",
        "",
        "A skipped cell is not a passing cell. Release A is complete only "
        "when every cell has either passed or been skipped for a reason "
        "that is a property of this machine rather than of the product.",
        "",
        "## Cells",
        "",
        "| cell | status | detail |",
        "|---|---|---|",
    ]
    for r in results:
        detail = r.reason
        if r.status == "passed" and r.measurement.get("generation_tps"):
            m = r.measurement
            detail = (f"{m['generation_tps']:.1f} t/s gen, "
                      f"load {m.get('load_seconds') or 0:.1f}s, "
                      f"{m.get('storage_activity') or 'n/a'}")
        lines.append(f"| `{r.name}` | {r.status} | {detail or '—'} |")

    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines.append(f"- **{r.name}** — {r.reason}")
    if skipped:
        lines += ["", "## Skipped, and why", ""]
        for r in skipped:
            lines.append(f"- **{r.name}** — {r.reason}")

    lines += ["", "## Descriptions", ""]
    for r in results:
        lines.append(f"- **{r.name}**: {r.cell.description}")
    return "\n".join(lines) + "\n"
