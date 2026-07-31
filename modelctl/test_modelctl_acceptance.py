"""Task E3: the hardware-matrix harness itself.

These are unit tests over the cell definitions, precondition logic and
report rendering -- the actual matrix execution needs real GPUs and lives
in docs/release-a-hardware-matrix-2026-07-30.md.

The property worth protecting: a skipped cell must never read as a pass.
"""
import unittest
from unittest import mock

import modelctl_acceptance as acc

GB = 1 << 30


class FakeGpu:
    def __init__(self, device, total=16 * GB, free=None):
        self.device = device
        self.name = device
        self.total_bytes = total
        self.free_bytes = total if free is None else free
        self.reserve_bytes = 0
        self.enabled = True
        self.role = ""


class FakeSnapshot:
    fingerprint = "hw123"

    def __init__(self, devices=("SYCL0", "SYCL1"), ram=32 * GB, free=None):
        self.gpus = tuple(FakeGpu(d, free=free) for d in devices)
        self.ram_available_bytes = ram


class FakeBackend:
    binary = "/bin/llama-server"
    binary_fingerprint = "bin123"
    capability_fingerprint = "cap123"

    def __init__(self, cache=True):
        self.capabilities = {
            "schema": 2, "features": {"moe_weight_transfer_cache": cache},
            "cli": {}, "_probe_status": "ok"}


MOE_PROFILE = {"name": "m1", "model_path": "/models/m.gguf",
               "config": {"ctx": 4096}}


def plan(profile=None, snapshot=None, backend=None, is_moe=True,
         include_heavy=False):
    layout = {"is_moe": is_moe, "weight_bytes": 16 * GB}
    with mock.patch("modelctl_vram.gguf_model_layout", return_value=layout):
        return acc.plan_matrix(profile or MOE_PROFILE,
                               snapshot or FakeSnapshot(),
                               backend or FakeBackend(),
                               include_heavy=include_heavy)


def by_name(results):
    return {r.name: r for r in results}


class TestCellDefinitions(unittest.TestCase):
    def test_every_roadmap_combination_has_a_cell(self):
        names = {c.name for c in acc.CELLS}
        for required in ("cpu-only", "single-gpu", "two-asymmetric-gpus",
                         "fits-one-gpu", "combined-vram", "ram-spill",
                         "mmap-storage-backed", "cache-disabled",
                         "cache-enabled", "concurrent-prompt-and-decode",
                         "main-plus-draft"):
            self.assertIn(required, names)

    def test_every_cell_declares_its_own_cache_state_or_needs_none(self):
        # Inheriting the fixture profile's moe_cache made cells fail for a
        # reason that had nothing to do with what they test.
        for cell in acc.CELLS:
            if cell.needs_moe_model or cell.needs_cache_capable:
                self.assertIsNotNone(
                    cell.moe_cache,
                    f"{cell.name} would inherit the fixture's cache config")

    def test_every_cell_has_a_description(self):
        for cell in acc.CELLS:
            self.assertTrue(cell.description)


class TestPreconditions(unittest.TestCase):
    def test_multi_gpu_cells_skip_on_one_gpu(self):
        r = by_name(plan(snapshot=FakeSnapshot(devices=("SYCL0",))))
        self.assertEqual(r["two-asymmetric-gpus"].status, "skipped")
        self.assertIn("GPU", r["two-asymmetric-gpus"].reason)
        self.assertEqual(r["single-gpu"].status, "pending")

    def test_cache_cells_skip_on_an_incapable_runtime(self):
        r = by_name(plan(backend=FakeBackend(cache=False)))
        self.assertEqual(r["cache-enabled"].status, "skipped")
        self.assertIn("moe_weight_transfer_cache", r["cache-enabled"].reason)

    def test_moe_cells_skip_on_a_dense_model(self):
        r = by_name(plan(is_moe=False))
        self.assertEqual(r["cache-enabled"].status, "skipped")
        self.assertIn("sparse MoE", r["cache-enabled"].reason)

    def test_draft_cell_skips_without_a_draft_model(self):
        r = by_name(plan(include_heavy=True))
        self.assertEqual(r["main-plus-draft"].status, "skipped")
        self.assertIn("draft", r["main-plus-draft"].reason)

    def test_ram_spill_skips_when_ram_is_short(self):
        r = by_name(plan(snapshot=FakeSnapshot(ram=8 * GB), include_heavy=True))
        self.assertEqual(r["ram-spill"].status, "skipped")
        self.assertIn("free RAM", r["ram-spill"].reason)

    def test_heavy_cells_need_an_explicit_opt_in(self):
        r = by_name(plan(include_heavy=False))
        self.assertEqual(r["combined-vram"].status, "skipped")
        self.assertIn("opt in", r["combined-vram"].reason)

    def test_cpu_only_always_runs(self):
        r = by_name(plan(snapshot=FakeSnapshot(devices=())))
        self.assertEqual(r["cpu-only"].status, "pending")

    def test_every_skip_carries_a_reason(self):
        for r in plan(snapshot=FakeSnapshot(devices=("SYCL0",)), is_moe=False):
            if r.status == "skipped":
                self.assertTrue(r.reason, f"{r.name} skipped with no reason")


class TestReport(unittest.TestCase):
    def results(self):
        cells = {c.name: c for c in acc.CELLS}
        return [
            acc.CellResult(cell=cells["cpu-only"], status="passed",
                           measurement={"generation_tps": 50.3,
                                        "load_seconds": 2.0,
                                        "storage_activity": "page-cache-served"}),
            acc.CellResult(cell=cells["ram-spill"], status="skipped",
                           reason="needs 24.0GB free RAM, have 22.6GB"),
            acc.CellResult(cell=cells["single-gpu"], status="failed",
                           reason="preflight_failed; cache unsupported"),
        ]

    def test_report_counts_skips_separately_from_passes(self):
        text = acc.render_report(self.results())
        self.assertIn("1 passed", text)
        self.assertIn("1 failed", text)
        self.assertIn("1 skipped", text)

    def test_report_states_that_a_skip_is_not_a_pass(self):
        # "The matrix went green" must never mean "most of it did not run".
        self.assertIn("A skipped cell is not a passing cell",
                      acc.render_report(self.results()))

    def test_report_lists_skip_reasons_and_failures(self):
        text = acc.render_report(self.results())
        self.assertIn("needs 24.0GB free RAM", text)
        self.assertIn("cache unsupported", text)

    def test_report_includes_provenance_when_given(self):
        text = acc.render_report(self.results(), MOE_PROFILE,
                                 FakeSnapshot(), FakeBackend())
        self.assertIn("bin123", text)
        self.assertIn("cap123", text)
        self.assertIn("hw123", text)

    def test_report_renders_measurements(self):
        self.assertIn("50.3 t/s", acc.render_report(self.results()))


if __name__ == "__main__":
    unittest.main()


class TestOffloadSweepCells(unittest.TestCase):
    """The offload-threshold sweep: four conditions that must stay
    distinguishable from each other and from the rest of the matrix."""

    def _cell(self, name):
        return next(c for c in acc.CELLS if c.name == name)

    def test_all_four_conditions_exist(self):
        names = {c.name for c in acc.CELLS}
        for n in ("offload-A-default", "offload-B-global-1",
                  "offload-D-moe-1", "offload-E-moe-1-no-cache"):
            self.assertIn(n, names)

    def test_conditions_differ_only_in_threshold_and_cache(self):
        # If placement drifted between cells the comparison would be
        # measuring placement, which is how the Q4_K_M run went wrong.
        cells = [self._cell(n) for n in
                 ("offload-A-default", "offload-B-global-1",
                  "offload-D-moe-1", "offload-E-moe-1-no-cache")]
        configs = {tuple(sorted(c.config.items())) for c in cells}
        self.assertEqual(len(configs), 1)

    def test_baseline_sets_no_threshold_override(self):
        self.assertEqual(self._cell("offload-A-default").env, ())

    def test_global_and_moe_conditions_set_different_variables(self):
        b = self._cell("offload-B-global-1").env
        d = self._cell("offload-D-moe-1").env
        self.assertIn("GGML_OP_OFFLOAD_MIN_BATCH=1", b)
        self.assertIn("GGML_OP_OFFLOAD_MOE_MIN_BATCH=1", d)
        self.assertNotEqual(set(b), set(d))

    def test_cache_on_conditions_share_one_cache_config(self):
        # Only the threshold may vary across A, B and D.
        caches = [self._cell(n).moe_cache for n in
                  ("offload-A-default", "offload-B-global-1", "offload-D-moe-1")]
        self.assertEqual(caches[0], caches[1])
        self.assertEqual(caches[1], caches[2])

    def test_condition_e_isolates_the_cache_contribution(self):
        e = self._cell("offload-E-moe-1-no-cache")
        d = self._cell("offload-D-moe-1")
        self.assertEqual(e.env, d.env)              # same threshold
        self.assertEqual(e.moe_cache, {"mode": "off"})

    def test_sweep_cells_are_heavy_and_need_a_real_moe(self):
        for n in ("offload-A-default", "offload-B-global-1",
                  "offload-D-moe-1", "offload-E-moe-1-no-cache"):
            c = self._cell(n)
            self.assertTrue(c.heavy, n)
            self.assertTrue(c.needs_moe_model, n)


class TestCellEnvironmentMerge(unittest.TestCase):
    """A cell's env must reach the launched process, and must win over the
    fixture profile's own settings."""

    def _merge(self, profile_env, cell_env):
        # Mirrors run_matrix()'s merge without launching anything.
        merged = {}
        for entry in list(profile_env) + list(cell_env):
            if "=" in entry:
                k, v = entry.split("=", 1)
                merged[k] = v
        return [f"{k}={v}" for k, v in sorted(merged.items())]

    def test_cell_env_overrides_the_profile(self):
        out = self._merge(["GGML_OP_OFFLOAD_MIN_BATCH=32"],
                          ["GGML_OP_OFFLOAD_MIN_BATCH=1"])
        self.assertEqual(out, ["GGML_OP_OFFLOAD_MIN_BATCH=1"])

    def test_unrelated_profile_env_survives(self):
        out = self._merge(["ZES_ENABLE_SYSMAN=1"],
                          ["GGML_OP_OFFLOAD_MOE_MIN_BATCH=1"])
        self.assertIn("ZES_ENABLE_SYSMAN=1", out)
        self.assertIn("GGML_OP_OFFLOAD_MOE_MIN_BATCH=1", out)


class TestMeasurementPrompts(unittest.TestCase):
    """Repeating one prompt is not a throughput measurement: llama.cpp
    reuses the KV cache for an identical prefix and skips expert
    restaging."""

    def test_default_rotates_several_distinct_prompts(self):
        import modelctl_tune
        seq = modelctl_tune._prompt_sequence(None)
        self.assertGreaterEqual(len(seq), 4)
        self.assertEqual(len(set(seq)), len(seq))

    def test_prompts_are_long_enough_to_restage_experts(self):
        # The staging path was observed to be skipped for short prompts.
        import modelctl_tune
        for p in modelctl_tune._prompt_sequence(None):
            self.assertGreater(len(p.split()), 32)

    def test_explicit_single_prompt_is_respected(self):
        import modelctl_tune
        self.assertEqual(modelctl_tune._prompt_sequence("hello"), ("hello",))

    def test_explicit_sequence_is_used_as_given(self):
        import modelctl_tune
        self.assertEqual(modelctl_tune._prompt_sequence(["a", "b"]), ("a", "b"))

    def test_empty_sequence_falls_back_to_the_default_set(self):
        import modelctl_tune
        self.assertEqual(modelctl_tune._prompt_sequence([]),
                         modelctl_tune._MEASURE_PROMPTS)
