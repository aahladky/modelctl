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


class TestCacheMetricsCapture(unittest.TestCase):
    """A throughput number without cache counters cannot distinguish "the
    cache did not help" from "the cache never ran"."""

    def test_parser_extracts_counters_per_device(self):
        import modelctl
        from unittest import mock
        import io
        text = (
            "# HELP llamacpp:moe_cache_hits_total MoE expert cache hits\n"
            "# TYPE llamacpp:moe_cache_hits_total counter\n"
            'llamacpp:moe_cache_hits_total{device="SYCL0"} 846\n'
            'llamacpp:moe_cache_misses_total{device="SYCL0"} 72\n'
            'llamacpp:moe_cache_hits_total{device="SYCL1"} 5\n'
            "unrelated_metric 1\n")
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(text.encode())
        with mock.patch("urllib.request.urlopen", return_value=cm):
            stats = modelctl.scrape_moe_cache_metrics(1234)
        self.assertEqual(stats["SYCL0"]["hits_total"], "846")
        self.assertEqual(stats["SYCL0"]["misses_total"], "72")
        self.assertEqual(stats["SYCL1"]["hits_total"], "5")
        self.assertNotIn("unrelated_metric", str(stats))

    def test_unreachable_server_reads_as_unknown_not_zero(self):
        # Reporting zero for an unreachable endpoint would look exactly
        # like an inert cache, which is the failure this exists to expose.
        import modelctl
        from unittest import mock
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertIsNone(modelctl.scrape_moe_cache_metrics(1234))

    def test_server_without_cache_metrics_reads_as_unknown(self):
        import modelctl
        from unittest import mock
        import io
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(b"other_metric 3\n")
        with mock.patch("urllib.request.urlopen", return_value=cm):
            self.assertIsNone(modelctl.scrape_moe_cache_metrics(1234))

    def test_console_telemetry_parses_the_same_counters_as_the_measurement(self):
        # Two copies could drift, and the one on screen would be the one
        # not used by the measurement. The console used to be pinned to the
        # measurement's parser by an alias (modelctl_web.app's
        # _scrape_moe_cache_metrics WAS modelctl.scrape_moe_cache_metrics);
        # the /v2 SSE stream now parses the worker's /metrics itself in
        # modelctl_web.telemetry, so identity can no longer carry the
        # invariant and equivalence has to. Same exposition text in, same
        # per-device counters out -- including the label shapes the
        # measurement's parser promises to survive: extra and reordered
        # labels, and a metric with no device label at all (the "" key).
        import modelctl
        from modelctl_web import telemetry
        from unittest import mock
        import io
        text = (
            "# HELP llamacpp:moe_cache_hits_total MoE expert cache hits\n"
            "# TYPE llamacpp:moe_cache_hits_total counter\n"
            'llamacpp:moe_cache_hits_total{device="SYCL0"} 846\n'
            'llamacpp:moe_cache_misses_total{device="SYCL0"} 72\n'
            'llamacpp:moe_cache_learning{device="SYCL0"} 1\n'
            'llamacpp:moe_cache_hit_ratio{model="q3",device="SYCL1"} 0.92\n'
            'llamacpp:moe_cache_hits_total{device="SYCL1"} 5\n'
            "llamacpp:moe_cache_bytes_resident 1048576\n"
            "llamacpp:prompt_tokens_total 40\n"
            "unrelated_metric 1\n")
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(text.encode())
        with mock.patch("urllib.request.urlopen", return_value=cm):
            measured = modelctl.scrape_moe_cache_metrics(1234)
        shown = telemetry.parse_worker_metrics(text)["moe"]
        self.assertEqual(shown, measured)
        self.assertEqual(shown["SYCL0"]["hits_total"], "846")
        self.assertEqual(shown["SYCL1"]["hit_ratio"], "0.92")
        self.assertNotIn("unrelated_metric", str(shown))

    def test_console_shows_no_counters_where_the_measurement_reads_unknown(self):
        # Same "absent is not zero" rule on the console side: a server with
        # no cache counters must leave the console with nothing to show,
        # not with zeros that look like an inert cache.
        import modelctl
        from modelctl_web import telemetry
        from unittest import mock
        import io
        text = "other_metric 3\n"
        cm = mock.MagicMock()
        cm.__enter__.return_value = io.BytesIO(text.encode())
        with mock.patch("urllib.request.urlopen", return_value=cm):
            self.assertIsNone(modelctl.scrape_moe_cache_metrics(1234))
        moe = telemetry.parse_worker_metrics(text)["moe"]
        self.assertEqual(moe, {})
        self.assertIsNone(telemetry.summarize_cache(moe))
