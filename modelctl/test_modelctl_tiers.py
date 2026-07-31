import re
import unittest
from pathlib import Path

import modelctl_tiers
import modelctl_vram

GIB = 1 << 30

B70 = {"device": "SYCL0", "name": "Intel(R) Arc(TM) Pro B70 Graphics",
       "total_bytes": 32656 * (1 << 20), "free_bytes": 0}
B580 = {"device": "SYCL1", "name": "Intel(R) Arc(TM) B580 Graphics",
        "total_bytes": 12216 * (1 << 20), "free_bytes": 0}
INVENTORY = [B70, B580]


def moe_layout(n_layers, per_layer_gib, non_expert_gib=4.0, arch="testmoe"):
    layers = {i: int(per_layer_gib * GIB) for i in range(1, n_layers + 1)}
    meta = {"general.architecture": arch, f"{arch}.block_count": n_layers + 1}
    weights = sum(layers.values()) + int(non_expert_gib * GIB)
    return {"arch": arch, "meta": meta, "block_count": n_layers + 1,
            "is_moe": True, "weight_bytes": weights,
            "non_expert_bytes": int(non_expert_gib * GIB),
            "other_bytes": GIB, "layer_bytes": int((non_expert_gib - 1) * GIB),
            "expert_bytes_per_layer": layers, "unknown_type_tensors": 0}


def dense_layout(n_layers, per_layer_gib, other_gib=2.0, arch="testdense"):
    layer_bytes = int(per_layer_gib * GIB) * n_layers
    other = int(other_gib * GIB)
    meta = {"general.architecture": arch, f"{arch}.block_count": n_layers}
    return {"arch": arch, "meta": meta, "block_count": n_layers,
            "is_moe": False, "weight_bytes": layer_bytes + other,
            "non_expert_bytes": layer_bytes + other, "other_bytes": other,
            "layer_bytes": layer_bytes, "expert_bytes_per_layer": {},
            "unknown_type_tensors": 0}


def profile(ctx=8192, extra=""):
    # tiny ctx keeps heuristic KV (no attention fields in fake metas) small
    return {"config": {"ctx": ctx, "cache_type_k": "q8_0",
                       "cache_type_v": "q8_0", "extra": extra},
            "model_path": "/fake/model.gguf"}


class TestRangeRegex(unittest.TestCase):
    def assert_range(self, a, b):
        rx = re.compile("^(?:" + modelctl_tiers.range_regex(a, b) + ")$")
        for n in range(max(0, a - 3), b + 4):
            self.assertEqual(bool(rx.match(str(n))), a <= n <= b,
                             f"range {a}-{b}, value {n}")

    def test_ranges(self):
        for a, b in [(0, 0), (5, 5), (0, 9), (1, 9), (9, 10), (1, 19),
                     (20, 28), (29, 47), (0, 47), (1, 47), (0, 99),
                     (95, 105), (1, 128)]:
            self.assert_range(a, b)

    def test_layers_regex_runs(self):
        rx = re.compile("^(?:" + modelctl_tiers.layers_regex([1, 2, 3, 7, 8, 20]) + ")$")
        for n, want in [(1, True), (2, True), (3, True), (4, False),
                        (7, True), (8, True), (9, False), (20, True), (21, False)]:
            self.assertEqual(bool(rx.match(str(n))), want, n)


class TestSplitExtraFlags(unittest.TestCase):
    def test_strips_placement_keeps_rest(self):
        placement, other = modelctl_tiers.split_extra_flags(
            "-ot 'ffn_.*_exps=CPU' -ngl 10 --no-mmap --ubatch-size 128 --fit off")
        self.assertIn("-ot", placement)
        self.assertIn("-ngl", placement)
        self.assertIn("--no-mmap", placement)
        self.assertIn("--fit", placement)
        self.assertEqual(other, ["--ubatch-size", "128"])

    def test_empty(self):
        self.assertEqual(modelctl_tiers.split_extra_flags(""), ([], []))
        self.assertEqual(modelctl_tiers.split_extra_flags(None), ([], []))


class TestPlanTiers(unittest.TestCase):
    def test_tier1_fits_primary(self):
        plan = modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=dense_layout(10, 0.5))
        self.assertEqual(plan["tier"], 1)
        self.assertEqual(plan["config"]["device"], "SYCL0")
        self.assertEqual(plan["config"]["extra"], "")

    def test_tier2_splits_gpus(self):
        # ~36 GiB dense: over the primary card alone, under both combined
        plan = modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=dense_layout(30, 1.0))
        self.assertEqual(plan["tier"], 2)
        self.assertEqual(plan["config"]["split_mode"], "layer")
        # The split ratio comes from USABLE bytes (limit_pct, reserves and
        # any cache reservation applied), not raw card capacity: llama.cpp
        # distributes the model by these weights, so a raw-capacity ratio
        # ("8,3" here) over-filled whichever card had the bigger haircut.
        # 90% of 32 GiB -> 29, 90% of 12 GiB -> 11.
        self.assertEqual(plan["config"]["tensor_split"], "29,11")

    def test_moe_spill_assigns_fastest_first(self):
        # Laguna-like: 47 layers x 1.1 GiB experts, 4 GiB non-expert
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(47, 1.1))
        self.assertIn(plan["tier"], (3, 4))
        extra = plan["config"]["extra"]
        # multi-device: needs split layer, explicit --device, --fit off
        self.assertEqual(plan["config"]["split_mode"], "layer")
        self.assertIn("--device SYCL0,SYCL1", extra)
        self.assertIn("--fit off", extra)
        # -ot: specific ranges before the CPU catch-all (first-match-wins)
        ot_pos = {tag: extra.index(tag) for tag in
                  ("=SYCL0", "=SYCL1", "=CPU")}
        self.assertLess(ot_pos["=SYCL0"], ot_pos["=CPU"])
        self.assertLess(ot_pos["=SYCL1"], ot_pos["=CPU"])
        # every expert layer appears in a specific range or under the catch-all
        toks = extra.split()
        ot = toks[toks.index("-ot") + 1]
        covered = set()
        has_catchall = False
        for part in ot.split(","):
            pattern = part.rsplit("=", 1)[0]
            # planner emits doubled backslashes for shlex; undo to get the regex
            pattern = pattern.replace("\\\\", "\\")
            m = re.match(r"blk\\\.(.*)\\\.ffn_", pattern)
            if not m:
                if pattern.startswith("ffn_"):
                    has_catchall = True
                continue
            rrx = re.compile("^(?:" + m.group(1) + ")$")
            for n in range(1, 48):
                if rrx.match(str(n)):
                    covered.add(n)
        if has_catchall:
            covered |= set(range(1, 48))
        self.assertEqual(covered, set(range(1, 48)))

    def test_moe_gpu_layers_respect_bandwidth_order(self):
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(47, 1.1))
        rows = {label: (gib, desc) for label, gib, desc in plan["layout"]}
        # B70 gets more expert GiB than the B580
        self.assertGreater(rows["SYCL0"][0], rows["SYCL1"][0])
        # CPU holds the remainder
        self.assertIn("CPU", rows)

    def test_tier4_when_exceeding_ram(self):
        # Ornith-like: 60 layers x 2.9 GiB experts -- way past RAM
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=4096), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(60, 2.9))
        self.assertEqual(plan["tier"], 4)
        self.assertNotIn("--no-mmap", plan["config"]["extra"])
        self.assertTrue(any("SSD" in w for w in plan["warnings"]))

    def test_tier3_gets_no_mmap(self):
        # MoE past combined VRAM but comfortably inside GPU+RAM
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0", ram_available=64 * GIB,
            layout=moe_layout(40, 1.2))
        self.assertEqual(plan["tier"], 3)
        self.assertIn("--no-mmap", plan["config"]["extra"])

    def test_dense_spill_computes_ngl(self):
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=4096), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=dense_layout(60, 1.5))
        self.assertIn(plan["tier"], (3, 4))
        m = re.search(r"-ngl (\d+)", plan["config"]["extra"])
        self.assertIsNotNone(m)
        ngl = int(m.group(1))
        self.assertGreater(ngl, 0)
        self.assertLess(ngl, 60)

    def test_replan_strips_old_placement_flags(self):
        p = profile(extra="-ot 'ffn_.*_exps=CPU' -ngl 10 --no-mmap --ubatch-size 128")
        plan = modelctl_tiers.plan_tiers(
            p, INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(47, 1.1))
        extra = plan["config"]["extra"]
        self.assertIn("--ubatch-size 128", extra)
        self.assertNotIn("-ngl 10", extra)
        self.assertEqual(extra.count("--no-mmap"), 1 if "--no-mmap" in extra else 0)

    def test_none_on_empty_layout(self):
        self.assertIsNone(modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", layout=None))


if __name__ == "__main__":
    unittest.main()


class TestRecommendQuantGroup(unittest.TestCase):
    def _groups(self):
        return [
            {"label": "model-BF16", "total_size": 54 * GIB, "sharded": True},
            {"label": "model-Q8_0", "total_size": 28 * GIB, "sharded": False},
            {"label": "model-Q6_K", "total_size": 22 * GIB, "sharded": False},
            {"label": "model-Q4_K_M", "total_size": 16 * GIB, "sharded": False},
            {"label": "model-IQ2_XXS", "total_size": 9 * GIB, "sharded": False},
            {"label": "imatrix_unsloth", "total_size": 350 * (1 << 20), "sharded": False},
        ]

    def test_picks_largest_that_fits(self):
        # 28.7 GiB budget: Q8_0 (28 + ~3.8 overhead/KV) doesn't fit, Q6_K does
        rec = modelctl_tiers.recommend_quant_group(
            self._groups(), int(28.7 * GIB), 64000)
        self.assertEqual(rec["group"]["label"], "model-Q6_K")
        self.assertTrue(rec["fits"])

    def test_excludes_imatrix(self):
        rec = modelctl_tiers.recommend_quant_group(
            [{"label": "imatrix_unsloth", "total_size": 1 << 20}],
            30 * GIB, 64000)
        self.assertIsNone(rec)

    def test_fallback_is_smallest(self):
        rec = modelctl_tiers.recommend_quant_group(
            self._groups(), 2 * GIB, 64000)
        self.assertEqual(rec["group"]["label"], "model-IQ2_XXS")
        self.assertFalse(rec["fits"])

    def test_tiny_model_takes_bf16(self):
        groups = [{"label": "m-BF16", "total_size": 2 * GIB},
                  {"label": "m-Q4_K_M", "total_size": 1 * GIB}]
        rec = modelctl_tiers.recommend_quant_group(groups, 30 * GIB, 32768)
        self.assertEqual(rec["group"]["label"], "m-BF16")


class TestAutoCtx(unittest.TestCase):
    """auto_ctx: largest CTX_STEP fitting weights+exact KV+overhead."""
    def _write_model(self, context_length=262144):
        from test_modelctl_vram import gguf_bytes
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "m.gguf"
        path.write_bytes(gguf_bytes({
            "general.architecture": (8, "testarch"),
            "testarch.block_count": (4, 32),
            "testarch.context_length": (4, context_length),
            "testarch.embedding_length": (4, 512),
            "testarch.attention.head_count": (4, 8),
            "testarch.attention.head_count_kv": (4, 4),
            "testarch.attention.key_length": (4, 128),
            "testarch.attention.value_length": (4, 128),
        }))
        return str(path)

    # KV/token here: 32 layers * 4 heads * 128 dims * 2 (K+V) * 1.0625 (q8_0)
    #   = 34816 B -> 131072 ctx ~= 4.25 GiB, 65536 ~= 2.1 GiB, 32768 ~= 1.1 GiB

    def test_picks_largest_fitting_step(self):
        rec = modelctl_tiers.auto_ctx(self._write_model(), 6 * GIB, "q8_0")
        self.assertEqual(rec["ctx"], 131072)
        self.assertTrue(rec["fits"])

    def test_drops_to_affordable_step(self):
        # 131072 (4.25 GiB KV) and 65536 (2.1 GiB) exceed 3 GiB with the
        # 1 GiB overhead; 32768 (1.06 GiB) fits.
        rec = modelctl_tiers.auto_ctx(self._write_model(), 3 * GIB, "q8_0")
        self.assertEqual(rec["ctx"], 32768)

    def test_model_max_caps(self):
        rec = modelctl_tiers.auto_ctx(self._write_model(context_length=32768),
                                      100 * GIB, "q8_0")
        self.assertEqual(rec["ctx"], 32768)
        self.assertEqual(rec["ctx"], rec["model_max"])

    def test_floor_when_nothing_fits(self):
        rec = modelctl_tiers.auto_ctx(self._write_model(), 1 << 20, "q8_0")
        self.assertEqual(rec["ctx"], modelctl_tiers.AUTO_CTX_FLOOR)
        self.assertFalse(rec["fits"])


class TestCacheRequest(unittest.TestCase):
    """cache_request: the planner reserves a UNIFORM per-GPU cache budget
    (the fork applies one --moe-cache-bytes to every device) before static
    expert placement."""

    def _req(self, budgets, mode="auto"):
        return {"mode": mode, "gpu": {"budgets_bytes": budgets}}

    def _plan(self, cache_request=None, **kw):
        args = dict(profile=profile(ctx=kw.pop("ctx", 8192)),
                    inventory=INVENTORY, limit_pct=90, primary="SYCL0",
                    ram_available=kw.pop("ram", 64 * GIB),
                    layout=kw.pop("layout", moe_layout(40, 1.2)))
        return modelctl_tiers.plan_tiers(cache_request=cache_request, **args)

    def test_budget_subtracted_shrinks_gpu_share(self):
        no_cache = self._plan()
        with_cache = self._plan(self._req({"SYCL0": 8 * GIB}))
        # uniform budget reserved on EVERY participating device
        self.assertEqual(with_cache["cache_budgets"],
                         {"SYCL0": 8 * GIB, "SYCL1": 8 * GIB})
        rows_nc = {l: g for l, g, _ in no_cache["layout"]}
        rows_wc = {l: g for l, g, _ in with_cache["layout"]}
        # reserved cache pushes more expert layers to CPU
        self.assertGreater(rows_wc["CPU"], rows_nc["CPU"])

    def test_per_device_budgets_collapse_to_uniform_max(self):
        plan = self._plan(self._req({"SYCL0": 4 * GIB, "SYCL1": 6 * GIB}))
        self.assertEqual(plan["cache_budgets"],
                         {"SYCL0": 6 * GIB, "SYCL1": 6 * GIB})
        self.assertEqual(plan["analysis"]["cache_budgets_gib"],
                         {"SYCL0": 6.0, "SYCL1": 6.0})

    def test_oversize_budget_disables_cache_loudly(self):
        # B580 usable is ~10.7 GiB; 20 GiB can't be reserved there, and the
        # fork has no per-device budgets -- so the whole cache is disabled.
        req = self._req({"SYCL0": 20 * GIB})
        plan = self._plan(req)
        self.assertIsNone(plan["cache_budgets"])
        self.assertIsNone(plan["analysis"]["cache_budgets_gib"])
        self.assertTrue(any("disabled" in w for w in plan["warnings"]))
        # ...and the placement is exactly the no-cache placement
        no_cache = self._plan()
        self.assertEqual(plan["tier"], no_cache["tier"])
        self.assertEqual(plan["config"], no_cache["config"])

    def test_layout_has_cache_rows(self):
        plan = self._plan(self._req({"SYCL0": 8 * GIB}))
        rows = {l: (g, d) for l, g, d in plan["layout"]}
        self.assertIn("SYCL0 (cache)", rows)
        self.assertIn("SYCL1 (cache)", rows)
        self.assertEqual(rows["SYCL0 (cache)"][0], 8.0)
        self.assertIn("cache", rows["SYCL0 (cache)"][1])

    def test_off_zero_and_absent_budgets(self):
        for req in (None,
                    self._req({}, mode="off"),
                    self._req({"SYCL0": 0}),
                    self._req({})):
            plan = self._plan(req)
            self.assertIsNone(plan["cache_budgets"], req)
            # docstring contract: the key is always present, value or None
            self.assertIn("cache_budgets_gib", plan["analysis"])
            self.assertIsNone(plan["analysis"]["cache_budgets_gib"], req)
