import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_vram


def gguf_bytes(kvs, version=3, magic=b"GGUF"):
    """Build a minimal GGUF file: header + metadata KVs, zero tensors.

    kvs: dict of key -> (type_id, value). Supported type_ids here:
    4=uint32, 8=string, 9=array-of-uint32 (value = list of ints),
    10=uint64.
    """
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = [magic, struct.pack("<I", version), struct.pack("<QQ", 0, len(kvs))]
    for key, (type_id, value) in kvs.items():
        out.append(s(key))
        out.append(struct.pack("<I", type_id))
        if type_id == 4:
            out.append(struct.pack("<I", value))
        elif type_id == 8:
            out.append(s(value))
        elif type_id == 9:
            out.append(struct.pack("<I", 4) + struct.pack("<Q", len(value)))
            out.extend(struct.pack("<I", v) for v in value)
        elif type_id == 10:
            out.append(struct.pack("<Q", value))
        else:
            raise AssertionError(f"fixture doesn't support type {type_id}")
    return b"".join(out)


class TestReadGgufKvMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"

    def test_reads_scalar_and_string_kvs(self):
        self.path.write_bytes(gguf_bytes({
            "general.architecture": (8, "qwen3"),
            "qwen3.block_count": (4, 48),
            "qwen3.embedding_length": (4, 5120),
            "qwen3.attention.head_count": (4, 40),
            "qwen3.attention.head_count_kv": (4, 8),
        }))
        meta = modelctl_vram.read_gguf_kv_metadata(str(self.path))
        self.assertEqual(meta["general.architecture"], "qwen3")
        self.assertEqual(meta["qwen3.block_count"], 48)
        self.assertEqual(meta["qwen3.attention.head_count_kv"], 8)

    def test_reads_int_arrays(self):
        self.path.write_bytes(gguf_bytes({
            "general.architecture": (8, "x"),
            "x.attention.head_count_kv": (9, [8, 8, 4, 4]),
        }))
        meta = modelctl_vram.read_gguf_kv_metadata(str(self.path))
        self.assertEqual(meta["x.attention.head_count_kv"], [8, 8, 4, 4])

    def test_uint64_values(self):
        self.path.write_bytes(gguf_bytes({"x.block_count": (10, 32)}))
        meta = modelctl_vram.read_gguf_kv_metadata(str(self.path))
        self.assertEqual(meta["x.block_count"], 32)

    def test_wrong_magic_returns_empty(self):
        self.path.write_bytes(gguf_bytes({}, magic=b"NOPE"))
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})

    def test_truncated_file_returns_empty(self):
        full = gguf_bytes({"general.architecture": (8, "qwen3")})
        self.path.write_bytes(full[: len(full) - 3])
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})

    def test_missing_file_returns_empty(self):
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})

    def test_v1_unsupported_returns_empty(self):
        self.path.write_bytes(gguf_bytes({}, version=1))
        self.assertEqual(modelctl_vram.read_gguf_kv_metadata(str(self.path)), {})


class TestGgufKvParams(unittest.TestCase):
    def test_derives_head_dim_from_embedding(self):
        meta = {
            "general.architecture": "qwen3",
            "qwen3.block_count": 48,
            "qwen3.embedding_length": 5120,
            "qwen3.attention.head_count": 40,
            "qwen3.attention.head_count_kv": 8,
        }
        p = modelctl_vram.gguf_kv_params(meta)
        self.assertEqual(p, {"block_count": 48, "n_kv_heads": 8,
                             "k_dim": 128.0, "v_dim": 128.0})

    def test_explicit_key_value_length_wins(self):
        meta = {
            "general.architecture": "x",
            "x.block_count": 32,
            "x.attention.head_count": 32,
            "x.attention.head_count_kv": 8,
            "x.embedding_length": 4096,
            "x.attention.key_length": 192,
            "x.attention.value_length": 128,
        }
        p = modelctl_vram.gguf_kv_params(meta)
        self.assertEqual(p["k_dim"], 192)
        self.assertEqual(p["v_dim"], 128)

    def test_per_layer_kv_head_array_averaged(self):
        meta = {
            "general.architecture": "x",
            "x.block_count": 4,
            "x.embedding_length": 1024,
            "x.attention.head_count": 8,
            "x.attention.head_count_kv": [8, 8, 4, 4],
        }
        p = modelctl_vram.gguf_kv_params(meta)
        self.assertEqual(p["n_kv_heads"], 6.0)

    def test_missing_arch_or_fields_returns_none(self):
        self.assertIsNone(modelctl_vram.gguf_kv_params({}))
        self.assertIsNone(modelctl_vram.gguf_kv_params({
            "general.architecture": "x", "x.block_count": 32,
        }))


class TestKvCacheBytes(unittest.TestCase):
    def test_f16_math(self):
        params = {"block_count": 48, "n_kv_heads": 8, "k_dim": 128, "v_dim": 128}
        # 48 layers * 32768 ctx * 8 heads * (128+128) dims * 2 bytes
        expected = 48 * 32768 * 8 * 256 * 2
        self.assertEqual(modelctl_vram.kv_cache_bytes(params, 32768, "f16"), expected)

    def test_q8_0_block_size(self):
        params = {"block_count": 1, "n_kv_heads": 1, "k_dim": 32, "v_dim": 32}
        # q8_0: 34 bytes per 32 elements = 1.0625 bytes/element
        self.assertEqual(modelctl_vram.kv_cache_bytes(params, 1, "q8_0"),
                         int(64 * 1.0625))

    def test_unknown_quant_conservative_f16(self):
        params = {"block_count": 1, "n_kv_heads": 1, "k_dim": 32, "v_dim": 32}
        self.assertEqual(modelctl_vram.kv_cache_bytes(params, 1, "weird"), 128)


class TestEstimateFromParts(unittest.TestCase):
    def test_exact_estimate(self):
        params = {"block_count": 48, "n_kv_heads": 8, "k_dim": 128, "v_dim": 128}
        weights = 18 * (1 << 30)
        est = modelctl_vram.estimate_from_parts(weights, 32768, "q8_0",
                                                gguf_params=params)
        self.assertEqual(est["quality"], "exact")
        self.assertEqual(est["weights"], weights)
        self.assertEqual(est["kv_bytes"],
                         modelctl_vram.kv_cache_bytes(params, 32768, "q8_0"))
        self.assertEqual(est["overhead"], int(weights * 0.10))  # >1GiB weights
        self.assertEqual(est["total"],
                         est["weights"] + est["kv_bytes"] + est["overhead"])

    def test_heuristic_fallback(self):
        est = modelctl_vram.estimate_from_parts(2 * (1 << 30), 8192, "f16",
                                                gguf_params=None)
        self.assertEqual(est["quality"], "heuristic")
        self.assertEqual(est["kv_bytes"], 8192 * modelctl_vram.HEURISTIC_KV_BYTES_PER_TOKEN)
        self.assertEqual(est["overhead"], 1 << 30)  # 10% of 2GiB < 1GiB floor

    def test_mmproj_included_in_weights(self):
        est = modelctl_vram.estimate_from_parts(100, 1, "f16", mmproj_bytes=50)
        self.assertEqual(est["weights"], 150)


if __name__ == "__main__":
    unittest.main()
