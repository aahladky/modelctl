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


if __name__ == "__main__":
    unittest.main()
