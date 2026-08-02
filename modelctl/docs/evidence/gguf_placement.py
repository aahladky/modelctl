#!/usr/bin/env python3
"""Derive the RPC expert-block placement from GGUF metadata.

Dependency-free GGUF parser (no numpy on this box). Tensor byte sizes are
computed two independent ways -- from the ggml type table, and from the
deltas between consecutive tensor data offsets -- and cross-checked, so
the arithmetic does not rest on a hardcoded table alone.

Reads headers only; no shard is pulled into RAM and the page cache is
not manipulated.
"""
import glob
import json
import struct
import sys
from collections import defaultdict

SHARDS = sorted(glob.glob(
    "/home/aaron/models/unsloth/Laguna-S-2.1-GGUF/UD-IQ4_NL/"
    "Laguna-S-2.1-UD-IQ4_NL-*-of-00003.gguf"))

# (blck_size, type_size) per ggml type id -- the standard ggml table.
TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18),
    3: ("q4_1", 32, 20), 6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36), 10: ("q2_K", 256, 84),
    11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144), 13: ("q5_K", 256, 176),
    14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74),
    18: ("iq3_xxs", 256, 98), 19: ("iq1_s", 256, 50),
    20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4),
    27: ("i64", 1, 8), 28: ("f64", 1, 8), 29: ("iq1_m", 256, 56),
    30: ("bf16", 1, 2), 39: ("mxfp4", 32, 17),
}


class R:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, vtype):
        simple = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
                  4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
                  10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
        if vtype in simple:
            fmt, size = simple[vtype]
            return struct.unpack(fmt, self.raw(size))[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = self.u32()
            count = self.u64()
            if etype == 8:
                return [self.string() for _ in range(count)]
            return [self.value(etype) for _ in range(count)]
        raise ValueError(f"unknown gguf value type {vtype}")


def read_shard(path):
    with open(path, "rb") as fh:
        r = R(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: bad magic {magic!r}")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            ndim = r.u32()
            shape = [r.u64() for _ in range(ndim)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append({"name": name, "shape": shape,
                            "type": ttype, "offset": offset})
        align = kv.get("general.alignment", 32)
        pos = fh.tell()
        data_start = pos + (-pos % align)
        fh.seek(0, 2)
        file_size = fh.tell()
    return {"version": version, "kv": kv, "tensors": tensors,
            "data_start": data_start, "file_size": file_size,
            "align": align, "path": path}


def type_bytes(t):
    name, blck, tsize = TYPES[t["type"]]
    n_elem = 1
    for d in t["shape"]:
        n_elem *= d
    if n_elem % blck:
        raise ValueError(f"{t['name']}: {n_elem} not divisible by {blck}")
    return n_elem // blck * tsize, name


shards = [read_shard(p) for p in SHARDS]

# --- metadata --------------------------------------------------------
kv = {}
for s in shards:
    for k, v in s["kv"].items():
        kv.setdefault(k, v)

print("=== GGUF metadata ===")
print(f"  gguf version        = {shards[0]['version']}")
print(f"  general.architecture= {kv.get('general.architecture')}")
arch = kv.get("general.architecture", "")
for suffix in ("block_count", "embedding_length", "expert_count",
               "expert_used_count", "feed_forward_length",
               "expert_feed_forward_length", "attention.head_count"):
    key = f"{arch}.{suffix}"
    if key in kv:
        print(f"  {key:<38}= {kv[key]}")
n_embd = kv.get(f"{arch}.embedding_length")

# --- per-tensor bytes, two ways --------------------------------------
mismatch = 0
per_block_exps = defaultdict(int)
per_block_exps_off = defaultdict(int)
per_block_shexp = defaultdict(int)
exps_detail = defaultdict(list)
blocks = set()

for s in shards:
    ordered = sorted(s["tensors"], key=lambda t: t["offset"])
    for i, t in enumerate(ordered):
        tb, tname = type_bytes(t)
        if i + 1 < len(ordered):
            span = ordered[i + 1]["offset"] - t["offset"]
        else:
            span = s["file_size"] - s["data_start"] - t["offset"]
        # span includes alignment padding; it must never be smaller
        if span < tb or span - tb >= s["align"]:
            mismatch += 1
        t["bytes"] = tb
        t["span"] = span
        t["tname"] = tname
        if not t["name"].startswith("blk."):
            continue
        blk = int(t["name"].split(".")[1])
        blocks.add(blk)
        if "_exps" in t["name"]:
            per_block_exps[blk] += tb
            per_block_exps_off[blk] += span
            exps_detail[blk].append((t["name"], tname, t["shape"], tb))
        elif "_shexp" in t["name"]:
            per_block_shexp[blk] += tb

print(f"\n  tensors: {sum(len(s['tensors']) for s in shards)}, "
      f"blocks {min(blocks)}..{max(blocks)} ({len(blocks)})")
print(f"  type-table vs offset-delta cross-check mismatches: {mismatch}")

print("\n=== routed-expert bytes per block ===")
sizes = sorted(set(per_block_exps.values()))
print(f"  distinct per-block totals: {[format(x, ',') for x in sizes]}")
sample = sorted(per_block_exps)[len(per_block_exps) // 2]
print(f"  sample block {sample}:")
for name, tname, shape, nb in sorted(exps_detail[sample]):
    print(f"    {name:<32} {tname:<8} {str(shape):<26} {nb:>14,} B")
print(f"    {'TOTAL':<32} {'':<8} {'':<26} "
      f"{per_block_exps[sample]:>14,} B "
      f"({per_block_exps[sample] / 2**30:.4f} GiB)")
if sample in per_block_shexp:
    print(f"    (shexp for this block, stays pinned: "
          f"{per_block_shexp[sample]:,} B)")

LIVE_SYCL0 = set(range(1, 20))
LIVE_SYCL1 = set(range(20, 29))
cpu_blocks = sorted(b for b in per_block_exps
                    if b not in LIVE_SYCL0 and b not in LIVE_SYCL1)
cpu_total = sum(per_block_exps[b] for b in cpu_blocks)
print("\n=== routed experts the live -ot rule leaves on CPU ===")
print(f"  blocks: {cpu_blocks}")
print(f"  total: {cpu_total:,} B ({cpu_total / 2**30:.3f} GiB)")

print("\n=== contiguous fill from blk 29 upward ===")
running = 0
fits = []
for b in [x for x in cpu_blocks if x >= 29]:
    running += per_block_exps[b]
    fits.append((b, per_block_exps[b], running))
    print(f"  through blk {b:>2}: +{per_block_exps[b]:>13,} B  "
          f"cumulative {running:>14,} B  {running / 2**30:8.4f} GiB")

json.dump(
    {"n_embd": n_embd, "arch": arch,
     "per_block_exps_bytes": {str(k): v for k, v in sorted(per_block_exps.items())},
     "per_block_shexp_bytes": {str(k): v for k, v in sorted(per_block_shexp.items())},
     "cpu_blocks_live": cpu_blocks, "cross_check_mismatches": mismatch},
    open("/home/aaron/workspace/.lanes/fleet-laguna-baseline/gguf-placement.json", "w"),
    indent=1)
print("\nwrote gguf-placement.json")
