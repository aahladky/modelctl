#!/usr/bin/env python3
"""Synthesize a tiny but real llama-arch GGUF for the RPC smoke.

Nothing on this box is small enough to be a "tiny fixture": every local
GGUF is either a vocab-only stub with no weights or the first shard of a
17-400 GB model. This writes ~1.5 MB of deterministic random weights in
the llama architecture -- enough layers (8) to place a contiguous
trailing range on a remote node, and small enough that the smoke proves
the plumbing rather than exercising the hardware.

Weights are seeded, so the file is byte-reproducible.
"""
import sys

sys.path.insert(0, "/home/aaron/workspace/moe-serving/llama.cpp/gguf-py")

import numpy as np
import gguf

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/aaron/models/fixtures/tiny-rpc-smoke.gguf"

N_VOCAB = 256
N_EMBD = 64
N_LAYER = 8
N_HEAD = 4
N_HEAD_KV = 4
N_FF = 128
HEAD_DIM = N_EMBD // N_HEAD

rng = np.random.default_rng(20260801)


def w(*shape):
    # Small magnitudes keep the forward pass numerically tame; the smoke
    # asserts determinism, not quality.
    return (rng.standard_normal(shape).astype(np.float32) * 0.02)


def main():
    writer = gguf.GGUFWriter(OUT, "llama")

    writer.add_name("tiny-rpc-smoke")
    writer.add_context_length(512)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)

    # An ASCII vocabulary, NOT byte tokens. Byte tokens detokenize to raw
    # bytes, and a random-weight model emits arbitrary sequences of them
    # -- which is invalid UTF-8, and llama-server's content parser
    # rejects the response with a 500 before the smoke can compare
    # anything. Every token here is printable ASCII, so any sequence the
    # model produces is valid UTF-8 by construction.
    tokens, scores, toktypes = [], [], []
    for i in range(N_VOCAB):
        tokens.append(f"x{i:03d}")
        scores.append(0.0)
        toktypes.append(gguf.TokenType.NORMAL)
    # The specials llama.cpp looks up by id.
    tokens[0] = "<unk>"
    toktypes[0] = gguf.TokenType.UNKNOWN
    tokens[1] = "<s>"
    toktypes[1] = gguf.TokenType.CONTROL
    tokens[2] = "</s>"
    toktypes[2] = gguf.TokenType.CONTROL

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    writer.add_tensor("token_embd.weight", w(N_VOCAB, N_EMBD))
    writer.add_tensor("output_norm.weight",
                      np.ones(N_EMBD, dtype=np.float32))
    writer.add_tensor("output.weight", w(N_VOCAB, N_EMBD))

    for i in range(N_LAYER):
        p = f"blk.{i}."
        writer.add_tensor(p + "attn_norm.weight",
                          np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(p + "attn_q.weight", w(N_EMBD, N_EMBD))
        writer.add_tensor(p + "attn_k.weight",
                          w(N_HEAD_KV * HEAD_DIM, N_EMBD))
        writer.add_tensor(p + "attn_v.weight",
                          w(N_HEAD_KV * HEAD_DIM, N_EMBD))
        writer.add_tensor(p + "attn_output.weight", w(N_EMBD, N_EMBD))
        writer.add_tensor(p + "ffn_norm.weight",
                          np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(p + "ffn_gate.weight", w(N_FF, N_EMBD))
        writer.add_tensor(p + "ffn_up.weight", w(N_FF, N_EMBD))
        writer.add_tensor(p + "ffn_down.weight", w(N_EMBD, N_FF))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
