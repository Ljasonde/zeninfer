#!/usr/bin/env python3
"""Convert a Qwen2 gguf (F32/F16/Q8_0/Q4_0) into a HuggingFace checkpoint directory.

Used to obtain an independent ground truth: the NumPy reference in ref_forward.py
shares my reading of the architecture, so a wrong-but-consistent forward pass would
still "pass". transformers' Qwen2ForCausalLM does not share it.

usage:
    python tools/gguf_to_hf.py <model.gguf> <out_dir> [--dtype float32|float16]
"""
import argparse
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_gguf  # noqa: E402
from ref_forward import Weights  # noqa: E402

# gguf name -> HF suffix, per block; {i} is the layer index.
BLOCK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    w = Weights(args.model)
    hp = w.info["hyper"]
    get = lambda k, d: hp.get(k, d)  # noqa: E731

    n_layer = int(get("qwen2.block_count", 24))
    n_embd = int(get("qwen2.embedding_length", 896))
    n_head = int(get("qwen2.attention.head_count", 14))
    n_kv = int(get("qwen2.attention.head_count_kv", 2))
    n_ff = int(get("qwen2.feed_forward_length", 4864))
    dtype = np.float32 if args.dtype == "float32" else np.float16

    state = {}
    for gguf_name, hf_name in GLOBAL_MAP.items():
        if gguf_name not in w.index:
            continue
        t = w.index[gguf_name]
        arr = w.vec(gguf_name) if len(t["dims"]) == 1 else w.mat(gguf_name)
        state[hf_name] = np.asarray(arr, dtype=dtype)

    for i in range(n_layer):
        for suffix, hf_suffix in BLOCK_MAP.items():
            gguf_name = f"blk.{i}.{suffix}"
            t = w.index[gguf_name]
            if len(t["dims"]) == 1:
                arr = w.vec(gguf_name)
            else:
                arr = w.mat(gguf_name)
            state[f"model.layers.{i}.{hf_suffix}"] = np.asarray(arr, dtype=dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in state.items()},
              os.path.join(args.out_dir, "model.safetensors"))

    tok = w.info.get("tokenizer", {})
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": n_embd,
        "intermediate_size": n_ff,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "vocab_size": int(state["lm_head.weight"].shape[0]),
        "hidden_act": "silu",
        "max_position_embeddings": int(get("qwen2.context_length", 32768)),
        "rms_norm_eps": float(get("qwen2.attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_theta": float(get("qwen2.rope.freq_base", 1e6)),
        "tie_word_embeddings": False,
        "torch_dtype": args.dtype,
        "bos_token_id": int(tok.get("tokenizer.ggml.bos_token_id", 151643)),
        "eos_token_id": int(tok.get("tokenizer.ggml.eos_token_id", 151645)),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    total = sum(v.nbytes for v in state.values())
    print(f"wrote {args.out_dir}: {len(state)} tensors, {total / 1048576:.1f} MiB ({args.dtype})")
    print("config:", json.dumps({k: config[k] for k in ("hidden_size", "num_attention_heads",
                                                       "num_key_value_heads", "vocab_size")}))


if __name__ == "__main__":
    main()
