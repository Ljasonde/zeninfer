#!/usr/bin/env python3
"""Ground truth via HuggingFace transformers' own Qwen2 implementation.

Loads the checkpoint produced by gguf_to_hf.py (same weights as the gguf, just
renamed) and dumps the last-position logits, so the C++ engine is compared against
an implementation written by someone else.

usage:
    python tools/hf_logits.py <hf_dir> --ids 1,2,3 --out hf.bin
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ids = [int(t) for t in args.ids.split(",") if t.strip() != ""]
    cfg = AutoConfig.from_pretrained(args.hf_dir)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    model.load_state_dict(
        {k: torch.from_numpy(np.asarray(v)) for k, v in _load_safetensors(args.hf_dir).items()},
        strict=True,
    )
    model.eval()

    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long))
    logits = out.logits[0, -1].to(torch.float32).numpy()

    top = np.argsort(-logits.astype(np.float64))[:5]
    print("ids            :", ids)
    print("source         :", os.path.basename(args.hf_dir))
    print("min/max        : %.6f / %.6f" % (logits.min(), logits.max()))
    print("top-5          :", [(int(i), float(logits[i])) for i in top])
    if args.out:
        logits.astype("<f4").tofile(args.out)
        print("wrote", args.out, logits.size * 4, "bytes")


def _load_safetensors(path):
    from safetensors.numpy import load_file
    return load_file(os.path.join(path, "model.safetensors"))


if __name__ == "__main__":
    sys.exit(main())
