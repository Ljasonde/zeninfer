#!/usr/bin/env python3
"""Greedy generation straight out of HuggingFace transformers.

Ground truth for tools/check_generate.py: same token ids in, the same token ids
out, no sampling involved.
"""
import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="directory with the HF checkpoint")
    ap.add_argument("--ids", required=True, help="comma separated prompt token ids")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32)
    model.eval()

    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=args.max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, len(ids) :].tolist()
    print(json.dumps({"ids": gen, "text": tok.decode(gen)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
