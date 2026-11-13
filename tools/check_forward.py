#!/usr/bin/env python3
"""End-to-end forward-pass parity: run zeninfer and the NumPy reference on the same
token ids and compare every logit.

Covers three regimes the single-token smoke test cannot reach:
  * S=1     -- embeddings + norms + all projections in isolation
  * S=9     -- one chat turn, exercises causal masking and GQA head mapping
  * S=21    -- longer context, catches off-by-one in the mask / rope positions

usage:
    python tools/check_forward.py --exe build/zeninfer --model models/x.gguf [--tol 1e-4]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROMPTS = [
    ("single-token", "Hello"),
    ("chat-turn", "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"),
    ("longer", "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\n用一句话解释什么是 KV cache。<|im_end|>\n"
               "<|im_start|>assistant\n"),
]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def ids_for(exe, model, text_file):
    out = run([exe, "tokenize", model, "--text-file", text_file, "--json"])
    return json.loads(out.strip().splitlines()[-1])["ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "zeninfer"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "qwen2.5-0.5b-instruct-q8_0.gguf"))
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--tol-hf", type=float, default=5e-4,
                    help="HF is the same weights run through f32 torch kernels, so its own "
                         "rounding noise is ~1e-4; the float64 NumPy reference stays at --tol")
    ap.add_argument("--hf", default="", help="HF checkpoint dir (see tools/gguf_to_hf.py) for a 3-way check")
    args = ap.parse_args()

    exe = args.exe
    if os.name == "nt" and not exe.endswith(".exe") and os.path.exists(exe + ".exe"):
        exe += ".exe"

    tmp = tempfile.mkdtemp(prefix="zenfwd-")
    failures = 0
    header = f"{'case':14} {'tokens':>6} {'max|diff|':>11} {'argmax':>14}  result"
    if args.hf:
        header += f" {'vs hf':>11}  hf"
    print(header)
    for name, prompt in PROMPTS:
        tfile = os.path.join(tmp, name + ".txt")
        with open(tfile, "w", encoding="utf-8", newline="") as f:
            f.write(prompt)
        ids = ids_for(exe, args.model, tfile)
        idarg = ",".join(str(i) for i in ids)

        cpp_bin = os.path.join(tmp, name + ".cpp.bin")
        py_bin = os.path.join(tmp, name + ".py.bin")
        run([exe, "logits", args.model, "--ids", idarg, "--out", cpp_bin, "--topk", "1"])
        run([sys.executable, os.path.join(HERE, "ref_forward.py"), args.model, "--ids", idarg, "--out", py_bin])

        a = np.fromfile(cpp_bin, dtype="<f4").astype(np.float64)
        b = np.fromfile(py_bin, dtype="<f4").astype(np.float64)
        d = float(np.abs(a - b).max())
        arg_a, arg_b = int(np.argmax(a)), int(np.argmax(b))
        ok = d <= args.tol and arg_a == arg_b and np.array_equal(np.argsort(-a)[:5], np.argsort(-b)[:5])
        line = f"{name:14} {len(ids):6d} {d:11.3e} {arg_a:7d}/{arg_b:<7d}  {'PASS' if ok else 'FAIL'}"

        if args.hf:
            hf_bin = os.path.join(tmp, name + ".hf.bin")
            run([sys.executable, os.path.join(HERE, "hf_logits.py"), args.hf, "--ids", idarg, "--out", hf_bin])
            c = np.fromfile(hf_bin, dtype="<f4").astype(np.float64)
            dh = float(np.abs(a - c).max())
            okh = dh <= args.tol_hf and int(np.argmax(c)) == arg_a
            ok = ok and okh
            line += f" {dh:11.3e}  {'PASS' if okh else 'FAIL'}"

        failures += 0 if ok else 1
        print(line)

    print(f"\n{'RESULT':14} {'':6} {'':11} {'':14}  {'PASS' if failures == 0 else 'FAIL'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
