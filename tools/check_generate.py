#!/usr/bin/env python3
"""Greedy-generation parity: zeninfer vs HuggingFace transformers.

Two independent checks:

  1. token ids   -- 32 greedy tokens must be identical, one by one
  2. stop reason -- the sequence must end at the same step (eos or max-new)

Logits are checked separately by check_forward.py; this test asks the different
question of whether a small numerical difference ever changes a decision.

Prompts go through --text-file, never --text: on Windows the C runtime hands
main() the ANSI (GBK) version of the command line, so non-ASCII argv reaches the
engine as mojibake. Files are read as UTF-8 and do not have that problem.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROMPTS = [
    ("single-token", "Hello"),
    ("chat-turn", "<|im_start|>user\n用一句话解释什么是 KV cache。<|im_end|>\n<|im_start|>assistant\n"),
    ("longer", "Write a haiku about the sea and explain the last line."),
]


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"command failed ({out.returncode}): {' '.join(cmd)}\n{out.stderr}")
    return out.stdout


def json_last_line(cmd):
    return json.loads(run(cmd).strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in PROMPTS:
            path = Path(tmp) / f"{name}.txt"
            path.write_text(text, encoding="utf-8")

            tok = json_last_line([args.exe, "tokenize", args.model, "--text-file", str(path), "--json"])
            ids = ",".join(str(i) for i in tok["ids"])

            zen = json_last_line(
                [args.exe, "generate", args.model, "--text-file", str(path),
                 "--max-new", str(args.max_new), "--json"]
            )
            hf = json_last_line(
                [args.python, "tools/hf_generate.py", "--hf", args.hf, "--ids", ids,
                 "--max-new", str(args.max_new)]
            )

            z, h = zen["ids"], hf["ids"]
            n = min(len(z), len(h))
            first_bad = next((i for i in range(n) if z[i] != h[i]), None)
            agree = first_bad is None and len(z) == len(h)

            print(f"{name:14s} prompt={len(tok['ids']):3d} tok  zen={len(z):3d}  hf={len(h):3d}  "
                  f"match={'yes' if agree else 'NO'}", flush=True)
            if hf.get("text"):
                # build/hf_f32 is a weights-only checkpoint, so this is empty unless a
                # tokenizer.json is dropped in next to it. Ids are the real test.
                print(f"               text: {hf['text'].strip()!r}")
            if not agree:
                failures += 1
                if first_bad is not None:
                    lo = max(0, first_bad - 3)
                    print(f"               first divergence at token {first_bad}: "
                          f"zen={z[lo:first_bad + 3]} hf={h[lo:first_bad + 3]}")

    print()
    if failures:
        print(f"RESULT: FAIL ({failures} prompt(s) diverged)")
        return 1
    print("RESULT: PASS (greedy argmax agrees token by token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
