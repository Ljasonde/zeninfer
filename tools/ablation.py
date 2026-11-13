#!/usr/bin/env python3
"""Ablation table: scalar vs AVX2 vs threads, on one prompt.

Runs `generate --json` once per configuration and prints prefill/decode cost.
Everything is measured, nothing is modelled: the numbers in README.md come from
this script.

    python tools/ablation.py --exe ./build/zeninfer --model models/q8_0.gguf \
        --text-file build/prompt.txt --max-new 16
"""
import argparse
import json
import subprocess
import sys


def run(exe, model, prompt, max_new, impl, threads):
    cmd = [exe, "generate", model, "--text-file", prompt, "--max-new", str(max_new),
           "--json", "--impl", impl, "--threads", str(threads)]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--threads", default="1,2,4,8,16")
    args = ap.parse_args()

    configs = [("scalar", 1)]
    for t in [int(x) for x in args.threads.split(",") if x.strip()]:
        configs.append(("avx2", t))

    rows = []
    for impl, threads in configs:
        r = run(args.exe, args.model, args.text_file, args.max_new, impl, threads)
        pre = r["prefill_ms"] / max(1, r["prompt_tokens"])
        dec = r["decode_ms"] / max(1, r["decode_tokens"])
        rows.append((impl, threads, pre, dec, r["decode_tokens"]))
        print(f"{impl:7s} threads={threads:2d}  prefill {pre:7.2f} ms/tok   "
              f"decode {dec:7.2f} ms/tok   ({1000.0 / dec:5.2f} tok/s)", flush=True)

    base = rows[0][3]
    print("\n| config | prefill ms/token | decode ms/token | decode tok/s | vs scalar |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for impl, threads, pre, dec, _ in rows:
        name = f"{impl}, {threads} thread" + ("s" if threads > 1 else "")
        print(f"| {name} | {pre:.1f} | {dec:.1f} | {1000.0 / dec:.2f} | {base / dec:.2f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
