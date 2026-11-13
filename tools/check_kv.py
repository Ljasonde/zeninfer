#!/usr/bin/env python3
"""KV-cache invariants.

The cache must be a pure speed optimisation: the logits it produces have to be the
same logits a full recompute would produce. Two independent statements of that:

  1. parity    -- `logits --all` (one full prefill) vs `logits --all --cache`
                  (S single-token steps through the cache), every logit compared
  2. decisions -- greedy generation must emit the same ids with and without the
                  cache, i.e. the small differences (if any) never flip an argmax

The same run also reports the per-token decode cost of both modes, which is the
M3 headline number.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROMPTS = [
    ("short", "Hello"),
    ("sentence", "The quick brown fox jumps over the lazy dog and then keeps running"),
    ("chinese", "用一句话解释什么是 KV cache。"),
]

# Kept small on purpose: the no-cache path is O(n^2), so this comparison is the
# slowest thing in the suite.
SPEED_PROMPT = "Hello, my name is"
SPEED_NEW = 8


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"command failed ({out.returncode}): {' '.join(cmd)}\n{out.stderr}")
    return out.stdout


def write_prompt(text, tmpdir, name):
    # --text-file, never --text: Windows hands main() the GBK form of the command
    # line, so non-ASCII argv reaches the engine as mojibake.
    path = Path(tmpdir) / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def logits_parity(exe, model, name, text, tmpdir, tol):
    prompt = write_prompt(text, tmpdir, name)
    no = Path(tmpdir) / f"{name}.nocache.bin"
    ca = Path(tmpdir) / f"{name}.cache.bin"
    run([exe, "logits", model, "--text-file", prompt, "--all", "--out", str(no)])
    run([exe, "logits", model, "--text-file", prompt, "--all", "--cache", "--out", str(ca)])
    a = np.fromfile(no, dtype=np.float32)
    b = np.fromfile(ca, dtype=np.float32)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    ok = float(diff.max()) <= tol
    return ok, f"{a.size} logits, max|diff| {diff.max():.3e}"


def generate(exe, model, prompt, max_new, no_cache=False):
    cmd = [exe, "generate", model, "--text-file", prompt, "--max-new", str(max_new), "--json"]
    if no_cache:
        cmd.append("--no-cache")
    return json.loads(run(cmd).strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        print("-- logit parity: full recompute vs KV cache --")
        for name, text in PROMPTS:
            ok, detail = logits_parity(args.exe, args.model, name, text, tmp, args.tol)
            print(f"{name:10s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
            if not ok:
                failures += 1

        print("\n-- greedy decisions: cached vs recomputed --")
        speed_prompt = write_prompt(SPEED_PROMPT, tmp, "speed")
        cached = generate(args.exe, args.model, speed_prompt, SPEED_NEW)
        plain = generate(args.exe, args.model, speed_prompt, SPEED_NEW, no_cache=True)
        same = cached["ids"] == plain["ids"]
        print(f"{'ids match' if same else 'IDS DIFFER'}: cached={cached['ids']}")
        if not same:
            print(f"                 recomputed={plain['ids']}")
            failures += 1

        ct = cached["decode_ms"] / max(1, cached["decode_tokens"])
        pt = plain["decode_ms"] / max(1, plain["decode_tokens"])
        print(f"\n-- decode cost, {SPEED_NEW} new tokens --")
        print(f"with cache : {ct:9.2f} ms/token")
        print(f"no cache   : {pt:9.2f} ms/token   ({pt / ct:.1f}x slower)")

    print()
    if failures:
        print(f"RESULT: FAIL ({failures})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
