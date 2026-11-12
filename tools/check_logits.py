#!/usr/bin/env python3
"""Compare two raw f32 logit dumps (little-endian) from zeninfer and ref_forward.

usage:
    python tools/check_logits.py --cpp build/cpp.bin --ref build/py.bin [--tol 1e-4]

Exit code 0 only if every logit is within tolerance and the argmax agrees.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    a = np.fromfile(args.cpp, dtype="<f4").astype(np.float64)
    b = np.fromfile(args.ref, dtype="<f4").astype(np.float64)
    if a.shape != b.shape:
        print(f"FAIL: shape mismatch cpp={a.shape} ref={b.shape}")
        return 1

    d = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1.0)
    rel = d / denom
    arg_a, arg_b = int(np.argmax(a)), int(np.argmax(b))

    print(f"elements       : {a.size}")
    print(f"max |diff|     : {d.max():.3e}")
    print(f"mean |diff|    : {d.mean():.3e}")
    print(f"max rel diff   : {rel.max():.3e}")
    print(f"argmax         : cpp={arg_a} ref={arg_b} {'(agree)' if arg_a == arg_b else '(DISAGREE)'}")
    # A top-5 comparison is the part a reviewer actually cares about: are the same
    # tokens ranked the same way?
    ka, kb = np.argsort(-a)[:5], np.argsort(-b)[:5]
    print(f"top-5 cpp      : {[int(i) for i in ka]}")
    print(f"top-5 ref      : {[int(i) for i in kb]}")
    same = bool(np.array_equal(ka, kb))

    ok = bool(d.max() <= args.tol) and arg_a == arg_b and same
    print(f"tolerance      : {args.tol:.1e}")
    print("RESULT         :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
