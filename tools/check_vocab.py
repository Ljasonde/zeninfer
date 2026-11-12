#!/usr/bin/env python3
"""Compare `zeninfer vocab-dump` output against a HuggingFace tokenizer.json.

Both sides are reduced to the same canonical form: token id -> byte-level
spelling (as UTF-8 hex) and merge rank -> (left, right).

    python tools/check_vocab.py --vocab build/q8_0.vocab --tokenizer models/tokenizer.json
"""
import argparse
import json
import re
import sys


def hexs(b: bytes) -> str:
    return b.hex()


def utf8h(s: str) -> str:
    return hexs(s.encode("utf-8"))


def load_hf(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    model = tj["model"]
    vocab = model["vocab"]
    tokens = [None] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    merges = []
    for m in model["merges"]:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m[0], m[1]
        merges.append((a, b))
    added = [t["content"] for t in tj.get("added_tokens", [])]
    return tokens, merges, added, tj.get("normalizer")


def load_flat(path: str):
    tokens, merges = [], []
    with open(path, "r", encoding="utf-8") as f:
        assert f.readline().strip() == "zenvocab 1", "bad magic"
        n = int(f.readline().split()[1])
        for _ in range(n):
            hx, typ = f.readline().split()
            tokens.append((hx, int(typ)))
        m = int(f.readline().split()[1])
        for _ in range(m):
            a, b = f.readline().split()
            merges.append((a, b))
    return tokens, merges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    hf_tokens, hf_merges, added, normalizer = load_hf(args.tokenizer)
    my_tokens, my_merges = load_flat(args.vocab)

    print(f"hf   tokens: {len(hf_tokens)}  merges: {len(hf_merges)}  added: {len(added)}")
    print(f"zen  tokens: {len(my_tokens)}  merges: {len(my_merges)}")
    print(f"normalizer: {normalizer}")

    bad = 0
    n = min(len(hf_tokens), len(my_tokens))
    for i in range(n):
        want = hf_tokens[i]
        got = bytes.fromhex(my_tokens[i][0]).decode("utf-8", "replace") if my_tokens[i][0] else ""
        if want != got:
            bad += 1
            if bad <= args.show:
                print(f"--- token id {i} ---")
                print(f"  hf  : {want!r} ({utf8h(want)})")
                print(f"  zen : {got!r} ({my_tokens[i][0]})")
    extra = len(my_tokens) - len(hf_tokens)
    extra_ok = False
    if extra > 0:
        trailer = [bytes.fromhex(h).decode("utf-8", "replace") for h, _ in my_tokens[len(hf_tokens):]]
        # llama.cpp appends the HF added tokens, then pads the vocab up to a
        # round number with "[PADnnnn]" placeholders.
        pads = [t for t in trailer if re.fullmatch(r"\[PAD\d+\]", t)]
        others = [t for t in trailer if t not in added and not re.fullmatch(r"\[PAD\d+\]", t)]
        extra_ok = not others
        tag = "OK: added_tokens + [PADn] padding" if extra_ok else f"MISMATCH: {others[:4]}"
        print(f"\nzen has {extra} extra trailing token(s) - {tag}")
        print(f"  {len(trailer) - len(pads)} added_tokens + {len(pads)} PAD placeholders"
              f" ({pads[0] if pads else '-'} .. {pads[-1] if pads else '-'})")
    elif extra < 0:
        print(f"\nzen is missing {-extra} trailing token(s)")

    bad_m = 0
    nm = min(len(hf_merges), len(my_merges))
    for i in range(nm):
        a, b = hf_merges[i]
        want = (utf8h(a), utf8h(b))
        if want != my_merges[i]:
            bad_m += 1
            if bad_m <= args.show:
                print(f"--- merge rank {i} ---")
                print(f"  hf  : {a!r} + {b!r} = {want}")
                print(f"  zen : {bytes.fromhex(my_merges[i][0]).decode('utf-8', 'replace')!r} + "
                      f"{bytes.fromhex(my_merges[i][1]).decode('utf-8', 'replace')!r} = {my_merges[i]}")
    print(f"\ntokens: {n - bad}/{n} identical, merges: {nm - bad_m}/{nm} identical in rank order")
    return 0 if (bad == 0 and bad_m == 0 and (extra == 0 or extra_ok)) else 1


if __name__ == "__main__":
    sys.exit(main())
