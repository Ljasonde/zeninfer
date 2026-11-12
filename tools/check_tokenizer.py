"""Cross-check zeninfer's BPE tokenizer against HuggingFace `tokenizers`.

The C++ side reads a corpus of one escaped case per line (\\n, \\r, \\t, \\xHH,
\\uXXXX) and prints {"ids": [...]} per line. This script builds the same cases,
asks the reference tokenizer for its ids, and diffs the two.

    python tools/check_tokenizer.py --tokenizer models/tokenizer.json \
        --exe build/zeninfer.exe --gguf models/qwen2.5-0.5b-instruct-q8_0.gguf

Exit code 0 only when every case matches.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
import unicodedata

from tokenizers import Tokenizer

# Hand-written edge cases: one per line of the pre-tokenizer regex, plus the
# mismatches that a naive implementation usually gets wrong.
CASES = [
    # plain text and contractions
    "hello world",
    "Hello, World!",
    "it's a test",
    "IT'S A TEST",
    "I'd've done it",
    "we'll see, they're here",
    "don't't't",
    "'s",
    "'",
    "'ll",
    "x'",
    # digits split per character
    "2024",
    "3.14159",
    "a1b2c3",
    "1 000 000",
    "版本2.5",
    # whitespace edge cases
    "hello  world",
    "hello   world",
    " hello",
    "  hello",
    "hello ",
    "hello  ",
    "\n",
    "\n\n",
    "\r\n",
    "a\n\nb",
    "a\r\n\r\nb",
    "a \n b",
    "a\n  \nb",
    "\t\ttab",
    "a\tb",
    " \n \n ",
    "line1\nline2\n",
    "para1\n\npara2",
    "\u00a0nbsp\u00a0",
    "\u3000ideographic space",
    "\u2028line sep",
    "",
    " ",
    "   ",
    # punctuation runs
    "!!!",
    "!!!\r\n",
    "...",
    "][]{}()",
    "a...b",
    "--->",
    "#@$%^&*",
    "e.g. i.e.",
    # CJK / mixed scripts / emoji
    "你好世界",
    "你好，世界！",
    "中文English混合2024年",
    "日本語テスト",
    "한국어 테스트",
    "café naïve résumé",
    "Übermensch",
    "Ελληνικά",
    "Привет мир",
    "🙂🙂 emoji 😀 test",
    "a🙂b",
    "\u200bzwsp\u200b",
    "combining a\u0301 e\u0301",
    # chat template fragments
    "<|im_start|>user\nhi<|im_end|>\n",
    "<|endoftext|>",
    "<|im_start|>",
    # code
    "def f(x):\n    return x + 1\n",
    "for (int i = 0; i < n; ++i) {}",
    "SELECT * FROM t WHERE a = 'b';",
    '{"key": "value", "n": 1}',
    "https://example.com/path?q=1&r=2",
    "日本語のコード: int x = 1;",
    # long runs
    "a" * 200,
    " " * 50 + "x",
    "1" * 100,
    "." * 100,
    "文字" * 100,
]

ALPHABET = (
    list("abcXYZ")
    + list("0123456789")
    + list(" \t\r\n")
    + list("'.,!?-_[]{}()<>|/\\`~@#$%^&*+=:")
    + list("\u00a0\u3000\u2028\u200b")
    + list("你好世界日本語한국어")
    + list("éüñøå")
    + ["🙂", "😀", "👍🏽", "\u0301"]
    + ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
)


def escape(text):
    """Inverse of the C++ unescape(): keep the corpus line one physical line."""
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def make_cases(n_fuzz, seed):
    cases = list(CASES)
    rng = random.Random(seed)
    for _ in range(n_fuzz):
        length = rng.randint(0, 40)
        cases.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    # The corpus format has no way to express an empty line (it would be
    # indistinguishable from a blank separator), and "" tokenizes to nothing.
    return [c for c in cases if c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--gguf", default="")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--fuzz", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--show", type=int, default=5, help="how many mismatches to print")
    args = ap.parse_args()

    ref = Tokenizer.from_file(args.tokenizer)
    cases = make_cases(args.fuzz, args.seed)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", newline="\n") as fh:
        corpus_path = fh.name
        for case in cases:
            fh.write(escape(case) + "\n")

    cmd = [args.exe, "tokenize"]
    if args.vocab:
        cmd += ["--vocab", args.vocab]
    else:
        cmd += [args.gguf]
    cmd += ["--batch", corpus_path]

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return 2

    got = [json.loads(line)["ids"] for line in proc.stdout.splitlines() if line.strip()]
    if len(got) != len(cases):
        print(f"line count mismatch: expected {len(cases)}, got {len(got)}")
        return 2

    bad = 0
    skipped_nfc = 0
    for idx, (text, ids) in enumerate(zip(cases, got)):
        # The reference applies NFC before splitting; zeninfer deliberately does
        # not (same as llama.cpp), so decomposed input is a known divergence.
        if unicodedata.normalize("NFC", text) != text:
            skipped_nfc += 1
            continue
        want = ref.encode(text, add_special_tokens=False).ids
        if want != ids:
            bad += 1
            if bad <= args.show:
                print(f"--- case {idx} ---")
                print(f"text     : {text!r}")
                print(f"expected : {want}")
                print(f"got      : {ids}")
                exp_tokens = ref.encode(text, add_special_tokens=False).tokens
                print(f"ref toks : {exp_tokens}")

    total = len(cases) - skipped_nfc
    print(f"\n{total - bad}/{total} cases match ({100.0 * (total - bad) / max(total, 1):.2f}%)")
    print(f"fuzz cases: {args.fuzz}, hand-written: {len(CASES)}, skipped (need NFC): {skipped_nfc}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
