# zeninfer

A CPU-only LLM inference engine for [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
written from scratch in C++17 with no BLAS, no ggml, and no ML framework.

The goal is not to beat `llama.cpp`. The goal is to be able to explain, line by line, where its
speed comes from — and to measure the gap instead of guessing it.

```
$ zeninfer bench-tokenizer models/qwen2.5-0.5b-instruct-q8_0.gguf corpus.txt
input            : corpus.txt (0.58 MiB)
tokens           : 130473 (4.674 bytes/token)
best             : 73.68 ms = 7.9 MB/s = 1770878 tok/s
```

## Status

| Milestone | What | State |
| --- | --- | --- |
| M1 | GGUF parser (mmap) + Qwen BPE tokenizer | **done** |
| M2 | Forward pass: RMSNorm / RoPE / GQA / SwiGLU, checked against numpy | next |
| M3 | KV cache + incremental decode + sampling | |
| M4 | AVX2 GEMM, threading, cache blocking | |
| M5 | Q8_0 / Q4_0 quantization, perplexity cost | |
| M6 | Benchmark harness + ablation table + OpenAI-compatible server | |

## Build

```sh
make                      # -> build/zeninfer (g++/clang++, -march=native)
make ARCH="-march=x86-64-v3"   # portable binary
```

Requirements: a C++17 compiler. No dependencies, no submodules, no CMake.

## Usage

```sh
zeninfer inspect    <model.gguf> [--json]        # metadata, hyper-parameters, tensor table
zeninfer tokenize   <model.gguf> --text "hello"  # ids, pieces, round-trip check
zeninfer spans      <model.gguf> --text "hi  there"   # pre-tokenizer spans
zeninfer vocab-dump <model.gguf> <out.vocab>     # flat, diffable vocabulary
zeninfer bench-tokenizer <model.gguf> <file> [--repeat N]
```

`--json` on `inspect` emits a schema that `tools/dump_gguf.py --json` also produces, so the two
independent implementations can be diffed tensor by tensor.

## M1: what is verified

Both halves of the front end are checked against independent implementations rather than against
themselves.

**GGUF parser** — `tools/dump_gguf.py` is a standalone pure-Python GGUF reader (no gguf, no
llama.cpp). Diffing its output against the C++ reader on `qwen2.5-0.5b-instruct-q8_0.gguf`:

```
OK: 291 tensors identical, 669763072 bytes, arch=qwen2
    tokens=151936 merges=151387 alignment=32
```

That covers dtype, dims, byte size and file offset for every tensor, plus the alignment-derived
`tensor_data_offset` and a payload out-of-bounds/overlap check.

**Tokenizer** — 75 hand-written edge cases plus 2000 fuzz strings (CJK, emoji, combining marks,
NBSP, ideographic space, U+2028, `<|im_start|>`, …) are compared against the HuggingFace
`tokenizers` library:

```
1979/1979 cases match (100.00%)
fuzz cases: 2000, hand-written: 75, skipped (need NFC): 43
```

And the vocabulary itself, token by token and merge by merge:

```
tokens: 151643/151643 identical, merges: 151387/151387 identical in rank order
```

### Two things that are easy to get wrong here

*The merged vocabulary is byte-level, not raw bytes.* llama.cpp keeps GPT-2's byte-to-codepoint
alphabet, so a space is stored as `Ġ` (U+0120) and a newline as `Ċ` (U+010A). A first version that
treated the tokens as raw bytes found no token for `" "` at all. `ByteLevelTable` in
`src/tokenizer.cpp` implements the forward and inverse mapping (68 bytes are remapped to U+0100+).

*The pre-tokenizer regex cannot be handed to `std::regex`.* ECMAScript `\s` is not the same set as
Python's, and `\s+(?!\S)` needs a lookahead that must agree with `\s*[\r\n]+` about which part of a
whitespace run belongs to which alternative. `pretokenize()` is a hand-written scanner instead: for a
run of whitespace it takes the position after the *last* newline, and falls back to giving back one
character so the next span can start with a space.

### Known limitation: no NFC

The reference pipeline normalizes to NFC before splitting; zeninfer, like llama.cpp, does not.
Input that is not already NFC-composed (e.g. `a` + U+0301) therefore tokenizes differently.
This is deliberate and counted separately in `tools/check_tokenizer.py` rather than hidden.

## Verifying it yourself

```sh
make
make check PYTHON=/path/to/python   # needs: pip install tokenizers
```

`make check` runs the three comparisons below and exits non-zero on any mismatch. Nothing it
reports is produced by the code under test.

| Check | Compared against | Expected |
| --- | --- | --- |
| GGUF structure | `tools/dump_gguf.py`, an independent pure-Python reader | `291 tensors identical` |
| vocabulary + merges | `tokenizer.json` from the HuggingFace repo | `151643/151643`, `151387/151387` |
| `encode()` | HuggingFace `tokenizers`, 2000 fuzz + 75 hand-written cases | `1979/1979` |

Three spot checks that need no Python at all:

* **291 is not a magic number.** 24 layers × 12 tensors (`attn_norm`, `attn_{q,k,v,o}`, `ffn_{norm,gate,up,down}`, with Q/K/V also having a bias) + `token_embd` + `output_norm` + `output` = 291.
* **File size arithmetic.** `tensor_data_offset (5947744) + sum of tensor bytes (669763072)` must equal the file size on disk exactly: `675710816` bytes.
* **Round trip.** `zeninfer tokenize --text "…" --pieces` prints `round trip : exact`, i.e. `decode(encode(x)) == x`, for every input it accepts.

## Repository layout

```
src/gguf.{h,cpp}        GGUF v2/v3 reader, mmap, typed metadata access, dequantization
src/tokenizer.{h,cpp}   byte-level BPE, Qwen2 pre-tokenizer, special tokens
src/unicode.{h,cpp}     UTF-8 decode/encode, Unicode 15.0.0 class tables
src/unicode_tables.h    generated by tools/gen_unicode_tables.py (do not edit)
src/main.cpp            CLI
tools/dump_gguf.py      independent GGUF reader (reference implementation)
tools/check_tokenizer.py  parity harness against HuggingFace tokenizers
tools/check_vocab.py    vocabulary/merge diff against tokenizer.json
tools/gen_unicode_tables.py
```

## Roadmap

M2 forward pass with a numpy cross-check at 1e-4, M3 KV cache, M4 AVX2 GEMM and threading,
M5 quantization and its perplexity cost, M6 benchmark harness and an OpenAI-compatible endpoint.
Numbers get published as they are measured; nothing here is an estimate dressed as a result.
