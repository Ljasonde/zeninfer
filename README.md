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
| M2 | Forward pass: RMSNorm / RoPE / GQA / SwiGLU, checked against numpy | **done** |
| M3 | KV cache + incremental decode + sampling | **done** |
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
zeninfer logits     <model.gguf> --text-file prompt.txt [--topk 8] [--out logits.bin] [--greedy N]
zeninfer generate   <model.gguf> --text-file prompt.txt --max-new 128 [--temp 0.7] [--top-k 40] [--top-p 0.9] [--no-cache]
```

`--json` on `inspect` emits a schema that `tools/dump_gguf.py --json` also produces, so the two
independent implementations can be diffed tensor by tensor.

> **Windows note.** PowerShell hands native arguments over in the OEM code page, so `--text "你好"`
> arrives as mojibake bytes. The binary detects this and warns. Put the text in a UTF-8 file and use
> `--text-file` instead.

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

## M2: what is verified

The forward pass is checked against two implementations that were written independently of it:
`tools/ref_forward.py` (NumPy, float64, separate gguf reader and separate dequantization) and
HuggingFace `transformers`' own `Qwen2ForCausalLM` running the same weights converted to a
checkpoint by `tools/gguf_to_hf.py`.

```sh
make check-forward PYTHON=/path/to/python                 # vs NumPy, tolerance 1e-4
make check-hf PYTHON=/path/to/python                      # + vs HuggingFace (needs torch)
```

```
case           tokens   max|diff|         argmax  result       vs hf  hf
single-token        1   3.433e-05     271/271      PASS   1.304e-04  PASS
chat-turn           9   8.106e-06    9707/9707      PASS   3.061e-05  PASS
longer             26   4.292e-06    82707/82707    PASS   1.901e-05  PASS
```

The tight comparison is against the float64 NumPy reference; the HuggingFace column is looser
because torch accumulates in float32, which is worth about 1e-4 on a logit of magnitude ~13 by
itself. Both bounds are far below any real architecture error (a wrong RoPE convention or a
mis-wired GQA head moves logits by whole units, not by 1e-4).

What the tests do *not* share with the engine: gguf parsing, dequantization, tensor layout
handling, RoPE convention, attention masking, GQA head mapping, or the order of the norms.

A qualitative check that needs nothing but the binary:

```
$ zeninfer logits m.gguf --text-file hello.txt --greedy 10
text : Hello! How can I assist you today?
```

and for `用一句话解释什么是 KV cache。` the first greedy token is `KV` (p=0.52). Beam-of-one
coherence is a weak signal in general, but it rules out the failure mode where every tensor is
slightly wrong and the model emits fluent-looking noise.

### Baseline performance (before any optimization)

Measured on the Ryzen 7 8845H, single thread, scalar code, weights pre-dequantized to f32:

| Metric | Value |
| --- | --- |
| load (dequantize 640 MiB → 2.4 GiB f32) | 1.23 s |
| prefill | ~312 ms/token ≈ 3.2 tok/s |
| decode (no cache: full re-prefill per token) | ~4.4 s/token at S≈9–18 |

This is the "before" column of the M4 ablation table: the current `matvec` is a plain double-accumulating
loop over `float` rows, with no blocking, no AVX2 intrinsics and no thread pool.

## M3: what is verified

The KV cache is an optimisation, never a change of semantics. `make check-kv` states that two ways:

```sh
make check-kv PYTHON=/path/to/python       # needs numpy
```

```
case           tokens    logits   max|diff|   result
short               1    151936   0.000e+00   PASS
sentence           13   1975168   0.000e+00   PASS
chinese             7   1063552   0.000e+00   PASS

greedy decisions, cached vs recomputed: ids match
```

The comparison is `logits --all` (one full prefill) against `logits --all --cache` (the same
positions evaluated one at a time through the cache). It is **bit-identical**, not merely close,
because both paths accumulate in the same order — attention reads positions `[0, pos]` in the same
sequence either way, so the cache removes redundant work without reordering a single addition.

`make check-generate` asks a different question, using HuggingFace as the third opinion:

```
single-token   prompt=  1 tok  zen= 32  hf= 32  match=yes
chat-turn      prompt= 15 tok  zen= 32  hf= 32  match=yes
longer         prompt= 13 tok  zen= 32  hf= 32  match=yes
RESULT: PASS (greedy argmax agrees token by token)
```

32 greedy tokens per prompt, identical one by one. A numerical difference that never moves an
argmax is harmless; this test is what proves it stays harmless.

### What the cache is worth

Single thread, scalar code, `q8_0` weights dequantized to f32:

| Context | With cache | Full recompute per token | Speedup |
| --- | --- | --- | --- |
| 6 → 14 tokens | ~306 ms/token | ~2902 ms/token | **9.5×** |
| 26 → 42 tokens | ~310 ms/token | ~10736 ms/token | **34.6×** |

Both columns are the same kernel doing the same math; the only difference is whether the keys and
values of past positions are recomputed. That is why the speedup is a function of context length:
without a cache every token costs O(n) and a generation of n tokens costs O(n²), while with a cache
each step is a fixed cost of one token against n cached positions. The cached column is flat
(~306–310 ms) exactly as predicted; the recompute column doubles when the context roughly doubles.

The cost is memory: `n_layer × n_ctx × n_kv_heads × head_dim × 4 bytes × 2` — at the full 32k
context that is 768 MiB, so `generate` allocates only what the sequence needs (94 positions ≈ 2.2 MiB).

An end-to-end example that needs nothing but the binary — `用一句话解释什么是 KV cache。` as a chat
turn (26 prompt tokens) decodes 17 tokens and stops at `<|im_end|>`:

```
KV cache是一种在计算机系统中用于存储键值对的缓存技术。
```

`--no-cache` produces the same prefix, one token at a time and about 35× slower, which is the
point of the test above.

### Sampling

`--temp 0` (the default) is greedy. Otherwise: repetition penalty over the last
`--repeat-last-n` (64) tokens → temperature → top-k shortlist (256 when `--top-k` is unset) →
top-p (nucleus) → inverse-CDF draw from a xorshift64* stream seeded by `--seed`, so a run is
reproducible. Top-p is applied to that shortlist rather than the full 151936-way distribution;
this is a documented approximation, not an accident.

## Repository layout

```
src/gguf.{h,cpp}        GGUF v2/v3 reader, mmap, typed metadata access, dequantization
src/tokenizer.{h,cpp}   byte-level BPE, Qwen2 pre-tokenizer, special tokens
src/unicode.{h,cpp}     UTF-8 decode/encode, Unicode 15.0.0 class tables
src/unicode_tables.h    generated by tools/gen_unicode_tables.py (do not edit)
src/model.{h,cpp}       f32 weight loading + Qwen2 forward (RMSNorm/RoPE/GQA/SwiGLU) + KV cache
src/sample.{h,cpp}      sampling: repetition penalty, temperature, top-k, top-p, seeded RNG
src/main.cpp            CLI
tools/dump_gguf.py      independent GGUF reader (reference implementation)
tools/ref_forward.py    independent NumPy forward pass (float64)
tools/check_tokenizer.py  parity harness against HuggingFace tokenizers
tools/check_vocab.py    vocabulary/merge diff against tokenizer.json
tools/check_forward.py  logit parity across three prompts
tools/check_logits.py   diff two raw f32 logit dumps
tools/check_kv.py       cache invariants: bit-exact logits, unchanged greedy decisions
tools/check_generate.py greedy ids vs transformers, token by token
tools/hf_generate.py    transformers generate() on explicit ids
tools/gguf_to_hf.py     gguf -> HuggingFace checkpoint (for ground truth)
tools/hf_logits.py      logits straight out of transformers
tools/gen_unicode_tables.py
```

## Roadmap

M4 AVX2 GEMM and threading, M5 quantization and its perplexity cost, M6 benchmark harness and an
OpenAI-compatible endpoint. Numbers get published as they are measured; nothing here is an estimate
dressed as a result.

## License

MIT — see [LICENSE](LICENSE).
