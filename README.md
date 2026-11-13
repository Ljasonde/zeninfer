# zeninfer

A CPU-only LLM inference engine for
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), written from scratch in
C++17 — no BLAS, no ggml, no ML framework, no dependencies beyond a compiler.

The point is not to beat `llama.cpp`. It is to explain, line by line, where its speed comes from —
and to *measure* the gap instead of guessing at it.

## Numbers

Measured on a Ryzen 7 8845H (8C/16T, DDR5-5600), q8_0 weights, `examples/prompt.txt`, 16 new tokens:

| | |
| --- | --- |
| decode | **20 tok/s** (48 ms/token, 4 threads) — **6.2×** the scalar baseline |
| prefill | 34 ms/token (29 tok/s) |
| tokenizer | **1.77M tok/s** — 130,473 tokens from 0.58 MiB of source in 73.7 ms |
| logits vs float64 NumPy | max abs diff **5.0e-05** (bound 1e-4) |
| KV cache vs full recompute | **bit-identical**: 3.2M logits, max diff exactly `0` |
| source | 3.4k lines of C++, one 2.6 MB binary |

## Quick start

```sh
git clone https://github.com/Ljasonde/zeninfer && cd zeninfer
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q8_0.gguf \
  https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
make
./build/zeninfer generate models/qwen2.5-0.5b-instruct-q8_0.gguf --text-file examples/prompt.txt
```

```
weights          : 638.74 MiB on disk -> 2403.90 MiB f32 (load 1466.8 ms)
kv cache         : 87 pos x 24 layers x 128 kv-dim = 2.04 MiB
prompt           : 15 tokens
prefill          : 571.1 ms (38.08 ms/token, 26.26 tok/s)
decode           : 28 tokens in 1403.8 ms (50.14 ms/token, 19.95 tok/s)
stop             : eos
text             : KV cache是一种用于存储键值对的缓存系统，通常使用键（key）和值（value）来存储数据。
```

Requirements: a C++17 compiler and `make`. `make ARCH="-march=x86-64-v3"` builds a portable binary.
Scope: this one model, batch size 1, CPU only.

### CLI

```sh
zeninfer inspect    <model.gguf> [--json]                 # metadata, hyper-parameters, tensors
zeninfer tokenize   <model.gguf> --text-file prompt.txt   # ids, pieces, round-trip check
zeninfer spans      <model.gguf> --text-file p.txt        # pre-tokenizer spans
zeninfer vocab-dump <model.gguf> out.vocab                # flat, diffable vocabulary
zeninfer logits     <model.gguf> --text-file p.txt [--topk 8] [--out logits.bin] [--greedy N]
zeninfer generate   <model.gguf> --text-file p.txt --max-new 128 [--temp 0.7] [--top-k 40]
zeninfer bench-tokenizer <model.gguf> <corpus.txt> [--repeat N]
```

`logits` and `generate` accept `--threads N` and `--impl scalar|avx2`; `ZEN_THREADS` and `ZEN_IMPL`
set the same defaults.

> **Windows:** PowerShell passes native arguments in the OEM code page, so `--text "你好"` arrives as
> mojibake. The binary warns when it detects that; use `--text-file` with UTF-8 text instead.

## Verify it yourself

Every check compares against an implementation that shares no code with the engine. They exit
non-zero on any mismatch.

| Command | Proves | Needs |
| --- | --- | --- |
| `make check` | GGUF structure, vocabulary, and tokenizer parity | `pip install tokenizers` |
| `make check-forward` | every logit of 3 prompts vs float64 NumPy (1e-4) | `numpy` |
| `make check-hf` | the same vs HuggingFace `transformers` (5e-4) | `torch`, `transformers` |
| `make check-kv` | cache == recompute, bit-exact; greedy ids unchanged | `numpy` |
| `make check-generate` | 32 greedy tokens match HF, token by token | `torch`, `transformers` |
| `make bench` | the ablation table below | — |

```
$ make check PYTHON=/path/to/python
OK: 291 tensors identical, 669763072 bytes, arch=qwen2
tokens: 151643/151643 identical, merges: 151387/151387 identical in rank order
1979/1979 cases match (100.00%)
```

Full tables, plus the dead ends and the two format traps that cost the most time, are in the
milestone reports: [M1](docs/M1-report.md) front end, [M2](docs/M2-report.md) forward pass,
[M3](docs/M3-report.md) KV cache, [M4](docs/M4-report.md) kernels (Chinese; the tables and numbers
are language-independent).

## How it works

```
bytes ──▶ BPE tokenizer ──▶ token embeddings
                                │
                    24 × transformer block
                    ├─ RMSNorm ─▶ GQA attention (RoPE, KV cache)
                    └─ RMSNorm ─▶ SwiGLU FFN
                                │
                         RMSNorm ─▶ lm_head ─▶ logits ──▶ sampler
```

- **Weights.** GGUF v2/v3 read through `mmap`, zero copy, rows dequantized once into f32. All 291
  tensors are cross-checked against an independent pure-Python parser: dtype, dims, byte size,
  offset, plus an out-of-bounds/overlap check on the payload.
- **Tokenizer.** A hand-written pre-tokenizer scanner (the Qwen2 regex, which `std::regex` cannot
  express) feeding a byte-level BPE. All 151,643 tokens and 151,387 merges are compared by value and
  rank against `tokenizer.json`.
- **Forward.** RMSNorm, NEOX RoPE, causal grouped-query attention (14 query / 2 KV heads), SwiGLU
  and the residual chain, all written out, with double accumulation inside each dot product.
- **KV cache.** Append-only, and bit-identical to a full recompute because both paths sum in the
  same order — the test asserts `0.0`, not "close".
- **Kernels.** AVX2+FMA dot product with four accumulators, plus a thread pool over output rows.
  At batch 1 the engine is one matrix-vector product per token, so it is bandwidth bound: a token
  streams 2.35 GiB of f32 weights, which at 48 ms is ~49 GiB/s — within 20% of what this laptop's
  DDR5 delivers. That is why 2, 4 and 8 threads land within 5% of each other and 16 threads lose,
  and why quantization, not vectorization, is the next lever.

## Layout

```
src/gguf.{h,cpp}         GGUF v2/v3 reader, mmap, metadata, dequantization
src/tokenizer.{h,cpp}    byte-level BPE + Qwen2 pre-tokenizer
src/unicode.{h,cpp}      UTF-8 encode/decode, Unicode 15.0.0 class tables
src/model.{h,cpp}        weights, forward pass, KV cache
src/kernels.{h,cpp}      AVX2+FMA dot product, thread pool, scalar fallback
src/sample.{h,cpp}       repetition penalty, temperature, top-k, top-p
src/main.cpp             CLI

tools/dump_gguf.py, ref_forward.py   independent reader and float64 forward pass (ground truth)
tools/check_*.py                     the parity harnesses behind `make check*`
tools/gguf_to_hf.py, hf_*.py         GGUF -> HuggingFace checkpoint; HF-side logits/generate
tools/ablation.py                    prints the table under Performance
tools/gen_unicode_tables.py          regenerates src/unicode_tables.h

docs/M1..M4-report.md    what each milestone proves, in detail
examples/prompt.txt      the prompt every number here was measured on
```

## Performance

Ablation, 16 new tokens from `examples/prompt.txt` — the output of `make bench`:

| config | prefill ms/token | decode ms/token | decode tok/s | vs scalar |
| --- | ---: | ---: | ---: | ---: |
| scalar, 1 thread | 230.8 | 307.8 | 3.25 | 1.00× |
| avx2, 1 thread | 43.7 | 58.9 | 16.99 | 5.23× |
| avx2, 2 threads | 36.3 | 49.2 | 20.32 | 6.26× |
| avx2, 4 threads | 34.3 | 50.1 | 19.98 | 6.15× |
| avx2, 8 threads | 36.7 | 51.3 | 19.51 | 6.01× |
| avx2, 16 threads | 50.1 | 65.0 | 15.37 | 4.73× |

Run-to-run spread is a few percent; the shape is the result, not the third digit.

## Known limitations

- f32 weights in RAM (2.4 GiB for this model); quantized matvec is M5, not done yet.
- Batch size 1. No continuous batching, no paged attention, one request at a time.
- No NFC normalization, like `llama.cpp` and unlike HuggingFace: non-composed input tokenizes
  differently. Counted separately in `check_tokenizer.py` rather than hidden.
- Top-p samples from a 256-token shortlist, not the full 151,936-way distribution. Documented, not
  accidental.
- AVX2 only; no AVX-512 or NEON path.

## Roadmap

M5 — Q8_0/Q4_0 matvec that reads the quantized blocks directly, plus the perplexity cost of each
format. M6 — benchmark harness and an OpenAI-compatible endpoint so `agentctl` can use this as its
local backend. Numbers get published when measured; nothing here is an estimate dressed as a result.

## License

MIT — see [LICENSE](LICENSE).
