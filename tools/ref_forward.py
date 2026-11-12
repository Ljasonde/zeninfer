#!/usr/bin/env python3
"""Independent NumPy reference forward pass for Qwen2 gguf weights.

Deliberately shares no code with the C++ engine: separate gguf reader (dump_gguf),
separate dequantization, separate forward math (float64 throughout). If the two
implementations agree to ~1e-5 on every logit, the C++ math is very unlikely to be
wrong in a way that a single implementation could hide.

usage:
    python tools/ref_forward.py <model.gguf> --ids 9707,11 --out ref.bin [--all]
"""
import argparse
import math
import mmap
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_gguf  # noqa: E402


# ---------------------------------------------------------------- weight loading
BLOCK = {"Q8_0": (32, 34), "Q4_0": (32, 18), "F16": (1, 2), "F32": (1, 4)}


def dequant(raw, dtype, nelems):
    """Return a float64 array of `nelems` values."""
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=nelems).astype(np.float64)
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=nelems).astype(np.float64)
    if dtype == "Q8_0":
        block_t = np.dtype([("d", "<f2"), ("q", "i1", 32)])
        blocks = np.frombuffer(raw, dtype=block_t, count=nelems // 32)
        out = blocks["d"].astype(np.float64)[:, None] * blocks["q"].astype(np.float64)
        return out.reshape(-1)
    if dtype == "Q4_0":
        block_t = np.dtype([("d", "<f2"), ("q", "u1", 16)])
        blocks = np.frombuffer(raw, dtype=block_t, count=nelems // 32)
        q = blocks["q"].astype(np.int32)
        lo = (q & 0x0F) - 8          # elements 0..15
        hi = ((q >> 4) & 0x0F) - 8   # elements 16..31
        vals = np.concatenate([lo, hi], axis=1).astype(np.float64)
        out = blocks["d"].astype(np.float64)[:, None] * vals
        return out.reshape(-1)
    raise SystemExit(f"ref_forward: unsupported dtype {dtype}")


class Weights:
    def __init__(self, path):
        self.info = dump_gguf.parse(path)
        self.f = open(path, "rb")
        self.buf = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.base = self.info["tensor_data_offset"]
        self.index = {t["name"]: t for t in self.info["tensors"]}
        self.hyper = self.info["hyper"]

    def raw(self, name):
        t = self.index[name]
        start = self.base + t["offset"]
        return self.buf[start:start + t["nbytes"]]

    def vec(self, name):
        t = self.index[name]
        return dequant(self.raw(name), t["dtype"], t["nelems"])

    def mat(self, name):
        """Returns W with shape (n_out, n_in) -- ggml dims are [ne0=n_in, ne1=n_out]."""
        t = self.index[name]
        flat = dequant(self.raw(name), t["dtype"], t["nelems"])
        ne0, ne1 = t["dims"][0], t["dims"][1]
        return flat.reshape(ne1, ne0)

    def get(self, key, default):
        return self.hyper.get(key, default)


# ---------------------------------------------------------------- ops
def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def rope_neox(x, n_head, head_dim, pos, base):
    """x: (n_head, head_dim) in place-ish; NEOX rotate-half convention."""
    half = head_dim // 2
    i = np.arange(half)
    theta = pos * np.power(base, -2.0 * i / head_dim)
    c, s = np.cos(theta), np.sin(theta)
    x = x.copy()
    x0 = x[:, :half].copy()
    x1 = x[:, half:].copy()
    x[:, :half] = x0 * c - x1 * s
    x[:, half:] = x0 * s + x1 * c
    return x


def silu(x):
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- forward
def forward(w, ids, all_positions=False):
    E = int(w.get("qwen2.embedding_length", 896))
    L = int(w.get("qwen2.block_count", 24))
    NH = int(w.get("qwen2.attention.head_count", 14))
    NKV = int(w.get("qwen2.attention.head_count_kv", 2))
    FF = int(w.get("qwen2.feed_forward_length", 4864))
    EPS = float(w.get("qwen2.attention.layer_norm_rms_epsilon", 1e-6))
    BASE = float(w.get("qwen2.rope.freq_base", 1e4))

    D = E // NH
    G = NH // NKV
    kv_dim = NKV * D

    emb = w.mat("token_embd.weight")          # (n_vocab, E)
    out_w = w.mat("output.weight")            # (n_vocab, E)
    out_norm = w.vec("output_norm.weight")

    x = emb[np.asarray(ids)]                  # (S, E)
    S = x.shape[0]

    for li in range(L):
        b = f"blk.{li}."
        wq, wk, wv = w.mat(b + "attn_q.weight"), w.mat(b + "attn_k.weight"), w.mat(b + "attn_v.weight")
        wo = w.mat(b + "attn_output.weight")
        bq, bk, bv = w.vec(b + "attn_q.bias"), w.vec(b + "attn_k.bias"), w.vec(b + "attn_v.bias")
        an, fn = w.vec(b + "attn_norm.weight"), w.vec(b + "ffn_norm.weight")
        wg, wu, wd = w.mat(b + "ffn_gate.weight"), w.mat(b + "ffn_up.weight"), w.mat(b + "ffn_down.weight")

        xn = np.stack([rmsnorm(x[t], an, EPS) for t in range(S)])
        q = xn @ wq.T + bq
        k = xn @ wk.T + bk
        v = xn @ wv.T + bv

        q = q.reshape(S, NH, D)
        k = k.reshape(S, NKV, D)
        v = v.reshape(S, NKV, D)
        q = np.stack([rope_neox(q[t], NH, D, t, BASE) for t in range(S)])
        k = np.stack([rope_neox(k[t], NKV, D, t, BASE) for t in range(S)])

        # (NH, S, D) views with kv-head broadcast by repeating each kv head G times
        qh = q.transpose(1, 0, 2)                       # (NH, S, D)
        kh = np.repeat(k.transpose(1, 0, 2), G, axis=0)  # (NH, S, D)
        vh = np.repeat(v.transpose(1, 0, 2), G, axis=0)  # (NH, S, D)

        attn = np.zeros((NH, S, D))
        scale = 1.0 / math.sqrt(D)
        for h in range(NH):
            scores = qh[h] @ kh[h].T * scale            # (S, S)
            mask = np.triu(np.ones((S, S), dtype=bool), k=1)
            scores = np.where(mask, -np.inf, scores)
            scores = scores - scores.max(axis=1, keepdims=True)
            e = np.exp(scores)
            p = e / e.sum(axis=1, keepdims=True)
            attn[h] = p @ vh[h]
        attn = attn.transpose(1, 0, 2).reshape(S, E)

        x = x + attn @ wo.T

        xn = np.stack([rmsnorm(x[t], fn, EPS) for t in range(S)])
        x = x + (silu(xn @ wg.T) * (xn @ wu.T)) @ wd.T

    xn = np.stack([rmsnorm(x[t], out_norm, EPS) for t in range(S)])
    logits = xn @ out_w.T                               # (S, n_vocab)
    return logits if all_positions else logits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ids", default="9707")
    ap.add_argument("--out", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    ids = [int(t) for t in args.ids.split(",") if t.strip() != ""]
    w = Weights(args.model)
    logits = forward(w, ids, args.all)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)

    top = np.argsort(-logits.astype(np.float64))[:5]
    print("ids            :", ids)
    print("shape          :", logits.shape)
    print("min/max        : %.6f / %.6f" % (logits.min(), logits.max()))
    print("top-5          :", [(int(i), float(logits[i])) for i in top])
    if args.out:
        logits.tofile(args.out)
        print("wrote", args.out, logits.size * 4, "bytes")


if __name__ == "__main__":
    main()
