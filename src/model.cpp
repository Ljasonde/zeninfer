// zeninfer - model.cpp : weight loading + Qwen2 forward pass.
#include "model.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace zen {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

std::vector<float> dequant(const TensorInfo& t) {
    const int64_t n = t.nelems();
    std::vector<float> out(static_cast<size_t>(n));
    if (t.type == DType::F32) {
        std::memcpy(out.data(), t.data, static_cast<size_t>(n) * sizeof(float));
    } else {
        dequantize(t.type, t.data, out.data(), n);
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------- ops

void rmsnorm(float* out, const float* x, const float* w, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
    const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + eps);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * scale * static_cast<double>(w[i]));
    }
}

void matvec(float* y, const float* x, const Mat& m) {
    for (int64_t o = 0; o < m.n_out; ++o) {
        const float* row = m.row(o);
        double acc = 0.0;
        for (int64_t i = 0; i < m.n_in; ++i) acc += static_cast<double>(row[i]) * x[i];
        y[o] = static_cast<float>(acc);
    }
}

void rope_neox(float* x, int64_t n_head, int64_t head_dim, int64_t pos, float base) {
    const int64_t half = head_dim / 2;
    for (int64_t h = 0; h < n_head; ++h) {
        float* head = x + h * head_dim;
        for (int64_t i = 0; i < half; ++i) {
            // theta = pos * base^(-2i/head_dim)
            const double freq = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(pos) * freq;
            const double c = std::cos(angle), s = std::sin(angle);
            const double x0 = head[i], x1 = head[i + half];
            head[i] = static_cast<float>(x0 * c - x1 * s);
            head[i + half] = static_cast<float>(x0 * s + x1 * c);
        }
    }
}

void silu_inplace(float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const double v = x[i];
        x[i] = static_cast<float>(v / (1.0 + std::exp(-v)));
    }
}

// ---------------------------------------------------------------- loading

Model::Model(const GGUFFile& f) { load(f); }

Vec Model::read_vec(const GGUFFile& f, const std::string& name) {
    const TensorInfo& t = f.tensor(name);
    source_bytes_ += t.nbytes;
    Vec out;
    out.v = dequant(t);
    resident_bytes_ += out.v.size() * sizeof(float);
    return out;
}

Mat Model::read_mat(const GGUFFile& f, const std::string& name, int64_t n_in, int64_t n_out) {
    const TensorInfo& t = f.tensor(name);
    if (static_cast<int64_t>(t.nelems()) != n_in * n_out) {
        fail("tensor " + name + " has " + std::to_string(t.nelems()) + " elements, expected " +
             std::to_string(n_in * n_out));
    }
    source_bytes_ += t.nbytes;
    Mat m;
    m.n_in = n_in;
    m.n_out = n_out;
    m.w = dequant(t);
    resident_bytes_ += m.w.size() * sizeof(float);
    return m;
}

void Model::load(const GGUFFile& f) {
    arch_ = f.get_str("general.architecture", "qwen2");
    const std::string p = arch_ + ".";

    cfg_.n_layer = static_cast<int>(f.get_u32(p + "block_count", 0));
    cfg_.n_embd = static_cast<int>(f.get_u32(p + "embedding_length", 0));
    cfg_.n_ff = static_cast<int>(f.get_u32(p + "feed_forward_length", 0));
    cfg_.n_head = static_cast<int>(f.get_u32(p + "attention.head_count", 0));
    cfg_.n_head_kv = static_cast<int>(f.get_u32(p + "attention.head_count_kv", cfg_.n_head));
    cfg_.n_ctx = static_cast<int>(f.get_u32(p + "context_length", 0));
    cfg_.rms_eps = f.get_f32(p + "attention.layer_norm_rms_epsilon", 1e-6f);
    cfg_.rope_base = f.get_f32(p + "rope.freq_base", 10000.0f);
    if (cfg_.n_layer <= 0 || cfg_.n_embd <= 0 || cfg_.n_head <= 0 || cfg_.n_head_kv <= 0) {
        fail("model: missing qwen2 hyper parameters");
    }
    if (cfg_.n_embd % cfg_.n_head != 0) fail("model: n_embd is not divisible by n_head");
    if (cfg_.n_head % cfg_.n_head_kv != 0) fail("model: n_head is not divisible by n_head_kv");
    cfg_.head_dim = cfg_.n_embd / cfg_.n_head;

    const TensorInfo& emb = f.tensor("token_embd.weight");
    cfg_.n_vocab = static_cast<int>(emb.nelems() / cfg_.n_embd);

    token_embd_ = read_mat(f, "token_embd.weight", cfg_.n_embd, cfg_.n_vocab);
    output_norm_ = read_vec(f, "output_norm.weight");
    output_ = read_mat(f, "output.weight", cfg_.n_embd, cfg_.n_vocab);

    layers_.resize(static_cast<size_t>(cfg_.n_layer));
    for (int i = 0; i < cfg_.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        const int64_t kv_dim = static_cast<int64_t>(cfg_.n_head_kv) * cfg_.head_dim;
        Layer& l = layers_[static_cast<size_t>(i)];
        l.wq = read_mat(f, b + "attn_q.weight", cfg_.n_embd, cfg_.n_embd);
        l.wk = read_mat(f, b + "attn_k.weight", cfg_.n_embd, kv_dim);
        l.wv = read_mat(f, b + "attn_v.weight", cfg_.n_embd, kv_dim);
        l.wo = read_mat(f, b + "attn_output.weight", cfg_.n_embd, cfg_.n_embd);
        l.w_gate = read_mat(f, b + "ffn_gate.weight", cfg_.n_embd, cfg_.n_ff);
        l.w_up = read_mat(f, b + "ffn_up.weight", cfg_.n_embd, cfg_.n_ff);
        l.w_down = read_mat(f, b + "ffn_down.weight", cfg_.n_ff, cfg_.n_embd);
        l.bq = read_vec(f, b + "attn_q.bias");
        l.bk = read_vec(f, b + "attn_k.bias");
        l.bv = read_vec(f, b + "attn_v.bias");
        l.attn_norm = read_vec(f, b + "attn_norm.weight");
        l.ffn_norm = read_vec(f, b + "ffn_norm.weight");
    }
}

// ---------------------------------------------------------------- kv cache

void KVCache::reset(int layers, int ctx, int dim) {
    n_layer = layers;
    n_ctx = ctx;
    kv_dim = dim;
    len = 0;
    const size_t n = static_cast<size_t>(n_layer) * static_cast<size_t>(n_ctx) *
                     static_cast<size_t>(kv_dim);
    k.assign(n, 0.0f);
    v.assign(n, 0.0f);
}

float* KVCache::k_at(int layer, int64_t pos) {
    return k.data() + (static_cast<size_t>(layer) * static_cast<size_t>(n_ctx) +
                       static_cast<size_t>(pos)) * static_cast<size_t>(kv_dim);
}
float* KVCache::v_at(int layer, int64_t pos) {
    return v.data() + (static_cast<size_t>(layer) * static_cast<size_t>(n_ctx) +
                       static_cast<size_t>(pos)) * static_cast<size_t>(kv_dim);
}
const float* KVCache::k_at(int layer, int64_t pos) const {
    return k.data() + (static_cast<size_t>(layer) * static_cast<size_t>(n_ctx) +
                       static_cast<size_t>(pos)) * static_cast<size_t>(kv_dim);
}
const float* KVCache::v_at(int layer, int64_t pos) const {
    return v.data() + (static_cast<size_t>(layer) * static_cast<size_t>(n_ctx) +
                       static_cast<size_t>(pos)) * static_cast<size_t>(kv_dim);
}

// ---------------------------------------------------------------- forward

std::vector<float> Model::forward(const std::vector<int32_t>& tokens) {
    std::vector<float> all = forward_all(tokens);
    const size_t n_vocab = static_cast<size_t>(cfg_.n_vocab);
    return std::vector<float>(all.end() - static_cast<ptrdiff_t>(n_vocab), all.end());
}

std::vector<float> Model::forward_all(const std::vector<int32_t>& tokens) {
    const int64_t S = static_cast<int64_t>(tokens.size());
    if (S <= 0) fail("model: empty token sequence");
    const int64_t E = cfg_.n_embd, H = cfg_.n_head, KVH = cfg_.n_head_kv, D = cfg_.head_dim;
    const int64_t FF = cfg_.n_ff, V = cfg_.n_vocab, G = cfg_.n_group();
    const int64_t kv_dim = KVH * D;

    for (int32_t t : tokens) {
        if (t < 0 || t >= V) fail("model: token id out of range");
    }

    std::vector<float> x(static_cast<size_t>(S * E));
    for (int64_t t = 0; t < S; ++t) {
        const float* src = token_embd_.row(tokens[static_cast<size_t>(t)]);
        std::memcpy(x.data() + t * E, src, static_cast<size_t>(E) * sizeof(float));
    }

    std::vector<float> xn(static_cast<size_t>(E));
    std::vector<float> q(static_cast<size_t>(S * E)), k(static_cast<size_t>(S * kv_dim)),
        v(static_cast<size_t>(S * kv_dim));
    std::vector<float> attn(static_cast<size_t>(S * E));
    std::vector<float> proj(static_cast<size_t>(E));
    std::vector<float> gate(static_cast<size_t>(FF)), up(static_cast<size_t>(FF));
    std::vector<double> scores(static_cast<size_t>(S));

    for (int li = 0; li < cfg_.n_layer; ++li) {
        const Layer& l = layers_[static_cast<size_t>(li)];

        // ---- attention block
        for (int64_t t = 0; t < S; ++t) {
            const float* xt = x.data() + t * E;
            rmsnorm(xn.data(), xt, l.attn_norm.data(), E, cfg_.rms_eps);
            float* qt = q.data() + t * E;
            float* kt = k.data() + t * kv_dim;
            float* vt = v.data() + t * kv_dim;
            matvec(qt, xn.data(), l.wq);
            matvec(kt, xn.data(), l.wk);
            matvec(vt, xn.data(), l.wv);
            for (int64_t i = 0; i < E; ++i) qt[i] += l.bq.data()[i];
            for (int64_t i = 0; i < kv_dim; ++i) {
                kt[i] += l.bk.data()[i];
                vt[i] += l.bv.data()[i];
            }
            rope_neox(qt, H, D, t, cfg_.rope_base);
            rope_neox(kt, KVH, D, t, cfg_.rope_base);
        }

        const double scale = 1.0 / std::sqrt(static_cast<double>(D));
        std::fill(attn.begin(), attn.end(), 0.0f);
        for (int64_t h = 0; h < H; ++h) {
            const int64_t kvh = h / G;
            for (int64_t t = 0; t < S; ++t) {
                const float* qh = q.data() + t * E + h * D;
                double max_score = -1e30;
                for (int64_t s = 0; s <= t; ++s) {
                    const float* kh = k.data() + s * kv_dim + kvh * D;
                    double acc = 0.0;
                    for (int64_t d = 0; d < D; ++d) acc += static_cast<double>(qh[d]) * kh[d];
                    acc *= scale;
                    scores[static_cast<size_t>(s)] = acc;
                    if (acc > max_score) max_score = acc;
                }
                double sum = 0.0;
                for (int64_t s = 0; s <= t; ++s) {
                    const double e = std::exp(scores[static_cast<size_t>(s)] - max_score);
                    scores[static_cast<size_t>(s)] = e;
                    sum += e;
                }
                const double inv = 1.0 / sum;
                float* out = attn.data() + t * E + h * D;
                for (int64_t s = 0; s <= t; ++s) {
                    const double p = scores[static_cast<size_t>(s)] * inv;
                    const float* vh = v.data() + s * kv_dim + kvh * D;
                    for (int64_t d = 0; d < D; ++d) {
                        out[d] = static_cast<float>(static_cast<double>(out[d]) + p * vh[d]);
                    }
                }
            }
        }

        for (int64_t t = 0; t < S; ++t) {
            float* xt = x.data() + t * E;
            matvec(proj.data(), attn.data() + t * E, l.wo);
            for (int64_t i = 0; i < E; ++i) xt[i] = static_cast<float>(xt[i] + proj[static_cast<size_t>(i)]);
        }

        // ---- SwiGLU block
        for (int64_t t = 0; t < S; ++t) {
            float* xt = x.data() + t * E;
            rmsnorm(xn.data(), xt, l.ffn_norm.data(), E, cfg_.rms_eps);
            matvec(gate.data(), xn.data(), l.w_gate);
            matvec(up.data(), xn.data(), l.w_up);
            silu_inplace(gate.data(), FF);
            for (int64_t i = 0; i < FF; ++i) gate[static_cast<size_t>(i)] *= up[static_cast<size_t>(i)];
            matvec(proj.data(), gate.data(), l.w_down);
            for (int64_t i = 0; i < E; ++i) xt[i] = static_cast<float>(xt[i] + proj[static_cast<size_t>(i)]);
        }
    }

    std::vector<float> logits(static_cast<size_t>(S * V));
    std::vector<float> row(static_cast<size_t>(V));
    for (int64_t t = 0; t < S; ++t) {
        rmsnorm(xn.data(), x.data() + t * E, output_norm_.data(), E, cfg_.rms_eps);
        matvec(row.data(), xn.data(), output_);
        std::memcpy(logits.data() + t * V, row.data(), static_cast<size_t>(V) * sizeof(float));
    }
    return logits;
}

// ---------------------------------------------------------------- incremental

void Model::reserve_cache(int n_ctx) {
    if (n_ctx <= 0) fail("model: n_ctx must be positive");
    cache_.reset(cfg_.n_layer, n_ctx, cfg_.n_head_kv * cfg_.head_dim);
}

const std::vector<float>& Model::eval(const std::vector<int32_t>& tokens) {
    const int64_t S = static_cast<int64_t>(tokens.size());
    if (S <= 0) fail("model: empty token chunk");
    if (cache_.n_ctx <= 0) fail("model: call reserve_cache() before eval()");
    const int64_t P0 = cache_.len;
    if (P0 + S > cache_.n_ctx) fail("model: kv cache full (" + std::to_string(P0) + " + " +
                                    std::to_string(S) + " > " + std::to_string(cache_.n_ctx) + ")");

    const int64_t E = cfg_.n_embd, H = cfg_.n_head, KVH = cfg_.n_head_kv, D = cfg_.head_dim;
    const int64_t FF = cfg_.n_ff, V = cfg_.n_vocab, G = cfg_.n_group();
    const int64_t kv_dim = KVH * D;
    for (int32_t t : tokens) {
        if (t < 0 || t >= V) fail("model: token id out of range");
    }

    std::vector<float> x(static_cast<size_t>(S * E));
    for (int64_t t = 0; t < S; ++t) {
        const float* src = token_embd_.row(tokens[static_cast<size_t>(t)]);
        std::memcpy(x.data() + t * E, src, static_cast<size_t>(E) * sizeof(float));
    }

    std::vector<float> xn(static_cast<size_t>(E));
    std::vector<float> q(static_cast<size_t>(S * E));
    std::vector<float> kt(static_cast<size_t>(kv_dim)), vt(static_cast<size_t>(kv_dim));
    std::vector<float> attn(static_cast<size_t>(S * E));
    std::vector<float> proj(static_cast<size_t>(E));
    std::vector<float> gate(static_cast<size_t>(FF)), up(static_cast<size_t>(FF));
    std::vector<double> scores(static_cast<size_t>(P0 + S));

    for (int li = 0; li < cfg_.n_layer; ++li) {
        const Layer& l = layers_[static_cast<size_t>(li)];

        // ---- project this chunk of positions and append K/V to the cache
        for (int64_t t = 0; t < S; ++t) {
            const int64_t pos = P0 + t;
            rmsnorm(xn.data(), x.data() + t * E, l.attn_norm.data(), E, cfg_.rms_eps);
            float* qt = q.data() + t * E;
            matvec(qt, xn.data(), l.wq);
            matvec(kt.data(), xn.data(), l.wk);
            matvec(vt.data(), xn.data(), l.wv);
            for (int64_t i = 0; i < E; ++i) qt[i] += l.bq.data()[i];
            for (int64_t i = 0; i < kv_dim; ++i) {
                kt[static_cast<size_t>(i)] += l.bk.data()[i];
                vt[static_cast<size_t>(i)] += l.bv.data()[i];
            }
            rope_neox(qt, H, D, pos, cfg_.rope_base);
            rope_neox(kt.data(), KVH, D, pos, cfg_.rope_base);
            std::memcpy(cache_.k_at(li, pos), kt.data(), static_cast<size_t>(kv_dim) * sizeof(float));
            std::memcpy(cache_.v_at(li, pos), vt.data(), static_cast<size_t>(kv_dim) * sizeof(float));
        }

        // ---- attention: query at pos attends to every cached key <= pos
        const double scale = 1.0 / std::sqrt(static_cast<double>(D));
        std::fill(attn.begin(), attn.end(), 0.0f);
        for (int64_t h = 0; h < H; ++h) {
            const int64_t kvh = h / G;
            for (int64_t t = 0; t < S; ++t) {
                const int64_t pos = P0 + t;
                const int64_t n = pos + 1;
                const float* qh = q.data() + t * E + h * D;
                double max_score = -1e30;
                for (int64_t s = 0; s < n; ++s) {
                    const float* kh = cache_.k_at(li, s) + kvh * D;
                    double acc = 0.0;
                    for (int64_t d = 0; d < D; ++d) acc += static_cast<double>(qh[d]) * kh[d];
                    acc *= scale;
                    scores[static_cast<size_t>(s)] = acc;
                    if (acc > max_score) max_score = acc;
                }
                double sum = 0.0;
                for (int64_t s = 0; s < n; ++s) {
                    const double e = std::exp(scores[static_cast<size_t>(s)] - max_score);
                    scores[static_cast<size_t>(s)] = e;
                    sum += e;
                }
                const double inv = 1.0 / sum;
                float* out = attn.data() + t * E + h * D;
                for (int64_t s = 0; s < n; ++s) {
                    const double p = scores[static_cast<size_t>(s)] * inv;
                    const float* vh = cache_.v_at(li, s) + kvh * D;
                    for (int64_t d = 0; d < D; ++d) {
                        out[d] = static_cast<float>(static_cast<double>(out[d]) + p * vh[d]);
                    }
                }
            }
        }

        for (int64_t t = 0; t < S; ++t) {
            float* xt = x.data() + t * E;
            matvec(proj.data(), attn.data() + t * E, l.wo);
            for (int64_t i = 0; i < E; ++i) xt[i] = static_cast<float>(xt[i] + proj[static_cast<size_t>(i)]);
        }

        // ---- SwiGLU block
        for (int64_t t = 0; t < S; ++t) {
            float* xt = x.data() + t * E;
            rmsnorm(xn.data(), xt, l.ffn_norm.data(), E, cfg_.rms_eps);
            matvec(gate.data(), xn.data(), l.w_gate);
            matvec(up.data(), xn.data(), l.w_up);
            silu_inplace(gate.data(), FF);
            for (int64_t i = 0; i < FF; ++i) gate[static_cast<size_t>(i)] *= up[static_cast<size_t>(i)];
            matvec(proj.data(), gate.data(), l.w_down);
            for (int64_t i = 0; i < E; ++i) xt[i] = static_cast<float>(xt[i] + proj[static_cast<size_t>(i)]);
        }
    }

    rmsnorm(xn.data(), x.data() + (S - 1) * E, output_norm_.data(), E, cfg_.rms_eps);
    logits_.resize(static_cast<size_t>(V));
    matvec(logits_.data(), xn.data(), output_);
    cache_.len = P0 + S;
    return logits_;
}

}  // namespace zen
