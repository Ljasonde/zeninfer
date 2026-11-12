// zeninfer - model.h : Qwen2 forward pass on dequantized f32 weights.
//
// Layout note: ggml stores a matrix with dims = [ne0, ne1] where ne0 varies
// fastest, so a weight of shape [n_in, n_out] is already row-major [n_out][n_in]
// once flattened -- exactly the layout a row-per-output matvec wants.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"

namespace zen {

struct Mat {
    int64_t n_in = 0;    // columns
    int64_t n_out = 0;   // rows
    std::vector<float> w;

    const float* row(int64_t r) const { return w.data() + r * n_in; }
};

struct Vec {
    std::vector<float> v;

    const float* data() const { return v.data(); }
    float* data() { return v.data(); }
    int64_t size() const { return static_cast<int64_t>(v.size()); }
};

struct Layer {
    Mat wq, wk, wv, wo;        // attention projections
    Mat w_gate, w_up, w_down;  // SwiGLU FFN
    Vec bq, bk, bv;            // Qwen2 keeps Q/K/V biases
    Vec attn_norm, ffn_norm;
};

struct Config {
    int n_layer = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_ctx = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000.0f;
    int head_dim = 0;   // n_embd / n_head
    int n_vocab = 0;

    int n_group() const { return n_head / n_head_kv; }  // query heads per kv head
};

class Model {
  public:
    // Reads and dequantizes every weight into f32. `quantized` is reported for
    // the record: how much the file saved compared with these f32 buffers.
    explicit Model(const GGUFFile& f);

    const Config& cfg() const { return cfg_; }
    const std::string& arch() const { return arch_; }
    // Bytes of f32 weights held in RAM after dequantization.
    uint64_t resident_bytes() const { return resident_bytes_; }
    uint64_t source_bytes() const { return source_bytes_; }

    // Full-sequence prefill (no KV cache yet): returns logits of the LAST token,
    // n_vocab floats.
    std::vector<float> forward(const std::vector<int32_t>& tokens);
    // Logits for every position, row-major [n_tokens][n_vocab].
    std::vector<float> forward_all(const std::vector<int32_t>& tokens);

  private:
    void load(const GGUFFile& f);
    Vec read_vec(const GGUFFile& f, const std::string& name);
    Mat read_mat(const GGUFFile& f, const std::string& name, int64_t n_in, int64_t n_out);

    Config cfg_;
    std::string arch_;
    Mat token_embd_;  // [n_vocab][n_embd]
    Mat output_;      // [n_vocab][n_embd]
    Vec output_norm_;
    std::vector<Layer> layers_;
    uint64_t resident_bytes_ = 0;
    uint64_t source_bytes_ = 0;
};

// ---------------------------------------------------------------- ops (f32/f64 mix)
// RMSNorm: y = x / rms(x) * w. Accumulated in double; written back as f32.
void rmsnorm(float* out, const float* x, const float* w, int64_t n, float eps);

// y = W x with double accumulation (row-per-output, W[n_out][n_in]).
void matvec(float* y, const float* x, const Mat& m);

// Rotary position embedding, NEOX/"rotate-half" convention used by Qwen2:
// pairs (i, i + head_dim/2) inside each head, angle = pos * base^(-2i/head_dim).
void rope_neox(float* x, int64_t n_head, int64_t head_dim, int64_t pos, float base);

// SiLU in place: x * sigmoid(x)
void silu_inplace(float* x, int64_t n);

}  // namespace zen
