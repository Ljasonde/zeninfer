// zeninfer - sample.h : token sampling (greedy / temperature / top-k / top-p).
#pragma once

#include <cstdint>
#include <vector>

namespace zen {

struct SamplingParams {
    float temp = 0.0f;             // <= 0 means greedy
    int top_k = 0;                 // 0 = disabled
    float top_p = 1.0f;            // 1.0 = disabled (nucleus)
    float repeat_penalty = 1.0f;   // 1.0 = disabled
    int repeat_last_n = 64;
};

// xorshift64*, seeded by the caller so a sampled run can be reproduced exactly.
class Rng {
  public:
    explicit Rng(uint64_t seed) : s_(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next();
    double uniform();  // [0, 1)

  private:
    uint64_t s_;
};

int32_t argmax_token(const float* logits, int n);

// Samples one token id from `logits` (not modified). `history` is the tokens
// generated so far, used only for the repetition penalty.
//
// top-p is applied over the top-K shortlist rather than the full 151936-wide
// distribution: K defaults to 256 when --top-k is unset, which is far wider than
// any nucleus the model actually uses, and keeps decode time out of the sort.
int32_t sample_token(const float* logits, int n, const SamplingParams& p,
                     const std::vector<int32_t>& history, Rng& rng);

}  // namespace zen
