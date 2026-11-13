// zeninfer - sample.cpp : token sampling.
#include "sample.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace zen {

namespace {
constexpr int kTopPShortlist = 256;  // see sample.h
}

uint64_t Rng::next() {
    s_ ^= s_ >> 12;
    s_ ^= s_ << 25;
    s_ ^= s_ >> 27;
    return s_ * 2685821657736338717ull;
}

double Rng::uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

int32_t argmax_token(const float* logits, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

int32_t sample_token(const float* logits, int n, const SamplingParams& p,
                     const std::vector<int32_t>& history, Rng& rng) {
    if (n <= 0) return 0;

    std::vector<float> scratch(static_cast<size_t>(n),
                               std::numeric_limits<float>::lowest());
    std::copy(logits, logits + n, scratch.begin());

    // ---- repetition penalty over the tail of the history
    if (p.repeat_penalty != 1.0f && !history.empty()) {
        const size_t keep = static_cast<size_t>(std::max(0, p.repeat_last_n));
        const size_t begin = history.size() > keep ? history.size() - keep : 0;
        for (size_t i = begin; i < history.size(); ++i) {
            const int32_t id = history[i];
            if (id < 0 || id >= n) continue;
            float& v = scratch[static_cast<size_t>(id)];
            v = (v > 0.0f) ? v / p.repeat_penalty : v * p.repeat_penalty;
        }
    }

    if (p.temp <= 0.0f) return argmax_token(scratch.data(), n);

    const float inv_temp = 1.0f / p.temp;
    float max_l = scratch[0];
    for (int i = 1; i < n; ++i) max_l = std::max(max_l, scratch[static_cast<size_t>(i)]);

    // ---- top-k shortlist (nth_element: O(n) instead of a full sort)
    int k = (p.top_k > 0) ? std::min(p.top_k, n) : std::min(kTopPShortlist, n);
    std::vector<int32_t> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    if (k < n) {
        std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(),
                         [&](int32_t a, int32_t b) { return scratch[a] > scratch[b]; });
        idx.resize(static_cast<size_t>(k));
    }
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return scratch[a] > scratch[b]; });

    // ---- softmax over the shortlist
    std::vector<double> prob(idx.size());
    double sum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        const double e = std::exp((static_cast<double>(scratch[idx[i]]) - max_l) * inv_temp);
        prob[i] = e;
        sum += e;
    }
    for (double& v : prob) v /= sum;

    // ---- nucleus: keep the shortest prefix whose mass reaches top_p
    size_t m = prob.size();
    if (p.top_p < 1.0f) {
        double acc = 0.0;
        m = 0;
        while (m < prob.size() && acc < static_cast<double>(p.top_p)) {
            acc += prob[m];
            ++m;
        }
        if (m == 0) m = 1;
        double rescale = 0.0;
        for (size_t i = 0; i < m; ++i) rescale += prob[i];
        for (size_t i = 0; i < m; ++i) prob[i] /= rescale;
    }

    // ---- inverse-CDF draw
    const double r = rng.uniform();
    double acc = 0.0;
    for (size_t i = 0; i < m; ++i) {
        acc += prob[i];
        if (r < acc) return idx[i];
    }
    return idx[m - 1];
}

}  // namespace zen
