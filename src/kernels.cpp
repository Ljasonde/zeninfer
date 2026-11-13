#include "kernels.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace zen {

namespace {
Impl g_impl = Impl::Scalar;
std::unique_ptr<ThreadPool> g_pool;
}  // namespace

const char* impl_name(Impl impl) {
    switch (impl) {
        case Impl::Scalar: return "scalar";
        case Impl::Avx2: return "avx2";
    }
    return "?";
}

bool cpu_has_avx2() {
#if defined(__AVX2__)
    return true;   // -march=native / -mavx2: the whole binary assumes it
#else
    return false;
#endif
}

void set_impl(Impl impl) {
    g_impl = (impl == Impl::Avx2 && !cpu_has_avx2()) ? Impl::Scalar : impl;
}

Impl impl() { return g_impl; }

// ------------------------------------------------------------------ dot

float dot_f32_scalar(const float* a, const float* b, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(acc);
}

float dot_f32_avx2(const float* a, const float* b, int64_t n) {
#if defined(__AVX2__)
    // Four independent accumulators hide FMA latency; each held ~n/32 terms, so the
    // float32 partial sums stay short and the final reduction happens in double.
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    const __m256 s = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    const __m128 lo = _mm256_castps256_ps128(s);
    const __m128 hi = _mm256_extractf128_ps(s, 1);
    const __m128 t = _mm_add_ps(lo, hi);
    const __m128 t2 = _mm_add_ps(t, _mm_movehl_ps(t, t));
    const __m128 t3 = _mm_add_ss(t2, _mm_shuffle_ps(t2, t2, 1));
    double total = static_cast<double>(_mm_cvtss_f32(t3));
    for (; i < n; ++i) total += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(total);
#else
    return dot_f32_scalar(a, b, n);
#endif
}

// --------------------------------------------------------------- matvec

void matvec_f32_impl(float* y, const float* x, const float* w, int64_t n_out, int64_t n_in, Impl use) {
    if (use == Impl::Avx2) {
        for (int64_t o = 0; o < n_out; ++o) y[o] = dot_f32_avx2(w + o * n_in, x, n_in);
        return;
    }
    for (int64_t o = 0; o < n_out; ++o) y[o] = dot_f32_scalar(w + o * n_in, x, n_in);
}

void matvec_f32(float* y, const float* x, const float* w, int64_t n_out, int64_t n_in) {
    ThreadPool* p = g_pool.get();
    if (!p || p->size() <= 1 || n_out < p->size() * 2) {
        matvec_f32_impl(y, x, w, n_out, n_in, g_impl);
        return;
    }
    const Impl use = g_impl;
    p->run(n_out, [&](int64_t begin, int64_t end) {
        matvec_f32_impl(y + begin, x, w + begin * n_in, end - begin, n_in, use);
    });
}

// ------------------------------------------------------------------ pool

struct ThreadPool::State {
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    uint64_t epoch = 0;
    int remaining = 0;
    int parts = 1;
    int64_t n = 0;
    const std::function<void(int64_t, int64_t)>* fn = nullptr;
    bool stop = false;
};

ThreadPool::ThreadPool(int n_threads) : s_(new State()), parts_(std::max(1, n_threads)) {
    s_->parts = parts_;
    s_->workers.reserve(static_cast<size_t>(parts_ - 1));
    for (int i = 1; i < parts_; ++i) {
        s_->workers.emplace_back([this, i] {
            uint64_t seen = 0;
            for (;;) {
                const std::function<void(int64_t, int64_t)>* fn = nullptr;
                int64_t n = 0;
                int part = 0, parts = 1;
                {
                    std::unique_lock<std::mutex> lk(s_->m);
                    s_->work_cv.wait(lk, [&] { return s_->stop || s_->epoch != seen; });
                    if (s_->stop) return;
                    seen = s_->epoch;
                    fn = s_->fn;
                    n = s_->n;
                    parts = s_->parts;
                    part = i;
                }
                const int64_t b = n * part / parts;
                const int64_t e = n * (part + 1) / parts;
                (*fn)(b, e);
                {
                    std::lock_guard<std::mutex> lk(s_->m);
                    if (--s_->remaining == 0) s_->done_cv.notify_one();
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(s_->m);
        s_->stop = true;
    }
    s_->work_cv.notify_all();
    for (auto& t : s_->workers) t.join();
    delete s_;
}

void ThreadPool::run(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (parts_ <= 1 || n <= 1) {
        fn(0, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(s_->m);
        s_->fn = &fn;
        s_->n = n;
        s_->parts = parts_;
        s_->remaining = parts_ - 1;
        ++s_->epoch;
    }
    s_->work_cv.notify_all();
    fn(0, n / parts_);
    std::unique_lock<std::mutex> lk(s_->m);
    s_->done_cv.wait(lk, [&] { return s_->remaining == 0; });
}

void set_threads(int n_threads) {
    const int want = std::max(1, n_threads);
    if (g_pool && g_pool->size() == want) return;
    g_pool.reset(want > 1 ? new ThreadPool(want) : nullptr);
}

int threads() { return g_pool ? g_pool->size() : 1; }

ThreadPool* pool() { return g_pool.get(); }

}  // namespace zen
