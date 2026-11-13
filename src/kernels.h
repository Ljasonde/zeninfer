// Kernels and the thread pool behind them.
//
// The scalar path is the reference: it accumulates in double and is what the
// NumPy cross-check was written against. The AVX2 path trades a little accuracy
// for bandwidth-shaped speed; how much is measured by check_forward.py, not
// assumed here.
#pragma once

#include <cstdint>
#include <functional>

namespace zen {

enum class Impl { Scalar, Avx2 };

const char* impl_name(Impl impl);
bool cpu_has_avx2();

// Process-wide selection, set once from the CLI (and from ZEN_IMPL / ZEN_THREADS).
void set_impl(Impl impl);
Impl impl();

// y[o] = sum_i x[i] * w[o * n_in + i]
void matvec_f32(float* y, const float* x, const float* w, int64_t n_out, int64_t n_in);
void matvec_f32_impl(float* y, const float* x, const float* w, int64_t n_out, int64_t n_in, Impl impl);

float dot_f32_scalar(const float* a, const float* b, int64_t n);
float dot_f32_avx2(const float* a, const float* b, int64_t n);

// Splits [0, n) into contiguous chunks, one per thread, and joins.
// Contiguous ranges keep each thread's writes on its own cache lines.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return parts_; }
    void run(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

private:
    struct State;
    State* s_;
    int parts_;
};

// Process-wide pool used by matvec_f32 (0/1 threads => none).
void set_threads(int n_threads);
int threads();
ThreadPool* pool();

}  // namespace zen
