# M4 验收报告：AVX2 内核与多线程

本轮目标：把 M2/M3 的标量参考实现换成真正跑得动的内核，并把「快了多少、为什么只快这么多」
量化出来。**结论是加速 7.4×，但更重要的是知道了加速的上限由内存带宽而不是 FMA 吞吐决定。**

## 1. 交付

| 文件 | 内容 |
| --- | --- |
| `src/kernels.h` / `src/kernels.cpp` | `dot_f32_avx2`（4 路 `__m256` 累加器 + `_mm256_fmadd_ps`）、`dot_f32_scalar`、`matvec_f32` 分派、`ThreadPool`（常驻 worker + epoch 唤醒） |
| `src/model.cpp` | `matvec()` 改为单行转调 `zen::matvec_f32`；标量版本保留为参考路径，由 `--impl scalar` 选中 |
| `src/main.cpp` | `--threads N` / `--impl scalar\|avx2`；环境变量 `ZEN_THREADS` / `ZEN_IMPL` 设默认值；默认 `cpu_has_avx2() ? Avx2 : Scalar` |
| `tools/ablation.py` | 逐配置跑 `generate --json`，打印 prefill/decode 成本并输出 markdown 表 |

批大小为 1，所以这一层的算子本质是矩阵-向量乘（matvec）而不是 GEMM：每个权重每 token 只被
读一次、只参与一次乘加。这一点决定了后面的所有结论。

## 2. 正确性：快路径还是同一个模型

| 检查 | 容差 | 结果 |
| --- | --- | --- |
| AVX2 vs 标量，同一 prompt 全 151936 logits | — | `max 1.544e-05` / `mean 2.362e-06` / argmax 同为 576 |
| `make check-hf`：vs NumPy float64 | 1e-4 | 5.007e-05 / 1.454e-05 / 8.821e-06（三条 prompt） |
| `make check-hf`：vs HuggingFace | 5e-4 | 1.638e-04 / 2.825e-05 / 2.170e-05 |
| `make check-kv`：cache 位等价 | 0 | 三条全 `0.000e+00`；贪心 ids 与重算路径一致 |

误差比 M2 的纯标量路径略大（3.4e-5 → 5.0e-5），这是向量化改变累加顺序的必然结果，仍远小于
容差，且 argmax 与 top-5 未变。**cache 位等价在这个版本依然成立**，因为 cached 与 recompute
走的是同一个内核、同样的行切分顺序；一旦哪天不成立，这条测试会立刻炸出来。

## 3. 消融（17 token prompt，生成 16 token，Ryzen 7 8845H / DDR5-5600）

| 配置 | prefill ms/token | decode ms/token | decode tok/s | 相对标量 |
| --- | ---: | ---: | ---: | ---: |
| scalar, 1 线程 | 240.4 | 315.2 | 3.17 | 1.00× |
| avx2, 1 线程 | 40.9 | 55.4 | 18.06 | 5.69× |
| avx2, 2 线程 | 32.3 | 46.3 | 21.60 | 6.81× |
| avx2, 4 线程 | 28.8 | **42.4** | **23.58** | **7.43×** |
| avx2, 8 线程 | 31.6 | 47.6 | 21.02 | 6.62× |
| avx2, 16 线程 | 47.4 | 65.7 | 15.22 | 4.80× |

复现：`python tools/ablation.py --exe ./build/zeninfer --model models/qwen2.5-0.5b-instruct-q8_0.gguf --text-file prompt.txt`

AVX2 单独给 5.7×（向量宽度 8，但标量版本是 double 累加 + 无 FMA，且受限于单核取指），线程再给
1.3× 到 4 线程为止，之后倒退。

## 4. 归因：为什么 4 线程是终点

每 token 要读 2403.90 MiB = **2.3476 GiB** 的 f32 权重，批大小 1 时这些数据没有任何复用。

- 1 线程：`2.3476 GiB / 0.0554 s = 42.4 GiB/s ≈ 45.5 GB/s`
- 4 线程：`2.3476 GiB / 0.0424 s = 55.4 GiB/s ≈ 59.4 GB/s`

这台机器是双通道 DDR5-5600，理论带宽 89.6 GB/s，实测可用上限通常六到七成。**4 线程时已经贴到
这个上限**，所以第 5–8 个线程没有新带宽可分，只能互相抢；8/16 线程的倒退还叠加了超线程共享
L1/L2 与执行单元的争用。

由此得到本轮真正的结论：**内核从计算瓶颈变成了带宽瓶颈**。继续优化 FMA 数量没有意义，杠杆变成
「每 token 少读字节」。直接读现有 q8_0 文件（0.625 GiB/token）在同样 59 GB/s 下的天花板是
≈11 ms/token ≈ **88 tok/s**，比 f32 路径的理论极限高约 4×。这个数就是 M5 要去逼近的目标。

## 5. 为什么没做「GEMM 分块」

计划里写的是 GEMM + cache 分块。实测后确认 batch=1 时不存在可复用的块：A 是 1×n 的向量，W 的
每一行都只被用一次，分块只会增加循环开销。真正需要分块的是 batch>1（多请求批处理）或
prefill 阶段把 S 个 token 一起乘——那是 M6 之后如果做连续批处理才需要考虑的事。这里如实记录，
不把「没做」包装成「做了」。

## 6. 坑

1. `main.cpp` 用了 `zen::set_threads` 却忘了 `#include "kernels.h"`，报
   `'set_threads' is not a member of 'zen'`。错误信息指向调用点，很容易误以为是命名空间问题。
2. 长依赖链：如果只用一个 `__m256` 累加器做 `acc = fmadd(acc, x, w)`，每次迭代都依赖上一次的
   FMA 延迟（Zen4 上 4–5 cycle），吞吐会掉一半以上。用 4 个独立累加器、最后树形归约，才能喂满
   两条 FMA 端口。
3. 线程池必须是 RAII：worker 线程在 `~ThreadPool` 里置 `stop_`、`notify_all`、`join`。否则进程
   退出时静态 pool 先析构而 worker 还在跑，Windows 上表现为偶发挂起。
4. 按行切块而不是按元素交错切块，避免两个 worker 写同一 cache line 造成伪共享。

## 7. 已知限制

- 权重仍是 f32 常驻（2403.90 MiB），加载后内存占用与 M2 相同；这是 M5 要改的。
- 只有 matvec 有 SIMD；RMSNorm / SiLU / RoPE / softmax 还是标量。它们合计占比不到 1%，优化它们
  不会改变 tokens/s（已由上面的带宽计算证明）。
- 线程数固定为「物理核」最划算，尚未按 NUMA/异构核做自适应；当前机器同构，暂无必要。
- 采样仍是单 token 串行，没有连续批处理。

## 8. 下一步（M5）

把 `matvec` 换成直接吃量化块的版本：Q8_0 用 `_mm256_maddubs_epi16` + `_mm256_madd_epi16`
（或 AVX-512 VNNI 的 `_mm256_dpbusd`），Q4_0 用 nibble 解包后同样的整数点积。同时测量化代价：
同 prompt 下 f32 / q8_0 / q4_0 的 logits 偏差，以及一小段文本上的困惑度上升百分比。预期
decode 从 42 ms/token（f32）降到接近 11–14 ms/token（q8_0），即 70–90 tok/s 区间。
