# M2 验收报告 — 前向推理 + 三方对拍

日期：2026-11-12　项目：`D:\zeninfer`　模型：Qwen2.5-0.5B-Instruct（q8_0，24 层 / 896 维 / 14 Q head / 2 KV head / 4864 FFN）

## 1. 交付内容

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 权重加载 | `src/model.{h,cpp}` | gguf → f32 常驻（F32 直接 memcpy，其余走已有反量化）；按 `[n_out][n_in]` 行主序存，每行一个输出神经元 |
| 算子 | 同上 | `rmsnorm`（double 累加）、`matvec`（double 累加）、`rope_neox`（rotate-half，base 1e6）、`silu_inplace` |
| 前向 | 同上 | embedding → 24 × (RMSNorm→QKV→RoPE→因果 GQA attention→O→残差 → RMSNorm→SwiGLU→残差) → final norm → lm_head |
| CLI | `src/main.cpp` | `logits <gguf> [--text/--text-file/--ids] [--topk N] [--out F.bin] [--all] [--greedy N] [--text-out F]` |
| 参考实现 | `tools/ref_forward.py` | 独立 NumPy 前向（float64，自己读 gguf、自己反量化，不复用 C++ 任何逻辑） |
| 三方对拍 | `tools/check_forward.py`、`tools/check_logits.py` | 三条 prompt × 全词表逐 logit 比对；后者只负责 diff 两个 raw f32 dump |
| HF 真值 | `tools/gguf_to_hf.py`、`tools/hf_logits.py` | gguf → HF checkpoint（291 张量改名 + config.json），交给 `transformers` 的 `Qwen2ForCausalLM` 跑 |

## 2. 验收证据

```
$ make check-forward PYTHON=...        # vs NumPy float64，容差 1e-4
$ make check-hf PYTHON=...             # 再加一列 vs HuggingFace transformers

case           tokens   max|diff|         argmax  result       vs hf  hf
single-token        1   3.433e-05     271/271      PASS   1.304e-04  PASS
chat-turn           9   8.106e-06    9707/9707      PASS   3.061e-05  PASS
longer             26   4.292e-06    82707/82707    PASS   1.901e-05  PASS

RESULT                                            PASS
```

三条用例覆盖三种 regime：S=1（只有投影与 norm，最容易被 layout 错误打中）、S=9（一个完整 chat turn，考因果掩码与 GQA head 映射）、S=26（长一点，考 RoPE 位置与掩码边界）。

对拍的是**全部 151936 个 logit**，不是 top-k；`check_logits.py` 还会断言 argmax 与前 5 名顺序一致。

**为什么 HF 那列容差更松**：HF 用 float32 kernel，光是累加顺序就值 ~1e-4（logit 量级 13，896 项点积）。真正紧的界是 float64 的 NumPy 参考。两边都远低于任何"架构级错误"的量级 —— RoPE 约定写错、GQA 头接错，logits 会整段偏移，不会只差 1e-4。

**两个实现不共享的东西**：gguf 解析、反量化、张量 layout 处理、RoPE 约定、attention 掩码、GQA head 映射、两个 norm 的先后顺序。

## 3. 定性证据（不依赖任何对照实现）

```
$ zeninfer logits m.gguf --text-file hello.txt --greedy 10
text : Hello! How can I assist you today?
```

```
$ zeninfer logits m.gguf --ids <26 个 token> --topk 3
   1. id  82707  logit  21.2057  p 0.521326  KV
   2. id     42  logit  20.9710  p 0.412245  K
   3. id  18493  logit  17.4142  p 0.011762  在
```

问「用一句话解释什么是 KV cache」，第一个 token 是 `KV`（p=0.52）。贪心连贯性本身是很弱的证据，但它能排除「每个张量都错一点点、模型输出看起来像话」这种失败模式。

## 4. 性能基线（优化前）

Ryzen 7 8845H，单线程，标量代码，权重已预先反量化为 f32：

| 指标 | 实测 |
| --- | --- |
| 加载（640 MiB q8_0 → 2403.9 MiB f32） | 1231.9 ms |
| prefill | ~312 ms/token ≈ 3.2 tok/s（S=9 时 324 ms/token，S=26 时 312 ms/token） |
| decode（无 cache，每 token 整段重算） | ~4.4 s/token（S≈9–18） |

这就是 M4 消融表的「before」列：`matvec` 是朴素的 double 累加行循环，没有分块、没有 AVX2 intrinsic、没有线程池。decode 那一行是 O(n) 而非 O(1)，因为还没有 KV cache —— M3 会把它变成真正的一次前向一个 token。

## 5. 三处值得记下的坑

1. **RoPE 必须是 NEOX / rotate-half 约定。** Qwen2 把每个 head 的 64 维拆成前后两半旋转（pair 是 `(i, i+32)`），不是 GPT-J 的相邻元素交织 `(2i, 2i+1)`。用错不会崩，只会让输出变成语法还行的胡话 —— 这类错误只能靠逐 logit 对拍抓住。
2. **ggml 的 dims 顺序正好等于「一行一个输出」。** `dims = [ne0, ne1]` 且 `ne0` 变化最快，所以 `[896, 4864]` 展平后就是 `4864 × 896` 的行主序矩阵，正好是 `y[o] = Σ_i x[i]·W[o][i]` 想要的布局，不需要转置。attn_q/attn_output 都是 896×896 的方阵，写错了也照样跑通 —— 对拍是唯一保险。
3. **HF 只作为"第二意见"，不作为唯一真值。** 它自己也有 float32 舍入（见上），所以容差分了 `--tol`（对 NumPy）和 `--tol-hf`（对 HF）。

## 6. 已知限制（如实记录）

- 权重常驻 f32，2.4 GiB；q8_0 的原生 8-bit 路径还没接进前向（M5 做量化时会同时解决内存与带宽）。
- 无 KV cache，decode 是整段重算（M3）。
- 无采样：只有贪心；`--greedy` 只为验证数学，不是为了好用（M3）。
- 单线程，标量（M4）。
- batch = 1，不支持 padding/mask 的通用批推理（不在计划内）。

## 7. M3 计划

1. per-layer KV cache（`n_ctx` 大小的 k/v 缓冲区，按层分配），forward 改成 `prefill(ids)` + `decode(id, pos)` 两段。
2. 采样：temperature / top-k / top-p / repetition penalty，`zeninfer run` 子命令 + `--seed`。
3. **新增不变量测试**：`decode` 逐 token 走 cache 得到的 logits，必须与「每次整段 prefill」的 logits 一致（容差同上）—— 这是 cache 写错唯一能立刻暴露的地方。
4. 验收：连续生成 64 token 与 HF `generate` 在贪心下的 token 序列完全一致。
