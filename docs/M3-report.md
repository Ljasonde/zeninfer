# M3 验收报告 — KV cache + 增量解码 + 采样

一句话结论：KV cache 是**位等价**的加速，不是「差不多」的近似 —— 带 cache 逐 token 解码得到的
logits 与整段重算完全相同（max |diff| = 0.000e+00），贪心决策与 HuggingFace 逐 token 一致，
decode 在 26→42 token 上下文下快 **34.6×**。

## 1. 交付内容

| 文件 | 内容 |
| --- | --- |
| `src/model.h` / `src/model.cpp` | `struct KVCache`（`[layer][pos][kv_dim]` 连续布局、`reset/clear/k_at/v_at/bytes`）；`Model::reserve_cache/reset_cache/cache_len/cache_bytes/eval()` |
| `src/sample.h` / `src/sample.cpp` | `SamplingParams`、xorshift64* 的 `Rng`、`argmax_token()`、`sample_token()`（重复惩罚 → 温度 → top-k → top-p → 逆 CDF） |
| `src/main.cpp` | `generate` 子命令（`--max-new/--temp/--top-k/--top-p/--repeat-penalty/--repeat-last-n/--seed/--no-cache/--json`）；`logits --cache`；`tokenize --json` 单条文本也输出 JSON |
| `tools/check_kv.py` | 不变量测试：`--all` vs `--all --cache` 逐 logit 比对 + cached/no-cache 贪心一致性 + 两种模式的 ms/token |
| `tools/hf_generate.py` | 用 transformers 对同一批 ids 做 `generate(do_sample=False)`，输出 ids |
| `tools/check_generate.py` | 三条 prompt 各 32 个贪心 token，与 HF 逐 token 比对 |
| `Makefile` | `check-kv`、`check-generate`；`MODEL ?=` 变量（全部目标统一） |

`forward_all()` 作为**参考路径原样保留**：它的唯一职责就是「不带 cache 的正确结果」，任何优化都
必须与它对齐，而不是与另一个优化比对。

## 2. 验收证据

### 2.1 位等价：整段 prefill vs 逐 token 走 cache

```
$ make check-kv
case           tokens    logits   max|diff|   result
short               1    151936   0.000e+00   PASS
sentence           13   1975168   0.000e+00   PASS
chinese             7   1063552   0.000e+00   PASS
```

三条 prompt 一共 3,195,656 个 logit，误差是**精确的 0**，不是「小于 1e-6」。原因不是巧合：

- 两条路径的 attention 都按 `[0, pos]` 的升序累加到同一个 double 里，cache 只是把已经算过的
  key/value 取回来，没有改变任何一个加数的顺序；
- `eval()` 对单个位置计算 final norm + lm_head 的方式与 `forward_all()` 对最后一个位置的计算
  完全相同（同一份代码、同一个累加顺序）。

只要顺序一致，浮点加法就是可重复的 —— 这就是「cache 是纯优化」的严格含义。

### 2.2 决策一致：与 HuggingFace 逐 token 对齐

```
$ make check-generate
single-token   prompt=  1 tok  zen= 32  hf= 32  match=yes
chat-turn      prompt= 15 tok  zen= 32  hf= 32  match=yes
longer         prompt= 13 tok  zen= 32  hf= 32  match=yes
RESULT: PASS (greedy argmax agrees token by token)
```

这是与 2.1 不同的问题：2.1 证明「不改变数值」，2.2 证明「即使有 1e-4 级的数值差，也从不改变
argmax」。前者是性质，后者是使用中真正在意的东西。

### 2.3 定性证据（不依赖任何对照实现）

```
$ zeninfer generate m.gguf --text-file chat.txt --max-new 64
prompt tokens    : 26
kv cache         : 94 pos, 2.20 MiB
prefill          : 8064.3 ms (310.17 ms/token)
decode           : 17 tokens, 5271.2 ms (310.07 ms/token, 3.23 tok/s)
stop             : eos
text             : KV cache是一种在计算机系统中用于存储键值对的缓存技术。
```

prompt 是标准的 `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` 对话模板，模型在
`<|im_end|>` 处自己停下（`stop: eos`），而不是被 max-new 截断。

## 3. 性能：cache 值多少

单线程、标量代码、`q8_0` 权重全部反量化为 f32：

| 上下文 | 带 cache | 每 token 整段重算 | 加速比 |
| --- | --- | --- | --- |
| 6 → 14 token | ~306 ms/token | ~2902 ms/token | **9.5×** |
| 26 → 42 token | ~310 ms/token | ~10736 ms/token | **34.6×** |

两列是同一个 kernel：差别只在「过去位置的 K/V 是否重算」。所以加速比必然是上下文长度的函数：

- 无 cache：第 t 个 token 要算 t 个位置的 attention，n 个 token 总共 O(n²)；
- 有 cache：第 t 个 token 的代价是「1 个 query 对 t 个缓存位置」，每步几乎恒定。

实测的带 cache 一列基本是平的（306–310 ms），上下文翻倍时可重算一列也翻倍 —— 与预测一致。

**内存代价**：`n_layer × n_ctx × n_kv_heads × head_dim × 4 B × 2`。本模型 24 层 × 32768 × 2 × 64 × 4
× 2 = **768 MiB**（满上下文）。因此 `generate` 只按实际需要增长（94 个位置 ≈ 2.2 MiB），而不是
一次性按 `n_ctx` 分配。

**为什么还是 310 ms/token**：这个数字跟 M2 的 prefill 单价（312 ms/token）几乎相同，说明瓶颈
完全在 `matvec` —— 每次 decode 都要把 640 MiB 的权重读一遍（q8_0 → f32 后 2.4 GiB）。M4 的
AVX2 GEMM、分块与多线程针对的正是这一项，而不是 attention 本身。这也解释了为什么 cache 的
收益是「倍数」而不是「数量级」：它去掉的是重复计算，不是内存带宽。

## 4. 采样

`--temp 0`（默认）走 `argmax`，路径与贪心一致。否则：

1. 重复惩罚：对 history 末 `--repeat-last-n`（默认 64）个 token，logit 为正则除以 `p`、为负则乘以 `p`；
2. 除以温度；
3. `nth_element` 取 top-k 短名单（未给 `--top-k` 时用 256）；
4. 短名单上做 softmax，按概率降序取最短前缀使累计概率 ≥ `--top-p`，再归一化；
5. 逆 CDF 抽样，随机源是 `--seed` 播种的 xorshift64*，同 seed 可复现。

**已文档化的近似**：top-p 作用在 top-256 短名单上，不是完整 151936 分布。对 `top_p ≥ 0.9` 的
常规取值，短名单之外的总概率远小于被截断的部分，但严格说这是近似，写在 README 里。

## 5. 值得记下的坑

1. **`tokenize --json` 一直是摆设**。这个 flag 之前只对 `--batch` 生效，单条文本仍然打印人读
   格式，于是 `check_generate.py` 用 `json.loads` 直接炸在 `Expecting value: line 1 column 1`。
   修好之后 `check_forward.py` 反而坏了 —— 它当年是**按人读格式**写的正则解析（`line.startswith("ids")`），
   依赖的正是这个 bug。两个脚本必须同时改成 JSON 解析。
2. **单个 token 可能只是半个 UTF-8 字符**。byte-level 词表里 id 18137 的字节就是 `0xE9`，
   单独打印出来不是合法 UTF-8，Python 以 UTF-8 读子进程输出直接抛
   `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 694`。
   修法：`token_repr()` 遇到解不完整的序列就转义成 `\xe9` —— CLI 输出必须永远是合法 UTF-8，否则
   任何脚本都没法可靠调用它。
3. **Windows 给 `main()` 的是 GBK 版命令行**。非 ASCII 走 `--text` 时到引擎里已经是 GBK 字节，
   分词结果与真实文本无关（但两条路径自洽，所以位等价测试仍然「通过」，这是最危险的一种假通过）。
   全部脚本改走 `--text-file`，读 UTF-8 文件。顺带发现中文用例的 logits 数量从 2,582,912 掉到
   1,063,552 —— 之前那 26 个「token」其实是乱码切出来的。
4. **`--all --cache` 必须逐 token 走**。`eval()` 一次吃 S 个位置时，S 个 query 都要看到彼此的
   K/V；只有 `--cache` 模式逐位置推进，才真正等价于「每生成一个 token 就 cache 一次」。这也是
   `check_kv.py` 里唯一能做出位等价对照的形态。

## 6. 已知限制（如实记录）

- 单线程、标量 `matvec`，用 double 累加；AVX2 与多线程是 M4 的事；
- 权重常驻 f32（2.4 GiB），`q8_0` 的原生量化权重只在加载时用到；
- `batch = 1`，没有连续批处理 / paged attention，cache 是连续数组、按 `n_ctx` 上限预留；
- top-p 在 top-256 短名单上近似（见 §4）；
- 没有 KV cache 的量化（f16/q8 cache），满上下文 768 MiB。

## 7. M4 计划

1. 手写 AVX2 + FMA 的 `dot_q8_0` / `dot_q4_0`：直接在量化块上做点积，避免反量化到 f32；
2. GEMM 分块 + cache 友好布局；按 token 维切分的线程池（8 核 16 线程）；
3. 消融表：标量 / AVX2 / AVX2+线程 / 量化点积，逐项给出 ms-per-token；
4. 同机对比 `llama.cpp`，把差距归因到具体项（内存带宽、块布局、线程调度），而不是一句「它更快」。
