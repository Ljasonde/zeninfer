# M1 验收报告 — GGUF 解析器 + Qwen BPE 分词器

日期：2026-11-12　项目：`D:\zeninfer`　模型：Qwen2.5-0.5B-Instruct（q8_0 / q4_0 / fp16 三份权重）

## 1. 交付内容

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| GGUF v2/v3 读取 | `src/gguf.h`, `src/gguf.cpp` | Win32 `CreateFileW`+`MapViewOfFile` 零拷贝映射；KV / tensor info 解析；按 `general.alignment` 求 `tensor_data_offset`；越界与张量重叠即抛错 |
| 反量化 | 同上 | F32 / F16 / Q8_0 / Q4_0（Q4_0 nibble 低半字节 = 元素 i，高半字节 = 元素 i+16，偏移 −8） |
| UTF-8 与字符分类 | `src/unicode.{h,cpp}`, `src/unicode_tables.h` | 严格解码（拒绝 overlong / 代理 / 越界），Unicode 15.0.0 的 `\p{L}`(659 段) / `\p{N}`(137 段) / `\s`(10 段) 表 |
| 分词器 | `src/tokenizer.{h,cpp}` | GPT-2 byte-level 字母表双向映射；手写 qwen2 pre-tokenizer 扫描器；special token 优先匹配；BPE 用 hash map + 按 rank 合并 |
| CLI | `src/main.cpp` | `inspect [--json]`、`tokenize`、`spans`、`vocab-dump`、`bench-tokenizer` |

## 2. 验收证据

三条命令，全部在 `make check` 里自动跑（退出码 0）。

### 2.1 解析器 vs 独立实现

`tools/dump_gguf.py` 是一个不复用任何 C++ 代码的纯 Python GGUF 解析器。两边对同一份权重各导一份 JSON，逐张量比对 dtype / dims / nbytes / offset，并各自检查 payload 越界与重叠：

```
OK: 291 tensors identical, 669763072 bytes, arch=qwen2
    tokens=151936 merges=151387 alignment=32
```

### 2.2 词表与 merge 表 vs `tokenizer.json`

```
tokens: 151643/151643 identical, merges: 151387/151387 identical in rank order
```

gguf 比 HuggingFace 多 293 条：22 条 added token（`<|im_start|>` 等）+ 271 条 `[PADnnnn]` 占位（llama.cpp 把词表补齐到 151936），已单独识别、不算差异。

### 2.3 编码结果 vs HuggingFace `tokenizers`

75 条手写边界用例 + 2000 条 fuzz（字母/数字/**连续空格**/NBSP/表意空格/U+2028/ZWSP/中日韩/重音拉丁/emoji/皮肤修饰/组合符/special token 混排）：

```
1979/1979 cases match (100.00%)
fuzz cases: 2000, hand-written: 75, skipped (need NFC): 43
```

43 条被跳过的是**含未 NFC 化组合字符**的输入：HF 在切分前做 NFC 归一化，zeninfer 与 llama.cpp 都不做，因此这类输入必然不同。这是有意保留的差异，不是掩盖的失败。

## 3. 过程中值得记下的两处坑

1. **合并后的词表是 byte-level 字母表，不是原始字节。** llama.cpp 保留了 GPT-2 的字节→码位映射，空格在词表里是 `Ġ`(U+0120)、换行是 `Ċ`(U+010A)。第一版按原始字节处理，`id_of(" ")` 直接 −1。现在由 `ByteLevelTable` 做双向映射（68 个字节重映射到 U+0100 起）。
2. **pre-tokenizer 的正则不能交给 `std::regex`。** Python 的 `\s` 与 ECMAScript 的不同；`\s+(?!\S)` 与 `\s*[\r\n]+` 对同一段空白的归属必须严格一致（前者取**最后一个换行之后**的位置，EOF 时吃满、否则退回一个字符）。改为手写扫描器，7 个 alternative 按顺序匹配。

## 4. 性能（已测）

`bench-tokenizer` 在 596 KB Python 源码语料上，5 次取最好：

```
tokens : 130473 (4.674 bytes/token)
best   : 73.68 ms = 7.9 MB/s = 1,770,878 tok/s
```

单词表查找是 `unordered_map<string_view>`，合并表按 rank 排序后线性扫描候选对；目前单线程，后续可并行切 span。

## 5. 三份权重均已验证可解析

| 文件 | 大小 | 张量构成 |
| --- | --- | --- |
| `qwen2.5-0.5b-instruct-fp16.gguf` | 1.17 GiB | F16 ×170 |
| `qwen2.5-0.5b-instruct-q8_0.gguf` | 644.4 MiB | Q8_0 ×170 |
| `qwen2.5-0.5b-instruct-q4_0.gguf` | 403.2 MiB | Q4_0 ×169 + Q8_0 ×1（`output.weight` 按 llama.cpp 惯例保留 8-bit） |

后两者是 M5 量化消融的对照组：同一份 fp16 权重压到 Q8_0 / Q4_0，用来量「压缩比 ↔ 困惑度上升」的代价。

## 6. M2 计划（待验收后开工）

1. 张量加载：mmap → 反量化到 fp32 常驻内存（0.5B 约 2 GB，可行）。
2. 手写算子：RMSNorm、RoPE（base 1e6）、GQA（14 Q head / 2 KV head，每 KV head 服务 7 个 Q head）、SwiGLU FFN（4864 中间维）、embedding + lm_head（tie 与否以 gguf 里是否有 `output.weight` 为准）。
3. 对拍目标：与 HF transformers 在 fp16 权重上的 logits 比，max |Δ| < 1e-4（fp32 累加，允许 fp16 舍入差异处放宽并记录）。
4. 验收：`zeninfer logits <gguf> --text "..."` 输出前若干位置的 top-k，与参考实现逐项一致；并给出单次前向耗时。
