# gpt-oss

gpt-oss 模型结构与 kernel 调研笔记(面向 XPU / pre-silicon enable)。

## 内容

- [gpt-oss 调研(精简版)](./gpt-oss-调研.md) — 与标准 Transformer 的差异、3 个硬骨头、XPU kernel 来源。

## 摘要

gpt-oss 骨架是标准 decoder-only Transformer,但有 5 个非标准改造,其中 3 个是 XPU enable 的重点:

1. **Attention Sinks** — softmax 多一项,标准 SDPA 不支持。
2. **MXFP4 grouped MoE** — 128 专家 top-4 + 4-bit 量化,最难 enable。
3. **Clamped/Interleaved SwiGLU** — clamp ±7、(up+1)、α=1.702,数值行为特殊。

> 来源:HF Transformers `modeling_gpt_oss.py` / `configuration_gpt_oss.py` @ `6f075c5`。
