# gpt-oss 调研(精简版)

> 骨架是标准 decoder-only Transformer,但有 5 个非标准改造,3 个是 XPU enable 的硬骨头。

## 与标准 Transformer 的差异

| 组件 | 标准做法 | gpt-oss | 难度 |
|---|---|---|---|
| Attention | 普通 softmax | **+ Attention Sinks**(分母多一项,泄压阀) | 高 |
| MLP 激活 | 标准 SwiGLU | **clamp + interleave + (up+1) + α=1.702** | 中 |
| FFN | dense / 普通 MoE | **128专家 top-4 + MXFP4 grouped GEMM** | 最高 |
| Attention 范围 | 全 full 或全 sliding | **隔层交替** full / sliding(128) | 中 |
| 位置编码 | 标准 RoPE | **YaRN**(4k→128k) | 中 |
| 归一化 | RMSNorm | RMSNorm(标准) | 低 |

## 三个硬骨头(重点)

1. **Attention Sinks** — 每 head 一个可学习 sink 拼进 softmax 再丢弃 → `_supports_sdpa=False`,必须用改过的 flash kernel(传 `s_aux`)。
2. **MXFP4 grouped MoE** — 128 专家选 top-4,4-bit 量化 + gather/scatter grouped GEMM,最难 enable。
3. **Clamped/Interleaved SwiGLU** — gate/up 交错、clamp ±7、(up+1)、α=1.702,数值行为特殊,不能套标准 kernel。

## Kernel 来源(XPU)

| 模块 | repo(huggingface.co) | 名称 |
|---|---|---|
| RMSNorm | kernels-community/rmsnorm | `RMSNorm` |
| MoE | kernels-community/megablocks | `MegaBlocksMoeMLP` |
| RoPE | kernels-community/rotary | `apply_rotary_transformers` |
| Attention | 动态加载(vllm-flash-attn3) | `flash_attn_varlen_func` |

> gpt-oss 走 HF hub kernels(huggingface.co);Qwen 走 GitHub 源码库(causal-conv1d 等)—— 分发方式不同。
