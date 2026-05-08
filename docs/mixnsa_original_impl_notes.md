# MixNSA 原始实现位置与机制（Qwen2.5-VL）

你问的“原始实现在哪里、怎么做的”，在这个仓库里主要是：

- 文件：`lmms-eval/lmms_eval/models/qwen25_vl/modeling_qwen2_5_vl.py`
- 核心类：`Qwen2_5_VLMixNSA`

## 1) 入口与挂载位置

`QWEN2_5_VL_ATTENTION_CLASSES` 把 `flash_attention_2` 映射到 `Qwen2_5_VLMixNSA`，并在 decoder layer 里通过 `self.self_attn = ...` 实例化。

## 2) token 类型判定

`_detect_token_types(...)` 用 `position_ids` 判定视觉/文本 token：

- 期望 `position_ids` 形状为 `[3, batch, seq]`（t/h/w）
- 若 `t==h==w` 则视为文本 token
- 否则视为视觉 token

## 3) 混合注意力主路径

`_mixed_attention(...)` 里把 q/k/v 按视觉 mask 与文本 mask 分流：

- 视觉分支走 `nsa_func(...)`
- 文本分支走 `_flash_attention_forward(...)`
- 最后两个分支输出相加融合

## 4) 两个纯路径回退

还提供了：

- `_pure_flash_attention(...)`
- `_pure_nsa_attention(...)`

用于在“全是文本”或“全是视觉”等场景走单一路径。

## 5) 对 Qwen3-VL 继续设计的直接启发

在 Qwen3-VL 上可以保持同一框架：

1. `forward` 里先 `detect token types`
2. 根据 mask 分流 q/k/v
3. vision 分支接 NSA；text 分支复用现有 attention interface
4. 合并输出，最后过 `o_proj`

