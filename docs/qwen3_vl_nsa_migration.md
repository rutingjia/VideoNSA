# Qwen3-VL 接入 Mixed-NSA 的实现建议

本文给出一个最小、可维护的迁移方案：把 Qwen2.5-VL 里“视觉 token 走 NSA、文本 token 走 Full/Flash Attention”的思路，迁移到 `third_party/transformers_qwen3_vl`。

## 1. 总体策略

- **不要直接改 `modeling_qwen3_vl.py`**：该文件顶部已声明是由 `modular_qwen3_vl.py` 自动生成。
- 在 `modular_qwen3_vl.py` 中新增一个 `Qwen3VLMixNSA` 注意力类，并在文本 decoder 层里替换默认注意力实现。
- 继续沿用当前项目在 Qwen2.5-VL 的分流逻辑：
  - 视觉 token：NSA
  - 文本 token：Flash/Eager（按 `config._attn_implementation`）

## 2. 推荐代码骨架

```python
class Qwen3VLMixNSA(Qwen3VLTextAttention):
    """Mixed attention for Qwen3-VL text decoder.

    Vision tokens -> NSA
    Text tokens   -> standard attention interface (flash/eager)
    """

    def _detect_token_types(self, hidden_states, position_ids=None):
        # Expect shape [3, bsz, seq_len] for multimodal position ids: (t, h, w)
        bsz, seq_len, _ = hidden_states.shape
        if position_ids is None:
            return torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)

        if position_ids.dim() == 3 and position_ids.shape[0] == 3:
            t, h, w = position_ids[0], position_ids[1], position_ids[2]
            is_text = (t == h) & (t == w)
            return ~is_text

        # fallback: all text
        return torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)

    def _nsa_attention(self, q, k, v):
        from nsa.nsa import nsa_func
        # q/k/v: [bsz, heads, seq, head_dim]
        # 调用签名按你本地 nsa_func 的实际参数调整
        out = nsa_func(q, k, v, block_size=64, window_size=512)
        return out

    def _text_attention(self, attention_interface, q, k, v, attention_mask, **kwargs):
        out, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=True,
            **kwargs,
        )
        return out

    def _mixed_attention(self, attention_interface, q, k, v, is_vision_tokens, attention_mask, **kwargs):
        # is_vision_tokens: [bsz, seq] -> [bsz, 1, seq, 1]
        vision_mask = is_vision_tokens.unsqueeze(1).unsqueeze(-1)
        text_mask = ~vision_mask

        qv, kv, vv = q * vision_mask, k * vision_mask, v * vision_mask
        qt, kt, vt = q * text_mask,  k * text_mask,  v * text_mask

        nsa_out = self._nsa_attention(qv, kv, vv)
        txt_out = self._text_attention(attention_interface, qt, kt, vt, attention_mask, **kwargs)

        return nsa_out + txt_out

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        # 1) project qkv + rope（沿用 Qwen3 原实现）
        # 2) detect token types
        is_vision_tokens = self._detect_token_types(hidden_states, position_ids)
        # 3) mixed branch
        # 4) o_proj + return
        ...
```

## 3. 在 decoder 中挂载

- 在文本 decoder layer 的 attention 构造处，把 `_attn_implementation == "flash_attention_2"` 的分支映射到 `Qwen3VLMixNSA`。
- 保留 eager/sdpa 分支用于回退调试。

示意：

```python
QWEN3_VL_ATTENTION_CLASSES = {
    "eager": Qwen3VLTextAttention,
    "sdpa": Qwen3VLTextAttention,
    "flash_attention_2": Qwen3VLMixNSA,
}
```

## 4. 三个容易踩坑的点

1. **position_ids 语义**：先打印一个真实 batch，确认 Qwen3-VL 的 `position_ids` 仍然是 `[3, bsz, seq]` 且 `(t,h,w)` 语义一致。若不一致，分流规则要改。
2. **KV cache**：生成阶段带 cache 时，token 分流需要只作用于本 step 的 query 位置，避免历史缓存位置错位。
3. **mask 乘法的数值行为**：`q/k/v * mask` 会把另一分支置零，等价于“共享序列、分支计算后再相加”。如发现边界 token 被稀释，可改成 gather/scatter 版（按索引抽取视觉/文本子序列再回填）。

## 5. 最小验证清单

1. 纯文本输入：输出应与原 Qwen3-VL 基本一致（允许轻微数值差）。
2. 图文输入：确认 `is_vision_tokens.sum() > 0`。
3. 关闭 NSA（全部走文本 attention）作为对照，确保可回退。
4. 短视频输入跑 1 个 batch，检查显存与吞吐。

---

如果你准备正式落地，建议第一版先做“**只在 inference 路径启用 Mixed-NSA**”，训练路径保持原生 attention，先把稳定性跑通再扩展到训练。
