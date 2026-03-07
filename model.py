"""
Minimal from-scratch Qwen2.5-0.5B-Instruct transformer implementation.

Implements the full model architecture in PyTorch with no HuggingFace model classes.
Loads weights directly from safetensors files.

Supports two attention backends:
  - PyTorch SDPA (CPU/fallback): manual scaled dot-product attention with padded KV cache
  - FlashInfer (GPU): paged KV cache with fused attention kernels
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

try:
    import flashinfer.page

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Qwen2Config:
    hidden_size: int = 896
    num_attention_heads: int = 14  # Q heads
    num_key_value_heads: int = 2  # KV heads (GQA group size = 7)
    head_dim: int = 64  # hidden_size / num_attention_heads
    num_hidden_layers: int = 24
    intermediate_size: int = 4864  # SwiGLU MLP
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen2Config":
        """Load config from a HuggingFace model directory's config.json."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            raw = json.load(f)
        return cls(
            hidden_size=raw["hidden_size"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            num_hidden_layers=raw["num_hidden_layers"],
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 1_000_000.0),
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
            tie_word_embeddings=raw.get("tie_word_embeddings", True),
        )


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def precompute_rope(
    head_dim: int, max_seq_len: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
    return emb.cos(), emb.sin()


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings via rotate-half.

    x:   [batch, n_heads, seq_len, head_dim]
    cos: [batch, seq_len, head_dim]  (from position_ids indexing)
    sin: [batch, seq_len, head_dim]
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    # [batch, 1, seq_len, head_dim] for broadcasting over heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(orig_dtype)


class Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        # Pad Q heads to next power-of-2 group size for FlashInfer compatibility
        padded_group = 1 << (self.num_groups - 1).bit_length()  # next power of 2
        self.padded_num_heads = padded_group * self.num_kv_heads
        self.needs_head_padding = self.padded_num_heads != self.num_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attn_backend=None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Rotary position embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Dispatch to the appropriate attention backend
        if attn_backend is not None:
            out = self._flashinfer_attention(q, k, v, attn_backend, bsz, seq_len)
        else:
            out = self._pytorch_attention(
                q, k, v, mask, kv_cache, position_ids, bsz, seq_len
            )

        return self.o_proj(out)

    def _flashinfer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        backend,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        """FlashInfer paged attention path.

        1. Reshape K, V to [total_tokens, kv_heads, head_dim]
        2. Append into paged KV cache via flashinfer.page.append_paged_kv_cache
        3. Reshape Q for FlashInfer
        4. Run the fused attention kernel
        5. Reshape output back to [bsz, seq_len, hidden_size]
        """
        layer_idx = backend.next_layer()
        kv_data = backend.paged_kv_cache.kv_data[layer_idx]

        # K, V: [bsz, kv_heads, seq_len, head_dim] -> [total_tokens, kv_heads, head_dim]
        k_append = k.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        v_append = v.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)

        # Append K, V into paged cache (fused CUDA kernel)
        flashinfer.page.append_paged_kv_cache(
            k_append,
            v_append,
            backend.batch_indices,
            backend.positions,
            kv_data,
            backend.kv_page_indices,
            backend.kv_page_indptr,
            backend.kv_last_page_len,
        )

        # Q: [bsz, heads, seq_len, head_dim] -> FlashInfer format
        if backend.mode == "prefill":
            # Ragged: [total_tokens, num_heads, head_dim]
            q_fi = q.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
        else:
            # Decode: [batch_size, num_heads, head_dim]
            q_fi = q.squeeze(2)

        # Pad Q heads per-group for non-power-of-2 GQA (e.g. 14 -> 16 heads)
        if self.needs_head_padding:
            leading = q_fi.shape[:-2]  # [total_tokens] or [bsz]
            padded_group = self.padded_num_heads // self.num_kv_heads
            q_fi = q_fi.view(*leading, self.num_kv_heads, self.num_groups, self.head_dim)
            q_fi = F.pad(q_fi, (0, 0, 0, padded_group - self.num_groups))
            q_fi = q_fi.reshape(*leading, self.padded_num_heads, self.head_dim)

        # Run fused attention kernel (handles causal mask + GQA internally)
        out = backend.wrapper.run(q_fi, kv_data)

        # Strip padded heads
        if self.needs_head_padding:
            leading = out.shape[:-2]
            padded_group = self.padded_num_heads // self.num_kv_heads
            out = out.view(*leading, self.num_kv_heads, padded_group, self.head_dim)
            out = out[:, :, :self.num_groups, :] if out.dim() == 4 else out[..., :self.num_groups, :]
            out = out.reshape(*leading, self.num_heads, self.head_dim)

        # Reshape output back to [bsz, seq_len, hidden_size]
        if backend.mode == "prefill":
            out = out.view(bsz, seq_len, self.num_heads, self.head_dim)
        else:
            out = out.view(bsz, 1, self.num_heads, self.head_dim)
        out = out.reshape(bsz, seq_len, -1).contiguous()

        return out

    def _pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.Tensor | None,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        """PyTorch SDPA path (CPU/fallback with padded KV cache)."""
        # KV cache update
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if seq_len == 1 and bsz > 1:
                # Batched decode: each element writes to its own position
                for i in range(bsz):
                    pos = position_ids[i, 0].item()
                    k_cache[i, :, pos, :] = k[i, :, 0, :]
                    v_cache[i, :, pos, :] = v[i, :, 0, :]
                max_kv_len = position_ids.max().item() + 1
                k = k_cache[:, :, :max_kv_len]
                v = v_cache[:, :, :max_kv_len]
            else:
                # Prefill (bsz=1) or single decode: contiguous slice
                start = position_ids[0, 0].item()
                k_cache[:, :, start : start + seq_len] = k
                v_cache[:, :, start : start + seq_len] = v
                k = k_cache[:, :, : start + seq_len]
                v = v_cache[:, :, : start + seq_len]

        # GQA: expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return out


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attn_backend=None,
    ) -> torch.Tensor:
        # Pre-norm attention + residual
        residual = x
        x = self.self_attn(
            self.input_layernorm(x), cos, sin, mask, kv_cache, position_ids,
            attn_backend,
        )
        x = residual + x

        # Pre-norm MLP + residual
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Separate lm_head when embeddings are not tied
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        # Precompute RoPE tables (registered as buffers so they move with .to())
        cos, sin = precompute_rope(
            config.head_dim, config.max_position_embeddings, config.rope_theta, torch.device("cpu")
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attn_backend=None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [batch, seq_len]
            kv_caches:      list of (k_cache, v_cache) tuples, or None (padded path)
            position_ids:   [batch, seq_len] — position for each token
            attention_mask:  [batch, 1, seq_len, kv_len] or None (padded path)
            attn_backend:   AttentionBackend or None (FlashInfer paged path)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)

        # RoPE from position_ids: index into cached tables
        # cos_cached: [max_pos, head_dim] -> cos: [batch, seq_len, head_dim]
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        # Build attention mask (only for PyTorch SDPA path)
        if attn_backend is not None:
            # FlashInfer handles causal masking internally
            mask = None
        elif attention_mask is not None:
            mask = attention_mask
        elif seq_len > 1:
            # Prefill: build causal mask (bsz=1 expected)
            start_pos = position_ids[0, 0].item()
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=h.device, dtype=h.dtype
            )
            mask = torch.triu(mask, diagonal=1)
            if start_pos > 0:
                mask = torch.cat(
                    [torch.zeros(seq_len, start_pos, device=h.device, dtype=h.dtype), mask],
                    dim=-1,
                )
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]
        else:
            mask = None  # Single decode: attends to everything

        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            h = layer(h, cos, sin, mask, kv, position_ids, attn_backend)

        h = self.norm(h)

        # lm_head: separate weight or tied to embedding
        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = F.linear(h, self.embed_tokens.weight)
        return logits


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_weights(
    model: Qwen2Model, model_path: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> None:
    """Load weights from safetensors files into the model."""
    pattern = os.path.join(model_path, "*.safetensors")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for f in files:
        state_dict.update(load_file(f, device=device))

    # Strip "model." prefix to match our module hierarchy
    cleaned: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key == "lm_head.weight":
            # Keep lm_head if model has a separate one (not tied)
            if model.lm_head is not None:
                cleaned[key] = tensor.to(dtype)
            continue
        new_key = key.removeprefix("model.")
        cleaned[new_key] = tensor.to(dtype)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # cos_cached / sin_cached are buffers, will show as missing — that's fine
    real_missing = [k for k in missing if "cached" not in k]
    if real_missing:
        print(f"Warning: missing keys: {real_missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
    print(f"Loaded {len(cleaned)} tensors from {len(files)} file(s)")
