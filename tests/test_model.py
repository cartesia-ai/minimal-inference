"""Tests for model.py — transformer components."""

import torch
import torch.nn.functional as F

from model import (
    Attention,
    MLP,
    RMSNorm,
    apply_rotary_emb,
    precompute_rope,
)
from tests.conftest import DEVICE, DTYPE, TINY_CONFIG


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_shape_and_dtype(self):
        norm = RMSNorm(64, eps=1e-6).to(DEVICE, DTYPE)
        x = torch.randn(2, 10, 64, device=DEVICE, dtype=DTYPE)
        out = norm(x)
        assert out.shape == x.shape
        assert out.dtype == DTYPE

    def test_normalization(self):
        norm = RMSNorm(64, eps=1e-6).to(DEVICE, DTYPE)
        x = torch.randn(1, 5, 64, device=DEVICE, dtype=DTYPE) * 10
        out = norm(x)
        # RMS of output (before weight) should be ~1
        rms = out.float().pow(2).mean(-1).sqrt()
        assert rms.mean().item() < 3.0  # not exploding


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_shape(self):
        cos, sin = precompute_rope(16, 256, 1e6, DEVICE)
        assert cos.shape == (256, 16)
        assert sin.shape == (256, 16)

    def test_values_bounded(self):
        cos, sin = precompute_rope(16, 256, 1e6, DEVICE)
        assert cos.min() >= -1.0
        assert cos.max() <= 1.0
        assert sin.min() >= -1.0
        assert sin.max() <= 1.0

    def test_apply_rotary_emb_shape(self):
        cos_table, sin_table = precompute_rope(16, 256, 1e6, DEVICE)
        # Simulate [batch, seq] position_ids
        position_ids = torch.arange(5, device=DEVICE).unsqueeze(0)  # [1, 5]
        cos = cos_table[position_ids]  # [1, 5, 16]
        sin = sin_table[position_ids]  # [1, 5, 16]
        x = torch.randn(1, 4, 5, 16, device=DEVICE, dtype=DTYPE)  # [batch, heads, seq, dim]
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_rotary_changes_values(self):
        cos_table, sin_table = precompute_rope(16, 256, 1e6, DEVICE)
        position_ids = torch.arange(5, device=DEVICE).unsqueeze(0)
        cos = cos_table[position_ids]
        sin = sin_table[position_ids]
        x = torch.randn(1, 4, 5, 16, device=DEVICE, dtype=DTYPE)
        out = apply_rotary_emb(x, cos, sin)
        # Output should differ from input (rotation applied)
        assert not torch.allclose(x, out, atol=1e-5)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class TestMLP:
    def test_shape(self):
        mlp = MLP(TINY_CONFIG).to(DEVICE, DTYPE)
        x = torch.randn(2, 10, 64, device=DEVICE, dtype=DTYPE)
        out = mlp(x)
        assert out.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestAttention:
    def test_output_shape(self):
        attn = Attention(TINY_CONFIG).to(DEVICE, DTYPE)
        cos_table, sin_table = precompute_rope(16, 256, 1e6, DEVICE)
        pos = torch.arange(10, device=DEVICE).unsqueeze(0)
        cos = cos_table[pos]
        sin = sin_table[pos]

        x = torch.randn(1, 10, 64, device=DEVICE, dtype=DTYPE)
        out = attn(x, cos, sin, mask=None)
        assert out.shape == (1, 10, 64)

    def test_kv_cache_prefill(self):
        """Prefill should populate KV cache at positions 0..seq_len-1."""
        attn = Attention(TINY_CONFIG).to(DEVICE, DTYPE)
        cos_table, sin_table = precompute_rope(16, 256, 1e6, DEVICE)

        seq_len = 5
        pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        cos = cos_table[pos]
        sin = sin_table[pos]

        k_cache = torch.zeros(1, 2, 64, 16, device=DEVICE, dtype=DTYPE)
        v_cache = torch.zeros(1, 2, 64, 16, device=DEVICE, dtype=DTYPE)

        x = torch.randn(1, seq_len, 64, device=DEVICE, dtype=DTYPE)
        attn(x, cos, sin, mask=None, kv_cache=(k_cache, v_cache), position_ids=pos)

        # First seq_len positions should be non-zero
        assert k_cache[:, :, :seq_len].abs().sum() > 0
        # Positions beyond seq_len should still be zero
        assert k_cache[:, :, seq_len:].abs().sum() == 0

    def test_kv_cache_decode(self):
        """Single-token decode should write to the correct position."""
        attn = Attention(TINY_CONFIG).to(DEVICE, DTYPE)
        cos_table, sin_table = precompute_rope(16, 256, 1e6, DEVICE)

        # First prefill 5 tokens
        seq_len = 5
        pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        cos = cos_table[pos]
        sin = sin_table[pos]
        k_cache = torch.zeros(1, 2, 64, 16, device=DEVICE, dtype=DTYPE)
        v_cache = torch.zeros(1, 2, 64, 16, device=DEVICE, dtype=DTYPE)
        x = torch.randn(1, seq_len, 64, device=DEVICE, dtype=DTYPE)
        attn(x, cos, sin, mask=None, kv_cache=(k_cache, v_cache), position_ids=pos)

        # Now decode at position 5
        decode_pos = torch.tensor([[5]], device=DEVICE)
        cos_d = cos_table[decode_pos]
        sin_d = sin_table[decode_pos]
        x_d = torch.randn(1, 1, 64, device=DEVICE, dtype=DTYPE)
        attn(x_d, cos_d, sin_d, mask=None, kv_cache=(k_cache, v_cache), position_ids=decode_pos)

        # Position 5 should now be populated
        assert k_cache[:, :, 5].abs().sum() > 0


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class TestQwen2Model:
    def test_forward_prefill(self, tiny_model):
        seq_len = 8
        input_ids = torch.randint(0, 1000, (1, seq_len), device=DEVICE)
        pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

        with torch.inference_mode():
            logits = tiny_model(input_ids, position_ids=pos)

        assert logits.shape == (1, seq_len, 1000)

    def test_forward_decode_with_cache(self, tiny_model, kv_cache):
        """Prefill then decode — logits shape should be [1, 1, vocab]."""
        seq_len = 5
        input_ids = torch.randint(0, 1000, (1, seq_len), device=DEVICE)
        pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        slot = kv_cache.allocate_slot()
        slot_caches = kv_cache.get_slot_caches(slot)

        with torch.inference_mode():
            logits = tiny_model(input_ids, kv_caches=slot_caches, position_ids=pos)
            assert logits.shape == (1, seq_len, 1000)

            # Decode step
            next_token = logits[0, -1].argmax().item()
            decode_ids = torch.tensor([[next_token]], device=DEVICE)
            decode_pos = torch.tensor([[seq_len]], device=DEVICE)
            logits2 = tiny_model(decode_ids, kv_caches=slot_caches, position_ids=decode_pos)
            assert logits2.shape == (1, 1, 1000)

    def test_causal_mask(self, tiny_model):
        """Earlier tokens should not attend to later tokens during prefill."""
        seq_len = 6
        input_ids = torch.randint(0, 1000, (1, seq_len), device=DEVICE)
        pos = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

        with torch.inference_mode():
            logits = tiny_model(input_ids, position_ids=pos)

        # If causal masking works, changing the last token shouldn't affect
        # the logits at position 0
        input_ids2 = input_ids.clone()
        input_ids2[0, -1] = (input_ids[0, -1] + 1) % 1000

        with torch.inference_mode():
            logits2 = tiny_model(input_ids2, position_ids=pos)

        # Position 0 logits should be identical (causal — can't see future)
        assert torch.allclose(logits[0, 0], logits2[0, 0], atol=1e-5)

    def test_tied_embeddings(self, tiny_model):
        """lm_head should use embed_tokens.weight (no separate lm_head layer)."""
        # The model computes logits via F.linear(h, embed_tokens.weight)
        # Verify there's no separate lm_head parameter
        param_names = [name for name, _ in tiny_model.named_parameters()]
        assert not any("lm_head" in n for n in param_names)

        # Verify logits shape matches vocab size from embedding
        input_ids = torch.randint(0, 1000, (1, 3), device=DEVICE)
        pos = torch.arange(3, device=DEVICE).unsqueeze(0)
        with torch.inference_mode():
            logits = tiny_model(input_ids, position_ids=pos)
        assert logits.shape[-1] == tiny_model.embed_tokens.weight.shape[0]
