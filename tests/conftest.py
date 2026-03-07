"""Shared fixtures for inference engine tests."""

import sys
import os

# Add inference/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import pytest
import torch

from model import Qwen2Config, Qwen2Model
from scheduler import BatchedKVCache, PagedKVCache, Scheduler


# ---------------------------------------------------------------------------
# Tiny config — 2 layers, small dims, fast tests
# ---------------------------------------------------------------------------

TINY_CONFIG = Qwen2Config(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=2,
    intermediate_size=128,
    vocab_size=1000,
    max_position_embeddings=256,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float32
MAX_BATCH = 4
MAX_SEQ = 64
PAGE_SIZE = 4


@pytest.fixture
def tiny_config():
    return TINY_CONFIG


@pytest.fixture
def tiny_model():
    """Randomly initialized tiny model (no weights needed)."""
    model = Qwen2Model(TINY_CONFIG).to(device=DEVICE, dtype=DTYPE).eval()
    return model


@pytest.fixture
def kv_cache():
    """BatchedKVCache for the tiny config."""
    return BatchedKVCache(TINY_CONFIG, MAX_BATCH, MAX_SEQ, DEVICE, DTYPE)


@pytest.fixture
def paged_kv_cache():
    """PagedKVCache for the tiny config."""
    return PagedKVCache(TINY_CONFIG, MAX_BATCH, MAX_SEQ, PAGE_SIZE, DEVICE, DTYPE)


@pytest.fixture
def scheduler_with_model(tiny_model):
    """A Scheduler backed by the tiny model, started and stopped per test."""
    sched = Scheduler(
        model=tiny_model,
        config=TINY_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
        max_batch_size=MAX_BATCH,
        max_seq_len=MAX_SEQ,
        page_size=PAGE_SIZE,
    )
    sched.start()
    yield sched
    sched.stop()
