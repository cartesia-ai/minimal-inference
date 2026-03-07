"""Tests for scheduler.py — KV cache, sampling, scheduler lifecycle."""

import asyncio
import time

import pytest
import torch

from scheduler import (
    BatchedKVCache,
    PagedKVCache,
    Request,
    RequestStatus,
    Scheduler,
    build_csr_metadata,
    sample,
)
from tests.conftest import DEVICE, DTYPE, MAX_BATCH, MAX_SEQ, PAGE_SIZE, TINY_CONFIG


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------


class TestSample:
    def test_greedy(self):
        logits = torch.zeros(1, 1, 1000)
        logits[0, 0, 42] = 100.0  # token 42 is clearly the argmax
        assert sample(logits, temperature=0) == 42

    def test_temperature(self):
        logits = torch.randn(1, 1, 1000)
        token = sample(logits, temperature=0.8)
        assert 0 <= token < 1000


# ---------------------------------------------------------------------------
# BatchedKVCache
# ---------------------------------------------------------------------------


class TestBatchedKVCache:
    def test_allocate_and_release(self, kv_cache):
        initial_free = len(kv_cache.free_slots)
        s0 = kv_cache.allocate_slot()
        s1 = kv_cache.allocate_slot()
        assert s0 != s1
        assert len(kv_cache.free_slots) == initial_free - 2

        kv_cache.release_slot(s0)
        assert s0 in kv_cache.free_slots
        assert len(kv_cache.free_slots) == initial_free - 1

    def test_full(self, kv_cache):
        # Exhaust all slots
        for _ in range(MAX_BATCH):
            kv_cache.allocate_slot()
        with pytest.raises(RuntimeError, match="No free KV cache slots"):
            kv_cache.allocate_slot()

    def test_release_zeros(self, kv_cache):
        slot = kv_cache.allocate_slot()
        # Write something non-zero
        for k, v in kv_cache.caches:
            k[slot] = 1.0
            v[slot] = 1.0
        kv_cache.release_slot(slot)
        # Should be zeroed
        for k, v in kv_cache.caches:
            assert k[slot].abs().sum() == 0
            assert v[slot].abs().sum() == 0

    def test_slot_view_shape(self, kv_cache):
        slot = kv_cache.allocate_slot()
        caches = kv_cache.get_slot_caches(slot)
        assert len(caches) == TINY_CONFIG.num_hidden_layers
        for k, v in caches:
            assert k.shape == (1, TINY_CONFIG.num_key_value_heads, MAX_SEQ, TINY_CONFIG.head_dim)
            assert v.shape == k.shape

    def test_batch_caches_shape(self, kv_cache):
        slots = [kv_cache.allocate_slot() for _ in range(3)]
        caches = kv_cache.get_batch_caches(slots)
        assert len(caches) == TINY_CONFIG.num_hidden_layers
        for k, v in caches:
            assert k.shape[0] == 3
            assert k.shape == (3, TINY_CONFIG.num_key_value_heads, MAX_SEQ, TINY_CONFIG.head_dim)

    def test_write_back(self, kv_cache):
        slots = [kv_cache.allocate_slot() for _ in range(2)]
        positions = [5, 10]

        # Get batch caches and write some data at the positions
        batch_caches = kv_cache.get_batch_caches(slots)
        for k, v in batch_caches:
            k[0, :, 5, :] = 1.0
            k[1, :, 10, :] = 2.0
            v[0, :, 5, :] = 3.0
            v[1, :, 10, :] = 4.0

        kv_cache.write_back_decode(slots, positions, batch_caches)

        # Verify master cache was updated at correct positions
        for k_master, v_master in kv_cache.caches:
            assert torch.allclose(k_master[slots[0], :, 5, :], torch.tensor(1.0))
            assert torch.allclose(k_master[slots[1], :, 10, :], torch.tensor(2.0))
            assert torch.allclose(v_master[slots[0], :, 5, :], torch.tensor(3.0))
            assert torch.allclose(v_master[slots[1], :, 10, :], torch.tensor(4.0))


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class TestRequest:
    def test_defaults(self):
        req = Request(
            request_id="test-1",
            input_ids=[1, 2, 3],
            max_new_tokens=10,
            temperature=0.0,
        )
        assert req.status == RequestStatus.WAITING
        assert req.generated_tokens == []
        assert req.current_position == 0
        assert req.batch_slot == -1
        assert req.finish_reason is None


# ---------------------------------------------------------------------------
# Scheduler integration (uses tiny model)
# ---------------------------------------------------------------------------


class TestSchedulerIntegration:
    def _collect_tokens(self, req: Request, timeout: float = 10.0) -> list[int]:
        """Collect all tokens from a request's queue (blocking)."""
        tokens = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                tok = req.token_queue.get_nowait()
            except asyncio.QueueEmpty:
                time.sleep(0.05)
                continue
            if tok is None:
                break
            tokens.append(tok)
        return tokens

    def test_single_request_completes(self, scheduler_with_model):
        req = Request(
            request_id="test-single",
            input_ids=[1, 2, 3, 4, 5],
            max_new_tokens=5,
            temperature=0.0,
        )
        scheduler_with_model.submit_request(req)
        tokens = self._collect_tokens(req)

        assert len(tokens) > 0
        assert len(tokens) <= 5
        assert req.status == RequestStatus.FINISHED
        assert req.finish_reason in ("stop", "length")

    def test_concurrent_requests(self, scheduler_with_model):
        reqs = []
        for i in range(3):
            req = Request(
                request_id=f"test-concurrent-{i}",
                input_ids=[10 + i, 20 + i, 30 + i],
                max_new_tokens=5,
                temperature=0.0,
            )
            scheduler_with_model.submit_request(req)
            reqs.append(req)

        for req in reqs:
            tokens = self._collect_tokens(req)
            assert len(tokens) > 0
            assert req.status == RequestStatus.FINISHED

    def test_max_tokens_stops(self, scheduler_with_model):
        req = Request(
            request_id="test-maxlen",
            input_ids=[1, 2, 3],
            max_new_tokens=3,
            temperature=0.0,
        )
        scheduler_with_model.submit_request(req)
        tokens = self._collect_tokens(req)

        assert len(tokens) <= 3
        assert req.status == RequestStatus.FINISHED
        # If it didn't hit EOS naturally, should be "length"
        assert req.finish_reason in ("stop", "length")

    def test_scheduler_start_stop(self, tiny_model):
        sched = Scheduler(
            model=tiny_model,
            config=TINY_CONFIG,
            device=DEVICE,
            dtype=DTYPE,
            max_batch_size=MAX_BATCH,
            max_seq_len=MAX_SEQ,
        )
        sched.start()
        assert sched._running is True
        assert sched._thread is not None
        assert sched._thread.is_alive()

        sched.stop()
        assert sched._running is False
        assert not sched._thread.is_alive()

    def test_slot_reuse(self, scheduler_with_model):
        sched = scheduler_with_model
        if sched.use_flashinfer:
            initial_free = len(sched.kv_cache.free_pages)
        else:
            initial_free = len(sched.kv_cache.free_slots)

        req = Request(
            request_id="test-reuse",
            input_ids=[1, 2, 3],
            max_new_tokens=2,
            temperature=0.0,
        )
        sched.submit_request(req)
        self._collect_tokens(req)

        # After completion, resources should be returned
        # Give scheduler a moment to retire
        time.sleep(0.2)
        if sched.use_flashinfer:
            assert len(sched.kv_cache.free_pages) == initial_free
        else:
            assert len(sched.kv_cache.free_slots) == initial_free


# ---------------------------------------------------------------------------
# PagedKVCache
# ---------------------------------------------------------------------------


class TestPagedKVCache:
    def test_allocate_and_free(self, paged_kv_cache):
        initial_free = len(paged_kv_cache.free_pages)
        pages = paged_kv_cache.allocate_pages(3)
        assert len(pages) == 3
        assert len(set(pages)) == 3  # all unique
        assert len(paged_kv_cache.free_pages) == initial_free - 3

        paged_kv_cache.release_pages(pages)
        assert len(paged_kv_cache.free_pages) == initial_free

    def test_exhaustion(self, paged_kv_cache):
        total = len(paged_kv_cache.free_pages)
        paged_kv_cache.allocate_pages(total)
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            paged_kv_cache.allocate_pages(1)

    def test_kv_data_shape(self, paged_kv_cache):
        assert len(paged_kv_cache.kv_data) == TINY_CONFIG.num_hidden_layers
        for kv in paged_kv_cache.kv_data:
            assert kv.shape == (
                paged_kv_cache.max_num_pages, 2, PAGE_SIZE,
                TINY_CONFIG.num_key_value_heads, TINY_CONFIG.head_dim,
            )

    def test_release_zeros_pages(self, paged_kv_cache):
        pages = paged_kv_cache.allocate_pages(1)
        for kv in paged_kv_cache.kv_data:
            kv[pages[0]] = 1.0
        paged_kv_cache.release_pages(pages)
        for kv in paged_kv_cache.kv_data:
            assert kv[pages[0]].abs().sum() == 0


# ---------------------------------------------------------------------------
# build_csr_metadata
# ---------------------------------------------------------------------------


class TestBuildCSRMetadata:
    def test_single_request(self):
        indptr, indices, last_len = build_csr_metadata(
            page_lists=[[0, 1, 2]],
            seq_lens=[40],
            page_size=16,
            device=DEVICE,
        )
        assert indptr.tolist() == [0, 3]
        assert indices.tolist() == [0, 1, 2]
        assert last_len.tolist() == [8]  # 40 % 16 = 8

    def test_multiple_requests(self):
        indptr, indices, last_len = build_csr_metadata(
            page_lists=[[0, 1], [2, 3, 4]],
            seq_lens=[32, 48],
            page_size=16,
            device=DEVICE,
        )
        assert indptr.tolist() == [0, 2, 5]
        assert indices.tolist() == [0, 1, 2, 3, 4]
        assert last_len.tolist() == [16, 16]  # exact multiples -> page_size

    def test_exact_page_boundary(self):
        indptr, indices, last_len = build_csr_metadata(
            page_lists=[[5]],
            seq_lens=[16],
            page_size=16,
            device=DEVICE,
        )
        assert last_len.tolist() == [16]

    def test_dtype_is_int32(self):
        indptr, indices, last_len = build_csr_metadata(
            page_lists=[[0]],
            seq_lens=[1],
            page_size=16,
            device=DEVICE,
        )
        assert indptr.dtype == torch.int32
        assert indices.dtype == torch.int32
        assert last_len.dtype == torch.int32
