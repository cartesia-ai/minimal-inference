"""
Continuous batching scheduler with paged KV cache (FlashInfer) or padded fallback.

Runs a step loop in a background thread. Requests submit via queue,
get tokens back via per-request asyncio.Queue.

When FlashInfer is available and the device is CUDA, the scheduler uses
paged attention for memory-efficient KV cache management. Otherwise it
falls back to the padded batching path a padded batching path.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from model import ModelConfig, Model

# ---------------------------------------------------------------------------
# Optional FlashInfer import
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

try:
    import flashinfer
    import flashinfer.page

    _FLASHINFER_AVAILABLE = True
    logger.info("FlashInfer is available (version %s)", getattr(flashinfer, "__version__", "unknown"))
except ImportError:
    _FLASHINFER_AVAILABLE = False
    logger.info("FlashInfer not found; will use padded attention fallback")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EOS_TOKEN_IDS: set[int] = {151645, 151643}  # default; overridden by Scheduler


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    input_ids: list[int]
    max_new_tokens: int
    temperature: float

    # Mutable state
    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    current_position: int = 0  # next write position in KV cache

    # Padded path (fallback)
    batch_slot: int = -1  # index in BatchedKVCache

    # Paged path (FlashInfer)
    pages: list[int] = field(default_factory=list)  # allocated page indices

    finish_reason: str | None = None

    # Timing
    prefill_done_time: float = 0.0  # time.perf_counter() after first token
    last_token_time: float = 0.0  # updated after each decode token

    # Communication back to async server
    token_queue: asyncio.Queue[int | None] = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = None


# ---------------------------------------------------------------------------
# Batched KV Cache (padded fallback)
# ---------------------------------------------------------------------------


class BatchedKVCache:
    """Pre-allocated KV cache with fixed batch slots.

    Shape per layer: [max_batch_size, num_kv_heads, max_seq_len, dim]
    where dim is head_dim normally, or kv_proj_dim when JL compression is active.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # When JL projection is enabled, K is stored in the projected
        # dimension, saving memory. V stays at full head_dim.
        k_dim = config.kv_proj_dim if config.kv_proj_dim > 0 else config.head_dim
        v_dim = config.head_dim

        self.caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(config.num_hidden_layers):
            k = torch.zeros(
                max_batch_size, config.num_key_value_heads, max_seq_len, k_dim,
                device=device, dtype=dtype,
            )
            v = torch.zeros(
                max_batch_size, config.num_key_value_heads, max_seq_len, v_dim,
                device=device, dtype=dtype,
            )
            self.caches.append((k, v))

        self.free_slots: list[int] = list(range(max_batch_size))

    def allocate_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("No free KV cache slots")
        return self.free_slots.pop(0)

    def release_slot(self, slot: int) -> None:
        for k_cache, v_cache in self.caches:
            k_cache[slot].zero_()
            v_cache[slot].zero_()
        self.free_slots.append(slot)

    def get_slot_caches(self, slot: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return a view of a single slot's caches for prefill (no copy)."""
        return [(k[slot : slot + 1], v[slot : slot + 1]) for k, v in self.caches]

    def get_batch_caches(
        self, slots: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return indexed caches for a batch of slots (copies via fancy index)."""
        return [(k[slots], v[slots]) for k, v in self.caches]

    def write_back_decode(
        self,
        slots: list[int],
        positions: list[int],
        batch_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Write back the single new KV entry per slot from decode copies."""
        for layer_idx, (k_master, v_master) in enumerate(self.caches):
            k_batch, v_batch = batch_caches[layer_idx]
            for i, (slot, pos) in enumerate(zip(slots, positions)):
                k_master[slot, :, pos, :] = k_batch[i, :, pos, :]
                v_master[slot, :, pos, :] = v_batch[i, :, pos, :]


# ---------------------------------------------------------------------------
# Paged KV Cache (FlashInfer)
# ---------------------------------------------------------------------------


class PagedKVCache:
    """Paged KV cache with FlashInfer-compatible layout.

    KV data per layer: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
    The "2" dimension stores K at index 0 and V at index 1.

    Pages are allocated on-demand from a free list, like virtual memory.
    """

    def __init__(
        self,
        config: ModelConfig,
        max_batch_size: int,
        max_seq_len: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.page_size = page_size
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        max_pages_per_seq = math.ceil(max_seq_len / page_size)
        self.max_num_pages = max_batch_size * max_pages_per_seq

        # One [max_pages, 2, page_size, kv_heads, head_dim] tensor per layer
        self.kv_data: list[torch.Tensor] = [
            torch.zeros(
                self.max_num_pages, 2, page_size,
                config.num_key_value_heads, config.head_dim,
                device=device, dtype=dtype,
            )
            for _ in range(self.num_layers)
        ]

        # Reserve last page as scratch for CUDA graph padding (never allocated to real requests)
        self.scratch_page = self.max_num_pages - 1
        self.free_pages: deque[int] = deque(range(self.max_num_pages - 1))

    def allocate_pages(self, num_pages: int) -> list[int]:
        """Allocate pages from the free list."""
        if len(self.free_pages) < num_pages:
            raise RuntimeError(
                f"Cannot allocate {num_pages} pages, only {len(self.free_pages)} free"
            )
        return [self.free_pages.popleft() for _ in range(num_pages)]

    def release_pages(self, page_indices: list[int]) -> None:
        """Return pages to the free list and zero them out."""
        for idx in page_indices:
            for layer_kv in self.kv_data:
                layer_kv[idx].zero_()
            self.free_pages.append(idx)


# ---------------------------------------------------------------------------
# FlashInfer attention backend (passed through model forward)
# ---------------------------------------------------------------------------


@dataclass
class AttentionBackend:
    """Carries FlashInfer state through the model's forward pass.

    Created fresh each scheduler step, consumed by Attention layers.
    """

    paged_kv_cache: PagedKVCache
    wrapper: object  # Prefill or Decode wrapper (already plan()'d)
    mode: str  # "prefill" or "decode"

    # CSR page-table metadata (for append_paged_kv_cache)
    kv_page_indptr: torch.Tensor  # [batch_size + 1], int32
    kv_page_indices: torch.Tensor  # [total_pages], int32
    kv_last_page_len: torch.Tensor  # [batch_size], int32

    # For KV append: maps new tokens to requests
    append_indptr: torch.Tensor  # [batch_size + 1], int32

    # For KV append: batch index and position of each appended token
    batch_indices: torch.Tensor  # [total_tokens], int32
    positions: torch.Tensor  # [total_tokens], int32

    # Layer counter — each Attention.forward() call advances this
    _layer_idx: int = 0

    def next_layer(self) -> int:
        """Return current layer index and advance to next."""
        idx = self._layer_idx
        self._layer_idx += 1
        return idx


# ---------------------------------------------------------------------------
# CSR metadata builder
# ---------------------------------------------------------------------------


def build_csr_metadata(
    page_lists: list[list[int]],
    seq_lens: list[int],
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CSR page-table arrays for FlashInfer.

    Args:
        page_lists: list of page index lists, one per request.
        seq_lens:   list of sequence lengths, one per request.
        page_size:  tokens per page.
        device:     target device.

    Returns:
        kv_page_indptr:   [batch_size + 1] int32
        kv_page_indices:  [total_pages] int32
        kv_last_page_len: [batch_size] int32
    """
    indptr = [0]
    flat_indices: list[int] = []
    last_page_lens: list[int] = []

    for pages, seq_len in zip(page_lists, seq_lens):
        flat_indices.extend(pages)
        indptr.append(len(flat_indices))
        last_len = seq_len % page_size
        if last_len == 0 and seq_len > 0:
            last_len = page_size
        last_page_lens.append(last_len)

    return (
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(flat_indices, dtype=torch.int32, device=device),
        torch.tensor(last_page_lens, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample(logits: torch.Tensor, temperature: float) -> int:
    """Sample from logits [1, seq_len, vocab_size] for a single request."""
    if temperature <= 0:
        return logits[0, -1].argmax(dim=-1).item()
    scaled = logits[0, -1] / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_batch(logits: torch.Tensor, temperatures: list[float]) -> list[int]:
    """Sample from logits [B, 1, vocab_size] for a batch of requests.

    Single GPU sync instead of one per request.
    """
    last_logits = logits[:, -1, :]  # [B, vocab_size]

    # Build temperature tensor; 0 means greedy
    temps = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
    greedy_mask = temps <= 0

    # Scale by temperature (avoid div-by-zero for greedy)
    temps = temps.clamp(min=1e-6).unsqueeze(1)  # [B, 1]
    scaled = last_logits / temps
    probs = F.softmax(scaled, dim=-1, dtype=torch.float32)

    # Sample all at once
    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

    # Override with argmax for greedy requests
    if greedy_mask.any():
        greedy_tokens = last_logits.argmax(dim=-1)
        sampled[greedy_mask] = greedy_tokens[greedy_mask]

    return sampled.tolist()  # single GPU->CPU sync


# ---------------------------------------------------------------------------
# CUDA Graph Runner
# ---------------------------------------------------------------------------

GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]


class CUDAGraphRunner:
    """Captures and replays a CUDA graph for a fixed decode batch size."""

    def __init__(self, batch_size: int, max_num_pages: int, scratch_page: int, device: torch.device):
        self.batch_size = batch_size
        self.device = device
        self._scratch_page = scratch_page
        self.graph: torch.cuda.CUDAGraph | None = None

        # Static input buffers (filled before replay)
        self.input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Static metadata buffers for FlashInfer append_paged_kv_cache
        # Initialize kv_page_indices to scratch_page so dummy padding writes
        # go to the scratch page instead of corrupting real requests' KV cache
        self.kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.kv_page_indices = torch.full((max_num_pages,), scratch_page, dtype=torch.int32, device=device)
        self.kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)
        self.append_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        self.batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        self.positions = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # Output buffer (set during capture)
        self.logits: torch.Tensor | None = None

    def capture(
        self,
        model: Model,
        decode_wrapper,
        paged_kv_cache,
        padded_num_qo_heads: int,
        config: ModelConfig,
        page_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Warmup and capture the decode forward pass."""
        backend = AttentionBackend(
            paged_kv_cache=paged_kv_cache,
            wrapper=decode_wrapper,
            mode="decode",
            kv_page_indptr=self.kv_page_indptr,
            kv_page_indices=self.kv_page_indices,
            kv_last_page_len=self.kv_last_page_len,
            append_indptr=self.append_indptr,
            batch_indices=self.batch_indices,
            positions=self.positions,
        )

        # Plan with dummy metadata for this batch size
        decode_wrapper.plan(
            indptr=self.kv_page_indptr,
            indices=self.kv_page_indices[:self.batch_size],
            last_page_len=self.kv_last_page_len,
            num_qo_heads=padded_num_qo_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            data_type=dtype,
        )

        # Warmup on a side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                backend._layer_idx = 0
                logits = model(
                    self.input_ids,
                    position_ids=self.position_ids,
                    attn_backend=backend,
                )
        s.synchronize()
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            backend._layer_idx = 0
            self.logits = model(
                self.input_ids,
                position_ids=self.position_ids,
                attn_backend=backend,
            )

        self._backend = backend
        logger.info("Captured CUDA graph for batch_size=%d", self.batch_size)

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Copy new data into static buffers and replay the graph."""
        # Copy inputs
        self.input_ids[:input_ids.shape[0]].copy_(input_ids)
        self.position_ids[:position_ids.shape[0]].copy_(position_ids)

        # Copy metadata
        self.kv_page_indptr[:kv_page_indptr.shape[0]].copy_(kv_page_indptr)
        self.kv_page_indices[:kv_page_indices.shape[0]].copy_(kv_page_indices)
        # Reset tail to scratch_page so stale indices from previous steps
        # don't point to pages now owned by different requests
        n = kv_page_indices.shape[0]
        if n < self.kv_page_indices.shape[0]:
            self.kv_page_indices[n:].fill_(self._scratch_page)
        self.kv_last_page_len[:kv_last_page_len.shape[0]].copy_(kv_last_page_len)
        self.batch_indices[:batch_indices.shape[0]].copy_(batch_indices)
        self.positions[:positions.shape[0]].copy_(positions)

        # Reset layer counter
        self._backend._layer_idx = 0

        # Replay
        self.graph.replay()
        return self.logits


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    def __init__(
        self,
        model: Model,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        max_batch_size: int = 8,
        max_seq_len: int = 4096,
        page_size: int = 16,
        eos_token_ids: set[int] | None = None,
    ):
        global EOS_TOKEN_IDS
        if eos_token_ids is not None:
            EOS_TOKEN_IDS = eos_token_ids
            logger.info("EOS token IDs: %s", EOS_TOKEN_IDS)

        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.page_size = page_size

        # Decide which attention backend to use.
        # JL projection needs K at proj_dim and V at head_dim — incompatible with
        # FlashInfer's paged KV cache which requires a single head_dim for both.
        self.use_flashinfer = (
            device.type == "cuda"
            and _FLASHINFER_AVAILABLE
            and config.kv_proj_dim == 0
        )
        if device.type == "cuda" and _FLASHINFER_AVAILABLE and config.kv_proj_dim > 0:
            logger.info("JL projection enabled (kv_proj_dim=%d) — using padded SDPA fallback", config.kv_proj_dim)

        # Pad Q heads to next power-of-2 group size for FlashInfer GQA compatibility
        gqa_group_size = config.num_attention_heads // config.num_key_value_heads
        padded_group = 1 << (gqa_group_size - 1).bit_length()
        self.padded_num_qo_heads = padded_group * config.num_key_value_heads
        if self.padded_num_qo_heads != config.num_attention_heads:
            logger.info(
                "Padding Q heads %d -> %d for FlashInfer (group_size %d -> %d)",
                config.num_attention_heads, self.padded_num_qo_heads,
                gqa_group_size, padded_group,
            )

        if self.use_flashinfer:
            logger.info("Using FlashInfer paged attention backend (device=%s)", device)
            self.kv_cache = PagedKVCache(
                config, max_batch_size, max_seq_len, page_size, device, dtype
            )
            # FlashInfer workspace buffers (128 MB each)
            workspace_size = 128 * 1024 * 1024
            self._prefill_workspace = torch.empty(
                workspace_size, dtype=torch.uint8, device=device
            )
            self._decode_workspace = torch.empty(
                workspace_size, dtype=torch.uint8, device=device
            )
            self._prefill_wrapper = (
                flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                    self._prefill_workspace, kv_layout="NHD"
                )
            )
            self._decode_wrapper = (
                flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                    self._decode_workspace, kv_layout="NHD"
                )
            )
        else:
            logger.info("Using padded attention fallback (device=%s, flashinfer_available=%s)", device, _FLASHINFER_AVAILABLE)
            self.kv_cache = BatchedKVCache(
                config, max_batch_size, max_seq_len, device, dtype
            )

        # CUDA graph runners for decode (only for FlashInfer path)
        self._graph_runners: dict[int, CUDAGraphRunner] = {}
        if self.use_flashinfer:
            max_pages = self.kv_cache.max_num_pages
            for bs in GRAPH_BATCH_SIZES:
                if bs > max_batch_size:
                    break
                runner = CUDAGraphRunner(bs, max_pages, self.kv_cache.scratch_page, device)
                runner.capture(
                    model, self._decode_wrapper, self.kv_cache,
                    self.padded_num_qo_heads, config, page_size, dtype,
                )
                self._graph_runners[bs] = runner
            logger.info("Captured CUDA graphs for batch sizes: %s", list(self._graph_runners.keys()))

        self.waiting_queue: deque[Request] = deque()
        self.active_requests: list[Request] = []
        self.lock = threading.Lock()
        self.new_request_event = threading.Event()

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.new_request_event.set()
        if self._thread:
            self._thread.join()

    def submit_request(self, request: Request) -> None:
        with self.lock:
            self.waiting_queue.append(request)
        self.new_request_event.set()

    # -------------------------------------------------------------------
    # Step loop
    # -------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._running:
            if not self.active_requests and not self.waiting_queue:
                self.new_request_event.wait(timeout=0.1)
                self.new_request_event.clear()
                continue

            self._admit_new_requests()
            self._prefill_new_requests()
            if any(r.status == RequestStatus.DECODING for r in self.active_requests):
                self._decode_step()
            self._retire_finished_requests()

    # -------------------------------------------------------------------
    # Admit
    # -------------------------------------------------------------------

    def _admit_new_requests(self) -> None:
        if self.use_flashinfer:
            self._admit_paged()
        else:
            self._admit_padded()

    def _admit_padded(self) -> None:
        with self.lock:
            while (
                self.waiting_queue
                and len(self.active_requests) < self.max_batch_size
                and self.kv_cache.free_slots
            ):
                req = self.waiting_queue.popleft()
                req.batch_slot = self.kv_cache.allocate_slot()
                req.status = RequestStatus.PREFILLING
                self.active_requests.append(req)

    def _admit_paged(self) -> None:
        with self.lock:
            while (
                self.waiting_queue
                and len(self.active_requests) < self.max_batch_size
            ):
                req = self.waiting_queue[0]
                pages_needed = math.ceil(len(req.input_ids) / self.page_size)
                if len(self.kv_cache.free_pages) < pages_needed:
                    break
                self.waiting_queue.popleft()
                req.pages = self.kv_cache.allocate_pages(pages_needed)
                req.status = RequestStatus.PREFILLING
                self.active_requests.append(req)

    # -------------------------------------------------------------------
    # Prefill
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def _prefill_new_requests(self) -> None:
        for req in self.active_requests:
            if req.status != RequestStatus.PREFILLING:
                continue

            if self.use_flashinfer:
                self._prefill_paged(req)
            else:
                self._prefill_padded(req)

    def _prefill_padded(self, req: Request) -> None:
        prompt_len = len(req.input_ids)
        input_ids = torch.tensor(
            [req.input_ids], dtype=torch.long, device=self.device
        )
        position_ids = torch.arange(
            prompt_len, device=self.device, dtype=torch.long
        ).unsqueeze(0)

        slot_caches = self.kv_cache.get_slot_caches(req.batch_slot)

        logits = self.model(
            input_ids,
            kv_caches=slot_caches,
            position_ids=position_ids,
        )

        next_token = sample(logits, req.temperature)
        req.generated_tokens.append(next_token)
        req.current_position = prompt_len
        req.status = RequestStatus.DECODING
        req.prefill_done_time = time.perf_counter()
        req.last_token_time = req.prefill_done_time

        self._push_token(req, next_token)

        if next_token in EOS_TOKEN_IDS or len(req.generated_tokens) >= req.max_new_tokens:
            req.status = RequestStatus.FINISHED
            req.finish_reason = "stop" if next_token in EOS_TOKEN_IDS else "length"

    def _prefill_paged(self, req: Request) -> None:
        prompt_len = len(req.input_ids)
        input_ids = torch.tensor(
            [req.input_ids], dtype=torch.long, device=self.device
        )
        position_ids = torch.arange(
            prompt_len, device=self.device, dtype=torch.long
        ).unsqueeze(0)

        # Build CSR metadata for this single request
        kv_page_indptr, kv_page_indices, kv_last_page_len = build_csr_metadata(
            [req.pages], [prompt_len], self.page_size, self.device
        )

        # qo_indptr: single request with prompt_len query tokens
        qo_indptr = torch.tensor(
            [0, prompt_len], dtype=torch.int32, device=self.device
        )

        # Plan the prefill wrapper
        self._prefill_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_page_indptr,
            paged_kv_indices=kv_page_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=self.padded_num_qo_heads,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim_qk=self.config.head_dim,
            page_size=self.page_size,
            causal=True,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
        )

        # batch_indices: all tokens belong to request 0; positions: 0..prompt_len-1
        batch_indices = torch.zeros(prompt_len, dtype=torch.int32, device=self.device)
        positions_append = torch.arange(prompt_len, dtype=torch.int32, device=self.device)

        # Build attention backend
        backend = AttentionBackend(
            paged_kv_cache=self.kv_cache,
            wrapper=self._prefill_wrapper,
            mode="prefill",
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            append_indptr=qo_indptr,  # same as qo_indptr for prefill
            batch_indices=batch_indices,
            positions=positions_append,
        )

        logits = self.model(
            input_ids,
            position_ids=position_ids,
            attn_backend=backend,
        )

        next_token = sample(logits, req.temperature)
        req.generated_tokens.append(next_token)
        req.current_position = prompt_len
        req.status = RequestStatus.DECODING
        req.prefill_done_time = time.perf_counter()
        req.last_token_time = req.prefill_done_time

        self._push_token(req, next_token)

        if next_token in EOS_TOKEN_IDS or len(req.generated_tokens) >= req.max_new_tokens:
            req.status = RequestStatus.FINISHED
            req.finish_reason = "stop" if next_token in EOS_TOKEN_IDS else "length"

    # -------------------------------------------------------------------
    # Decode
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def _decode_step(self) -> None:
        if self.use_flashinfer:
            self._decode_paged()
        else:
            self._decode_padded()

    def _decode_padded(self) -> None:
        decoding = [r for r in self.active_requests if r.status == RequestStatus.DECODING]
        if not decoding:
            return

        batch_size = len(decoding)
        slots = [r.batch_slot for r in decoding]
        positions = [r.current_position for r in decoding]

        # Build input_ids [B, 1]
        last_tokens = [r.generated_tokens[-1] for r in decoding]
        input_ids = torch.tensor(
            last_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        # Build position_ids [B, 1]
        position_ids = torch.tensor(
            positions, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        # Gather KV caches and trim to max needed length
        max_kv_len = max(positions) + 1
        batch_caches = self.kv_cache.get_batch_caches(slots)
        trimmed_caches = [
            (k[:, :, :max_kv_len].contiguous(), v[:, :, :max_kv_len].contiguous())
            for k, v in batch_caches
        ]

        # Build attention mask [B, 1, 1, max_kv_len]
        attn_mask = torch.zeros(
            batch_size, 1, 1, max_kv_len, device=self.device, dtype=self.dtype
        )
        for i, pos in enumerate(positions):
            valid_len = pos + 1
            if valid_len < max_kv_len:
                attn_mask[i, 0, 0, valid_len:] = float("-inf")

        # Forward pass
        logits = self.model(
            input_ids,
            kv_caches=trimmed_caches,
            position_ids=position_ids,
            attention_mask=attn_mask,
        )

        # Write back new KV entries to master cache
        self.kv_cache.write_back_decode(slots, positions, trimmed_caches)

        # Batched sampling: single GPU->CPU sync for all requests
        temperatures = [r.temperature for r in decoding]
        next_tokens = sample_batch(logits, temperatures)

        now = time.perf_counter()
        for i, (req, next_token) in enumerate(zip(decoding, next_tokens)):
            req.generated_tokens.append(next_token)
            req.current_position += 1
            req.last_token_time = now

            self._push_token(req, next_token)

            if (
                next_token in EOS_TOKEN_IDS
                or len(req.generated_tokens) >= req.max_new_tokens
                or req.current_position >= self.max_seq_len
            ):
                req.status = RequestStatus.FINISHED
                req.finish_reason = (
                    "stop" if next_token in EOS_TOKEN_IDS else "length"
                )

    def _decode_paged(self) -> None:
        decoding = [r for r in self.active_requests if r.status == RequestStatus.DECODING]
        if not decoding:
            return

        batch_size = len(decoding)

        # Allocate new pages if any request crosses a page boundary
        for req in decoding:
            pages_needed = math.ceil((req.current_position + 1) / self.page_size)
            if pages_needed > len(req.pages):
                new_pages = self.kv_cache.allocate_pages(pages_needed - len(req.pages))
                req.pages.extend(new_pages)

        # Build input_ids [B, 1]
        last_tokens = [r.generated_tokens[-1] for r in decoding]
        input_ids = torch.tensor(
            last_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        # Build position_ids [B, 1]
        positions = [r.current_position for r in decoding]
        position_ids = torch.tensor(
            positions, dtype=torch.long, device=self.device
        ).unsqueeze(1)

        # Sequence lengths include the new token about to be appended
        seq_lens = [r.current_position + 1 for r in decoding]
        page_lists = [r.pages for r in decoding]

        # Build CSR metadata
        kv_page_indptr, kv_page_indices, kv_last_page_len = build_csr_metadata(
            page_lists, seq_lens, self.page_size, self.device
        )

        # append_indptr: each request appends 1 token
        append_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        )

        # batch_indices: one token per request; positions: current_position for each
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=self.device)
        positions_append = torch.tensor(
            [r.current_position for r in decoding], dtype=torch.int32, device=self.device
        )

        # Find the nearest captured graph batch size
        # TEMPORARILY DISABLED — plan()/replay() interaction causes corruption
        padded_bs = 0
        runner = None

        if runner is not None:
            # Pad inputs to the captured batch size
            if batch_size < padded_bs:
                pad = padded_bs - batch_size
                input_ids = F.pad(input_ids, (0, 0, 0, pad))
                position_ids = F.pad(position_ids, (0, 0, 0, pad))
                # Pad CSR metadata: extend indptr with repeated last value
                last_val = kv_page_indptr[-1]
                kv_page_indptr = F.pad(kv_page_indptr, (0, pad), value=last_val.item())
                kv_last_page_len = F.pad(kv_last_page_len, (0, pad), value=1)
                batch_indices = torch.arange(padded_bs, dtype=torch.int32, device=self.device)
                positions_append = F.pad(positions_append, (0, pad))

            # Copy metadata into runner's static buffers FIRST, then plan
            # (plan reads from these buffers, graph replay will too)
            runner.kv_page_indptr[:kv_page_indptr.shape[0]].copy_(kv_page_indptr)
            runner.kv_page_indices[:kv_page_indices.shape[0]].copy_(kv_page_indices)
            # Reset tail to scratch_page BEFORE plan() so dummy padding
            # slots don't reference stale pages from previous steps
            n_idx = kv_page_indices.shape[0]
            if n_idx < runner.kv_page_indices.shape[0]:
                runner.kv_page_indices[n_idx:].fill_(self.kv_cache.scratch_page)
            runner.kv_last_page_len[:kv_last_page_len.shape[0]].copy_(kv_last_page_len)

            # Plan with the runner's static buffers (outside graph)
            self._decode_wrapper.plan(
                indptr=runner.kv_page_indptr[:padded_bs + 1],
                indices=runner.kv_page_indices[:kv_page_indices.shape[0]],
                last_page_len=runner.kv_last_page_len[:padded_bs],
                num_qo_heads=self.padded_num_qo_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                page_size=self.page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                data_type=self.dtype,
            )

            # Replay the graph (copies same data into static buffers + resets tail)
            logits = runner.replay(
                input_ids, position_ids,
                kv_page_indptr, kv_page_indices, kv_last_page_len,
                batch_indices, positions_append,
            )
            logits = logits[:batch_size]  # strip padding
        else:
            # Fallback: eager mode for batch sizes without a captured graph
            self._decode_wrapper.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_len,
                num_qo_heads=self.padded_num_qo_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                page_size=self.page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.dtype,
                data_type=self.dtype,
            )
            backend = AttentionBackend(
                paged_kv_cache=self.kv_cache,
                wrapper=self._decode_wrapper,
                mode="decode",
                kv_page_indptr=kv_page_indptr,
                kv_page_indices=kv_page_indices,
                kv_last_page_len=kv_last_page_len,
                append_indptr=append_indptr,
                batch_indices=batch_indices,
                positions=positions_append,
            )
            logits = self.model(
                input_ids,
                position_ids=position_ids,
                attn_backend=backend,
            )

        # No write-back needed — KV written directly into paged cache

        # Batched sampling: single GPU->CPU sync for all requests
        temperatures = [r.temperature for r in decoding]
        next_tokens = sample_batch(logits, temperatures)

        now = time.perf_counter()
        for i, (req, next_token) in enumerate(zip(decoding, next_tokens)):
            req.generated_tokens.append(next_token)
            req.current_position += 1
            req.last_token_time = now

            self._push_token(req, next_token)

            if (
                next_token in EOS_TOKEN_IDS
                or len(req.generated_tokens) >= req.max_new_tokens
                or req.current_position >= self.max_seq_len
            ):
                req.status = RequestStatus.FINISHED
                req.finish_reason = (
                    "stop" if next_token in EOS_TOKEN_IDS else "length"
                )

    # -------------------------------------------------------------------
    # Retire
    # -------------------------------------------------------------------

    def _retire_finished_requests(self) -> None:
        still_active = []
        for req in self.active_requests:
            if req.status == RequestStatus.FINISHED:
                num_tokens = len(req.generated_tokens)
                if num_tokens > 1 and req.prefill_done_time > 0:
                    decode_time = req.last_token_time - req.prefill_done_time
                    avg_t2t = decode_time / (num_tokens - 1) * 1000  # ms
                    logger.info(
                        "Request %s finished: %d tokens, avg token-to-token %.1f ms (%.1f tok/s)",
                        req.request_id, num_tokens, avg_t2t, 1000 / avg_t2t,
                    )
                self._push_token(req, None)  # sentinel
                if self.use_flashinfer:
                    self.kv_cache.release_pages(req.pages)
                else:
                    self.kv_cache.release_slot(req.batch_slot)
            else:
                still_active.append(req)
        self.active_requests = still_active

    def _push_token(self, req: Request, token: int | None) -> None:
        if req.loop is not None:
            req.loop.call_soon_threadsafe(req.token_queue.put_nowait, token)
        else:
            req.token_queue.put_nowait(token)
