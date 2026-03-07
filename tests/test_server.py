"""Tests for server.py — HTTP API integration tests.

Uses a tiny model + real scheduler so no model download is needed.
"""

import asyncio
import json

import httpx
import pytest
import torch

from model import Qwen2Model
from scheduler import Scheduler
from tests.conftest import DEVICE, DTYPE, MAX_BATCH, MAX_SEQ, PAGE_SIZE, TINY_CONFIG

# Import the FastAPI app and globals we need to patch
import server as server_module


# ---------------------------------------------------------------------------
# Module-level setup: create tiny model + scheduler once
# ---------------------------------------------------------------------------

_original_tokenize = None


def _fake_tokenize(messages):
    """Convert message text to token IDs (simple hash-based)."""
    text = " ".join(m.content for m in messages)
    tokens = [hash(c) % TINY_CONFIG.vocab_size for c in text]
    return tokens if tokens else [1]


class _FakeTokenizer:
    """Minimal tokenizer stub for tests — maps token IDs to 'tN' strings."""

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(f"t{tid}" for tid in token_ids)


@pytest.fixture(scope="module", autouse=True)
def setup_server():
    """Initialize server globals with a tiny model for the entire test module."""
    global _original_tokenize

    model = Qwen2Model(TINY_CONFIG).to(device=DEVICE, dtype=DTYPE).eval()
    sched = Scheduler(
        model=model,
        config=TINY_CONFIG,
        device=DEVICE,
        dtype=DTYPE,
        max_batch_size=MAX_BATCH,
        max_seq_len=MAX_SEQ,
        page_size=PAGE_SIZE,
    )
    sched.start()
    server_module.scheduler = sched
    server_module.tokenizer = _FakeTokenizer()

    _original_tokenize = server_module.tokenize_messages
    server_module.tokenize_messages = _fake_tokenize

    yield

    sched.stop()
    server_module.tokenize_messages = _original_tokenize
    server_module.scheduler = None
    server_module.tokenizer = None


@pytest.fixture
def client():
    """Async HTTP client against the FastAPI app."""
    transport = httpx.ASGITransport(app=server_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _make_body(stream=False, max_tokens=10, temperature=0.0):
    return {
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


class TestNonStreaming:
    async def test_response_schema(self, client):
        resp = await client.post("/v1/chat/completions", json=_make_body())
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

    async def test_choice_structure(self, client):
        resp = await client.post("/v1/chat/completions", json=_make_body())
        data = resp.json()
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in ("stop", "length")

    async def test_usage_math(self, client):
        resp = await client.post("/v1/chat/completions", json=_make_body())
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0

    async def test_finish_length(self, client):
        resp = await client.post("/v1/chat/completions", json=_make_body(max_tokens=1))
        data = resp.json()
        assert data["usage"]["completion_tokens"] <= 1

    async def test_deterministic_greedy(self, client):
        body = _make_body(temperature=0.0, max_tokens=5)
        resp1 = await client.post("/v1/chat/completions", json=body)
        resp2 = await client.post("/v1/chat/completions", json=body)
        text1 = resp1.json()["choices"][0]["message"]["content"]
        text2 = resp2.json()["choices"][0]["message"]["content"]
        assert text1 == text2


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestStreaming:
    async def _collect_stream(self, client, body=None):
        """Send streaming request and return list of raw SSE lines."""
        if body is None:
            body = _make_body(stream=True)
        lines = []
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.strip():
                    lines.append(line)
        return lines

    async def test_sse_format(self, client):
        lines = await self._collect_stream(client)
        for line in lines:
            assert line.startswith("data: "), f"Bad SSE line: {line}"

    async def test_first_chunk_role(self, client):
        lines = await self._collect_stream(client)
        first_data = json.loads(lines[0].removeprefix("data: "))
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    async def test_done_sentinel(self, client):
        lines = await self._collect_stream(client)
        assert lines[-1] == "data: [DONE]"

    async def test_finish_reason(self, client):
        lines = await self._collect_stream(client)
        # Second-to-last data line should have finish_reason set
        last_json_line = lines[-2]
        data = json.loads(last_json_line.removeprefix("data: "))
        assert data["choices"][0]["finish_reason"] in ("stop", "length")


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------


class TestConcurrent:
    async def test_concurrent_http_requests(self, client):
        bodies = [
            _make_body(max_tokens=3, temperature=0.0),
            _make_body(max_tokens=5, temperature=0.0),
            _make_body(max_tokens=4, temperature=0.0),
        ]
        tasks = [client.post("/v1/chat/completions", json=b) for b in bodies]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            assert "choices" in data
            assert data["choices"][0]["finish_reason"] in ("stop", "length")
