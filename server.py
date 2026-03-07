"""
FastAPI server providing an OpenAI-compatible /v1/chat/completions endpoint.

Uses the continuous batching scheduler for concurrent request handling.

Usage:
    python server.py                          # uses config.yaml
    python server.py --config my_config.yaml  # custom config
"""

import argparse
import asyncio
import time
import uuid

import torch
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from model import Qwen2Config, Qwen2Model, load_weights
from scheduler import EOS_TOKEN_IDS, Request, Scheduler

# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible schema)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-0.5b-instruct"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# Streaming chunk models


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ---------------------------------------------------------------------------
# Globals (set during startup)
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen2.5-0.5B Inference Engine")
scheduler: Scheduler | None = None
tokenizer: AutoTokenizer | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def tokenize_messages(messages: list[ChatMessage]) -> list[int]:
    """Apply chat template and tokenize."""
    dicts = [m.model_dump() for m in messages]
    text = tokenizer.apply_chat_template(
        dicts, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    input_ids = tokenize_messages(request.messages)
    loop = asyncio.get_event_loop()

    req = Request(
        request_id=make_id(),
        input_ids=input_ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        loop=loop,
    )
    scheduler.submit_request(req)

    if request.stream:
        return StreamingResponse(
            stream_response(req, request),
            media_type="text/event-stream",
        )
    return await non_stream_response(req, request)


async def non_stream_response(
    req: Request, request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Collect all tokens from the scheduler, then return the full response."""
    all_ids: list[int] = []
    finish_reason = "length"

    while True:
        token_id = await req.token_queue.get()
        if token_id is None:
            finish_reason = req.finish_reason or "length"
            break
        if token_id in EOS_TOKEN_IDS:
            finish_reason = "stop"
            continue
        all_ids.append(token_id)

    text = tokenizer.decode(all_ids, skip_special_tokens=True)

    return ChatCompletionResponse(
        id=req.request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=len(req.input_ids),
            completion_tokens=len(all_ids),
            total_tokens=len(req.input_ids) + len(all_ids),
        ),
    )


async def stream_response(req: Request, request: ChatCompletionRequest):
    """Read tokens from the scheduler's queue and stream as SSE."""
    request_id = req.request_id
    created = int(time.time())

    def _make_chunk(delta: DeltaMessage, finish: str | None = None) -> str:
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=request.model,
            choices=[StreamChoice(delta=delta, finish_reason=finish)],
        )
        return f"data: {chunk.model_dump_json()}\n\n"

    # First chunk: role
    yield _make_chunk(DeltaMessage(role="assistant"))

    # Incremental decoding from scheduler's token queue
    all_ids: list[int] = []
    prev_text = ""
    finish_reason = "length"

    while True:
        token_id = await req.token_queue.get()
        if token_id is None:
            finish_reason = req.finish_reason or "length"
            break
        if token_id in EOS_TOKEN_IDS:
            finish_reason = "stop"
            continue

        all_ids.append(token_id)
        full_text = tokenizer.decode(all_ids, skip_special_tokens=True)
        delta_text = full_text[len(prev_text) :]
        prev_text = full_text

        if delta_text:
            yield _make_chunk(DeltaMessage(content=delta_text))

    # Final chunk with finish_reason
    yield _make_chunk(DeltaMessage(), finish=finish_reason)
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def init_engine(config: dict):
    global scheduler, tokenizer

    model_path = config["model_path"]
    device = config["device"]
    dtype_str = config["dtype"]
    max_seq_len = config["scheduler"]["max_seq_len"]
    max_batch_size = config["scheduler"]["max_batch_size"]
    page_size = config["scheduler"]["page_size"]

    dtype = getattr(torch, dtype_str)
    print(f"Loading model from {model_path} on {device} ({dtype_str})...")

    model_config = Qwen2Config()
    model = Qwen2Model(model_config)
    load_weights(model, model_path, device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype).eval()

    scheduler = Scheduler(
        model=model,
        config=model_config,
        device=torch.device(device),
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
    )
    scheduler.start()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Engine ready (max_batch_size={max_batch_size}, page_size={page_size}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B Inference Server")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    init_engine(config)
    uvicorn.run(
        app,
        host=config["server"]["host"],
        port=config["server"]["port"],
    )


if __name__ == "__main__":
    main()
