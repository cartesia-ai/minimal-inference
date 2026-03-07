# minimal-inference

A from-scratch LLM inference engine for Qwen2.5 models. Implements the full transformer architecture in PyTorch with no HuggingFace model classes, a continuous batching scheduler, and an OpenAI-compatible API server.

Supports two attention backends:
- **FlashInfer** (GPU): Paged KV cache with fused attention kernels — memory-efficient, scales to long sequences
- **PyTorch SDPA** (CPU/fallback): Padded batched KV cache with manual attention

## Architecture

```
HTTP Clients (concurrent)
    │  POST /v1/chat/completions
    ▼
server.py        FastAPI, tokenization, submit to scheduler queue
    │
    ▼
scheduler.py     Background thread step loop, continuous batching
    │             Admit → Prefill → Batched Decode → Retire
    ▼
model.py         From-scratch Qwen2 transformer (GQA, RoPE, SwiGLU)
```

**Scheduler step loop** (runs continuously):
1. **Admit** — dequeue waiting requests, allocate KV cache pages (or slots)
2. **Prefill** — run full prompt through model (bsz=1 per new request)
3. **Decode** — batch all active requests into one `[B, 1]` forward pass
4. **Retire** — free resources for finished requests, push sentinel

## Quickstart

```bash
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
```

Edit `config.yaml` to set your model path, device, and scheduler settings:

```yaml
model_path: ./Qwen2.5-0.5B-Instruct
device: cuda
dtype: bfloat16

server:
  host: 0.0.0.0
  port: 8000

scheduler:
  max_seq_len: 4096
  max_batch_size: 8
  page_size: 16
```

```bash
python server.py                          # uses config.yaml
python server.py --config my_config.yaml  # custom config
```

**Send concurrent requests:**
```bash
for i in 1 2 3; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Count to $i\"}], \"temperature\": 0, \"max_tokens\": 50}" &
done
wait
```

**Run tests:**
```bash
pytest tests/ -v
```

## Paged vs Padded KV Cache

**Padded batching** (CPU fallback) pre-allocates `[max_seq_len]` per request regardless of actual length. Wastes memory with long sequences or many concurrent requests.

**Paged attention** (GPU with FlashInfer) allocates KV cache in small pages on-demand — a request at token 100 only uses ~100 tokens of pages, not the full max. Pages are freed and reused dynamically:

```
Req A (100 tokens):  [page 0][page 1][page 2][page 3]...
Req B (2000 tokens): [page 7][page 8][page 9]...[page 131]
Req C (50 tokens):   [page 4][page 5][page 6]
                     ↑ pages allocated from shared pool
```

## Limitations

- Sequential prefill (one request at a time)
- No prefix caching
- No speculative decoding
