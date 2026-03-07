# minimal-inference

A from-scratch LLM inference engine. Pure PyTorch transformer, continuous batching scheduler, FlashInfer paged attention, CUDA graph decode, and an OpenAI-compatible API — all in ~2500 lines across three files.

## Quickstart

```bash
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./Qwen2.5-7B-Instruct
python server.py
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

Config lives in `config.yaml`:

```yaml
model_path: ./Qwen2.5-7B-Instruct
device: cuda
dtype: bfloat16

scheduler:
  max_seq_len: 4096
  max_batch_size: 48
  page_size: 16
```

## Supported Models

Any HuggingFace checkpoint with a Qwen2 or Mistral architecture. Config auto-loads from `config.json`:
- Qwen2.5 (0.5B through 72B, Instruct variants)
- Mistral-7B-Instruct-v0.3

## Key Optimizations

| Optimization | Impact | Description |
|---|---|---|
| **Fused RMSNorm** | -5ms T2T (-26%) | `F.rms_norm` replaces 5-kernel manual implementation. Eliminated 224 kernel launches per decode step. |
| **CUDA Graph Decode** | -2.5ms T2T (-17%) | Full forward pass captured per batch size. `plan()` runs outside graph, `replay()` inside. Scratch page prevents KV corruption from padded dummy requests. |
| **Batched Sampling** | -1ms T2T (-6%) | Single `multinomial` + `argmax` over all requests. One GPU-CPU sync instead of one per request. |
| **Paged KV Cache** | Memory efficiency | FlashInfer paged attention allocates pages on-demand. Requests only use memory proportional to their actual sequence length. |
| **GQA Head Padding** | Compatibility | Pads Q heads to next power-of-2 group size per KV head for FlashInfer. Enables paged attention for Qwen (group_size=7). |

## Benchmarks

Mistral-7B-Instruct-v0.3, single H100, bfloat16, page_size=16, max_tokens=50.

| Concurrent Requests | TTFT (avg) | T2T (avg/stream) | Aggregate tok/s |
|--------------------:|----------:|-----------------:|----------------:|
| 2 | 370ms | 9.8ms | 109 |
| 20 | 258ms | 11.2ms | 1,114 |
| 32 | 337ms | 12.9ms | 1,532 |

**vs vLLM** (same model, same hardware):

| Batch | Our T2T | vLLM T2T | Our tok/s | vLLM tok/s |
|------:|--------:|---------:|----------:|-----------:|
| 2 | 9.8ms | 6.5ms | 109 | 178 |
| 20 | 11.2ms | 6.1ms | 1,114 | 1,935 |
| 32 | 12.9ms | 6.0ms | 1,532 | 2,928 |

TTFT = time to first token (includes prefill). T2T = average inter-token interval during decode only (excludes first token). These are separated because prefill is a full forward pass over the prompt, while each decode step is a single `[B, 1]` pass.

```bash
python benchmark.py --batch-size 32 --runs 3
```

## Architecture

```
server.py        FastAPI + tokenization + request routing
scheduler.py     Continuous batching: admit → prefill → decode → retire
model.py         From-scratch transformer (GQA, RoPE, SwiGLU, paged attention)
benchmark.py     Async streaming benchmark with concurrent requests
```

## Limitations

- Sequential prefill (one request at a time)
- No prefix caching
- No speculative decoding
- No FP8 quantization

## Chat UI

- Run python server.py on your interactive job
- Server starts on 0.0.0.0:8000
- Run this on your laptop: ssh -L 8000:<JOB_NODE>:8000 USER@b65c909e-hn-0.cloud.together.ai
- Open chat.html on your laptop

<img width="1710" height="1025" alt="Screenshot 2026-03-07 at 6 16 29 AM" src="https://github.com/user-attachments/assets/fb786db7-b413-4b72-a014-4cf152d98bcf" />

