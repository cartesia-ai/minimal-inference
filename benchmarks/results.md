# KV Cache Compression — Benchmark Results

**Model:** Mistral-7B-Instruct-v0.3
**Branch:** `JL_spda` (orthogonal projection)
**Eval:** GSM8K (first 200 problems), max_tokens=512, temperature=0, batch_size=20

---

## GSM8K Accuracy

| kv_proj_dim | Compression | Accuracy | Correct/Total | Errors | Time |
|-------------|-------------|----------|---------------|--------|------|
| 0 (baseline)| 0%          | 55.5%    | 111/200       | 3      | 163s |
| 108         | 15.6%       | 30.0%    | 60/200        | 1      | 179s |
| 96          | 25%         | 11.0%    | 22/200        | 3      | 164s |

## Latency (batch_size=10, max_tokens=500, prompt="Write a short poem about the ocean")

| kv_proj_dim | TTFT (ms) | T2T (ms) | Throughput (tok/s) |
|-------------|-----------|----------|--------------------|
| 0 (baseline)| 566.5     | 33.0     | 255.9              |
| 108         | 258.3     | 35.5     | 261.1              |
| 96          | 236.7     | 33.2     | 286.7              |
