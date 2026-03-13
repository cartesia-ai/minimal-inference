# JL Attention Benchmark Results

Mistral-7B-Instruct-v0.3, single H100, bfloat16, head_dim=128.

## GSM8K Accuracy (eval_gsm8k.py, two-turn, n=200)

| Metric   | Baseline | kv_proj=96 | kv_proj=110 |
|---------:|---------:|-----------:|------------:|
| Accuracy | 56.5%    |            |             |
| Avg T2T  | 16.0ms   |            |             |
