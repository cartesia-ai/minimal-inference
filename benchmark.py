"""
Benchmark token-to-token latency against an OpenAI-compatible server.

Usage:
    # Single request
    python benchmark.py --url http://localhost:8000

    # Concurrent requests (batch_size=4)
    python benchmark.py --url http://localhost:8000 --batch-size 4

    # Against vLLM
    python benchmark.py --url http://localhost:8001 --model Qwen/Qwen2.5-7B-Instruct --batch-size 8
"""

import argparse
import asyncio
import json
import time

import httpx


async def stream_one_request(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict,
    request_id: int,
) -> dict:
    """Send one streaming request and collect timing stats."""
    token_times: list[float] = []
    start = time.perf_counter()
    first_token_time = None
    num_tokens = 0

    async with client.stream("POST", endpoint, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                num_tokens += 1

    total = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else 0

    t2t_intervals = [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]
    avg_t2t_ms = (sum(t2t_intervals) / len(t2t_intervals) * 1000) if t2t_intervals else 0

    return {
        "request_id": request_id,
        "num_tokens": num_tokens,
        "ttft": ttft,
        "avg_t2t_ms": avg_t2t_ms,
        "total": total,
    }


async def benchmark(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    batch_size: int,
    num_runs: int,
) -> None:
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    all_t2t: list[float] = []
    all_ttft: list[float] = []
    all_throughput: list[float] = []

    for run in range(1, num_runs + 1):
        print(f"Run {run}: sending {batch_size} concurrent request(s)...")
        batch_start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120) as client:
            tasks = [
                stream_one_request(client, endpoint, payload, i)
                for i in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)

        batch_total = time.perf_counter() - batch_start
        total_tokens = sum(r["num_tokens"] for r in results)
        batch_throughput = total_tokens / batch_total

        for r in results:
            if r["ttft"] > 0:
                all_ttft.append(r["ttft"])
            if r["avg_t2t_ms"] > 0:
                all_t2t.append(r["avg_t2t_ms"])

            print(
                f"  req {r['request_id']}: {r['num_tokens']} tokens, "
                f"TTFT {r['ttft']*1000:.1f}ms, "
                f"avg T2T {r['avg_t2t_ms']:.1f}ms, "
                f"total {r['total']:.2f}s"
            )

        all_throughput.append(batch_throughput)
        print(
            f"  batch: {total_tokens} tokens in {batch_total:.2f}s "
            f"({batch_throughput:.1f} tok/s aggregate)\n"
        )

    print("--- Summary ---")
    if all_ttft:
        print(f"  TTFT (time to first token):    avg {sum(all_ttft)/len(all_ttft)*1000:.1f}ms")
    if all_t2t:
        avg = sum(all_t2t) / len(all_t2t)
        print(f"  T2T  (per-request avg):        avg {avg:.1f}ms ({1000/avg:.1f} tok/s per stream)")
    if all_throughput:
        print(f"  Aggregate throughput:          avg {sum(all_throughput)/len(all_throughput):.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark token-to-token latency")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--model", default="", help="Model name for API request")
    parser.add_argument("--prompt", default="Write a short poem about the ocean", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    print(f"Benchmarking {args.url} (model={args.model or 'default'})")
    print(f"Prompt: {args.prompt!r}, max_tokens={args.max_tokens}, batch_size={args.batch_size}, runs={args.runs}\n")

    asyncio.run(benchmark(args.url, args.model, args.prompt, args.max_tokens, args.batch_size, args.runs))


if __name__ == "__main__":
    main()
