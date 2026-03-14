"""
GSM8K accuracy evaluation against an OpenAI-compatible server.

Downloads the GSM8K test set, sends problems to the server,
extracts the final numeric answer, and compares against ground truth.

Usage:
    python benchmarks/eval_gsm8k.py --url http://localhost:8000
    python benchmarks/eval_gsm8k.py --url http://localhost:8000 --num-samples 200 --batch-size 20
"""

import argparse
import asyncio
import json
import re
import time

import httpx
from datasets import load_dataset


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output."""
    # #### <number>
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # "the answer is <number>"
    m = re.search(r"the answer is\s*[:\s]*\$?([+-]?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # "= <number>" at end of line
    m = re.search(r"=\s*\$?([+-]?[\d,]+\.?\d*)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", "")

    # **<number>**
    m = re.search(r"\*\*\$?([+-]?[\d,]+\.?\d*)\*\*", text)
    if m:
        return m.group(1).replace(",", "")

    # Fallback: last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(ans: str | None) -> float | None:
    if ans is None:
        return None
    try:
        return float(ans)
    except ValueError:
        return None


def extract_ground_truth(answer_str: str) -> str:
    return answer_str.split("####")[-1].strip().replace(",", "")


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    question: str,
    model: str,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step. "
                    "Write your final numeric answer on the last line "
                    "in the format: #### <number>\n\n"
                    f"{question}"
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    resp = await client.post(endpoint, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def eval_batch(
    client: httpx.AsyncClient,
    endpoint: str,
    batch: list[dict],
    model: str,
    max_tokens: int,
) -> list[dict]:
    tasks = [
        send_request(client, endpoint, item["question"], model, max_tokens)
        for item in batch
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for item, resp in zip(batch, responses):
        gt = extract_ground_truth(item["answer"])
        if isinstance(resp, Exception):
            results.append({
                "question": item["question"],
                "ground_truth": gt,
                "model_output": f"ERROR: {resp}",
                "extracted": None,
                "correct": False,
            })
        else:
            extracted = extract_answer(resp)
            gt_val = normalize_answer(gt)
            ext_val = normalize_answer(extracted)
            correct = gt_val is not None and ext_val is not None and abs(gt_val - ext_val) < 1e-6
            results.append({
                "question": item["question"],
                "ground_truth": gt,
                "model_output": resp,
                "extracted": extracted,
                "correct": correct,
            })
    return results


async def run_eval(
    url: str,
    model: str,
    max_tokens: int,
    batch_size: int,
    num_samples: int,
) -> None:
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"

    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    total = len(dataset)
    print(f"Evaluating {total} problems (batch_size={batch_size})\n")

    all_results = []
    correct = 0
    errors = 0
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=180) as client:
        for i in range(0, total, batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, total))]
            batch_results = await eval_batch(client, endpoint, batch, model, max_tokens)
            all_results.extend(batch_results)

            for r in batch_results:
                if r["correct"]:
                    correct += 1
                if r["model_output"].startswith("ERROR"):
                    errors += 1

            done = len(all_results)
            acc = correct / done * 100
            elapsed = time.perf_counter() - start
            print(f"  [{done}/{total}] accuracy: {correct}/{done} ({acc:.1f}%)  elapsed: {elapsed:.0f}s")

    elapsed = time.perf_counter() - start
    acc = correct / total * 100

    print(f"\n{'='*50}")
    print(f"GSM8K Results")
    print(f"{'='*50}")
    print(f"  Samples:  {total}")
    print(f"  Correct:  {correct}")
    print(f"  Errors:   {errors}")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Time:     {elapsed:.0f}s")
    print(f"{'='*50}")

    # Show some wrong answers
    wrong = [r for r in all_results if not r["correct"] and not r["model_output"].startswith("ERROR")]
    if wrong:
        print(f"\nSample wrong answers (up to 5):")
        for r in wrong[:5]:
            print(f"\n  Q: {r['question'][:120]}...")
            print(f"  Ground truth: {r['ground_truth']}")
            print(f"  Extracted:    {r['extracted']}")
            print(f"  Model (last 200 chars): ...{r['model_output'][-200:]}")


def main():
    parser = argparse.ArgumentParser(description="GSM8K accuracy evaluation")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--model", default="", help="Model name for API request")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--batch-size", type=int, default=20, help="Concurrent requests")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to evaluate")
    args = parser.parse_args()

    print(f"GSM8K Eval — {args.url} (model={args.model or 'default'})")
    print(f"max_tokens={args.max_tokens}, batch_size={args.batch_size}, num_samples={args.num_samples}\n")

    asyncio.run(run_eval(args.url, args.model, args.max_tokens, args.batch_size, args.num_samples))


if __name__ == "__main__":
    main()
