"""
Evaluate GSM8K accuracy against an OpenAI-compatible server (two-turn).

Turn 1: Ask the question, let the model solve it.
Turn 2: Ask the model to extract its final numerical answer as JSON.

Usage:
    python eval_gsm8k.py --url http://localhost:8000 --n 200
    python eval_gsm8k.py --url http://localhost:8000 --n 0   # full test set (1319)
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


EXTRACT_PROMPT = (
    'Present your final answer like shown below. I am not interested in words '
    'in the final answer. It should just be the final number / decimal / '
    'numerical you solved for\n\n{"answer": "..."}'
)


def parse_answer(text: str) -> str | None:
    """Extract the answer value from model JSON output."""
    try:
        obj = json.loads(text.strip())
        if "answer" in obj:
            return str(obj["answer"]).strip()
    except json.JSONDecodeError:
        pass

    m = re.search(r'\{\s*"answer"\s*:\s*"([^"]*)"\s*\}', text)
    if m:
        return m.group(1).strip()

    m = re.search(r'\{\s*"answer"\s*:\s*([0-9.eE+-]+)\s*\}', text)
    if m:
        return m.group(1).strip()

    return None


def normalize_answer(ans: str) -> str:
    """Normalize numeric answer for comparison."""
    ans = ans.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


def evaluate_one(endpoint: str, question: str, gold: str, idx: int) -> dict:
    """Two-turn evaluation of a single GSM8K problem using a fresh session."""

    session = requests.Session()
    messages = [{"role": "user", "content": question}]
    total_tokens = 0
    start = time.perf_counter()

    # Turn 1: solve the problem
    try:
        resp = session.post(
            endpoint,
            json={"messages": messages, "max_tokens": 512, "temperature": 0, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data1 = resp.json()
        turn1 = data1["choices"][0]["message"]["content"]
        total_tokens += data1.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        return {"idx": idx, "correct": False, "error": f"turn1: {e}", "tokens": 0, "elapsed": 0}

    # Turn 2: extract answer
    messages.append({"role": "assistant", "content": turn1})
    messages.append({"role": "user", "content": EXTRACT_PROMPT})

    try:
        resp = session.post(
            endpoint,
            json={"messages": messages, "max_tokens": 64, "temperature": 0, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data2 = resp.json()
        turn2 = data2["choices"][0]["message"]["content"]
        total_tokens += data2.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        return {"idx": idx, "correct": False, "error": f"turn2: {e}", "tokens": total_tokens, "elapsed": time.perf_counter() - start}

    elapsed = time.perf_counter() - start
    session.close()

    pred = parse_answer(turn2)
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred) if pred else ""
    correct = pred_norm == gold_norm

    return {
        "idx": idx,
        "correct": correct,
        "gold": gold_norm,
        "pred": pred_norm,
        "turn2_raw": turn2[:200],
        "tokens": total_tokens,
        "elapsed": elapsed,
    }


def run_eval(url: str, n: int, concurrency: int) -> None:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n > 0:
        ds = ds.select(range(min(n, len(ds))))

    print(f"Evaluating {len(ds)} GSM8K problems (concurrency={concurrency})\n")

    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    problems = [
        (row["question"], row["answer"].split("####")[-1].strip(), i)
        for i, row in enumerate(ds)
    ]

    wall_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(evaluate_one, endpoint, q, g, i): i
            for q, g, i in problems
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["idx"])
    wall_elapsed = time.perf_counter() - wall_start

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    total_tokens = sum(r.get("tokens", 0) for r in results)

    # Per-request T2T: tokens / elapsed for each request
    per_req_t2t = []
    for r in results:
        if r.get("tokens", 0) > 1 and r.get("elapsed", 0) > 0:
            t2t_ms = r["elapsed"] / r["tokens"] * 1000
            per_req_t2t.append(t2t_ms)

    print(f"--- GSM8K Results ---")
    print(f"  Accuracy:   {correct}/{total} ({100 * correct / total:.1f}%)")
    if per_req_t2t:
        avg_t2t = sum(per_req_t2t) / len(per_req_t2t)
        print(f"  Avg T2T:    {avg_t2t:.1f}ms ({1000 / avg_t2t:.1f} tok/s per request)")
    print(f"  Throughput: {total_tokens / wall_elapsed:.1f} tok/s aggregate")
    print(f"  Wall time:  {wall_elapsed:.1f}s")
    if errors:
        print(f"  Errors:     {errors}")

    wrong = [r for r in results if not r["correct"] and "error" not in r][:5]
    if wrong:
        print(f"\n  Sample incorrect (showing up to 5):")
        for r in wrong:
            print(f"    #{r['idx']}: gold={r['gold']}, pred={r['pred']}")
            print(f"           {r['turn2_raw'][:100]}")


def main():
    parser = argparse.ArgumentParser(description="GSM8K eval (two-turn)")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="Number of problems (0=all)")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    run_eval(args.url, args.n, args.concurrency)


if __name__ == "__main__":
    main()
