#!/usr/bin/env python3
"""Benchmark MLX-LM inference performance."""

import argparse
import statistics
import time
from openai import OpenAI


PROMPTS = [
    "用 Python 写一个快速排序算法",
    "Explain quantum computing in 100 words",
    "What are the key differences between TCP and UDP?",
    "Write a SQL query to find the top 5 customers by total purchases",
    "介绍一下中国历史上的唐朝",
    "How does RSA encryption work? Explain step by step",
    "Write a function to detect cycles in a directed graph",
    "Compare and contrast REST and GraphQL APIs",
    "解释一下机器学习中的过拟合和欠拟合",
    "Write a bash script to monitor disk usage and send alerts",
    "What is the CAP theorem and why does it matter?",
    "用 Rust 写一个简单的 HTTP server",
]


def run_benchmark(client, model: str, prompts: list[str], max_tokens: int) -> dict:
    ttfts = []
    tpots = []
    throughputs = []

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        tokens_out = 0
        ttft = None

        response = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=0.0, stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.time() - t0
                    ttfts.append(ttft)
                tokens_out += 1

        elapsed = time.time() - t0
        tpot = (elapsed - ttft) / (tokens_out - 1) if ttft and tokens_out > 1 else 0
        tpots.append(tpot)
        throughput = tokens_out / elapsed if elapsed > 0 else 0
        throughputs.append(throughput)

        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}... "
              f"TTFT: {ttft:.2f}s, {tokens_out} tokens, {throughput:.1f} tokens/s")

    n = len(ttfts)
    return {
        "num_prompts": len(prompts),
        "ttft_p50": statistics.median(ttfts),
        "ttft_p95": statistics.quantiles(ttfts, n=20)[18] if n >= 20 else max(ttfts),
        "ttft_p99": statistics.quantiles(ttfts, n=100)[98] if n >= 100 else max(ttfts),
        "tpot_avg": statistics.mean(tpots),
        "throughput_avg": statistics.mean(throughputs),
    }


def main():
    parser = argparse.ArgumentParser(description="MLX-LM Benchmark")
    parser.add_argument("--model", default="default_model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")
    prompts = PROMPTS[:args.prompts]

    print(f"Benchmark: model={args.model}, prompts={len(prompts)}, max_tokens={args.max_tokens}\n")
    results = run_benchmark(client, args.model, prompts, args.max_tokens)

    n = len(prompts)
    print(f"\n{'='*50}")
    print(f"Results ({n} prompts):")
    print(f"  TTFT P50:      {results['ttft_p50']:.3f}s")
    if n >= 20:
        print(f"  TTFT P95:      {results['ttft_p95']:.3f}s")
    if n >= 100:
        print(f"  TTFT P99:      {results['ttft_p99']:.3f}s")
    if n < 20:
        print(f"  (P95/P99 require 20+/100+ prompts, use --prompts for more)")
    print(f"  TPOT avg:      {results['tpot_avg']:.3f}s/token")
    print(f"  Throughput:    {results['throughput_avg']:.1f} tokens/s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
