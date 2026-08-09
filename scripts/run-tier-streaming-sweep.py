#!/usr/bin/env python3
"""Run the real-FlashInfer tier-streaming crossover matrix."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "serving" / "FlashInferTierStreaming.py"


def positive_vector(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-tokens", default="1,64,256,512,1024")
    parser.add_argument("--context-tokens", default="16384")
    parser.add_argument("--group-tokens", default="512,1024,2048,4096")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resident-fractions", default="0,0.25,0.5,1")
    parser.add_argument("--qo-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if (
        min(
            arguments.batch_size,
            arguments.qo_heads,
            arguments.kv_heads,
            arguments.head_dim,
            arguments.warmup,
            arguments.iterations,
            arguments.trials,
        )
        <= 0
    ):
        parser.error("all dimensions and sampling counts must be positive")
    try:
        arguments.query_vector = positive_vector(arguments.query_tokens, "query tokens")
        arguments.context_vector = positive_vector(
            arguments.context_tokens, "context tokens"
        )
        arguments.group_vector = positive_vector(arguments.group_tokens, "group tokens")
    except ValueError as error:
        parser.error(str(error))
    return arguments


def run_point(
    arguments: argparse.Namespace,
    query_tokens: int,
    context_tokens: int,
    group_tokens: int,
    artifact: pathlib.Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(BENCHMARK),
        "--batch-size",
        str(arguments.batch_size),
        "--query-tokens",
        str(query_tokens),
        "--context-tokens",
        str(context_tokens),
        "--resident-fractions",
        arguments.resident_fractions,
        "--group-tokens",
        str(group_tokens),
        "--qo-heads",
        str(arguments.qo_heads),
        "--kv-heads",
        str(arguments.kv_heads),
        "--head-dim",
        str(arguments.head_dim),
        "--warmup",
        str(arguments.warmup),
        "--iterations",
        str(arguments.iterations),
        "--trials",
        str(arguments.trials),
        "--output",
        str(artifact),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"tier-streaming point q={query_tokens} c={context_tokens} "
            f"g={group_tokens} failed:\n{completed.stderr}"
        )
    return json.loads(completed.stdout.splitlines()[-1])


def main() -> int:
    arguments = parse_args()
    output = arguments.output.resolve()
    raw = output.with_name(f"{output.stem}-raw")
    raw.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for context_tokens in arguments.context_vector:
        for query_tokens in arguments.query_vector:
            for group_tokens in arguments.group_vector:
                artifact = raw / (
                    f"q{query_tokens}-c{context_tokens}-g{group_tokens}.json"
                )
                result = run_point(
                    arguments,
                    query_tokens,
                    context_tokens,
                    group_tokens,
                    artifact,
                )
                points.append(
                    {
                        "query_tokens": query_tokens,
                        "context_tokens": context_tokens,
                        "group_tokens": group_tokens,
                        "streaming_speedup_over_atomic": result[
                            "streaming_speedup_over_atomic"
                        ],
                        "streaming_speedup_95ci": result["streaming_speedup_95ci"],
                        "staging_capacity_reduction": result[
                            "staging_capacity_reduction"
                        ],
                        "direct_us": result["direct_us"]["median"],
                        "atomic_us": result["atomic_us"]["median"],
                        "streaming_us": result["streaming_us"]["median"],
                        "bulk_copy_bandwidth_gbps": result["bulk_copy_bandwidth_gbps"],
                        "output_parity": result["output_parity"],
                        "artifact": str(artifact),
                    }
                )

    qualified = [
        point for point in points if point["staging_capacity_reduction"] >= 4.0
    ]
    best = max(
        qualified or points,
        key=lambda point: point["streaming_speedup_over_atomic"],
    )
    speeds = [point["streaming_speedup_over_atomic"] for point in points]
    report = {
        "schema": 1,
        "classification": "flashinfer-tier-streaming-crossover-sweep",
        "point_count": len(points),
        "all_outputs_exact": all(point["output_parity"] for point in points),
        "query_tokens": list(arguments.query_vector),
        "context_tokens": list(arguments.context_vector),
        "group_tokens": list(arguments.group_vector),
        "resident_fractions": arguments.resident_fractions,
        "points": points,
        "best_capacity_qualified_point": best,
        "crossover_measured": min(speeds) < 1.0 < max(speeds),
        "speedup_gate": {
            "minimum_speedup": 1.15,
            "minimum_staging_capacity_reduction": 4.0,
            "passed": (
                best["streaming_speedup_over_atomic"] >= 1.15
                and best["staging_capacity_reduction"] >= 4.0
                and best["streaming_speedup_95ci"]["lower"] > 1.0
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
