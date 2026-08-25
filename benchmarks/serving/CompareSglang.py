#!/usr/bin/env python3
"""Run matched stock and NTA SGLang trials in isolated processes."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument(
        "--cuda-graph-decode",
        choices=("disabled", "full"),
        default="full",
    )
    parser.add_argument(
        "--skip-cache-prime",
        action="store_true",
        help="skip the unmeasured per-backend JIT-cache priming runs",
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=ROOT / "results" / "serving" / "sglang-cache",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "results" / "serving" / "sglang-comparison.json",
    )
    return parser.parse_args()


def parse_report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("engine") == "sglang":
            return value
    raise RuntimeError("SGLang trial did not emit a JSON report")


def require_clean_mechanism(
    report: dict[str, Any], *, require_graph_replay: bool
) -> dict[str, Any]:
    stats = [
        entry
        for entry in report.get("engine_stats", [])
        if entry.get("backend") == "nta_flashinfer"
    ]
    if not stats:
        raise RuntimeError("NTA trial did not publish engine statistics")
    fallbacks = sum(int(entry.get("hicache_fallback_batches", 0)) for entry in stats)
    if fallbacks != 0:
        raise RuntimeError(f"NTA trial used {fallbacks} HiCache fallback batches")
    transformed = sum(
        int(entry.get("transformed_direct_launches", 0)) for entry in stats
    )
    if transformed == 0:
        raise RuntimeError("NTA trial did not execute transformed FlashInfer")
    stock = sum(int(entry.get("stock_attention_launches", 0)) for entry in stats)
    if stock != 0:
        raise RuntimeError(f"NTA trial executed {stock} stock attention launches")
    ticketed = sum(
        int(entry.get("ticketed_incremental_launches", 0)) for entry in stats
    )
    total = sum(
        int(entry.get("decode_launches", 0)) + int(entry.get("prefill_launches", 0))
        for entry in stats
    )
    if total == 0 or transformed + ticketed != total:
        raise RuntimeError(
            "NTA did not account every attention launch to a transformed form "
            f"({transformed} + {ticketed} != {total})"
        )
    verified_modules = sum(
        int(entry.get("verified_operator_modules", 0)) for entry in stats
    )
    contracts = [
        contract for entry in stats for contract in entry.get("operator_contracts", [])
    ]
    if verified_modules == 0 or not contracts:
        raise RuntimeError("NTA trial did not verify compiler operator contracts")
    plans = [plan for entry in stats for plan in entry.get("operator_plans", [])]
    verified_pairs = sum(
        int(entry.get("verified_operator_pairs", 0)) for entry in stats
    )
    verified_plan_pairs = sum(
        int(entry.get("verified_operator_plan_pairs", 0)) for entry in stats
    )
    if not plans:
        raise RuntimeError("NTA trial did not verify compiler operator plans")
    if ticketed and (verified_pairs == 0 or verified_plan_pairs == 0):
        raise RuntimeError(
            "incremental attention ran without a paired direct execution plan"
        )
    graph_captures = sum(int(entry.get("graph_captures", 0)) for entry in stats)
    graph_replays = sum(int(entry.get("graph_replays", 0)) for entry in stats)
    if require_graph_replay and (graph_captures == 0 or graph_replays == 0):
        raise RuntimeError(
            "full decode graph was requested but transformed capture/replay was absent"
        )
    return {
        "all_attention_transformed": True,
        "active_forms": [
            name
            for name, count in (
                ("direct", transformed),
                ("incremental", ticketed),
            )
            if count > 0
        ],
        "transformed_direct_launches": transformed,
        "ticketed_incremental_launches": ticketed,
        "total_attention_launches": total,
        "stock_launches": stock,
        "fallback_batches": fallbacks,
        "verified_operator_modules": verified_modules,
        "verified_operator_pairs": verified_pairs,
        "verified_operator_plan_pairs": verified_plan_pairs,
        "graph_captures": graph_captures,
        "graph_replays": graph_replays,
    }


def run(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    workspace = args.cache_root.resolve() / backend
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "SglangSmoke.py"),
        "--model",
        str(args.model.resolve()),
        "--requests",
        str(args.requests),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--iterations",
        str(args.iterations),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cuda-graph-decode",
        args.cuda_graph_decode,
        "--attention-backend",
        backend,
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"SGLang {backend} trial failed with exit code "
            f"{completed.returncode}:\n{tail}"
        )
    return parse_report(completed.stdout)


def main() -> int:
    args = parse_args()
    if not args.skip_cache_prime:
        run(args, "flashinfer")
        run(args, "nta_flashinfer")
    baseline = run(args, "flashinfer")
    mechanism = run(args, "nta_flashinfer")
    activation = require_clean_mechanism(
        mechanism, require_graph_replay=args.cuda_graph_decode == "full"
    )
    if baseline["generated_text_sha256"] != mechanism["generated_text_sha256"]:
        raise RuntimeError("stock and NTA SGLang runs generated different output")
    baseline_time = float(baseline["median_batch_seconds"])
    mechanism_time = float(mechanism["median_batch_seconds"])
    report = {
        "schema": 1,
        "classification": "matched-sglang-serving-comparison",
        "correctness": True,
        "jit_cache_primed": not args.skip_cache_prime,
        "baseline": baseline,
        "mechanism": mechanism,
        "mechanism_activation": activation,
        "throughput_ratio": baseline_time / mechanism_time,
        "latency_overhead_fraction": mechanism_time / baseline_time - 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
