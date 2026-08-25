#!/usr/bin/env python3
"""RQ2/2A: measure real per-layer promotion, attention, and barrier-stall time.

This harness answers one question before any further integration work: on a
real SGLang HiCache serving workload, how much compute-stream time is actually
blocked waiting for layer readiness, and how does it compare with promotion
and attention time? It drives the existing `SglangHiCacheLoad.py` benchmark
with the engine's CUDA-event profiling enabled and derives the opportunity
metrics the evaluation plan requires. It performs no simulation: every number
comes from device events recorded during real serving execution.

Run inside an activated NTA JIT environment, exactly like the trials runner:

    tools/jit/activate.py --build-dir build --flashinfer-hook -- \
      python3 benchmarks/serving/OpportunityCharacterize.py \
        --model /path/to/model \
        --flashinfer-workspace-base /path/to/workspace \
        --context-length 32768 \
        --external-token-points 2048,8192,24576 \
        --output results/serving/opportunity-characterization.json

    The KV pool and churn are sized per point so every point both fits and is
    forced to evict its hot prefix to the host tier before measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOAD_BENCHMARK = ROOT / "benchmarks" / "serving" / "SglangHiCacheLoad.py"

PROFILE_ENVIRONMENT = {
    "NTA_PROFILE_BARRIER": "1",
    "NTA_PROFILE_GPU": "1",
    "NTA_PROFILE_TRANSFER": "1",
}


def token_points(value: str) -> tuple[int, ...]:
    try:
        points = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "external token points must be integers"
        ) from error
    if not points or any(point <= 0 for point in points):
        raise argparse.ArgumentTypeError("token points must be positive")
    if len(set(points)) != len(points):
        raise argparse.ArgumentTypeError("token points must be unique")
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument(
        "--external-token-points", type=token_points, default=(4096, 16384, 65536)
    )
    parser.add_argument("--external-requests", type=int, default=1)
    parser.add_argument("--resident-requests", type=int, default=1)
    parser.add_argument("--resident-tokens", type=int, default=8192)
    parser.add_argument("--resident-output-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--request-rate", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--blocked-threshold",
        type=float,
        default=0.10,
        help="minimum blocked fraction that counts as incremental opportunity",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not LOAD_BENCHMARK.is_file():
        parser.error(f"load benchmark is missing: {LOAD_BENCHMARK}")
    if not 0 < args.blocked_threshold < 1:
        parser.error("blocked threshold must be inside (0, 1)")
    largest = max(args.external_token_points)
    if point_max_total_tokens(largest, args) >= args.context_length:
        parser.error("context length must exceed the largest point's KV pool size")
    return args


def point_max_total_tokens(external_tokens: int, args: argparse.Namespace) -> int:
    """KV pool sized so every timed context fits but the hot prefix must
    still evict to host before its measured promotion."""
    return (
        args.external_requests * external_tokens
        + args.resident_requests * (args.resident_tokens + args.resident_output_tokens)
        + 1024
    )


def point_churn_tokens(external_tokens: int, args: argparse.Namespace) -> int:
    """Churn must itself fit the pool while hot + churn exceeds it.

    fit:   churn <= max_total - generation overhead (1024 margin);
    evict: churn > max_total - external, which holds for external > 1024.
    """
    if external_tokens <= 1024:
        raise ValueError(
            "external token points below 1025 cannot force eviction with a "
            "pool-fitting churn prompt"
        )
    return point_max_total_tokens(external_tokens, args) - 1024


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def run_load_point(args: argparse.Namespace, external_tokens: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(LOAD_BENCHMARK),
        "--model",
        str(args.model),
        "--attention-backend",
        "nta_flashinfer",
        "--external-requests",
        str(args.external_requests),
        "--external-tokens",
        str(external_tokens),
        "--resident-requests",
        str(args.resident_requests),
        "--resident-tokens",
        str(args.resident_tokens),
        "--resident-output-tokens",
        str(args.resident_output_tokens),
        "--context-length",
        str(args.context_length),
        "--churn-tokens",
        str(point_churn_tokens(external_tokens, args)),
        "--max-total-tokens",
        str(point_max_total_tokens(external_tokens, args)),
        "--request-rate",
        str(args.request_rate),
        "--seed",
        str(args.seed),
        "--flashinfer-workspace-base",
        str(args.flashinfer_workspace_base),
    ]
    environment = dict(os.environ)
    environment.update(PROFILE_ENVIRONMENT)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"load benchmark produced no report at {external_tokens}")
    report = json.loads(lines[-1])
    if not isinstance(report, dict) or "engine_stats" not in report:
        raise RuntimeError(
            f"load benchmark emitted an invalid report at {external_tokens}"
        )
    return {"command": command, "report": report}


def merge_profiles(engine_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum profiled counters across engine processes; fail if none profiled."""
    merged: dict[str, float] = {}
    by_layer: dict[str, float] = {}
    profiled_entries = 0
    for entry in engine_stats:
        if "profiled_barrier_waits" not in entry:
            continue
        profiled_entries += 1
        for key, value in entry.items():
            if key == "profiled_barrier_stall_by_layer_ms":
                for layer, stall in value.items():
                    by_layer[layer] = by_layer.get(layer, 0.0) + float(stall)
                continue
            if key.startswith("profiled_") and isinstance(value, (int, float)):
                if key.endswith("max_stall_gpu_ms"):
                    merged[key] = max(merged.get(key, 0.0), float(value))
                else:
                    merged[key] = merged.get(key, 0.0) + float(value)
    if profiled_entries == 0:
        raise RuntimeError(
            "no engine process recorded barrier profiles; the profiling "
            "environment did not reach the NTA backend"
        )
    merged["profiled_engine_processes"] = profiled_entries
    if by_layer:
        merged["profiled_barrier_stall_by_layer_ms"] = dict(
            sorted(by_layer.items(), key=lambda item: int(item[0]))
        )
    return merged


def characterize_point(
    args: argparse.Namespace, external_tokens: int
) -> dict[str, Any]:
    outcome = run_load_point(args, external_tokens)
    report = outcome["report"]
    profiles = merge_profiles(report["engine_stats"])

    stall_ms = float(profiles.get("profiled_barrier_stall_gpu_ms", 0.0))
    waits = int(profiles.get("profiled_barrier_waits", 0))
    if waits == 0:
        raise RuntimeError(
            f"barrier profiling recorded zero waits at {external_tokens} tokens"
        )
    operator_ms = sum(
        value for key, value in profiles.items() if key.endswith("_operator_gpu_ms")
    )
    if operator_ms <= 0:
        raise RuntimeError(
            f"GPU operator profiling recorded no attention time at "
            f"{external_tokens} tokens"
        )
    transfer_ms = float(profiles.get("profiled_pipeline_transfer_gpu_ms", 0.0))
    transfer_bytes = int(profiles.get("profiled_pipeline_transfer_bytes", 0))
    blocked_fraction = stall_ms / (stall_ms + operator_ms)

    return {
        "external_tokens": external_tokens,
        "barrier_waits": waits,
        "barrier_stalled_waits": int(profiles.get("profiled_barrier_stalled_waits", 0)),
        "barrier_stall_gpu_ms": stall_ms,
        "barrier_max_stall_gpu_ms": float(
            profiles.get("profiled_barrier_max_stall_gpu_ms", 0.0)
        ),
        "attention_operator_gpu_ms": operator_ms,
        "pipeline_transfer_gpu_ms": transfer_ms,
        "pipeline_transfer_bytes": transfer_bytes,
        "blocked_fraction": blocked_fraction,
        "load_compute_ratio": (transfer_ms / operator_ms) if operator_ms else 0.0,
        "stall_by_layer_ms": profiles.get("profiled_barrier_stall_by_layer_ms", {}),
        "resident_p99_itl_seconds": report.get("resident_p99_itl_seconds"),
        "external_p95_ttft_seconds": report.get("external_p95_ttft_seconds"),
        "engine_processes": int(profiles.get("profiled_engine_processes", 0)),
        "command": outcome["command"],
    }


def main() -> int:
    args = parse_args()
    points = [
        characterize_point(args, external_tokens)
        for external_tokens in args.external_token_points
    ]
    opportunity_points = [
        point["external_tokens"]
        for point in points
        if point["blocked_fraction"] >= args.blocked_threshold
    ]
    report = {
        "schema": 1,
        "classification": "sglang-opportunity-characterization",
        "revision": os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
        "dirty": bool(git_value("status", "--porcelain")),
        "model": str(args.model),
        "blocked_threshold": args.blocked_threshold,
        "resident_tokens": args.resident_tokens,
        "external_token_points": list(args.external_token_points),
        "points": points,
        "opportunity_points": opportunity_points,
        "opportunity_present": bool(opportunity_points),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
