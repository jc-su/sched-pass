#!/usr/bin/env python3
"""Run matched stock and NTA HiCache promotion trials."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--hot-tokens", type=int, default=160)
    parser.add_argument("--hot-requests", type=int, default=1)
    parser.add_argument("--churn-tokens", type=int, default=240)
    parser.add_argument("--resident-tokens", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=320)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--cuda-graph-decode",
        choices=("disabled", "full"),
        default="disabled",
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--max-latency-regression-percent",
        type=float,
        help="fail when the NTA median exceeds stock by more than this percent",
    )
    parser.add_argument(
        "--verify-transfer",
        action="store_true",
        help=(
            "run a separate performance-excluded NTA arm that compares every "
            "promoted KV layer with its pinned-host source"
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "results" / "serving" / "sglang-hicache.json",
    )
    args = parser.parse_args()
    if (
        args.max_latency_regression_percent is not None
        and args.max_latency_regression_percent < 0
    ):
        parser.error("latency regression limit cannot be negative")
    return args


def parse_report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(report, dict)
            and report.get("classification") == "sglang-hicache-promotion"
        ):
            return report
    raise RuntimeError("HiCache trial did not emit a report")


def require_clean_mechanism(
    report: dict[str, Any], *, require_graph_replay: bool
) -> None:
    stats = [
        entry
        for entry in report.get("engine_stats", [])
        if entry.get("backend") == "nta_flashinfer"
    ]
    if not stats:
        raise RuntimeError("NTA HiCache trial did not publish engine statistics")
    fallbacks = sum(int(entry.get("hicache_fallback_batches", 0)) for entry in stats)
    claimed = sum(int(entry.get("hicache_claimed_batches", 0)) for entry in stats)
    if fallbacks != 0:
        raise RuntimeError(f"NTA HiCache trial used {fallbacks} fallback batches")
    if claimed == 0:
        raise RuntimeError("NTA HiCache trial did not claim an external batch")
    planless = sum(
        int(entry.get("planless_preacquired_launches", 0)) for entry in stats
    )
    if planless != 0:
        raise RuntimeError(
            f"NTA timed {planless} instrumented launches after acquisition"
        )
    stock_launches = sum(
        int(entry.get("stock_bulk_launches", 0)) for entry in stats
    )
    if stock_launches == 0:
        raise RuntimeError("NTA preacquired path did not execute stock FlashInfer")
    external_launches = sum(
        int(entry.get("external_launches", 0)) for entry in stats
    )
    prefetched_layers = sum(int(entry.get("prefetched_layers", 0)) for entry in stats)
    if external_launches == 0 or external_launches != prefetched_layers:
        raise RuntimeError(
            "NTA did not execute exactly one external attention layer for every "
            f"prefetched layer ({external_launches} != {prefetched_layers})"
        )
    if require_graph_replay:
        replays = sum(int(entry.get("graph_replays", 0)) for entry in stats)
        captures = sum(int(entry.get("graph_captures", 0)) for entry in stats)
        if captures == 0 or replays == 0:
            raise RuntimeError(
                "NTA HiCache graph trial did not capture and replay a decode graph"
            )


def run(
    args: argparse.Namespace, backend: str, *, verify_transfer: bool = False
) -> dict[str, Any]:
    workspace = ROOT / "results" / "serving" / "sglang-hicache-cache" / backend
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "SglangHiCache.py"),
        "--model",
        str(args.model.resolve()),
        "--attention-backend",
        backend,
        "--iterations",
        str(args.iterations),
        "--hot-tokens",
        str(args.hot_tokens),
        "--hot-requests",
        str(args.hot_requests),
        "--churn-tokens",
        str(args.churn_tokens),
        "--resident-tokens",
        str(args.resident_tokens),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--context-length",
        str(args.context_length),
        "--cuda-graph-decode",
        args.cuda_graph_decode,
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
    ]
    if args.max_attempts is not None:
        command.extend(("--max-attempts", str(args.max_attempts)))
    environment = os.environ.copy()
    environment.pop("NTA_SGLANG_VERIFY_TRANSFER", None)
    if verify_transfer:
        if backend != "nta_flashinfer":
            raise ValueError("transfer verification is defined only for NTA")
        environment["NTA_SGLANG_VERIFY_TRANSFER"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-100:])
        raise RuntimeError(
            f"HiCache {backend} trial failed with exit code "
            f"{completed.returncode}:\n{tail}"
        )
    return parse_report(completed.stdout)


def main() -> int:
    args = parse_args()
    execution_order = ["flashinfer", "nta_flashinfer"]
    random.Random(args.seed).shuffle(execution_order)
    reports = {backend: run(args, backend) for backend in execution_order}
    baseline = reports["flashinfer"]
    mechanism = reports["nta_flashinfer"]
    require_clean_mechanism(
        mechanism, require_graph_replay=args.cuda_graph_decode == "full"
    )
    if baseline.get("revision") != mechanism.get("revision"):
        raise RuntimeError("stock and NTA trials used different revisions")
    if baseline["generated_text_sha256"] != mechanism["generated_text_sha256"]:
        raise RuntimeError(
            "stock and NTA HiCache generations differ: "
            f"stock={baseline.get('generated_text_samples')} "
            f"NTA={mechanism.get('generated_text_samples')}"
        )
    if baseline["external_attempt_indices"] != mechanism["external_attempt_indices"]:
        raise RuntimeError(
            "stock and NTA observed different host-residency sequences: "
            f"stock={baseline['external_attempt_indices']} "
            f"NTA={mechanism['external_attempt_indices']}"
        )
    transfer_verification = None
    if args.verify_transfer:
        transfer_verification = run(args, "nta_flashinfer", verify_transfer=True)
        require_clean_mechanism(
            transfer_verification,
            require_graph_replay=args.cuda_graph_decode == "full",
        )
        if (
            transfer_verification["generated_text_sha256"]
            != baseline["generated_text_sha256"]
        ):
            raise RuntimeError("transfer-verification generation differs from stock")
        if (
            transfer_verification["external_attempt_indices"]
            != baseline["external_attempt_indices"]
        ):
            raise RuntimeError(
                "transfer-verification host-residency sequence differs from stock"
            )
    baseline_time = float(baseline["median_promotion_seconds"])
    mechanism_time = float(mechanism["median_promotion_seconds"])
    latency_change = mechanism_time / baseline_time - 1.0
    if (
        args.max_latency_regression_percent is not None
        and latency_change > args.max_latency_regression_percent / 100.0
    ):
        raise RuntimeError(
            f"NTA median latency changed by {100.0 * latency_change:.2f}%; "
            f"limit is {args.max_latency_regression_percent:.2f}%"
        )
    report = {
        "schema": 1,
        "classification": "matched-sglang-hicache-comparison",
        "revision": baseline["revision"],
        "dirty": bool(baseline.get("dirty") or mechanism.get("dirty")),
        "correctness": True,
        "execution_order": execution_order,
        "randomization_seed": args.seed,
        "baseline": baseline,
        "mechanism": mechanism,
        "promotion_throughput_ratio": baseline_time / mechanism_time,
        "promotion_latency_change_fraction": latency_change,
        "max_latency_regression_percent": args.max_latency_regression_percent,
        "transfer_verification": transfer_verification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
