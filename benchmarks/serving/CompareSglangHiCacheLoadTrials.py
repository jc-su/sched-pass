#!/usr/bin/env python3
"""Run arm-balanced, paired SGLang HiCache qualification trials."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
RATIO_FIELDS = (
    "output_throughput_ratio",
    "goodput_ratio",
    "resident_p95_ttft_ratio",
    "resident_p95_tpot_ratio",
    "resident_p99_itl_ratio",
    "external_p95_ttft_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260801)
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=ROOT / "results" / "serving" / "sglang-hicache-load-trials",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT
        / "results"
        / "serving"
        / "sglang-hicache-load-qualification.json",
    )
    parser.add_argument(
        "comparison_args",
        nargs=argparse.REMAINDER,
        help="arguments for CompareSglangHiCacheLoad.py after --",
    )
    args = parser.parse_args()
    if args.trials < 3:
        parser.error("qualification requires at least three paired trials")
    if args.comparison_args[:1] == ["--"]:
        args.comparison_args = args.comparison_args[1:]
    if not args.comparison_args:
        parser.error("comparison arguments are required after --")
    forbidden = {"--seed", "--output"}.intersection(args.comparison_args)
    if forbidden:
        parser.error(
            "the trial runner owns these comparison arguments: "
            + ", ".join(sorted(forbidden))
        )
    return args


def _seed_for_order(seed: int, first: str) -> int:
    while True:
        order = ["flashinfer", "nta_flashinfer"]
        random.Random(seed).shuffle(order)
        if order[0] == first:
            return seed
        seed += 1


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("ratios must be finite and positive")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _bootstrap_interval(
    values: list[float], *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = sorted(
        _geometric_mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(samples)
    )
    return estimates[round(0.025 * (samples - 1))], estimates[
        round(0.975 * (samples - 1))
    ]


def _aggregate(reports: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, field in enumerate(RATIO_FIELDS):
        values = [float(report[field]) for report in reports]
        low, high = _bootstrap_interval(values, seed=seed + index)
        result[field] = {
            "paired_values": values,
            "median": statistics.median(values),
            "geometric_mean": _geometric_mean(values),
            "bootstrap_95_percent_ci": [low, high],
        }
    return result


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    artifacts: list[str] = []
    next_seed = args.seed_base
    for trial in range(args.trials):
        first = "flashinfer" if trial % 2 == 0 else "nta_flashinfer"
        seed = _seed_for_order(next_seed, first)
        next_seed = seed + 1
        artifact = (args.artifact_dir / f"trial-{trial:02d}.json").resolve()
        command = [
            sys.executable,
            str(ROOT / "benchmarks" / "serving" / "CompareSglangHiCacheLoad.py"),
            *args.comparison_args,
            "--seed",
            str(seed),
            "--output",
            str(artifact),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"paired trial {trial} failed:\n"
                + "\n".join(completed.stdout.splitlines()[-120:])
            )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        if report.get("classification") != "sglang-hicache-load-comparison":
            raise RuntimeError(f"paired trial {trial} emitted an invalid artifact")
        if report.get("execution_order", [None])[0] != first:
            raise RuntimeError(f"paired trial {trial} did not preserve arm balancing")
        reports.append(report)
        artifacts.append(str(artifact))

    aggregate = {
        "schema": 1,
        "classification": "sglang-hicache-load-qualification",
        "trial_count": len(reports),
        # Ten process-level trials are the documented evidence standard for a
        # serving claim; smaller runs are diagnostics and must say so.
        "evidence_grade": "qualified" if len(reports) >= 10 else "diagnostic",
        "arm_order": [report["execution_order"] for report in reports],
        "artifacts": artifacts,
        "all_outputs_exact": all(
            report["stock"]["generated_text_sha256"]
            == report["nta"]["generated_text_sha256"]
            for report in reports
        ),
        "all_attention_transformed": all(
            bool(report["mechanism_activation"]["all_attention_transformed"])
            for report in reports
        ),
        "all_fallback_free": all(
            int(report["mechanism_activation"]["fallback_batches"]) == 0
            for report in reports
        ),
        "ratios": _aggregate(reports, args.seed_base),
    }
    if not all(
        aggregate[key]
        for key in (
            "all_outputs_exact",
            "all_attention_transformed",
            "all_fallback_free",
        )
    ):
        raise RuntimeError("qualification violated a mandatory mechanism invariant")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
