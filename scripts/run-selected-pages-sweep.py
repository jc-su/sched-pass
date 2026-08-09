#!/usr/bin/env python3
"""Run a predeclared FlashInfer selected-page candidate-ratio sweep."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "benchmarks" / "serving" / "FlashInferSelectedPages.py"


def positive_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate pages must be integers") from error
    if (
        not values
        or any(item <= 0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError(
            "candidate pages must be unique positive integers"
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-pages", type=positive_list, default=(16, 32, 64, 128, 256)
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--minimum-peak-speedup", type=float, default=2.0)
    parser.add_argument("--maximum-oracle-regret", type=float, default=2.0)
    parser.add_argument("--maximum-policy-regret", type=float, default=1.05)
    parser.add_argument("--require-peak-speedup", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if (
        min(
            arguments.batch_size,
            arguments.top_k,
            arguments.iterations,
            arguments.trials,
        )
        <= 0
    ):
        parser.error("batch size, top-k, iterations, and trials must be positive")
    if any(candidate < arguments.top_k for candidate in arguments.candidate_pages):
        parser.error("every candidate-page point must be at least top-k")
    if arguments.minimum_peak_speedup < 1:
        parser.error("minimum peak speedup must be at least one")
    if arguments.maximum_oracle_regret < 1:
        parser.error("maximum oracle regret must be at least one")
    if arguments.maximum_policy_regret < 1:
        parser.error("maximum policy regret must be at least one")

    raw_directory = arguments.output.parent / f"{arguments.output.stem}-raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for candidate_pages in arguments.candidate_pages:
        raw_path = raw_directory / f"candidate-{candidate_pages}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--batch-size",
                str(arguments.batch_size),
                "--candidate-pages",
                str(candidate_pages),
                "--top-k",
                str(arguments.top_k),
                "--iterations",
                str(arguments.iterations),
                "--trials",
                str(arguments.trials),
                "--output",
                str(raw_path),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        point = json.loads(completed.stdout.strip().splitlines()[-1])
        if not isinstance(point, dict):
            raise RuntimeError("selected-page benchmark returned non-object JSON")
        points.append(point)

    revisions = {point.get("revision") for point in points}
    if len(revisions) != 1 or not next(iter(revisions)):
        raise RuntimeError("selected-page sweep mixed revisions")
    if not all(
        point.get("gpu_selected_pages") is True
        and point.get("nta_hot_path_host_identity_round_trips") == 0
        and point.get("offline_oracle_precomputed") is True
        and point.get("real_flashinfer_selector") is True
        and point.get("real_flashinfer_attention") is True
        and point.get("all_policy_attention_transformed") is True
        and point.get("paired_operator_contract_verified") is True
        and isinstance(point.get("adaptive_us", {}).get("samples"), list)
        and point.get("adaptive_us", {}).get("derived_from")
        in {"nta_cold_us", "overfetch_us"}
        and isinstance(point.get("candidate_retained_us", {}).get("samples"), list)
        and all(
            float(value) >= 1.0
            for value in point.get("online_policy_regret_samples", ())
        )
        and point.get("stock_output_parity") is True
        for point in points
    ):
        raise RuntimeError("selected-page sweep contains an unqualified point")

    crossing = next(
        (
            int(point["candidate_pages_per_request"])
            for point in points
            if float(point["speedup_over_overfetch"]) >= 1.0
        ),
        None,
    )
    peak = max(
        float(point["online_policy_speedup_over_forced_overfetch"])
        for point in points
    )
    peak_point = max(
        points,
        key=lambda point: float(
            point["online_policy_speedup_over_forced_overfetch"]
        ),
    )
    maximum_regret = max(float(point["regret_to_offline_oracle"]) for point in points)
    maximum_policy_regret = max(
        float(point["online_policy_regret_to_best_measured"]) for point in points
    )
    minimum_online_speedup = min(
        float(point["online_policy_speedup_over_forced_overfetch"]) for point in points
    )
    cold_indexed_over_candidate_retained = [
        float(point["nta_cold_us"]["median_with_topk"])
        / float(point["candidate_retained_us"]["median_with_topk"])
        for point in points
    ]
    selective_points = [
        point for point in points if float(point["bytes_avoided_fraction"]) >= 0.75
    ]
    selective_speedup = min(
        (
            float(point["online_policy_speedup_over_forced_overfetch"])
            for point in selective_points
        ),
        default=0.0,
    )
    no_selectivity = next(
        (point for point in points if float(point["bytes_avoided_fraction"]) == 0.0),
        None,
    )
    proceed = (
        crossing is not None
        and peak >= arguments.minimum_peak_speedup
        and selective_speedup >= arguments.minimum_peak_speedup
        and maximum_regret <= arguments.maximum_oracle_regret
        and maximum_policy_regret <= arguments.maximum_policy_regret
    )
    report = {
        "schema": 1,
        "classification": "flashinfer-gpu-selected-host-pages-sweep",
        "revision": next(iter(revisions)),
        "gpu_selected_pages": True,
        "nta_hot_path_host_identity_round_trips": 0,
        "real_flashinfer_selector": True,
        "real_flashinfer_attention": True,
        "all_policy_attention_transformed": True,
        "paired_operator_contract_verified": True,
        "stock_output_parity": True,
        "batch_size": arguments.batch_size,
        "top_k": arguments.top_k,
        "candidate_pages": list(arguments.candidate_pages),
        "candidate_sweep_points": len(points),
        "crossover_candidate_pages": crossing,
        "selectivity_crossover_measured": crossing is not None,
        "peak_speedup_over_overfetch": peak,
        "peak_speedup_bootstrap_95_percent_ci": peak_point[
            "online_policy_speedup_bootstrap_95_percent_ci"
        ],
        "peak_candidate_pages": int(peak_point["candidate_pages_per_request"]),
        "minimum_speedup_at_or_above_75pct_bytes_avoided": selective_speedup,
        "maximum_regret_to_offline_oracle": maximum_regret,
        "maximum_online_policy_regret": maximum_policy_regret,
        "policy_regret_definition": "same_trial_chosen_over_best",
        "candidate_retained_baseline": True,
        "minimum_cold_indexed_latency_ratio_to_candidate_retained": min(
            cold_indexed_over_candidate_retained
        ),
        "maximum_cold_indexed_latency_ratio_to_candidate_retained": max(
            cold_indexed_over_candidate_retained
        ),
        "minimum_online_policy_speedup_over_forced_overfetch": (minimum_online_speedup),
        "no_selectivity_speedup": (
            None
            if no_selectivity is None
            else float(
                no_selectivity["online_policy_speedup_over_forced_overfetch"]
            )
        ),
        "no_selectivity_forced_indexed_throughput_ratio": (
            None
            if no_selectivity is None
            else float(no_selectivity["speedup_over_overfetch"])
        ),
        "no_selectivity_policy_mode": (
            None if no_selectivity is None else no_selectivity["online_transfer_mode"]
        ),
        "minimum_peak_speedup": arguments.minimum_peak_speedup,
        "maximum_oracle_regret": arguments.maximum_oracle_regret,
        "maximum_policy_regret": arguments.maximum_policy_regret,
        "proceed": proceed,
        "points": points,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if arguments.require_peak_speedup and not report["proceed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
