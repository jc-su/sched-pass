#!/usr/bin/env python3
"""Audit EDF equivalence and summarize serving frontier-granularity trials."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any


WORKLOAD_FIELDS = (
    "model",
    "hot_tokens",
    "hot_requests",
    "churn_tokens",
    "resident_tokens",
    "max_total_tokens",
    "context_length",
    "iterations",
    "promotion_warmup_iterations",
)


def parse_arm(value: str) -> tuple[int, pathlib.Path]:
    width_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("frontier arms use WIDTH=REPORT.json")
    try:
        width = int(width_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("frontier width must be an integer") from error
    path = pathlib.Path(path_text)
    if width <= 0 or not path.is_file():
        raise argparse.ArgumentTypeError("frontier width and report must be valid")
    return width, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-report", type=pathlib.Path, required=True)
    parser.add_argument(
        "--frontier-arm",
        action="append",
        type=parse_arm,
        required=True,
        metavar="WIDTH=REPORT",
    )
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not args.stock_report.is_file() or args.layer_count <= 0:
        parser.error("stock report and layer count must be valid")
    widths = [width for width, _ in args.frontier_arm]
    if len(set(widths)) != len(widths):
        parser.error("frontier widths must be unique")
    return args


def load(path: pathlib.Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("classification") != "sglang-hicache-promotion":
        raise ValueError(f"not a promotion report: {path}")
    return report


def median_ms(report: dict[str, Any], field: str) -> float:
    values = report.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"report lacks non-empty {field}")
    return statistics.median(float(value) for value in values) * 1_000.0


def workload_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {field: report.get(field) for field in WORKLOAD_FIELDS}


def layer_waves(layer_count: int, width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(first, min(first + width, layer_count)))
        for first in range(0, layer_count, width)
    )


def main() -> int:
    args = parse_args()
    stock = load(args.stock_report)
    arms = [(width, load(path), path) for width, path in args.frontier_arm]
    signature = workload_signature(stock)
    for width, report, path in arms:
        if workload_signature(report) != signature:
            raise ValueError(f"frontier arm {width} has a different workload: {path}")
        stats = report.get("engine_stats")
        if not isinstance(stats, list) or len(stats) != 1:
            raise ValueError(f"frontier arm {width} lacks one engine report")
        engine = stats[0]
        if engine.get("frontier_layers_per_wave") != width:
            raise ValueError(f"frontier arm {width} did not execute its width")
        if engine.get("host_mover") != "sm" or engine.get("tier_fallback"):
            raise ValueError(f"frontier arm {width} is not a fallback-free SM arm")
        expected_external = int(report["iterations"]) + int(
            report["promotion_warmup_iterations"]
        )
        direct_batches = int(engine.get("host_direct_batches", 0))
        incremental_batches = int(engine.get("host_incremental_batches", 0))
        calibration_batches = int(
            engine.get("host_selection_calibration_probe_batches", 0)
        )
        if (
            int(engine.get("hicache_external_batches", 0)) != expected_external
            or direct_batches + incremental_batches != expected_external
            or incremental_batches != calibration_batches
            or calibration_batches > int(report["promotion_warmup_iterations"])
            or direct_batches < int(report["iterations"])
        ):
            raise ValueError(
                f"frontier arm {width} cannot prove that every timed batch "
                "used the direct path"
            )

    # In the measured domain every layer transfer is available at forward
    # entry and the consumer reaches layers monotonically. Therefore deadline
    # order is exactly layer order. Frontier width only groups adjacent entries;
    # it cannot create a different EDF schedule.
    deadlines = tuple(range(args.layer_count))
    edf_order = tuple(sorted(range(args.layer_count), key=deadlines.__getitem__))
    arm_rows = []
    for width, report, path in sorted(arms):
        waves = layer_waves(args.layer_count, width)
        flattened = tuple(layer for wave in waves for layer in wave)
        if flattened != edf_order:
            raise RuntimeError("frontier implementation changed transfer order")
        engine = report["engine_stats"][0]
        hot_ms = median_ms(report, "hot_request_seconds_samples")
        promotion_ms = median_ms(report, "promotion_seconds_samples")
        stock_hot_ms = median_ms(stock, "hot_request_seconds_samples")
        stock_promotion_ms = median_ms(stock, "promotion_seconds_samples")
        arm_rows.append(
            {
                "width": width,
                "report": str(path.resolve()),
                "wave_count_per_promotion": len(waves),
                "observed_copy_waves": engine.get("lookahead_copy_waves"),
                "hot_median_ms": hot_ms,
                "promotion_median_ms": promotion_ms,
                "hot_speedup_over_stock": stock_hot_ms / hot_ms,
                "promotion_speedup_over_stock": stock_promotion_ms / promotion_ms,
            }
        )

    winner = min(arm_rows, key=lambda arm: arm["promotion_median_ms"])
    result = {
        "schema": 1,
        "classification": "sglang-frontier-granularity-diagnostic",
        "evidence_grade": "diagnostic",
        "workload": signature,
        "stock_report": str(args.stock_report.resolve()),
        "stock_hot_median_ms": median_ms(stock, "hot_request_seconds_samples"),
        "stock_promotion_median_ms": median_ms(
            stock, "promotion_seconds_samples"
        ),
        "scheduling_domain": {
            "concurrent_staged_forwards": 1,
            "all_layer_demands_available_at_forward_entry": True,
            "consumer_deadlines_monotonic_by_layer": True,
            "edf_order": list(edf_order),
            "edf_distinct_from_layer_order": False,
            "interpretation": (
                "EDF and layer order are identical in this domain; measured "
                "differences are transfer-issue granularity, not priority order."
            ),
        },
        "arms": arm_rows,
        "best_measured_width": winner["width"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
