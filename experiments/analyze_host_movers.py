#!/usr/bin/env python3
"""Validate and summarize paired SGLang host-mover diagnostics."""

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
    "iterations",
    "promotion_warmup_iterations",
)
REQUIRED_MOVERS = frozenset(("sm", "copy_engine", "auto"))


def parse_arm(value: str) -> tuple[str, pathlib.Path]:
    name, separator, path_text = value.partition("=")
    path = pathlib.Path(path_text)
    if not separator or name not in REQUIRED_MOVERS or not path.is_file():
        raise argparse.ArgumentTypeError(
            "mover arms use sm|copy_engine|auto=REPORT.json"
        )
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.arm]
    if set(names) != REQUIRED_MOVERS or len(names) != len(REQUIRED_MOVERS):
        parser.error("provide exactly one sm, copy_engine, and auto arm")
    return args


def load(path: pathlib.Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("classification") != "sglang-hicache-promotion":
        raise ValueError(f"not a SGLang promotion report: {path}")
    stats = report.get("engine_stats")
    if not isinstance(stats, list) or len(stats) != 1:
        raise ValueError(f"report does not contain one engine snapshot: {path}")
    return report


def median_ms(report: dict[str, Any], field: str) -> float:
    values = report.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"report lacks {field}")
    return 1_000.0 * statistics.median(float(value) for value in values)


def main() -> int:
    args = parse_args()
    reports = {name: (load(path), path) for name, path in args.arm}
    first = next(iter(reports.values()))[0]
    signature = {field: first.get(field) for field in WORKLOAD_FIELDS}
    rows: dict[str, dict[str, Any]] = {}
    for name, (report, path) in reports.items():
        if {field: report.get(field) for field in WORKLOAD_FIELDS} != signature:
            raise ValueError(f"mover arm {name} used a different workload")
        engine = report["engine_stats"][0]
        if engine.get("host_mover") != name or engine.get("tier_fallback"):
            raise ValueError(f"mover arm {name} did not execute fallback-free")
        pipeline_bytes = int(engine.get("profiled_pipeline_transfer_bytes", 0))
        pipeline_ms = float(engine.get("profiled_pipeline_transfer_gpu_ms", 0.0))
        if pipeline_bytes <= 0 or pipeline_ms <= 0:
            raise ValueError(f"mover arm {name} lacks a transfer profile")
        rows[name] = {
            "report": str(path.resolve()),
            "promotion_median_ms": median_ms(report, "promotion_seconds_samples"),
            "hot_median_ms": median_ms(report, "hot_request_seconds_samples"),
            "pipeline_bytes": pipeline_bytes,
            "pipeline_gpu_ms": pipeline_ms,
            "pipeline_gib_per_second": pipeline_bytes / pipeline_ms / (1 << 30) * 1_000,
            "copy_engine_bytes": int(engine.get("copy_engine_bytes", 0)),
            "sm_bytes": int(engine.get("sm_mover_bytes", 0)),
            "copy_engine_operations": int(engine.get("copy_engine_operations", 0)),
            "selected_copy_runs": int(engine.get("copy_engine_selected_runs", 0)),
            "selected_copy_rows": int(engine.get("copy_engine_selected_rows", 0)),
            "layout_runs": int(engine.get("indexed_layout_runs", 0)),
            "layout_rows": int(engine.get("indexed_layout_rows", 0)),
            "selected_mover_batches": {
                mover: int(engine.get(f"host_mover_{mover}_batches", 0))
                for mover in ("sm", "copy_engine", "hybrid")
            },
            "uncalibrated_copy_engine_batches": int(
                engine.get("host_mover_uncalibrated_copy_engine_batches", 0)
            ),
            "insufficient_gain_batches": int(
                engine.get("host_mover_insufficient_gain_batches", 0)
            ),
            "service_cost_batches": int(
                engine.get("host_mover_service_cost_batches", 0)
            ),
        }

    endpoint_ms = min(rows["sm"]["pipeline_gpu_ms"], rows["copy_engine"]["pipeline_gpu_ms"])
    auto_ms = rows["auto"]["pipeline_gpu_ms"]
    auto_rows = rows["auto"]["selected_copy_rows"]
    total_rows = rows["auto"]["layout_rows"]
    auto_activations = rows["auto"]["selected_mover_batches"]
    active_auto_movers = sorted(
        mover for mover, batches in auto_activations.items() if batches > 0
    )
    result = {
        "schema": 1,
        "classification": "sglang-host-mover-diagnostic",
        "evidence_grade": "diagnostic",
        "workload": signature,
        "arms": rows,
        "best_pipeline_mover": min(
            rows, key=lambda name: rows[name]["pipeline_gpu_ms"]
        ),
        "best_promotion_mover": min(
            rows, key=lambda name: rows[name]["promotion_median_ms"]
        ),
        "auto_selected_movers": active_auto_movers,
        "auto_speedup_over_best_forced_mover": endpoint_ms / auto_ms,
        "auto_copy_row_fraction": (
            auto_rows / total_rows if total_rows > 0 else 0.0
        ),
        "interpretation": (
            "The auto arm is accepted only when its measured service time "
            "matches or beats the best forced mover at this operating point. "
            "Its decision is driven by an explicit deployment calibration; "
            "without one it must fail closed to SM, never infer a portable "
            "byte threshold from this trial."
        ),
    }
    result["auto_matches_best_forced_mover"] = auto_ms <= endpoint_ms * 1.03
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
