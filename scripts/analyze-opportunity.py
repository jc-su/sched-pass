#!/usr/bin/env python3
"""Analyze request/tile arrival traces without assuming an online oracle."""

from __future__ import annotations

import argparse
import json
import pathlib

from nta_runtime.opportunity import (
    load_json_lines,
    summarize,
    summarize_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--launch-overhead-ns", type=int, default=0)
    parser.add_argument("--grouping-window-ns", type=int, default=0)
    parser.add_argument("--parallel-slots", type=int, default=1)
    parser.add_argument("--material-delay-ns", type=int, default=0)
    parser.add_argument("--speedup-threshold", type=float, default=1.2)
    parser.add_argument("--minimum-material-tile-fraction", type=float, default=0.25)
    parser.add_argument(
        "--minimum-material-operator-fraction", type=float, default=0.10
    )
    parser.add_argument("--minimum-aggregate-speedup", type=float, default=1.20)
    parser.add_argument(
        "--require-proceed",
        action="store_true",
        help="exit 2 when the predeclared opportunity gate does not pass",
    )
    arguments = parser.parse_args()
    if not 0 <= arguments.minimum_material_tile_fraction <= 1:
        parser.error("material tile fraction must be in [0, 1]")
    if not 0 <= arguments.minimum_material_operator_fraction <= 1:
        parser.error("material operator fraction must be in [0, 1]")
    if arguments.minimum_aggregate_speedup < 1:
        parser.error("minimum aggregate speedup must be at least one")
    with arguments.trace.open(encoding="utf-8") as source:
        records = load_json_lines(source)
    result = summarize(
        records,
        launch_overhead_ns=arguments.launch_overhead_ns,
        grouping_window_ns=arguments.grouping_window_ns,
        parallel_slots=arguments.parallel_slots,
        material_delay_ns=arguments.material_delay_ns,
        speedup_threshold=arguments.speedup_threshold,
    )
    provenance = summarize_provenance(records)
    if len(provenance.revisions) != 1 or not provenance.revisions[0]:
        raise ValueError("trace must contain one non-empty qualified revision")
    criteria = {
        "minimum_material_tile_fraction": arguments.minimum_material_tile_fraction,
        "minimum_material_operator_fraction": (
            arguments.minimum_material_operator_fraction
        ),
        "minimum_aggregate_speedup": arguments.minimum_aggregate_speedup,
    }
    checks = {
        "material_tiles": (
            result.material_available_before_atomic_launch
            >= arguments.minimum_material_tile_fraction
        ),
        "material_operators": (
            result.operators_with_material_opportunity
            >= arguments.minimum_material_operator_fraction
        ),
        "aggregate_speedup": (
            result.incremental_speedup >= arguments.minimum_aggregate_speedup
        ),
    }
    report = {
        "schema": 2,
        "revision": provenance.revisions[0],
        "classification": "incremental-execution-opportunity",
        "provenance": provenance.as_json(),
        "opportunity": result.as_json(),
        "kill_criteria": criteria,
        "checks": checks,
        "proceed": all(checks.values()),
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    if arguments.require_proceed and not report["proceed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
