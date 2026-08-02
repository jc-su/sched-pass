#!/usr/bin/env python3
"""Aggregate predeclared dense opportunity reports without cherry-picking."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def load_report(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != 2
        or value.get("classification") != "incremental-execution-opportunity"
    ):
        raise ValueError(f"invalid opportunity report: {path}")
    return value


def strings(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument("--minimum-passing-traces", type=int, default=2)
    parser.add_argument(
        "--required-tiers", default="host_staged,nvme", help="comma-separated"
    )
    parser.add_argument("--require-proceed", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.minimum_models, arguments.minimum_passing_traces) <= 0:
        parser.error("study coverage thresholds must be positive")
    required_tiers = {
        item.strip() for item in arguments.required_tiers.split(",") if item.strip()
    }
    if not required_tiers:
        parser.error("at least one required tier is needed")

    reports = [load_report(path) for path in arguments.reports]
    revisions = {report.get("revision") for report in reports}
    if len(revisions) != 1 or not next(iter(revisions)):
        raise ValueError("opportunity study reports must use one revision")
    models: set[str] = set()
    tiers: set[str] = set()
    measured_tiles = 0
    gpu_timestamped_tiles = 0
    resident_at_launch_tiles = 0
    total_tiles = 0
    passing = 0
    for report in reports:
        provenance = report.get("provenance")
        opportunity = report.get("opportunity")
        if not isinstance(provenance, dict) or not isinstance(opportunity, dict):
            raise ValueError("opportunity report is missing structured metrics")
        models.update(strings(provenance.get("models")))
        tiers.update(strings(provenance.get("tiers")))
        measured_tiles += int(provenance.get("measured_compute_tiles", 0))
        gpu_timestamped_tiles += int(provenance.get("gpu_timestamped_tiles", 0))
        resident_at_launch_tiles += int(provenance.get("resident_at_launch_tiles", 0))
        total_tiles += int(opportunity.get("tiles", 0))
        passing += report.get("proceed") is True

    checks = {
        "model_coverage": len(models) >= arguments.minimum_models,
        "tier_coverage": required_tiers.issubset(tiers),
        "measured_compute": total_tiles > 0 and measured_tiles == total_tiles,
        "arrival_provenance": (
            gpu_timestamped_tiles > 0
            and gpu_timestamped_tiles + resident_at_launch_tiles == total_tiles
        ),
        "material_traces": passing >= arguments.minimum_passing_traces,
    }
    result = {
        "schema": 1,
        "classification": "dense-opportunity-study",
        "revision": next(iter(revisions)),
        "trace_count": len(reports),
        "model_count": len(models),
        "models": sorted(models),
        "tiers": sorted(tiers),
        "measured_compute_tiles": measured_tiles,
        "gpu_timestamped_tiles": gpu_timestamped_tiles,
        "resident_at_launch_tiles": resident_at_launch_tiles,
        "total_tiles": total_tiles,
        "passing_trace_count": passing,
        "gpu_timestamped_arrivals": checks["arrival_provenance"],
        "material_barrier_cost": checks["material_traces"],
        "required_tiers": sorted(required_tiers),
        "checks": checks,
        "proceed": all(checks.values()),
        "reports": [str(path) for path in arguments.reports],
    }
    encoded = json.dumps(result, sort_keys=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["proceed"] or not arguments.require_proceed else 2


if __name__ == "__main__":
    raise SystemExit(main())
