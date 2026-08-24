#!/usr/bin/env python3
"""Validate the report files emitted by ``run_evaluation.py``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(output: Path) -> dict[str, Any]:
    report = json.loads((output / "evaluation-report.json").read_text(encoding="utf-8"))
    strata = json.loads((output / "strata-report.json").read_text(encoding="utf-8"))
    causal = json.loads((output / "causal-report.json").read_text(encoding="utf-8"))
    _require(report.get("classification") == "nta-osdi-evaluation-report", "invalid evaluation report")
    _require(strata.get("classification") == "nta-strata-report", "invalid strata report")
    _require(causal.get("classification") == "nta-causal-report", "invalid causal report")
    _require(report.get("strata") == strata.get("strata"), "strata report diverges from canonical report")
    _require(report.get("causal_comparisons") == causal.get("comparisons"), "causal report diverges from canonical report")
    _require(report.get("causal_comparisons"), "evaluation report has no causal comparisons")
    provenance = report.get("provenance")
    _require(isinstance(provenance, dict) and provenance.get("trial_count", 0) > 0, "report has no trial provenance")
    _require(
        isinstance(provenance.get("workload_demand_digest"), str)
        and bool(provenance["workload_demand_digest"]),
        "report has no exact workload/demand digest",
    )
    evaluation_metadata = json.loads(
        (output / "evaluation-metadata.json").read_text(encoding="utf-8")
    )
    _require(
        provenance["workload_demand_digest"]
        == evaluation_metadata.get("workload_demand_digest"),
        "report demand digest does not match evaluation metadata",
    )
    physical_tiers = {
        str(entry.get("tier"))
        for entry in report.get("strata", [])
        if entry.get("tier") in {"nvme", "dax"}
    }
    if physical_tiers:
        _require(
            isinstance(provenance.get("tier_qualification_digest"), str)
            and bool(provenance["tier_qualification_digest"]),
            "physical-tier report has no qualification digest",
        )
        _require(
            physical_tiers <= set(provenance.get("qualified_physical_tiers", [])),
            "report omits a qualified physical tier",
        )
    for entry in report.get("strata", []):
        _require(entry.get("repetitions", 0) >= 5, "stratum has too few repetitions")
        for metric, summary in entry.get("metrics", {}).items():
            _require(summary.get("count") == entry["repetitions"], f"incomplete metric summary: {metric}")
            _require(all(math.isfinite(float(summary[name])) for name in ("mean", "median", "p95", "p99")), f"non-finite stratum metric: {metric}")
    for comparison in report.get("causal_comparisons", []):
        _require(comparison.get("matched_metadata") is True, "causal comparison is not matched")
        _require(len(comparison.get("pairs", [])) >= 5, "causal comparison has too few pairs")
        bootstrap = comparison.get("paired_bootstrap", {})
        interval = bootstrap.get("ci95", [])
        _require(len(interval) == 2 and all(math.isfinite(float(value)) for value in interval), "invalid paired bootstrap interval")
    for entry in report.get("little_law", []):
        little = entry.get("report")
        if little is None:
            continue
        _require(little.get("method") == "finite_window_arrival_departure_accounting", "unknown Little's Law method")
        _require(math.isfinite(float(little.get("residual", math.nan))), "non-finite Little's Law residual")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate(args.output.resolve())
    print("evaluation_artifact=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
