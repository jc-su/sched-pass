#!/usr/bin/env python3
"""Validate the report files emitted by ``run_evaluation.py``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .consumer_contract import validate_consumer_contract
    from .mechanism_arms import ARMS
except ImportError:  # pragma: no cover - direct script execution
    from consumer_contract import validate_consumer_contract
    from mechanism_arms import ARMS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(output: Path) -> dict[str, Any]:
    report = json.loads((output / "evaluation-report.json").read_text(encoding="utf-8"))
    strata = json.loads((output / "strata-report.json").read_text(encoding="utf-8"))
    causal = json.loads((output / "causal-report.json").read_text(encoding="utf-8"))
    evaluation_metadata = json.loads(
        (output / "evaluation-metadata.json").read_text(encoding="utf-8")
    )
    evaluation_profile = evaluation_metadata.get("evaluation_profile", "contract")
    _require(
        evaluation_profile in {"contract", "mechanism-study"},
        "unknown evaluation profile",
    )
    _require(
        report.get("classification") == "nta-osdi-evaluation-report",
        "invalid evaluation report",
    )
    _require(
        strata.get("classification") == "nta-strata-report", "invalid strata report"
    )
    _require(
        causal.get("classification") == "nta-causal-report", "invalid causal report"
    )
    _require(
        report.get("strata") == strata.get("strata"),
        "strata report diverges from canonical report",
    )
    _require(
        report.get("causal_comparisons") == causal.get("comparisons"),
        "causal report diverges from canonical report",
    )
    _require(
        report.get("causal_comparisons"), "evaluation report has no causal comparisons"
    )
    provenance = report.get("provenance")
    _require(
        isinstance(provenance, dict) and provenance.get("trial_count", 0) > 0,
        "report has no trial provenance",
    )
    contracts = provenance.get("consumer_contracts", [])
    _require(
        isinstance(contracts, list),
        "report consumer contract provenance is not a list",
    )
    for contract in contracts:
        try:
            validate_consumer_contract(
                contract,
                require_formal_execution=evaluation_profile == "mechanism-study",
            )
        except ValueError as error:
            raise ValueError(
                f"invalid consumer contract provenance: {error}"
            ) from error
    demand_digests = provenance.get("workload_demand_digests")
    workloads = provenance.get("workloads")
    _require(
        isinstance(demand_digests, list)
        and demand_digests
        and demand_digests == sorted(set(demand_digests))
        and all(isinstance(value, str) and value for value in demand_digests),
        "report has no exact workload/demand identities",
    )
    _require(
        isinstance(workloads, list)
        and workloads
        and {
            str(entry.get("demand_trace_digest"))
            for entry in workloads
            if isinstance(entry, dict)
        }
        == set(demand_digests),
        "report workload descriptors disagree with demand identities",
    )
    _require(
        provenance.get("evaluation_profile", "contract") == evaluation_profile,
        "report evaluation profile diverges from evaluation metadata",
    )
    if evaluation_profile == "mechanism-study":
        contract = json.loads(
            (output / "evaluation-contract.json").read_text(encoding="utf-8")
        )
        expected_arms = list(ARMS)
        _require(
            evaluation_metadata.get("arm_set") == expected_arms,
            "mechanism-study artifact does not contain exactly A0-A3",
        )
        _require(
            isinstance(evaluation_metadata.get("tier_set"), list)
            and len(evaluation_metadata["tier_set"]) == 1,
            "mechanism-study artifact must measure one tier per paired spec",
        )
        _require(
            isinstance(evaluation_metadata.get("strata_count"), int)
            and evaluation_metadata["strata_count"] >= 6,
            "mechanism-study artifact has too few workload strata",
        )
        expected_pairs = sorted(
            {
                f"{pair['numerator']}>{pair['denominator']}"
                for pair in contract.get("mechanism_study", {}).get("causal_pairs", [])
            }
        )
        _require(
            evaluation_metadata.get("causal_pairs") == expected_pairs,
            "mechanism-study artifact is missing a canonical causal boundary",
        )
        consumer_kinds = evaluation_metadata.get("consumer_kinds")
        _require(
            isinstance(consumer_kinds, dict)
            and set(consumer_kinds) == set(expected_arms)
            and all(
                kind in {"native_work_unit", "framework_reference"}
                for kind in consumer_kinds.values()
            ),
            "mechanism-study artifact has no complete consumer-kind declaration",
        )
        observed_kinds = {
            str(entry.get("arm")): entry.get("consumer_kind")
            for entry in report.get("strata", [])
            if isinstance(entry, dict) and entry.get("arm") is not None
        }
        _require(
            observed_kinds == consumer_kinds,
            "strata consumer evidence diverges from evaluation metadata",
        )
        _require(
            "native_work_unit"
            in {
                contract.get("kind")
                for contract in contracts
                if isinstance(contract, dict)
            },
            "mechanism-study artifact has no native numerical consumer evidence",
        )
    _require(
        workloads == evaluation_metadata.get("workloads"),
        "report workloads do not match evaluation metadata",
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
            _require(
                summary.get("count") == entry["repetitions"],
                f"incomplete metric summary: {metric}",
            )
            _require(
                all(
                    math.isfinite(float(summary[name]))
                    for name in ("mean", "median", "p95", "p99")
                ),
                f"non-finite stratum metric: {metric}",
            )
    for comparison in report.get("causal_comparisons", []):
        _require(
            comparison.get("matched_metadata") is True,
            "causal comparison is not matched",
        )
        _require(
            len(comparison.get("pairs", [])) >= 5, "causal comparison has too few pairs"
        )
        bootstrap = comparison.get("paired_bootstrap", {})
        interval = bootstrap.get("ci95", [])
        _require(
            len(interval) == 2
            and all(math.isfinite(float(value)) for value in interval),
            "invalid paired bootstrap interval",
        )
    for entry in report.get("finite_window_accounting", []):
        accounting = entry.get("report")
        if accounting is None:
            continue
        _require(
            accounting.get("method")
            == "finite_window_arrival_departure_accounting",
            "unknown finite-window accounting method",
        )
        _require(
            accounting.get("interpretation")
            == "descriptive_client_timestamp_accounting",
            "finite-window accounting is mislabeled",
        )
        for field in (
            "arrival_rate_per_second",
            "completion_rate_per_second",
            "mean_in_system",
            "mean_system_time_seconds",
            "occupancy_area_request_seconds",
            "sum_residence_seconds",
        ):
            _require(
                math.isfinite(float(accounting.get(field, math.nan))),
                f"non-finite finite-window accounting field {field}",
            )
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
