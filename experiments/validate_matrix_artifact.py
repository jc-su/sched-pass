#!/usr/bin/env python3
"""Validate fairness and activation invariants of a work-unit artifact.

The validator deliberately knows nothing about GPU timings.  It checks the
properties that a dependency-free artifact can prove: one exact trace per
case/repetition, complete arm coverage, explicit mechanism activation, tier
and granularity strata, and Little's-law accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

_ABLATION_TARGETS = {
    "full": frozenset(),
    "host_demand": frozenset({"B3", "B4", "B5", "B6"}),
    "batch_readiness": frozenset({"B4", "B5", "B6"}),
    "coarse_granularity": frozenset({"B4", "B5", "B6"}),
    "manual_mapping": frozenset({"B4", "B5", "B6"}),
    "shadow_generation_checks": frozenset({"B4", "B5", "B6"}),
    "admission_feedback": frozenset({"B5", "B6"}),
    "unbounded_staging": frozenset({"B4", "B5", "B6"}),
}
_ARMS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
_REQUIRED_FIELDS = (
    "arm",
    "ablation",
    "demand_trace_hash",
    "candidate_units",
    "selected_units",
    "useful_bytes",
    "physical_bytes",
    "tier",
    "stratum",
    "pending_arrival_rate",
    "mean_pending_units",
    "mean_pending_us",
    "littles_law_residual",
    "activation_counters",
    "execution_mode",
    "ablation_applied",
    "work_units",
    "group_count",
    "staging_high_water_units",
    "measurement",
)
_REQUIRED_STRATUM_FIELDS = {
    "request_state",
    "granularity",
    "tier",
    "load_ratio",
    "availability_skew",
    "staging_pressure",
    "arrival",
}


def _case_key(record: dict[str, Any]) -> tuple[int, str]:
    return (
        int(record["repetition"]),
        json.dumps(record["case"], sort_keys=True, separators=(",", ":")),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_activation(record: dict[str, Any]) -> None:
    arm = record["arm"]
    ablation = record["ablation"]
    applied = arm in _ABLATION_TARGETS[ablation]
    _require(
        record["ablation_applied"] is applied,
        f"incorrect ablation scope for {arm}/{ablation}",
    )
    mode = record["execution_mode"]
    counters = record["activation_counters"]
    work_units = int(record["work_units"])
    _require(
        counters["exact_demand_bindings"] == work_units,
        f"exact demand did not execute for {arm}/{ablation}",
    )
    if not applied:
        return
    if ablation == "host_demand":
        _require(
            mode["demand_source"] == "host_demand",
            "host-demand ablation did not use host demand",
        )
        _require(
            counters["host_demand_materializations"] > 0,
            "host-demand ablation never materialized demand",
        )
    elif ablation == "batch_readiness":
        _require(
            mode["readiness"] == "batch",
            "batch-readiness ablation did not use a batch boundary",
        )
        _require(
            counters["batch_readiness_barriers"] > 0,
            "batch-readiness ablation never crossed a barrier",
        )
    elif ablation == "coarse_granularity":
        _require(
            mode["granularity"] == "coarse",
            "coarse-granularity ablation was not active",
        )
        _require(
            record["group_count"] == 1,
            "coarse-granularity ablation did not collapse groups",
        )
    elif ablation == "manual_mapping":
        _require(mode["mapping"] == "manual", "manual-mapping ablation was not active")
        _require(
            counters["manual_mapping_sites"] > 0,
            "manual-mapping ablation never mapped a site",
        )
    elif ablation == "shadow_generation_checks":
        _require(
            mode["generation_checks"] == "shadow",
            "generation checks stayed on the hot path",
        )
        _require(
            counters["request_generation_checks_shadow"] > 0,
            "shadow generation checks never executed",
        )
        _require(
            counters["request_generation_checks_hot"] == 0,
            "hot generation checks were not disabled",
        )
    elif ablation == "admission_feedback":
        _require(
            mode["admission_feedback"] is False, "admission feedback was not disabled"
        )
        _require(
            counters["admission_feedback_decisions"] == 0,
            "admission feedback still executed",
        )
    elif ablation == "unbounded_staging":
        _require(mode["bounded_staging"] is False, "bounded staging was not disabled")
        _require(
            record["staging_high_water_units"] == record["selected_units"],
            "full promotion was not recorded",
        )


def validate(artifact: dict[str, Any], *, require_all_ablations: bool) -> None:
    _require(artifact.get("schema") == 2, "unsupported work-unit artifact schema")
    measurement = artifact.get("measurement", {})
    _require(
        measurement.get("serving_evidence") is False,
        "synthetic artifact cannot claim serving evidence",
    )
    _require(
        measurement.get("timing_is_modeled") is True,
        "synthetic timing must be labeled modeled",
    )
    provenance = artifact.get("provenance", {})
    _require(bool(provenance.get("revision")), "artifact has no revision provenance")
    arms = tuple(artifact.get("arms", ()))
    _require(arms == _ARMS, "artifact arm order is not the canonical B0-B6 order")
    ablations = tuple(artifact.get("ablations", ()))
    _require(
        set(ablations) <= set(_ABLATION_TARGETS),
        "artifact contains an unknown ablation",
    )
    if require_all_ablations:
        _require(
            set(ablations) == set(_ABLATION_TARGETS),
            "artifact did not execute all ablations",
        )
    records = artifact.get("records", ())
    _require(records, "artifact contains no records")
    manifest_path = Path(artifact.get("manifest", ""))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    _require(manifest_path.is_file(), "artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_metrics = tuple(manifest.get("required_metrics", ()))
    _require(required_metrics, "manifest has no required metric contract")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        for field in {*_REQUIRED_FIELDS, *required_metrics}:
            _require(field in record, f"record is missing {field}")
        _require(record["ablation"] in ablations, "record uses an unrequested ablation")
        _require(
            record["measurement"]["serving_evidence"] is False,
            "record claims serving evidence",
        )
        _require(
            isinstance(record["stratum"], dict)
            and _REQUIRED_STRATUM_FIELDS <= set(record["stratum"]),
            "record has incomplete evaluation strata",
        )
        _require(
            record["tier"] in {"hbm", "host_mem", "nvme", "dax"},
            "record uses an undeclared tier",
        )
        _require(
            float(record["littles_law_residual"]) <= 1e-9,
            "Little's-law residual is non-zero",
        )
        _require(
            record["selected_units"] <= record["candidate_units"],
            "selected demand exceeds candidate demand",
        )
        _require(
            record["useful_bytes"] <= record["physical_bytes"] or record["arm"] == "B0",
            "physical bytes are less than useful bytes",
        )
        _validate_activation(record)
        grouped.setdefault(_case_key(record), []).append(record)
    expected_per_group = len(arms) * len(ablations)
    for key, group in grouped.items():
        _require(
            len(group) == expected_per_group,
            f"incomplete arm/ablation coverage for {key}",
        )
        trace_hashes = {record["demand_trace_hash"] for record in group}
        _require(
            len(trace_hashes) == 1,
            f"demand trace changed across matched arms for {key}",
        )
        demand_shapes = {
            (
                record["candidate_units"],
                record["selected_units"],
                record["useful_bytes"],
            )
            for record in group
        }
        _require(
            len(demand_shapes) == 1,
            f"exact demand shape changed across matched arms for {key}",
        )
        tiers = {record["tier"] for record in group}
        _require(len(tiers) == 1, f"tier changed across matched arms for {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--require-all-ablations", action="store_true")
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    validate(artifact, require_all_ablations=args.require_all_ablations)
    print("matrix_artifact=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
