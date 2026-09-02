#!/usr/bin/env python3
"""Validate a registered or completed four-RQ OSDI evaluation campaign.

The paired A0--A3 runner validates one mechanism study.  This validator owns
the paper-level boundary: model/workload diversity, current competitors, load
curves, independent observations, mechanism attribution, opportunity sweeps,
and deployment/isolation evidence.  A registered plan may omit result paths;
``--require-complete`` additionally verifies every declared artifact digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "evaluation-manifest.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from experiments.validate_evaluation import validate as validate_contract  # noqa: E402


RQ1_METRICS = {
    "ttft_p50_p95_p99",
    "tpot_p50_p95_p99",
    "itl_p99",
    "slo_goodput",
    "request_throughput",
    "output_token_throughput",
}
MECHANISM_ARMS = {"A0", "A1", "A1P", "A2", "A3"}
MECHANISM_PAIRS = {"A1>A0", "A1P>A1", "A2>A1", "A3>A2"}
RQ3_AXES = {"context", "locality", "load", "tier"}
RQ4_STUDIES = {"short_context_control", "resource_profile", "tenant_interference"}


def _objects(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"campaign {name} must be a non-empty array")
    if not all(isinstance(entry, dict) for entry in value):
        raise ValueError(f"campaign {name} entries must be objects")
    return value


def _unique_ids(entries: Iterable[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        identity = entry.get("id")
        if not isinstance(identity, str) or not identity or identity in indexed:
            raise ValueError(f"campaign {name} has a missing or duplicate id")
        indexed[identity] = entry
    return indexed


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validate_artifact(
    entry: dict[str, Any], *, base_dir: Path, require_complete: bool, label: str
) -> None:
    path_value = entry.get("artifact")
    digest = entry.get("artifact_sha256")
    if not require_complete and path_value is None and digest is None:
        return
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} has no artifact path")
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{label} artifact does not exist: {path}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} has no SHA-256 artifact identity")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise ValueError(f"{label} artifact digest mismatch")


def validate(
    document: dict[str, Any],
    *,
    base_dir: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    if document.get("schema") != 1 or document.get("classification") != (
        "nta-osdi-campaign"
    ):
        raise ValueError("unsupported OSDI campaign")
    status = document.get("status")
    if status not in {"registered", "complete"}:
        raise ValueError("campaign status must be registered or complete")
    if require_complete and status != "complete":
        raise ValueError("paper-level evidence is not complete")

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    validate_contract(contract)
    requirements = contract["paper_campaign_contract"]

    correctness = document.get("correctness_gate")
    if not isinstance(correctness, dict) or correctness.get("role") != (
        "validity_gate_not_research_question"
    ):
        raise ValueError("campaign must declare correctness as a validity gate")
    if require_complete and correctness.get("status") != "pass":
        raise ValueError("completed campaign has not passed its correctness gate")
    _validate_artifact(
        correctness,
        base_dir=base_dir,
        require_complete=require_complete,
        label="correctness gate",
    )

    models = _unique_ids(_objects(document.get("models"), "models"), "models")
    if len(models) < requirements["minimum_models"]:
        raise ValueError("campaign has too few models")
    families = {model.get("family") for model in models.values()}
    if None in families or len(families) < requirements["minimum_model_families"]:
        raise ValueError("campaign has too few model families")
    long_models = {
        identity
        for identity, model in models.items()
        if model.get("long_context") is True
        and isinstance(model.get("context_window_tokens"), int)
        and model["context_window_tokens"] >= 32768
    }
    if len(long_models) < requirements["minimum_long_context_models"]:
        raise ValueError("campaign has too few long-context models")

    declared_systems = _unique_ids(
        _objects(document.get("systems"), "systems"), "systems"
    )
    contract_systems = {entry["id"] for entry in contract["systems"]}
    if not set(declared_systems) <= contract_systems:
        raise ValueError("campaign declares a system outside the evaluation contract")
    if "nta-full" not in declared_systems:
        raise ValueError("campaign omits NTA")
    for group in requirements["required_competitor_groups"]:
        available = set(group["systems"]) & set(declared_systems)
        if len(available) < group["minimum"]:
            raise ValueError(f"campaign omits competitor group {group['id']}")

    load_selection = document.get("load_selection")
    if (
        not isinstance(load_selection, dict)
        or load_selection.get("policy") != "stock_only_frozen_knee"
        or load_selection.get("nta_observed_during_selection") is not False
    ):
        raise ValueError("campaign load points must be frozen from stock-only pilots")
    registered_fractions = load_selection.get("fractions_of_stock_knee")
    if not isinstance(registered_fractions, list) or len(registered_fractions) < (
        requirements["minimum_load_points_per_headline_curve"]
    ):
        raise ValueError("campaign load selection has too few registered fractions")
    load_fractions = {
        _positive_number(value, "stock-knee load fraction")
        for value in registered_fractions
    }
    if len(load_fractions) != len(registered_fractions):
        raise ValueError("campaign load selection repeats a stock-knee fraction")
    _validate_artifact(
        load_selection,
        base_dir=base_dir,
        require_complete=require_complete,
        label="stock-only load selection",
    )

    workloads = _unique_ids(
        _objects(document.get("workloads"), "workloads"), "workloads"
    )
    workload_families = {entry.get("family") for entry in workloads.values()}
    if None in workload_families or len(workload_families) < requirements[
        "minimum_workload_families"
    ]:
        raise ValueError("campaign has too few workload families")
    natural = {
        identity
        for identity, entry in workloads.items()
        if entry.get("provenance") == "natural_trace"
        and entry.get("statistical_independence") == "source_request_identity"
    }
    if len(natural) < requirements["minimum_natural_trace_workloads"]:
        raise ValueError("campaign has too few natural trace workloads")
    short_controls = {
        identity
        for identity, entry in workloads.items()
        if entry.get("short_context_control") is True
    }
    if requirements["require_short_context_control"] and not short_controls:
        raise ValueError("campaign omits the short-context no-regression workload")

    curves = _objects(document.get("headline_curves"), "headline_curves")
    curve_ids = _unique_ids(curves, "headline_curves")
    covered_models: set[str] = set()
    covered_workloads: set[str] = set()
    covered_systems: set[str] = set()
    cache_states: set[str] = set()
    independent_identity_counts: dict[str, int] = {}
    for identity, curve in curve_ids.items():
        model = curve.get("model")
        workload = curve.get("workload")
        systems = curve.get("systems")
        cache_state = curve.get("cache_state")
        if model not in models or workload not in workloads:
            raise ValueError(f"headline curve {identity} references an unknown model/workload")
        if not isinstance(systems, list) or len(systems) < 2:
            raise ValueError(f"headline curve {identity} needs at least two systems")
        if "nta-full" not in systems or not set(systems) <= set(declared_systems):
            raise ValueError(f"headline curve {identity} has an invalid system set")
        if cache_state not in {"warm", "cold"}:
            raise ValueError(f"headline curve {identity} must declare warm or cold cache")
        metrics = curve.get("metrics")
        if not isinstance(metrics, list) or not RQ1_METRICS <= set(metrics):
            raise ValueError(f"headline curve {identity} omits an RQ1 metric")
        load_points = _objects(curve.get("load_points"), f"{identity} load_points")
        if len(load_points) < requirements["minimum_load_points_per_headline_curve"]:
            raise ValueError(f"headline curve {identity} has too few load points")
        curve_fractions: set[float] = set()
        for index, point in enumerate(load_points):
            fraction = _positive_number(
                point.get("stock_knee_fraction"),
                f"headline curve {identity} load point {index} fraction",
            )
            if fraction in curve_fractions:
                raise ValueError(f"headline curve {identity} repeats a load point")
            curve_fractions.add(fraction)
            rate = point.get("offered_requests_per_second")
            if require_complete and rate is None:
                raise ValueError(
                    f"completed headline curve {identity} has no resolved offered rate"
                )
            if rate is not None:
                _positive_number(
                    rate, f"headline curve {identity} load point {index} rate"
                )
            independent_requests = _positive_integer(
                point.get("independent_requests"),
                f"headline curve {identity} independent requests",
                requirements["minimum_independent_requests_per_load_point"],
            )
            _positive_integer(
                point.get("repetitions"),
                f"headline curve {identity} repetitions",
                requirements["minimum_repetitions"],
            )
            identity_set = point.get("request_identity_set")
            if not isinstance(identity_set, str) or not identity_set:
                raise ValueError(f"headline curve {identity} has no request identity set")
            prior_count = independent_identity_counts.get(identity_set)
            if prior_count is not None and prior_count != independent_requests:
                raise ValueError(
                    f"request identity set {identity_set!r} has inconsistent size"
                )
            independent_identity_counts[identity_set] = independent_requests
            _validate_artifact(
                point,
                base_dir=base_dir,
                require_complete=require_complete,
                label=f"headline curve {identity} load point {index}",
            )
        if curve_fractions != load_fractions:
            raise ValueError(
                f"headline curve {identity} diverges from registered load fractions"
            )
        covered_models.add(model)
        covered_workloads.add(workload)
        covered_systems.update(systems)
        cache_states.add(cache_state)
    if covered_models != set(models):
        raise ValueError("not every declared model appears in a headline curve")
    if not set(natural) <= covered_workloads or not short_controls <= covered_workloads:
        raise ValueError("headline curves omit a natural or short-control workload")
    if set(declared_systems) != covered_systems:
        raise ValueError("headline curves omit a required available system")
    if cache_states != {"warm", "cold"}:
        raise ValueError("headline evidence must include both warm and cold cache")
    total_independent = sum(independent_identity_counts.values())
    if total_independent < requirements["minimum_independent_requests_for_p99"]:
        raise ValueError("campaign has too few independent requests for p99 claims")

    mechanism_studies = _objects(
        document.get("mechanism_studies"), "mechanism_studies"
    )
    for index, study in enumerate(mechanism_studies):
        if study.get("profile") != "mechanism-study":
            raise ValueError("RQ2 evidence must use the mechanism-study profile")
        if set(study.get("arms", ())) != MECHANISM_ARMS or set(
            study.get("causal_pairs", ())
        ) != MECHANISM_PAIRS:
            raise ValueError("RQ2 evidence omits a canonical causal boundary")
        _positive_integer(study.get("scenarios"), "mechanism scenarios", 6)
        if study.get("result_emitted_activation") is not True:
            raise ValueError("RQ2 evidence has no result-emitted activation proof")
        _validate_artifact(
            study,
            base_dir=base_dir,
            require_complete=require_complete,
            label=f"mechanism study {index}",
        )

    sweeps = _objects(document.get("opportunity_sweeps"), "opportunity_sweeps")
    sweep_axes = {sweep.get("axis") for sweep in sweeps}
    if not RQ3_AXES <= sweep_axes:
        raise ValueError("RQ3 evidence omits a required opportunity axis")
    for index, sweep in enumerate(sweeps):
        if sweep.get("axis") not in RQ3_AXES | {"heterogeneity", "granularity"}:
            raise ValueError("RQ3 evidence declares an unknown sweep axis")
        levels = sweep.get("levels")
        if not isinstance(levels, list) or len(levels) < 3:
            raise ValueError("every RQ3 sweep must predeclare at least three levels")
        _validate_artifact(
            sweep,
            base_dir=base_dir,
            require_complete=require_complete,
            label=f"opportunity sweep {index}",
        )

    deployment = _objects(document.get("deployment_studies"), "deployment_studies")
    kinds = {study.get("kind") for study in deployment}
    if not RQ4_STUDIES <= kinds:
        raise ValueError("RQ4 evidence omits no-op, resource, or tenant isolation")
    for index, study in enumerate(deployment):
        _positive_integer(
            study.get("repetitions"),
            f"deployment study {index} repetitions",
            requirements["minimum_repetitions"],
        )
        metrics = study.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("RQ4 studies must declare metrics")
        _validate_artifact(
            study,
            base_dir=base_dir,
            require_complete=require_complete,
            label=f"deployment study {index}",
        )

    return {
        "status": status,
        "models": len(models),
        "model_families": len(families),
        "workloads": len(workloads),
        "workload_families": len(workload_families),
        "systems": len(declared_systems),
        "headline_curves": len(curves),
        "load_points": sum(len(curve["load_points"]) for curve in curves),
        "independent_requests": total_independent,
        "request_identity_sets": len(independent_identity_counts),
        "mechanism_studies": len(mechanism_studies),
        "opportunity_sweeps": len(sweeps),
        "deployment_studies": len(deployment),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        document = json.loads(args.campaign.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("campaign is not an object")
        summary = validate(
            document,
            base_dir=args.campaign.resolve().parent,
            require_complete=args.require_complete,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"OSDI campaign refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
