#!/usr/bin/env python3
"""Build the auditable, strata-first report for a qualified evaluation.

This module deliberately consumes raw trial records rather than trusting the
runner's aggregate.  A causal comparison is emitted only when the two arms
share repetition, tier, stratum, exact-demand declaration, workload digest,
and metric contract.  The bootstrap is paired and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

try:
    from .consumer_contract import validate_consumer_contract
except ImportError:  # pragma: no cover - direct script execution
    from consumer_contract import validate_consumer_contract


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_STRATA = {
    "request_state",
    "granularity",
    "load_ratio",
    "availability_skew",
    "staging_pressure",
    "arrival",
}
_BOOTSTRAP_SAMPLES = 10_000


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty metric")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data or not all(math.isfinite(value) for value in data):
        raise ValueError("metric has no finite observations")
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": _percentile(data, 0.95),
        "p99": _percentile(data, 0.99),
        "minimum": min(data),
        "maximum": max(data),
    }


def _metadata_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("tier"),
        record.get("demand_semantics"),
        record.get("workload_demand_digest"),
        json.dumps(record.get("stratum"), sort_keys=True, separators=(",", ":")),
    )


def _workload_digest(
    record: dict[str, Any], evaluation_metadata: dict[str, Any]
) -> str | None:
    direct_record = record.get("workload_demand_digest")
    if isinstance(direct_record, str) and direct_record:
        return direct_record
    result = record["result"]
    direct = result.get("demand_trace_digest")
    if isinstance(direct, str) and direct:
        return direct
    workload = result.get("workload")
    if isinstance(workload, dict) and isinstance(
        workload.get("demand_trace_digest"), str
    ):
        return workload["demand_trace_digest"]
    # A file digest proves which manifest was used; it is not the identity of
    # the exact demand consumed by the engine. The only valid fallback is the
    # normalized manifest's demand digest copied into evaluation metadata.
    return evaluation_metadata.get("workload_demand_digest")


def _validate_result(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("trial result is not a JSON object")
    classification = result.get("classification")
    if not isinstance(classification, str) or not classification:
        raise ValueError("trial result has no machine-readable classification")
    failures = result.get("verification_failures")
    if failures is None and isinstance(result.get("correctness"), dict):
        failures = result["correctness"].get("verification_failures")
    if not isinstance(failures, int) or failures != 0:
        raise ValueError(
            f"trial {record.get('experiment')}/{record.get('variant')} "
            "does not prove zero verification failures"
        )
    stats = result.get("engine_stats")
    if stats is None:
        return ()
    if not isinstance(stats, list):
        raise ValueError("engine_stats must be a list when present")
    contracts: list[dict[str, Any]] = []
    for entry in stats:
        if not isinstance(entry, dict) or entry.get("backend") != "nta_flashinfer":
            continue
        contracts.append(
            validate_consumer_contract(
                entry.get("consumer_contract"),
                require_formal_execution=True,
            )
        )
    return tuple(contracts)


def _load(output: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads(
            (output / "evaluation-metadata.json").read_text(encoding="utf-8")
        )
        lines = (output / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"evaluation artifact is incomplete: {error}") from error
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("trials.jsonl contains invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("trial record is not an object")
        records.append(value)
    if (
        not isinstance(metadata, dict)
        or not isinstance(evaluation, dict)
        or not records
    ):
        raise ValueError("evaluation artifact has no metadata or trials")
    return metadata, evaluation, records


def _paired_bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot bootstrap an empty paired comparison")
    rng = random.Random(seed)
    observed = statistics.fmean(values)
    samples = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(_BOOTSTRAP_SAMPLES)
    ]
    return {
        "estimand": "mean_paired_delta",
        "observed": observed,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "samples": _BOOTSTRAP_SAMPLES,
        "seed": seed,
    }


def analyze(output: Path) -> dict[str, Any]:
    metadata, evaluation_metadata, records = _load(output.resolve())
    spec = metadata.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("runner metadata does not contain the trial specification")
    repetitions = spec.get("repetitions")
    experiments = spec.get("experiments")
    if (
        not isinstance(repetitions, int)
        or repetitions < 5
        or not isinstance(experiments, list)
    ):
        raise ValueError(
            "evaluation specification has no defensible repetition contract"
        )
    declared = {(entry["name"], entry["variant"]): entry for entry in experiments}
    expected_count = repetitions * len(declared)
    if len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} trial records, found {len(records)}"
        )
    physical_tiers = {
        str(entry["tier"])
        for entry in declared.values()
        if entry.get("tier") in {"nvme", "dax"}
    }
    qualification_digest = evaluation_metadata.get("tier_qualification_digest")
    if physical_tiers and not isinstance(qualification_digest, str):
        raise ValueError("physical-tier evaluation has no qualification digest")

    seen: set[tuple[str, str, int]] = set()
    workload_digests: set[str] = set()
    consumer_contracts: set[str] = set()
    for record in records:
        identity = (
            record.get("experiment"),
            record.get("variant"),
            record.get("repetition"),
        )
        if identity in seen:
            raise ValueError(f"duplicate trial identity: {identity}")
        seen.add(identity)
        if (identity[0], identity[1]) not in declared:
            raise ValueError(f"trial is not declared by the spec: {identity[:2]}")
        declaration = declared[(identity[0], identity[1])]
        if (
            record.get("arm") != declaration.get("arm")
            or record.get("tier") != declaration.get("tier")
            or record.get("stratum") != declaration.get("stratum")
        ):
            raise ValueError(
                f"trial metadata diverges from its declaration: {identity[:2]}"
            )
        if not isinstance(identity[2], int) or not 0 <= identity[2] < repetitions:
            raise ValueError(f"invalid repetition for {identity[:2]}")
        if record.get("arm") not in {f"B{index}" for index in range(7)}:
            raise ValueError("trial has no canonical B0-B6 arm")
        if record.get("demand_semantics") != "exact":
            raise ValueError("trial artifact is not exact-demand")
        if not isinstance(record.get("stratum"), dict) or not REQUIRED_STRATA <= set(
            record["stratum"]
        ):
            raise ValueError("trial artifact is missing required strata")
        for contract in _validate_result(record):
            consumer_contracts.add(
                json.dumps(contract, sort_keys=True, separators=(",", ":"))
            )
        if physical_tiers:
            if record.get("tier_qualification_digest") != qualification_digest:
                raise ValueError(
                    "trial qualification digest diverges from evaluation metadata"
                )
        digest = _workload_digest(record, evaluation_metadata)
        if not isinstance(digest, str) or not digest:
            raise ValueError("trial has no exact workload/demand digest")
        workload_digests.add(digest)
        for metric in declaration.get("metrics", []):
            if not _finite(record["result"].get(metric)):
                raise ValueError(
                    f"metric {metric} is missing or non-finite in {identity[:2]}"
                )
    if len(workload_digests) > 1:
        raise ValueError("paired trials use different workload/demand digests")

    strata: list[dict[str, Any]] = []
    for key, declaration in declared.items():
        selected = [
            record
            for record in records
            if (record["experiment"], record["variant"]) == key
        ]
        if len(selected) != repetitions:
            raise ValueError(f"incomplete repetitions for {key}")
        strata.append(
            {
                "experiment": key[0],
                "variant": key[1],
                "arm": declaration["arm"],
                "tier": declaration["tier"],
                "stratum": declaration["stratum"],
                "repetitions": repetitions,
                "metrics": {
                    metric: _summary(record["result"][metric] for record in selected)
                    for metric in declaration.get("metrics", [])
                },
            }
        )

    causal: list[dict[str, Any]] = []
    for index, comparison in enumerate(spec.get("comparisons", [])):
        name = comparison["name"]
        experiment = comparison["experiment"]
        numerator = comparison["numerator_variant"]
        denominator = comparison["denominator_variant"]
        metric = comparison["metric"]
        numerator_records = {
            record["repetition"]: record
            for record in records
            if record["experiment"] == experiment and record["variant"] == numerator
        }
        denominator_records = {
            record["repetition"]: record
            for record in records
            if record["experiment"] == experiment and record["variant"] == denominator
        }
        if (
            set(numerator_records) != set(denominator_records)
            or len(numerator_records) != repetitions
        ):
            raise ValueError(f"comparison {name} does not contain complete pairs")
        deltas: list[float] = []
        ratios: list[float] = []
        pair_rows: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            left = numerator_records[repetition]
            right = denominator_records[repetition]
            if _metadata_key(left) != _metadata_key(right):
                raise ValueError(
                    f"comparison {name} is not metadata-matched at repetition {repetition}"
                )
            left_value = float(left["result"][metric])
            right_value = float(right["result"][metric])
            if right_value == 0:
                raise ValueError(f"comparison {name} has a zero denominator")
            deltas.append(left_value - right_value)
            ratios.append(left_value / right_value)
            pair_rows.append(
                {
                    "repetition": repetition,
                    "numerator": left_value,
                    "denominator": right_value,
                }
            )
        seed_material = f"{evaluation_metadata.get('spec_digest', '')}:{name}:{index}"
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode()).digest()[:8], "big"
        )
        causal.append(
            {
                "name": name,
                "experiment": experiment,
                "metric": metric,
                "numerator_variant": numerator,
                "denominator_variant": denominator,
                "matched_metadata": True,
                "pairs": pair_rows,
                "delta": _summary(deltas),
                "ratio": _summary(ratios),
                "paired_bootstrap": _paired_bootstrap(deltas, seed),
            }
        )

    report = {
        "schema": 1,
        "classification": "nta-osdi-evaluation-report",
        "provenance": {
            "revision": metadata.get("revision"),
            "dirty": metadata.get("dirty"),
            "evaluation_profile": evaluation_metadata.get(
                "evaluation_profile", "contract"
            ),
            "spec_digest": evaluation_metadata.get("spec_digest"),
            "workload_manifest_digest": evaluation_metadata.get(
                "workload_manifest_digest"
            ),
            "workload_demand_digest": next(iter(workload_digests), None),
            "tier_qualification_digest": qualification_digest,
            "qualified_physical_tiers": sorted(physical_tiers),
            "trial_count": len(records),
            "consumer_contracts": [
                json.loads(value) for value in sorted(consumer_contracts)
            ],
        },
        "strata": strata,
        "causal_comparisons": causal,
        "little_law": [
            {
                "experiment": record["experiment"],
                "variant": record["variant"],
                "repetition": record["repetition"],
                "report": record["result"].get("littles_law"),
            }
            for record in records
            if isinstance(record["result"].get("littles_law"), dict)
        ],
    }
    (output / "evaluation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "strata-report.json").write_text(
        json.dumps(
            {"schema": 1, "classification": "nta-strata-report", "strata": strata},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "causal-report.json").write_text(
        json.dumps(
            {"schema": 1, "classification": "nta-causal-report", "comparisons": causal},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.output)
    print("evaluation_report=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
