"""Fail-closed registry for trial result schemas.

An experiment command is free to print logs, but a formal trial may be sealed
only when its final JSON satisfies a declared result contract.  Classification
selects a concrete validator; it is never accepted as evidence by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

try:
    from .validate_bailian_replay import (
        validate as validate_bailian_replay,
        validate_formal_arm as validate_formal_bailian_arm,
    )
    from .validate_serving_report import (
        validate as validate_sglang_serving,
        validate_formal_arm as validate_formal_sglang_arm,
    )
    from .validate_vllm_report import validate as validate_vllm_serving
except ImportError:  # pragma: no cover - direct script execution
    from validate_bailian_replay import (
        validate as validate_bailian_replay,
        validate_formal_arm as validate_formal_bailian_arm,
    )
    from validate_serving_report import (
        validate as validate_sglang_serving,
        validate_formal_arm as validate_formal_sglang_arm,
    )
    from validate_vllm_report import validate as validate_vllm_serving


@dataclass(frozen=True, slots=True)
class ResultContract:
    name: str
    classifications: frozenset[str]
    validator: Callable[[dict[str, Any]], dict[str, Any]]
    metric_extractor: Callable[[dict[str, Any], str], float]
    formal: bool = True


def _numeric(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"trial result has no finite canonical metric {name!r}")
    return float(value)


def _sglang_metric(report: dict[str, Any], name: str) -> float:
    classification = report.get("classification")
    if classification in {"sglang-hicache-load", "sglang-bailian-natural-replay"}:
        if name == "slo_goodput_requests_per_second":
            goodput = report.get("slo_goodput")
            if not isinstance(goodput, dict):
                raise ValueError("serving result has no SLO-goodput object")
            return _numeric(goodput.get("goodput_requests_per_second"), name)
        if name == "slo_attainment":
            goodput = report.get("slo_goodput")
            if not isinstance(goodput, dict):
                raise ValueError("serving result has no SLO-goodput object")
            return _numeric(goodput.get("attainment"), name)
        if name in {"mean_admission_delay_seconds", "p95_admission_delay_seconds"}:
            admission = report.get("admission_delay_seconds")
            if not isinstance(admission, dict):
                raise ValueError("serving result has no admission-delay object")
            field = "mean" if name.startswith("mean_") else "p95"
            return _numeric(admission.get(field), name)
        return _numeric(report.get(name), name)
    # A paired comparison is a complete diagnostic result, not one causal arm.
    # Its only canonical metrics are explicit scalar comparison fields.
    if classification in {
        "sglang-hicache-load-comparison",
        "sglang-bailian-natural-replay-comparison",
    }:
        return _numeric(report.get(name), name)
    raise ValueError("unsupported SGLang metric result classification")


def _vllm_metric(report: dict[str, Any], name: str) -> float:
    aliases = {
        "request_throughput": "requests_per_second",
        "output_token_throughput": "generated_tokens_per_second",
    }
    return _numeric(report.get(aliases.get(name, name)), name)


_CONTRACTS = {
    contract.name: contract
    for contract in (
        ResultContract(
            "sglang-serving",
            frozenset({"sglang-hicache-load"}),
            validate_formal_sglang_arm,
            _sglang_metric,
        ),
        ResultContract(
            "sglang-serving-comparison",
            frozenset({"sglang-hicache-load-comparison"}),
            validate_sglang_serving,
            _sglang_metric,
            formal=False,
        ),
        ResultContract(
            "sglang-bailian-replay",
            frozenset({"sglang-bailian-natural-replay"}),
            validate_formal_bailian_arm,
            _sglang_metric,
        ),
        ResultContract(
            "sglang-bailian-replay-comparison",
            frozenset({"sglang-bailian-natural-replay-comparison"}),
            validate_bailian_replay,
            _sglang_metric,
            formal=False,
        ),
        ResultContract(
            "vllm-serving",
            frozenset({"vllm-serving-integration-smoke"}),
            validate_vllm_serving,
            _vllm_metric,
        ),
    )
}

_BY_CLASSIFICATION: dict[str, ResultContract] = {}
for _contract in _CONTRACTS.values():
    for _classification in _contract.classifications:
        if _classification in _BY_CLASSIFICATION:
            raise RuntimeError(
                f"duplicate trial-result classification {_classification!r}"
            )
        _BY_CLASSIFICATION[_classification] = _contract


def result_contract_names(*, formal_only: bool = False) -> frozenset[str]:
    return frozenset(
        name
        for name, contract in _CONTRACTS.items()
        if not formal_only or contract.formal
    )


def validate_trial_result(
    report: dict[str, Any],
    *,
    expected_contract: str | None,
    formal: bool,
) -> dict[str, Any]:
    """Validate one command result against its preregistered schema."""

    if not isinstance(report, dict):
        raise ValueError("trial result is not a JSON object")
    classification = report.get("classification")
    observed = (
        _BY_CLASSIFICATION.get(classification)
        if isinstance(classification, str)
        else None
    )
    if expected_contract is None:
        if formal:
            raise ValueError("formal trial has no declared result_contract")
        # Contract-profile fixtures may intentionally exercise only the generic
        # runner. A recognized production result is nevertheless always fully
        # validated instead of receiving a weaker path.
        if observed is None:
            return report
        return observed.validator(report)
    declared = _CONTRACTS.get(expected_contract)
    if declared is None:
        raise ValueError(f"unknown trial result_contract: {expected_contract!r}")
    if formal and not declared.formal:
        raise ValueError(
            f"trial result_contract {expected_contract!r} is diagnostic-only"
        )
    if observed is not declared:
        raise ValueError(
            "trial result classification does not satisfy its declared contract: "
            f"contract={expected_contract!r}, classification={classification!r}"
        )
    return declared.validator(report)


def extract_trial_metrics(
    report: dict[str, Any],
    names: list[str] | tuple[str, ...],
    *,
    expected_contract: str | None,
    formal: bool,
) -> dict[str, float]:
    """Extract typed scalar metrics from a validated command result."""

    classification = report.get("classification")
    observed = (
        _BY_CLASSIFICATION.get(classification)
        if isinstance(classification, str)
        else None
    )
    if expected_contract is None:
        if formal:
            raise ValueError("formal trial has no declared result_contract")
        extractor = observed.metric_extractor if observed is not None else None
    else:
        declared = _CONTRACTS.get(expected_contract)
        if declared is None or observed is not declared:
            raise ValueError("trial metric extraction contract does not match result")
        extractor = declared.metric_extractor
    metrics: dict[str, float] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("trial metric name is invalid")
        metrics[name] = (
            extractor(report, name)
            if extractor is not None
            else _numeric(report.get(name), name)
        )
    return metrics


def _direct_demand_digest(report: dict[str, Any]) -> str | None:
    direct = report.get("demand_trace_digest")
    if isinstance(direct, str) and direct:
        return direct
    workload = report.get("workload")
    if isinstance(workload, dict):
        for name in ("demand_trace_digest", "selected_demand_trace_digest"):
            value = workload.get(name)
            if isinstance(value, str) and value:
                return value
    correctness = report.get("correctness")
    if isinstance(correctness, dict):
        value = correctness.get("demand_trace_digest")
        if isinstance(value, str) and value:
            return value
    return None


def result_demand_digest(report: dict[str, Any]) -> str:
    """Return demand identity emitted by the measured command itself.

    Paired reports often keep this identity in their stock and NTA children.
    Every observed child must agree; evaluation metadata is intentionally not
    accepted as a fallback because it proves only what the runner intended to
    execute, not what the engine consumed.
    """

    digests: set[str] = set()
    direct = _direct_demand_digest(report)
    if direct is not None:
        digests.add(direct)
    for child_name in ("stock", "nta"):
        child = report.get(child_name)
        if isinstance(child, dict):
            child_digest = _direct_demand_digest(child)
            if child_digest is not None:
                digests.add(child_digest)
    if not digests:
        raise ValueError("trial result did not emit its consumed demand digest")
    if len(digests) != 1:
        raise ValueError("trial result contains conflicting consumed demand digests")
    return next(iter(digests))
