#!/usr/bin/env python3
"""Freeze one stock-derived serving overload rate before NTA evaluation.

The input reports must be independently timed, clean, stock-only serving arms.
The rule deliberately has no access to NTA measurements: it identifies the
first transition from a sustainable stock rate to multi-signal saturation and
freezes the smallest measured overload candidate above that knee.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from .atomic_io import atomic_write_json
except ImportError:  # pragma: no cover - direct script execution
    from atomic_io import atomic_write_json


CLASSIFICATION = "stock-serving-overload-rate-freeze"
RULE_NAME = "first_sustained_stock_multisignal_knee_v1"
_HEX = frozenset("0123456789abcdef")
_WORKLOAD_CONFIGURATION_FIELDS = (
    "model",
    "engine",
    "engine_version",
    "batch_mode",
    "serving_tier",
    "seed",
    "context_length",
    "resident_requests",
    "external_requests",
    "resident_tokens",
    "external_tokens",
    "resident_input_cache_tokens",
    "external_input_cache_tokens",
    "shared_input_cache_tokens",
    "resident_output_tokens",
    "external_output_tokens",
    "chunked_prefill_size",
    "mixed_chunk_enabled",
)


@dataclass(frozen=True, slots=True)
class PilotInput:
    offered_rate: float
    path: Path


@dataclass(frozen=True, slots=True)
class FreezeRule:
    throughput_shortfall_fraction: float = 0.10
    p95_growth_factor: float = 1.50
    p99_growth_factor: float = 1.50
    slo_attainment_drop: float = 0.05
    minimum_presaturation_slo_attainment: float = 0.95
    minimum_overload_margin_fraction: float = 0.10
    maximum_rate_step_factor: float = 2.0
    maximum_throughput_overshoot_fraction: float = 0.05
    maximum_throughput_reversal_fraction: float = 0.15
    maximum_latency_reversal_fraction: float = 0.20
    maximum_slo_reversal: float = 0.05
    maximum_replicate_relative_range: float = 0.10
    maximum_slo_replicate_range: float = 0.05
    minimum_replicates_per_rate: int = 1
    minimum_requests_per_report: int = 100

    def __post_init__(self) -> None:
        numeric_thresholds = (
            self.throughput_shortfall_fraction,
            self.p95_growth_factor,
            self.p99_growth_factor,
            self.slo_attainment_drop,
            self.minimum_presaturation_slo_attainment,
            self.minimum_overload_margin_fraction,
            self.maximum_rate_step_factor,
            self.maximum_throughput_overshoot_fraction,
            self.maximum_throughput_reversal_fraction,
            self.maximum_latency_reversal_fraction,
            self.maximum_slo_reversal,
            self.maximum_replicate_relative_range,
            self.maximum_slo_replicate_range,
        )
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0
            for value in numeric_thresholds
        ):
            raise ValueError("freeze thresholds must be finite and nonnegative")
        if not 0.0 < self.throughput_shortfall_fraction < 1.0:
            raise ValueError("throughput shortfall must be in (0, 1)")
        if not 0.0 < self.slo_attainment_drop <= 1.0:
            raise ValueError("SLO attainment drop must be in (0, 1]")
        if not 0.0 < self.minimum_presaturation_slo_attainment <= 1.0:
            raise ValueError("pre-saturation SLO attainment must be in (0, 1]")
        if min(self.p95_growth_factor, self.p99_growth_factor) <= 1.0:
            raise ValueError("latency growth factors must exceed one")
        if self.maximum_rate_step_factor <= 1.0:
            raise ValueError("maximum rate-step factor must exceed one")
        if (
            max(
                self.maximum_throughput_overshoot_fraction,
                self.maximum_throughput_reversal_fraction,
                self.maximum_latency_reversal_fraction,
                self.maximum_slo_reversal,
                self.maximum_replicate_relative_range,
                self.maximum_slo_replicate_range,
            )
            >= 1.0
        ):
            raise ValueError("stability fractions must be below one")
        if (
            not isinstance(self.minimum_replicates_per_rate, int)
            or isinstance(self.minimum_replicates_per_rate, bool)
            or self.minimum_replicates_per_rate <= 0
        ):
            raise ValueError("minimum replicates per rate must be positive")
        if (
            not isinstance(self.minimum_requests_per_report, int)
            or isinstance(self.minimum_requests_per_report, bool)
            or self.minimum_requests_per_report < 100
        ):
            raise ValueError("p99 saturation pilots require at least 100 requests")


@dataclass(frozen=True, slots=True)
class _Observation:
    offered_rate: float
    path: str
    sha256: str
    achieved_request_throughput: float
    p95_system_latency_seconds: float
    p99_system_latency_seconds: float
    slo_attainment: float
    request_count: int
    generated_text_sha256: str
    revision: str
    machine_digest: str
    workload_family_digest: str
    output_identity_digest: str
    slo_thresholds_digest: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    offered_rate: float
    replicates: int
    achieved_request_throughput: float
    throughput_efficiency: float
    p95_system_latency_seconds: float
    p99_system_latency_seconds: float
    slo_attainment: float
    input_sha256: tuple[str, ...]


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive " if positive else "finite "
        raise ValueError(f"serving pilot lacks {qualifier}{name}")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"serving pilot has invalid {name}")
    return value


def _close(actual: Any, expected: float, name: str) -> None:
    value = _finite(actual, name)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"serving pilot {name} does not match request records")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _strict_json(raw: bytes, path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"serving pilot {path} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"serving pilot {path} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"serving pilot {path} is not a JSON object")
    return value


def _reject_nta_data(value: Any, location: str = "report") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized == "nta" or normalized.startswith("nta_"):
                raise ValueError(
                    f"stock saturation pilot contains NTA data at {location}.{key}"
                )
            _reject_nta_data(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nta_data(child, f"{location}[{index}]")


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("serving pilot has no system-latency samples")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _request_identities(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    workload: list[dict[str, Any]] = []
    outputs: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("serving pilot has missing or duplicate request identity")
        seen.add(request_id)
        kind = record.get("kind")
        if kind not in {"resident", "external"}:
            raise ValueError(f"serving pilot record {index} has invalid request kind")
        signature: dict[str, Any] = {"request_id": request_id, "kind": kind}
        for field in (
            "input_tokens",
            "completion_tokens",
            "host_cached_tokens",
            "device_cached_tokens",
        ):
            signature[field] = _integer(record.get(field), f"record {index} {field}")
        if signature["completion_tokens"] < 2:
            raise ValueError("saturation pilot contains vacuous TPOT/ITL output")
        text_digest = record.get("text_sha256")
        if not _valid_sha256(text_digest):
            raise ValueError(f"serving pilot record {index} lacks exact output digest")
        workload.append(signature)
        outputs.append({"request_id": request_id, "text_sha256": text_digest})
    workload.sort(key=lambda item: item["request_id"])
    outputs.sort(key=lambda item: item["request_id"])
    return workload, outputs


def _workload_family(report: Mapping[str, Any], requests: list[dict[str, Any]]) -> Any:
    workload = report.get("workload")
    provenance: dict[str, Any] | None = None
    if isinstance(workload, dict):
        provenance = {
            key: workload[key]
            for key in (
                "manifest_digest",
                "records_digest",
                "token_input_identity_digest",
            )
            if key in workload
        }
    return {
        "explicit_family": report.get("workload_family"),
        "provenance": provenance,
        "configuration": {
            field: report.get(field) for field in _WORKLOAD_CONFIGURATION_FIELDS
        },
        # Completion order can change under saturation; request identity and
        # exact shape must not.
        "requests": requests,
    }


def _load_observation(spec: PilotInput, rule: FreezeRule) -> _Observation:
    rate = _finite(spec.offered_rate, "offered request rate", positive=True)
    path = spec.path.resolve()
    raw = path.read_bytes()
    report = _strict_json(raw, path)
    _reject_nta_data(report)
    if (
        report.get("schema") != 1
        or report.get("classification") != "sglang-hicache-load"
    ):
        raise ValueError("saturation input must be one stock serving-arm report")
    if report.get("attention_backend") != "flashinfer":
        raise ValueError("saturation input is not the stock FlashInfer arm")
    if report.get("engine_stats") != []:
        raise ValueError("stock saturation input contains engine mechanism data")
    if report.get("dirty") is not False:
        raise ValueError("stock saturation pilot did not use a clean revision")
    revision = report.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in _HEX for character in revision)
    ):
        raise ValueError("stock saturation pilot lacks a full clean revision")
    machine = report.get("machine")
    if not isinstance(machine, dict) or not machine:
        raise ValueError("stock saturation pilot lacks machine identity")
    if report.get("demand_semantics") != "exact":
        raise ValueError("stock saturation pilot does not use exact demand")
    if report.get("placement_proven") is not True:
        raise ValueError("stock saturation pilot has unproven tier placement")
    if report.get("load_warmup_excluded") is not True:
        raise ValueError("stock saturation pilot includes load warmup")
    if report.get("verification_failures") != 0:
        raise ValueError("stock saturation pilot has verification failures")
    correctness = report.get("correctness")
    aggregate_output = report.get("generated_text_sha256")
    if (
        not isinstance(correctness, dict)
        or correctness.get("verification_failures") != 0
        or not _valid_sha256(correctness.get("generated_text_sha256"))
        or correctness["generated_text_sha256"] != aggregate_output
    ):
        raise ValueError("stock saturation pilot lacks exact correctness evidence")

    report_rate = _finite(
        report.get("request_rate"), "recorded request rate", positive=True
    )
    if not math.isclose(report_rate, rate, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("CLI offered rate disagrees with the stock report")
    arrival_schedule = report.get("arrival_schedule")
    if not isinstance(arrival_schedule, dict) or arrival_schedule.get("method") not in {
        "seeded_exponential",
        "manifest_exact",
        "uniform_manifest_time_dilation",
    }:
        raise ValueError("stock saturation pilot has no proved arrival schedule")
    schedule_rate = _finite(
        arrival_schedule.get("target_rate_per_second"),
        "arrival-schedule target rate",
        positive=True,
    )
    if not math.isclose(schedule_rate, rate, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("arrival schedule disagrees with the offered rate")
    if arrival_schedule.get("request_order_preserved") is not True:
        raise ValueError("arrival schedule does not preserve request order")
    scale = _finite(
        arrival_schedule.get("uniform_time_scale"),
        "arrival-schedule time scale",
        positive=True,
    )
    if arrival_schedule["method"] == "uniform_manifest_time_dilation":
        source_rate = _finite(
            arrival_schedule.get("source_target_rate_per_second"),
            "arrival-schedule source rate",
            positive=True,
        )
        _close(scale, source_rate / rate, "arrival-schedule time scale")
    records_value = report.get("records")
    if not isinstance(records_value, list) or any(
        not isinstance(record, dict) for record in records_value
    ):
        raise ValueError("stock saturation pilot has invalid request records")
    records: list[dict[str, Any]] = records_value
    if len(records) < rule.minimum_requests_per_report:
        raise ValueError("stock saturation pilot has too few requests for p99")
    requests, outputs = _request_identities(records)

    system_latencies: list[float] = []
    for index, record in enumerate(records):
        system_latencies.append(
            _finite(
                record.get("system_time_seconds"),
                f"record {index} system latency",
                positive=True,
            )
        )
        if (
            record.get("token_timestamps_exact") is not True
            or _integer(
                record.get("itl_sample_count"), f"record {index} ITL count", minimum=1
            )
            != _integer(
                record.get("completion_tokens"), f"record {index} completion tokens"
            )
            - 1
        ):
            raise ValueError("stock saturation pilot lacks exact token timing")
        for field in ("ttft_seconds", "tpot_seconds", "p99_itl_seconds"):
            _finite(record.get(field), f"record {index} {field}", positive=True)

    p95 = _percentile(system_latencies, 0.95)
    p99 = _percentile(system_latencies, 0.99)
    if "p95_system_time_seconds" in report:
        _close(report["p95_system_time_seconds"], p95, "p95 system latency")
    if "p99_system_time_seconds" in report:
        _close(report["p99_system_time_seconds"], p99, "p99 system latency")

    elapsed = _finite(report.get("elapsed_seconds"), "elapsed time", positive=True)
    throughput = _finite(
        report.get("request_throughput"), "achieved request throughput", positive=True
    )
    _close(throughput, len(records) / elapsed, "request throughput")
    if throughput > rate * (1.0 + rule.maximum_throughput_overshoot_fraction):
        raise ValueError("achieved throughput exceeds offered rate beyond tolerance")

    goodput = report.get("slo_goodput")
    if not isinstance(goodput, dict):
        raise ValueError("stock saturation pilot lacks SLO attainment")
    thresholds = goodput.get("thresholds_seconds")
    if not isinstance(thresholds, dict):
        raise ValueError("stock saturation pilot lacks fixed SLO thresholds")
    ttft_limit = _finite(thresholds.get("ttft"), "SLO TTFT threshold", positive=True)
    tpot_limit = _finite(thresholds.get("tpot"), "SLO TPOT threshold", positive=True)
    itl_limit = _finite(thresholds.get("p99_itl"), "SLO ITL threshold", positive=True)
    qualified = sum(
        float(record["ttft_seconds"]) <= ttft_limit
        and float(record["tpot_seconds"]) <= tpot_limit
        and float(record["p99_itl_seconds"]) <= itl_limit
        for record in records
    )
    if (
        _integer(goodput.get("qualified_requests"), "qualified request count")
        != qualified
        or _integer(goodput.get("total_requests"), "SLO request count") != len(records)
        or _integer(
            goodput.get("requests_with_token_level_itl"),
            "token-level SLO request count",
        )
        != len(records)
    ):
        raise ValueError("stock saturation SLO counts do not match request records")
    attainment = qualified / len(records)
    _close(goodput.get("attainment"), attainment, "SLO attainment")
    _close(
        goodput.get("goodput_requests_per_second"),
        qualified / elapsed,
        "SLO goodput",
    )
    return _Observation(
        rate,
        str(path),
        hashlib.sha256(raw).hexdigest(),
        throughput,
        p95,
        p99,
        attainment,
        len(records),
        aggregate_output,
        revision,
        _digest(machine),
        _digest(_workload_family(report, requests)),
        _digest(outputs),
        _digest(thresholds),
    )


def _relative_range(values: Sequence[float]) -> float:
    center = statistics.median(values)
    if center <= 0.0:
        raise ValueError("pilot replicate metric has no positive median")
    return (max(values) - min(values)) / center


def _aggregate(
    observations: Sequence[_Observation], rule: FreezeRule
) -> list[_Candidate]:
    by_rate: dict[float, list[_Observation]] = defaultdict(list)
    for observation in observations:
        by_rate[observation.offered_rate].append(observation)
    if len(by_rate) < 3:
        raise ValueError("saturation pilot needs at least three candidate rates")
    replicate_counts = {len(values) for values in by_rate.values()}
    if len(replicate_counts) != 1:
        raise ValueError("candidate rates have unbalanced replicate counts")
    if min(replicate_counts) < rule.minimum_replicates_per_rate:
        raise ValueError("candidate rate has too few pilot replicates")

    candidates: list[_Candidate] = []
    for rate in sorted(by_rate):
        values = by_rate[rate]
        throughputs = [value.achieved_request_throughput for value in values]
        p95_values = [value.p95_system_latency_seconds for value in values]
        p99_values = [value.p99_system_latency_seconds for value in values]
        attainments = [value.slo_attainment for value in values]
        for name, samples in (
            ("throughput", throughputs),
            ("p95 system latency", p95_values),
            ("p99 system latency", p99_values),
        ):
            if _relative_range(samples) > rule.maximum_replicate_relative_range:
                raise ValueError(f"unstable {name} replicates at offered rate {rate:g}")
        if max(attainments) - min(attainments) > rule.maximum_slo_replicate_range:
            raise ValueError(f"unstable SLO replicates at offered rate {rate:g}")
        throughput = statistics.median(throughputs)
        candidates.append(
            _Candidate(
                rate,
                len(values),
                throughput,
                throughput / rate,
                statistics.median(p95_values),
                statistics.median(p99_values),
                statistics.median(attainments),
                tuple(sorted(value.sha256 for value in values)),
            )
        )
    return candidates


def _validate_sequence(candidates: Sequence[_Candidate], rule: FreezeRule) -> None:
    for lower, upper in zip(candidates, candidates[1:]):
        if upper.offered_rate / lower.offered_rate > rule.maximum_rate_step_factor:
            raise ValueError("candidate rate grid is too coarse to freeze a knee")
        if upper.achieved_request_throughput < lower.achieved_request_throughput * (
            1.0 - rule.maximum_throughput_reversal_fraction
        ):
            raise ValueError("unstable achieved-throughput reversal across rates")
        if upper.p95_system_latency_seconds < lower.p95_system_latency_seconds * (
            1.0 - rule.maximum_latency_reversal_fraction
        ):
            raise ValueError("unstable p95 system-latency reversal across rates")
        if upper.p99_system_latency_seconds < lower.p99_system_latency_seconds * (
            1.0 - rule.maximum_latency_reversal_fraction
        ):
            raise ValueError("unstable p99 system-latency reversal across rates")
        if upper.slo_attainment > lower.slo_attainment + rule.maximum_slo_reversal:
            raise ValueError("unstable SLO-attainment reversal across rates")


def _saturation_signals(
    knee: _Candidate, candidate: _Candidate, rule: FreezeRule
) -> dict[str, Any]:
    return {
        "throughput_shortfall": (
            1.0 - candidate.throughput_efficiency >= rule.throughput_shortfall_fraction
        ),
        "p95_growth": (
            candidate.p95_system_latency_seconds
            >= knee.p95_system_latency_seconds * rule.p95_growth_factor
        ),
        "p99_growth": (
            candidate.p99_system_latency_seconds
            >= knee.p99_system_latency_seconds * rule.p99_growth_factor
        ),
        "slo_drop": (
            knee.slo_attainment - candidate.slo_attainment >= rule.slo_attainment_drop
        ),
        "p95_growth_ratio": (
            candidate.p95_system_latency_seconds / knee.p95_system_latency_seconds
        ),
        "p99_growth_ratio": (
            candidate.p99_system_latency_seconds / knee.p99_system_latency_seconds
        ),
        "slo_attainment_drop": knee.slo_attainment - candidate.slo_attainment,
    }


def build_freeze(inputs: Sequence[PilotInput], rule: FreezeRule) -> dict[str, Any]:
    if not inputs:
        raise ValueError("no stock saturation pilots were provided")
    resolved_paths = [item.path.resolve() for item in inputs]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("one serving pilot file was supplied more than once")
    observations = sorted(
        (_load_observation(item, rule) for item in inputs),
        key=lambda value: (value.offered_rate, value.path, value.sha256),
    )
    identity_fields = (
        "revision",
        "machine_digest",
        "workload_family_digest",
        "output_identity_digest",
        "slo_thresholds_digest",
    )
    for field in identity_fields:
        if len({getattr(value, field) for value in observations}) != 1:
            label = field.replace("_", " ")
            raise ValueError(f"stock saturation pilots do not share one {label}")

    candidates = _aggregate(observations, rule)
    _validate_sequence(candidates, rule)
    efficiency_floor = 1.0 - rule.throughput_shortfall_fraction
    if (
        candidates[0].throughput_efficiency <= efficiency_floor
        or candidates[0].slo_attainment < rule.minimum_presaturation_slo_attainment
    ):
        raise ValueError("lowest stock pilot rate is already saturated")

    transition_index: int | None = None
    transition_signals: dict[str, Any] | None = None
    for index in range(1, len(candidates)):
        knee = candidates[index - 1]
        candidate = candidates[index]
        if any(
            value.throughput_efficiency <= efficiency_floor
            or value.slo_attainment < rule.minimum_presaturation_slo_attainment
            for value in candidates[:index]
        ):
            break
        signals = _saturation_signals(knee, candidate, rule)
        if all(
            signals[name]
            for name in (
                "throughput_shortfall",
                "p95_growth",
                "p99_growth",
                "slo_drop",
            )
        ):
            transition_index = index
            transition_signals = signals
            break
    if transition_index is None or transition_signals is None:
        raise ValueError("no stable multi-signal stock saturation knee was found")

    knee = candidates[transition_index - 1]
    minimum_overload_rate = knee.offered_rate * (
        1.0 + rule.minimum_overload_margin_fraction
    )
    selected: _Candidate | None = None
    selected_signals: dict[str, Any] | None = None
    for candidate in candidates[transition_index:]:
        signals = _saturation_signals(knee, candidate, rule)
        if candidate.offered_rate >= minimum_overload_rate and all(
            signals[name]
            for name in (
                "throughput_shortfall",
                "p95_growth",
                "p99_growth",
                "slo_drop",
            )
        ):
            selected = candidate
            selected_signals = signals
            break
    if selected is None or selected_signals is None:
        raise ValueError("no measured formal overload rate exists above the knee")

    first = observations[0]
    return {
        "schema": 1,
        "classification": CLASSIFICATION,
        "arm": "stock",
        "revision": first.revision,
        "machine_digest": first.machine_digest,
        "workload_family_digest": first.workload_family_digest,
        "output_identity_digest": first.output_identity_digest,
        "slo_thresholds_digest": first.slo_thresholds_digest,
        "rule": {
            "name": RULE_NAME,
            "thresholds": asdict(rule),
            "knee_definition": (
                "highest measured sustainable stock rate immediately before the "
                "first throughput-shortfall, p95-growth, p99-growth, and SLO-drop "
                "transition"
            ),
            "overload_selection": (
                "smallest measured rate above the knee and minimum margin that "
                "retains all four stock saturation signals"
            ),
            "aggregation": "median_per_rate_with_balanced_replicates",
        },
        "inputs": [asdict(value) for value in observations],
        "candidates": [asdict(value) for value in candidates],
        "knee": {
            "offered_rate": knee.offered_rate,
            "candidate": asdict(knee),
            "saturation_onset_rate": candidates[transition_index].offered_rate,
            "onset_signals": transition_signals,
        },
        "formal_overload": {
            "offered_rate": selected.offered_rate,
            "candidate": asdict(selected),
            "signals_relative_to_knee": selected_signals,
            "input_sha256": list(selected.input_sha256),
        },
        "frozen_overload_rate": selected.offered_rate,
    }


def freeze_overload_rate(
    inputs: Sequence[PilotInput], output: Path, rule: FreezeRule
) -> dict[str, Any]:
    destination = output.resolve()
    if destination in {item.path.resolve() for item in inputs}:
        raise ValueError("freeze output cannot overwrite a pilot input")
    result = build_freeze(inputs, rule)
    atomic_write_json(destination, result)
    return result


def _pilot_input(value: str) -> PilotInput:
    try:
        raw_rate, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("input must be RATE=PATH") from error
    if not raw_path:
        raise argparse.ArgumentTypeError("input path cannot be empty")
    try:
        rate = float(raw_rate)
    except ValueError as error:
        raise argparse.ArgumentTypeError("input rate must be numeric") from error
    if not math.isfinite(rate) or rate <= 0.0:
        raise argparse.ArgumentTypeError("input rate must be finite and positive")
    return PilotInput(rate, Path(raw_path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=_pilot_input)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--throughput-shortfall-fraction", type=float, default=0.10)
    parser.add_argument("--p95-growth-factor", type=float, default=1.50)
    parser.add_argument("--p99-growth-factor", type=float, default=1.50)
    parser.add_argument("--slo-attainment-drop", type=float, default=0.05)
    parser.add_argument(
        "--minimum-presaturation-slo-attainment", type=float, default=0.95
    )
    parser.add_argument("--minimum-overload-margin-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-rate-step-factor", type=float, default=2.0)
    parser.add_argument(
        "--maximum-throughput-overshoot-fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--maximum-throughput-reversal-fraction", type=float, default=0.15
    )
    parser.add_argument("--maximum-latency-reversal-fraction", type=float, default=0.20)
    parser.add_argument("--maximum-slo-reversal", type=float, default=0.05)
    parser.add_argument("--maximum-replicate-relative-range", type=float, default=0.10)
    parser.add_argument("--maximum-slo-replicate-range", type=float, default=0.05)
    parser.add_argument("--minimum-replicates-per-rate", type=int, default=1)
    parser.add_argument("--minimum-requests-per-report", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        rule = FreezeRule(
            throughput_shortfall_fraction=args.throughput_shortfall_fraction,
            p95_growth_factor=args.p95_growth_factor,
            p99_growth_factor=args.p99_growth_factor,
            slo_attainment_drop=args.slo_attainment_drop,
            minimum_presaturation_slo_attainment=(
                args.minimum_presaturation_slo_attainment
            ),
            minimum_overload_margin_fraction=args.minimum_overload_margin_fraction,
            maximum_rate_step_factor=args.maximum_rate_step_factor,
            maximum_throughput_overshoot_fraction=(
                args.maximum_throughput_overshoot_fraction
            ),
            maximum_throughput_reversal_fraction=(
                args.maximum_throughput_reversal_fraction
            ),
            maximum_latency_reversal_fraction=(args.maximum_latency_reversal_fraction),
            maximum_slo_reversal=args.maximum_slo_reversal,
            maximum_replicate_relative_range=(args.maximum_replicate_relative_range),
            maximum_slo_replicate_range=args.maximum_slo_replicate_range,
            minimum_replicates_per_rate=args.minimum_replicates_per_rate,
            minimum_requests_per_report=args.minimum_requests_per_report,
        )
        freeze_overload_rate(args.input, args.output, rule)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"serving_overload_rate_frozen={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
