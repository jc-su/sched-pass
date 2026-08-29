"""Canonical causal arms and result-derived activation proofs.

The serving mechanism has one production form: exact heterogeneous work is
bound to acquisition groups and released when its dependencies are ready.
Transport engine, frontier depth, granularity, tier, and workload skew are
orthogonal axes; they are not separate mechanisms.  The four arms below are
the minimum causal decomposition needed to distinguish framework control,
exact preacquisition, device-side demand discovery, and progressive work-unit
consumption.

An arm name is never accepted as evidence.  :func:`validate_arm_result`
reconstructs the executed form from the timed report and its engine counters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ARMS = ("A0", "A1", "A2", "A3")

ARM_DEFINITIONS: dict[str, dict[str, Any]] = {
    "A0": {
        "name": "framework-bulk",
        "consumer_kind": "framework_reference",
        "role": "upstream SGLang HiCache control",
    },
    "A1": {
        "name": "exact-preacquired-stock",
        "consumer_kind": "framework_reference",
        "role": "exact NTA acquisition with the stock numerical consumer",
    },
    "A2": {
        "name": "device-discovery-bulk",
        "consumer_kind": "native_work_unit",
        "role": "GPU demand discovery and exact acquisition with one bulk readiness boundary",
    },
    "A3": {
        "name": "late-bound-work-unit",
        "consumer_kind": "native_work_unit",
        "role": "exact heterogeneous work released at acquisition-group readiness",
    },
}

CAUSAL_PAIRS = (
    ("A1", "A0", "exact_acquisition_boundary"),
    ("A2", "A1", "device_discovery_boundary"),
    ("A3", "A2", "late_bound_work_unit_boundary"),
)

FORMAL_SERVING_METRICS = (
    "slo_goodput_requests_per_second",
    "slo_attainment",
    "request_throughput",
    "output_token_throughput",
    "p50_ttft_seconds",
    "p95_ttft_seconds",
    "p99_ttft_seconds",
    "p50_tpot_seconds",
    "p95_tpot_seconds",
    "p99_tpot_seconds",
    "p99_itl_seconds",
    "mean_admission_delay_seconds",
    "p95_admission_delay_seconds",
    "verification_failures",
)


def arm_environment(arm: str) -> dict[str, str]:
    """Return the explicit execution form for one canonical arm.

    The caller still owns tier, mover, graph, and verification settings.  A2
    names a real device-bulk form; accepting the configuration before that
    form is implemented would turn an intended ablation into a mislabeled A3
    run, so framework adapters must fail closed on unknown values.
    """

    if arm == "A0":
        return {}
    if arm == "A1":
        return {
            "NTA_EXECUTION_PROTOCOL": "late_bound",
            "NTA_EXECUTION_HOST_FORM": "direct",
            "NTA_EXECUTION_HOST_MOVER": "sm",
            "NTA_EXECUTION_CALIBRATION_PROBES": "0",
        }
    if arm == "A2":
        return {
            "NTA_EXECUTION_PROTOCOL": "late_bound",
            "NTA_EXECUTION_HOST_FORM": "device_bulk",
            "NTA_EXECUTION_HOST_MOVER": "sm",
            "NTA_EXECUTION_CALIBRATION_PROBES": "0",
            "NTA_EXECUTION_MAX_ROUNDS": "1",
        }
    if arm == "A3":
        return {
            "NTA_EXECUTION_PROTOCOL": "late_bound",
            "NTA_EXECUTION_HOST_FORM": "dependency_aware",
            "NTA_EXECUTION_HOST_MOVER": "sm",
            "NTA_EXECUTION_CALIBRATION_PROBES": "0",
        }
    raise ValueError(f"unknown mechanism arm: {arm!r}")


def arm_backend(arm: str) -> str:
    if arm not in ARM_DEFINITIONS:
        raise ValueError(f"unknown mechanism arm: {arm!r}")
    return "flashinfer" if arm == "A0" else "nta_flashinfer"


def arm_consumer_kind(arm: str) -> str:
    try:
        return str(ARM_DEFINITIONS[arm]["consumer_kind"])
    except KeyError as error:
        raise ValueError(f"unknown mechanism arm: {arm!r}") from error


def _counter(stats: Sequence[Mapping[str, Any]], name: str) -> int:
    total = 0
    for entry in stats:
        value = entry.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"mechanism counter {name!r} is not nonnegative")
        total += value
    return total


def _identity(stats: Sequence[Mapping[str, Any]], name: str) -> str | None:
    values = {
        str(entry[name])
        for entry in stats
        if name in entry and isinstance(entry[name], str) and entry[name]
    }
    if len(values) > 1:
        raise ValueError(f"engine workers disagree on mechanism identity {name!r}")
    return next(iter(values), None)


def _consumer_kind(report: Mapping[str, Any]) -> str | None:
    contract = report.get("consumer_contract")
    if isinstance(contract, Mapping) and isinstance(contract.get("kind"), str):
        return str(contract["kind"])
    return None


def _external_request_count(report: Mapping[str, Any]) -> int:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("mechanism report has no request records")
    if any(isinstance(record, Mapping) and "kind" in record for record in records):
        return sum(
            isinstance(record, Mapping) and record.get("kind") == "external"
            for record in records
        )
    # Natural replay intentionally does not force a placement label.  A
    # measured host-cached prefix is its result-emitted evidence that the
    # request exercised external acquisition.
    return sum(
        isinstance(record, Mapping) and int(record.get("host_cached_tokens", 0)) > 0
        for record in records
    )


def _heterogeneous_native_batch(report: Mapping[str, Any]) -> bool:
    measured = report.get("batch_heterogeneity")
    if isinstance(measured, Mapping):
        return bool(
            measured.get("proven") is True
            and measured.get("native_mixed_consumer_proven") is True
        )
    natural = report.get("heterogeneity")
    if not isinstance(natural, Mapping):
        return False
    engine = natural.get("engine_forward")
    return bool(
        natural.get("scope") == "batch_internal"
        and natural.get("batch_internal_geometry_proven") is True
        and natural.get("batch_internal_availability_proven") is True
        and int(natural.get("maximum_concurrent_requests", 0)) > 1
        and isinstance(engine, Mapping)
        and int(engine.get("mixed_dependency_layers", 0)) > 0
    )


def validate_arm_result(report: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Validate and summarize one measured causal arm.

    This function is deliberately independent of benchmark command lines and
    environment metadata.  Only the numerical report and timed-window engine
    counters can prove which form executed.
    """

    if arm not in ARM_DEFINITIONS:
        raise ValueError(f"unknown mechanism arm: {arm!r}")
    expected_backend = arm_backend(arm)
    if report.get("attention_backend") != expected_backend:
        raise ValueError(
            f"{arm} requires attention backend {expected_backend!r}, observed "
            f"{report.get('attention_backend')!r}"
        )
    if _external_request_count(report) <= 0:
        raise ValueError(f"{arm} did not measure an external request")
    expected_consumer = arm_consumer_kind(arm)
    observed_consumer = _consumer_kind(report)
    if observed_consumer != expected_consumer:
        raise ValueError(
            f"{arm} requires consumer {expected_consumer!r}, observed "
            f"{observed_consumer!r}"
        )

    raw_stats = report.get("engine_stats")
    if not isinstance(raw_stats, list) or any(
        not isinstance(entry, Mapping) for entry in raw_stats
    ):
        raise ValueError("mechanism report has invalid engine statistics")
    stats = [
        entry for entry in raw_stats if entry.get("backend") == "nta_flashinfer"
    ]
    if arm == "A0":
        if stats:
            raise ValueError("A0 framework control contains NTA engine statistics")
        return {
            "schema": 1,
            "arm": arm,
            "name": ARM_DEFINITIONS[arm]["name"],
            "attention_backend": expected_backend,
            "consumer_kind": observed_consumer,
            "external_requests": _external_request_count(report),
            "execution_form": "framework_bulk",
        }
    if not stats:
        raise ValueError(f"{arm} contains no NTA engine statistics")

    counters = {
        name: _counter(stats, name)
        for name in (
            "hicache_fallback_batches",
            "hicache_external_batches",
            "host_direct_batches",
            "host_device_bulk_batches",
            "host_incremental_batches",
            "external_launches",
            "native_external_attention_launches",
            "stock_prefetched_external_attention_launches",
            "ticketed_incremental_launches",
            "event_ordered_incremental_launches",
            "mixed_dependency_layers",
            "progressive_consumer_batch_observations",
            "progressive_consumer_batches",
            "progressive_consumer_layers",
            "request_acquisition_groups",
            "prefetch_mover_plan_calibration_probe_sm_leases",
            "prefetch_mover_plan_calibration_probe_copy_leases",
            "verified_operator_modules",
        )
    }
    if counters["hicache_fallback_batches"] != 0:
        raise ValueError(f"{arm} used a HiCache fallback")
    if counters["hicache_external_batches"] <= 0:
        raise ValueError(f"{arm} did not execute an external batch")
    if counters["external_launches"] != (
        counters["native_external_attention_launches"]
        + counters["stock_prefetched_external_attention_launches"]
    ):
        raise ValueError(f"{arm} external numerical accounting is not exact")
    if (
        counters["prefetch_mover_plan_calibration_probe_sm_leases"]
        + counters["prefetch_mover_plan_calibration_probe_copy_leases"]
        != 0
    ):
        raise ValueError(f"{arm} timed a host-mover calibration probe")
    if arm in {"A2", "A3"} and counters["verified_operator_modules"] <= 0:
        raise ValueError(f"{arm} did not verify its compiler/operator contract")

    protocol = _identity(stats, "execution_protocol")
    host_form = _identity(stats, "host_execution_mode")
    if protocol != "late_bound":
        raise ValueError(f"{arm} did not use the canonical exact protocol")

    if arm == "A1":
        valid = (
            host_form == "direct"
            and counters["host_direct_batches"] > 0
            and counters["host_device_bulk_batches"] == 0
            and counters["host_incremental_batches"] == 0
            and counters["stock_prefetched_external_attention_launches"] > 0
            and counters["native_external_attention_launches"] == 0
            and counters["ticketed_incremental_launches"] == 0
            and counters["event_ordered_incremental_launches"] == 0
            and counters["progressive_consumer_layers"] == 0
        )
        execution_form = "exact_preacquired_stock"
    elif arm == "A2":
        valid = (
            host_form == "device_bulk"
            and counters["host_device_bulk_batches"] > 0
            and counters["host_direct_batches"] == 0
            and counters["host_incremental_batches"] == 0
            and counters["native_external_attention_launches"] > 0
            and counters["stock_prefetched_external_attention_launches"] == 0
            and counters["ticketed_incremental_launches"] > 0
            and counters["request_acquisition_groups"] > 0
            and counters["event_ordered_incremental_launches"] == 0
            and counters["progressive_consumer_batches"] == 0
            and counters["progressive_consumer_layers"] == 0
        )
        execution_form = "device_discovery_bulk"
    else:
        valid = (
            host_form == "dependency_aware"
            and counters["host_incremental_batches"] > 0
            and counters["host_direct_batches"] == 0
            and counters["host_device_bulk_batches"] == 0
            and counters["native_external_attention_launches"] > 0
            # A proactive wave event is a useful partial-consumer path, but it
            # does not prove A3's device-scheduled acquisition boundary.  At
            # least one layer must materialize exact request acquisition groups
            # and submit them through the ticketed runtime.
            and counters["ticketed_incremental_launches"] > 0
            and counters["request_acquisition_groups"] > 0
            and counters["mixed_dependency_layers"] > 0
            and counters["progressive_consumer_batch_observations"] > 0
            and counters["progressive_consumer_batches"] > 0
            and counters["progressive_consumer_layers"] > 0
            and _heterogeneous_native_batch(report)
        )
        execution_form = "late_bound_heterogeneous_work_unit"
    if not valid:
        raise ValueError(
            f"{arm} timed counters do not prove {execution_form}: {counters}, "
            f"protocol={protocol!r}, host_form={host_form!r}"
        )
    return {
        "schema": 1,
        "arm": arm,
        "name": ARM_DEFINITIONS[arm]["name"],
        "attention_backend": expected_backend,
        "consumer_kind": observed_consumer,
        "execution_protocol": protocol,
        "host_execution_mode": host_form,
        "execution_form": execution_form,
        "external_requests": _external_request_count(report),
        "counters": counters,
    }
