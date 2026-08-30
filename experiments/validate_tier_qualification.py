#!/usr/bin/env python3
"""Validate the machine-readable qualification contract for physical tiers.

The native attention executable, the VFIO-NVMe qualification runner, and the
devdax probe have deliberately different output formats.  This validator is
the single admission boundary for an evaluation: a skipped or failed device
is never silently promoted to a passing tier result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


ALL_TIERS = ("hbm", "host_mem", "nvme", "dax")
NATIVE_CLASSIFICATION = "nta-paged-attention"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_report(path: Path) -> dict[str, Any]:
    """Read a JSON report or the last JSON object in a native stdout log."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read tier report {path}: {error}") from error
    candidates = [text.strip()]
    candidates.extend(line.strip() for line in text.splitlines())
    for candidate in reversed(candidates):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"tier report {path} contains no JSON object")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_physical_identity(report: dict[str, Any], tier: str) -> None:
    identity = report.get("platform_identity")
    _require(isinstance(identity, dict), f"{tier} report has no platform identity")
    _require(
        isinstance(identity.get("boot_id"), str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            identity["boot_id"],
        )
        is not None,
        f"{tier} report has no valid boot identity",
    )
    _require(
        isinstance(identity.get("kernel"), str) and bool(identity["kernel"]),
        f"{tier} report has no kernel identity",
    )
    drivers = identity.get("nvidia_driver_versions")
    _require(
        isinstance(drivers, list)
        and bool(drivers)
        and all(isinstance(value, str) and value for value in drivers),
        f"{tier} report has no NVIDIA driver identity",
    )


def _validate_native(report: dict[str, Any], tier: str) -> None:
    _require(
        report.get("classification") == NATIVE_CLASSIFICATION,
        f"{tier} report is not native paged-attention output",
    )
    _require(
        report.get("tier") == tier,
        f"native report declares tier {report.get('tier')!r}, expected {tier!r}",
    )
    _require(
        report.get("demand_semantics") == "exact", f"{tier} report is not exact-demand"
    )
    _require(
        report.get("verification_failures") == 0,
        f"{tier} report has verification failures",
    )
    qualification = report.get("qualification")
    _require(
        isinstance(qualification, dict), f"{tier} report has no qualification object"
    )
    _require(
        qualification.get("qualified") is True,
        f"{tier} native qualification is not true",
    )
    _require(
        _finite(report.get("graph_ms")) and float(report["graph_ms"]) >= 0,
        f"{tier} report has no finite timing",
    )


def _validate_nvme_tuning(report: dict[str, Any], gpu: dict[str, Any]) -> None:
    """Prove that the reported datapath and baseline are the calibrated winners."""
    identity = report.get("target_identity")
    _require(isinstance(identity, dict), "NVMe report has no stable target identity")
    bdf = identity.get("bdf")
    _require(
        isinstance(bdf, str)
        and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf)
        is not None,
        "NVMe target identity has no valid PCI BDF",
    )
    for field in ("model", "serial", "kernel_driver", "block_device"):
        _require(
            isinstance(identity.get(field), str) and bool(identity[field]),
            f"NVMe target identity has no {field}",
        )
    _require(
        gpu.get("device") == f"vfio:{bdf}"
        and gpu.get("namespace_id") == identity.get("namespace_id"),
        "NVMe GPU result does not match the identified namespace",
    )

    baseline = report.get("baseline")
    baseline_candidates = report.get("baseline_candidates")
    _require(
        isinstance(baseline, dict)
        and isinstance(baseline_candidates, list)
        and len(baseline_candidates) >= 2
        and all(isinstance(candidate, dict) for candidate in baseline_candidates),
        "NVMe report has no baseline calibration",
    )
    baseline_depths = [
        candidate.get("queue_depth") for candidate in baseline_candidates
    ]
    _require(
        baseline.get("block_device") == identity.get("block_device")
        and all(
            candidate.get("block_device") == identity.get("block_device")
            for candidate in baseline_candidates
        )
        and all(isinstance(depth, int) and depth > 0 for depth in baseline_depths)
        and len(set(baseline_depths)) == len(baseline_depths)
        and all(
            candidate.get("block_bytes") == gpu.get("bytes_per_request")
            for candidate in baseline_candidates
        ),
        "NVMe fio calibration used a different namespace",
    )
    baseline_bandwidths = [
        candidate.get("bandwidth_mib_per_second") for candidate in baseline_candidates
    ]
    _require(
        all(_finite(value) and float(value) > 0 for value in baseline_bandwidths)
        and _finite(baseline.get("bandwidth_mib_per_second"))
        and math.isclose(
            float(baseline["bandwidth_mib_per_second"]),
            max(float(value) for value in baseline_bandwidths),
            rel_tol=1e-9,
        ),
        "NVMe report did not select the fastest fio baseline",
    )

    calibration = report.get("gpu_queue_depth_calibration")
    _require(
        isinstance(calibration, list) and len(calibration) >= 2,
        "NVMe report has no GPU queue-depth calibration",
    )
    depths: set[int] = set()
    winners: list[tuple[float, int]] = []
    for candidate in calibration:
        _require(isinstance(candidate, dict), "NVMe GPU calibration is malformed")
        depth = candidate.get("queue_depth")
        samples = candidate.get("bandwidth_mib_per_second")
        _require(
            isinstance(depth, int)
            and depth >= 2
            and depth not in depths
            and isinstance(samples, list)
            and len(samples) >= 3
            and len(samples) % 2 == 1
            and candidate.get("trials") == len(samples)
            and all(_finite(value) and float(value) > 0 for value in samples),
            "NVMe GPU calibration candidate is invalid",
        )
        depths.add(depth)
        median = sorted(float(value) for value in samples)[len(samples) // 2]
        _require(
            _finite(candidate.get("median_bandwidth_mib_per_second"))
            and math.isclose(
                float(candidate["median_bandwidth_mib_per_second"]),
                median,
                rel_tol=1e-9,
            ),
            "NVMe GPU calibration median is inconsistent",
        )
        winners.append((median, depth))
    selected_median, selected_depth = min(winners, key=lambda item: (-item[0], item[1]))
    recommendation = report.get("recommended_serving_config")
    _require(
        gpu.get("queue_depth") == selected_depth
        and _finite(gpu.get("end_to_end_mib_per_second"))
        and math.isclose(
            float(gpu["end_to_end_mib_per_second"]), selected_median, rel_tol=1e-9
        )
        and isinstance(recommendation, dict)
        and recommendation.get("NTA_NVME_QUEUE_DEPTH") == selected_depth
        and recommendation.get("transfer_bytes") == gpu.get("bytes_per_request")
        and recommendation.get("selection_metric")
        == "median_exact_end_to_end_mib_per_second",
        "NVMe report did not select its calibrated GPU queue depth",
    )
    ratio = report.get("matched_bandwidth_ratio")
    _require(
        _finite(ratio)
        and math.isclose(
            float(ratio),
            selected_median / float(baseline["bandwidth_mib_per_second"]),
            rel_tol=1e-9,
        ),
        "NVMe matched bandwidth ratio is inconsistent",
    )


def _validate_entry(entry: dict[str, Any], tier: str) -> None:
    _require(
        entry.get("tier") == tier,
        f"qualification entry declares {entry.get('tier')!r}, expected {tier!r}",
    )
    _require(
        entry.get("status") == "qualified",
        f"{tier} qualification status is {entry.get('status')!r}, not qualified",
    )
    _require(entry.get("qualified") is True, f"{tier} qualification is not true")
    report = entry.get("report")
    _require(isinstance(report, dict), f"{tier} qualification has no embedded report")
    classification = report.get("classification")
    if tier in {"hbm", "host_mem"}:
        _validate_native(report, tier)
        return
    if tier == "dax":
        _require(
            classification == "nta-dax-qualification",
            "DAX report is not the devdax qualification report",
        )
        _require(
            report.get("qualified") is True, "DAX mapping qualification is not true"
        )
        _require(
            report.get("verification_failures") == 0,
            "DAX qualification has verification failures",
        )
        _require(
            report.get("direct_device_visible") is True,
            "DAX report does not prove CUDA visibility",
        )
        return
    _require(
        classification == "nta-vfio-nvme-qualification",
        "NVMe report is not the VFIO qualification report",
    )
    _validate_physical_identity(report, tier)
    _require(
        isinstance(report.get("revision"), str)
        and re.fullmatch(r"[0-9a-f]{40}", report["revision"]) is not None,
        "NVMe report has no immutable git revision",
    )
    _require(report.get("dirty") is False, "NVMe report was produced from dirty code")
    if "provenance_ready" in report:
        _require(
            report.get("provenance_ready") is True,
            "NVMe qualification provenance is not ready",
        )
    _require(
        report.get("demand_semantics") == "exact", "NVMe report is not exact-demand"
    )
    _require(
        report.get("transport_ready", report.get("ready")) is True
        and report.get("qualified") is True,
        "NVMe transport qualification is not ready",
    )
    gpu = report.get("gpu_controlled")
    _require(isinstance(gpu, dict), "NVMe report has no GPU-controlled result")
    runtime_abi = report.get("runtime_abi")
    _require(
        isinstance(runtime_abi, int)
        and runtime_abi > 0
        and gpu.get("runtime_abi") == runtime_abi,
        "NVMe report does not match its runtime ABI",
    )
    _require(
        gpu.get("revision") == report.get("revision"),
        "NVMe GPU result does not match its qualification revision",
    )
    for field in (
        "verified",
        "selected_data_path_verified",
        "translated_iommu",
        "gpu_doorbell_mapping_validated",
        "hbm_peer_dma_supported",
    ):
        _require(gpu.get(field) is True, f"NVMe report does not prove {field}")
    _require(
        gpu.get("destination") == "hbm-peer",
        "NVMe qualification is not the direct-HBM data path",
    )
    _require(
        gpu.get("hbm_mapping_backend") in {"cuda-dmabuf-ioas", "nvidia-peer-pages"},
        "NVMe report does not prove a direct-HBM mapping backend",
    )
    required_backend = report.get("required_hbm_backend")
    selected_backend = report.get("selected_hbm_backend")
    _require(
        required_backend in {"auto", "cuda-dmabuf-ioas", "nvidia-peer-pages"},
        "NVMe report has no valid native HBM mapping requirement",
    )
    _require(
        report.get("reported_hbm_mapping_policy") == required_backend
        and gpu.get("hbm_mapping_policy") == required_backend,
        "NVMe report does not prove its HBM policy was enforced natively",
    )
    _require(
        selected_backend == gpu.get("hbm_mapping_backend"),
        "NVMe report contradicts its selected HBM mapping backend",
    )
    _require(
        required_backend == "auto" or selected_backend == required_backend,
        "NVMe selected backend does not satisfy its explicit mapping policy",
    )
    _require(
        report.get("iommu_fault_free") is True,
        "NVMe qualification observed a new target IOMMU fault",
    )
    _require(
        gpu.get("verification_failures") == 0 and gpu.get("failed") == 0,
        "NVMe GPU-controlled run has failures",
    )
    _require(
        gpu.get("outstanding") == 0, "NVMe GPU-controlled run has outstanding commands"
    )
    _validate_nvme_tuning(report, gpu)
    ratio = report.get("matched_bandwidth_ratio")
    _require(_finite(ratio), "NVMe report has no finite matched bandwidth ratio")
    threshold = report.get("minimum_bandwidth_ratio")
    _require(
        _finite(threshold) and float(threshold) > 0,
        "NVMe report has no finite performance threshold",
    )
    if "performance_qualified" in report:
        _require(
            isinstance(report["performance_qualified"], bool),
            "NVMe performance qualification flag is not boolean",
        )
        _require(
            report["performance_qualified"] == (float(ratio) >= float(threshold)),
            "NVMe performance qualification contradicts its matched ratio",
        )
    _require(
        report.get("performance_qualified") is True,
        "NVMe qualification does not meet its performance threshold",
    )


def validate(
    document: dict[str, Any], *, required_tiers: Iterable[str] = ALL_TIERS
) -> dict[str, Any]:
    """Validate and return a tier qualification document."""

    required = tuple(dict.fromkeys(required_tiers))
    _require(
        document.get("schema") == 1
        and document.get("classification") == "nta-tier-qualification",
        "unsupported tier qualification document",
    )
    _require(
        document.get("demand_semantics") == "exact",
        "tier qualification must use exact demand",
    )
    declared_required = document.get("required_tiers")
    _require(
        isinstance(declared_required, list),
        "tier qualification has no required_tiers declaration",
    )
    _require(
        tuple(declared_required) == required,
        "tier qualification required_tiers does not match the validation scope",
    )
    entries = document.get("entries")
    _require(isinstance(entries, list), "tier qualification has no entries")
    by_tier: dict[str, dict[str, Any]] = {}
    for entry in entries:
        _require(isinstance(entry, dict), "tier qualification entry is not an object")
        tier = entry.get("tier")
        _require(tier in ALL_TIERS, f"unknown qualification tier: {tier!r}")
        _require(tier not in by_tier, f"duplicate qualification tier: {tier}")
        by_tier[tier] = entry
    for tier in required:
        _require(tier in ALL_TIERS, f"unknown required tier: {tier}")
        _require(tier in by_tier, f"qualification document lacks required tier: {tier}")
        _validate_entry(by_tier[tier], tier)
    document["required_tiers"] = list(required)
    return document


def validate_file(
    path: Path, *, required_tiers: Iterable[str] = ALL_TIERS
) -> dict[str, Any]:
    return validate(_read_report(path.resolve()), required_tiers=required_tiers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path)
    parser.add_argument(
        "--required-tier",
        action="append",
        dest="required_tiers",
        choices=ALL_TIERS,
        help="tier that must be qualified (repeatable; default: all)",
    )
    args = parser.parse_args()
    validate_file(args.qualification, required_tiers=args.required_tiers or ALL_TIERS)
    print("tier_qualification=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
