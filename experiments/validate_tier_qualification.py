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
        gpu.get("hbm_mapping_backend") == "nvidia-peer-pages",
        "NVMe report does not prove the peer-page mapping backend",
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
