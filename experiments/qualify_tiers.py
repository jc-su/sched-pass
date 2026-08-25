#!/usr/bin/env python3
"""Assemble native and transport reports into one qualified-tier artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .validate_tier_qualification import ALL_TIERS, _read_report, validate
except ImportError:  # Direct CLI execution.
    from validate_tier_qualification import ALL_TIERS, _read_report, validate


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(
    reports: dict[str, Path],
    output: Path,
    *,
    required_tiers: tuple[str, ...] = ALL_TIERS,
) -> dict[str, Any]:
    required_tiers = tuple(dict.fromkeys(required_tiers))
    if not required_tiers:
        raise ValueError("at least one tier must be required")
    entries = []
    for tier in required_tiers:
        if tier not in reports:
            raise ValueError(f"missing report for required tier: {tier}")
        path = reports[tier].resolve()
        report = _read_report(path)
        qualified = report.get("qualified", report.get("ready"))
        if qualified is None and isinstance(report.get("qualification"), dict):
            qualified = report["qualification"].get("qualified")
        entries.append(
            {
                "tier": tier,
                "status": "qualified" if qualified is True else "not_qualified",
                "qualified": qualified is True,
                "source": {"path": str(path), "sha256": _digest(path)},
                "report": report,
            }
        )
    document: dict[str, Any] = {
        "schema": 1,
        "classification": "nta-tier-qualification",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "demand_semantics": "exact",
        "required_tiers": list(required_tiers),
        "entries": entries,
        "policy": {
            "missing_hardware": "skip_not_pass",
            "regular_file_dax": "reject",
            "nvme_without_translated_iommu": "reject",
        },
    }
    validate(document, required_tiers=required_tiers)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for tier in ALL_TIERS:
        parser.add_argument(f"--{tier.replace('_', '-')}-report", type=Path)
    parser.add_argument(
        "--required-tier",
        action="append",
        choices=ALL_TIERS,
        help="required tier (repeatable; default: all four tiers)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required_tiers = tuple(args.required_tier or ALL_TIERS)
    assemble(
        {
            tier: getattr(args, f"{tier.replace('-', '_')}_report")
            for tier in required_tiers
            if getattr(args, f"{tier.replace('-', '_')}_report") is not None
        },
        args.output,
        required_tiers=required_tiers,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
