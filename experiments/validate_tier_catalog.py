#!/usr/bin/env python3
"""Validate an exact physical-tier page catalog without opening hardware."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.tier import (  # noqa: E402
    PHYSICAL_SERVING_TIERS,
    ServingTier,
    TierPageCatalog,
)


def validate(path: Path, tier: str) -> dict[str, object]:
    try:
        selected = ServingTier(tier)
    except ValueError as error:
        raise ValueError(f"unsupported catalog tier {tier!r}") from error
    catalog = TierPageCatalog.load(path.resolve(), expected_tier=selected)
    return {
        "schema": TierPageCatalog.SCHEMA,
        "tier": catalog.tier.value,
        "format": catalog.FORMAT,
        "namespace": catalog.namespace,
        "page_tokens": catalog.page_tokens,
        "layer_count": catalog.layer_count,
        "components": list(catalog.components),
        "storage_keys": catalog.page_count,
        "alignment_bytes": catalog.alignment_bytes,
        "window_bytes": catalog.window_bytes,
        "digest": catalog.digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument(
        "--tier",
        required=True,
        choices=sorted(tier.value for tier in PHYSICAL_SERVING_TIERS),
    )
    args = parser.parse_args()
    report = validate(args.catalog, args.tier)
    print(
        "tier_catalog=valid "
        f"storage_keys={report['storage_keys']} "
        f"namespace={report['namespace']} digest={report['digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
