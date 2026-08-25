#!/usr/bin/env python3
"""Normalize an anonymized Bailian trace into a reproducible workload bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .bailian import normalize, read_jsonl, write_workload
except ImportError:
    from bailian import normalize, read_jsonl, write_workload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--arrival-mode",
        choices=("trace", "batch_release", "calibrated_open_loop"),
        default="batch_release",
    )
    parser.add_argument("--arrival-reference", type=Path)
    parser.add_argument(
        "--timestamp-unit", choices=("auto", "seconds", "milliseconds"), default="auto"
    )
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--target-rate", type=float)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--max-requests",
        type=int,
        help=(
            "keep the first N source rows for a bounded replay; the full "
            "source digest remains recorded in the manifest"
        ),
    )
    parser.add_argument("--synthesize-prompts", action="store_true")
    parser.add_argument(
        "--state-policy",
        choices=("preserve", "root_resident"),
        default="preserve",
        help=(
            "preserve source request_state labels, or derive a reproducible "
            "root-resident/follow-up-external serving setup"
        ),
    )
    args = parser.parse_args(argv)
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    try:
        rows = read_jsonl(args.input)
        source_request_count = len(rows)
        if args.max_requests is not None:
            rows = rows[: args.max_requests]
            if not rows:
                raise ValueError("--max-requests selected no source rows")
        reference = (
            read_jsonl(args.arrival_reference) if args.arrival_reference else None
        )
        manifest, records = normalize(
            rows,
            arrival_mode=args.arrival_mode,
            timestamp_unit=args.timestamp_unit,
            time_scale=args.time_scale,
            target_rate=args.target_rate,
            seed=args.seed,
            reference_rows=reference,
            synthesize_prompts=args.synthesize_prompts,
            state_policy=args.state_policy,
        )
        # Keep the normalized bundle portable.  The source digest is the
        # immutable identity; an absolute checkout path would make a copied
        # artifact appear machine-specific even when its records are intact.
        manifest["source_file"] = args.input.name
        manifest["source_digest"] = (
            __import__("hashlib").sha256(args.input.read_bytes()).hexdigest()
        )
        manifest["selection"] = {
            "mode": "source_prefix" if args.max_requests is not None else "all_rows",
            "max_requests": args.max_requests,
            "source_request_count": source_request_count,
        }
        write_workload(args.manifest, args.records, manifest, records)
    except (OSError, ValueError) as error:
        print(f"prepare_bailian failed: {error}", file=sys.stderr)
        return 2
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
