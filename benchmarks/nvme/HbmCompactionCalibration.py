#!/usr/bin/env python3
"""Calibrate the cold-HBM service envelope for exact NVMe compaction.

The production selector needs a launch cost and an effective read+write
bandwidth.  Replaying one address table measures L2, not transport scratch, so
this benchmark samples non-repeating rows from a pool larger than cache and
reports a conservative p90 envelope.  It runs once during deployment or
artifact qualification; no request content is compiled or probed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import ceil, floor
from pathlib import Path
import random
import statistics
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.atomic_io import atomic_write_text  # noqa: E402
from nta_runtime.runtime import JitPhaseProgram  # noqa: E402


@dataclass(frozen=True, slots=True)
class _Point:
    rows: int
    median_ns: int
    p90_ns: int

    @property
    def exact_bytes(self) -> int:
        return self.rows * 4096


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("program", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pool-mib", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0x4E5441)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.gpu < 0 or min(arguments.pool_mib, arguments.trials) <= 0:
        parser.error("GPU, pool size, and trial count must be valid")
    if arguments.trials < 11 or arguments.trials % 2 == 0:
        parser.error("calibration requires an odd trial count of at least 11")
    return arguments


def _percentile(samples: list[float], percentile: float) -> int:
    ordered = sorted(samples)
    return ceil(ordered[ceil(percentile * len(ordered)) - 1])


def _recommend(points: tuple[_Point, ...]) -> tuple[int, int]:
    launch_points = tuple(point for point in points if point.rows <= 64)
    bandwidth_points = tuple(point for point in points if point.rows >= 256)
    if not launch_points or not bandwidth_points:
        raise RuntimeError("compaction calibration has no service envelope")
    # A maximum over four small samples makes one launch outlier masquerade as
    # a bandwidth change. Use their median p90 as the fixed intercept, fit the
    # byte-dependent term over large median points, then retain 20% headroom.
    launch_ns = ceil(statistics.median(point.p90_ns for point in launch_points))
    fit = [
        (2 * point.exact_bytes, point.median_ns - launch_ns)
        for point in bandwidth_points
        if point.median_ns > launch_ns
    ]
    if not fit:
        raise RuntimeError("compaction calibration cannot resolve HBM bandwidth")
    slope_ns_per_byte = sum(x * y for x, y in fit) / sum(x * x for x, _y in fit)
    bandwidth_bps = floor(0.8 * 1_000_000_000 / slope_ns_per_byte)
    if bandwidth_bps <= 0:
        raise RuntimeError("compaction calibration returned invalid HBM bandwidth")
    return launch_ns, bandwidth_bps


def main() -> None:
    arguments = _arguments()
    program_path = arguments.program.resolve(strict=True)
    row_bytes = 4096
    row_counts = (1, 4, 16, 64, 256, 1024, 4096)
    pool_rows = arguments.pool_mib * 1024 * 1024 // row_bytes
    required_rows = max(row_counts) * arguments.trials + 8192
    if pool_rows < required_rows:
        raise RuntimeError(
            "cold-HBM pool is too small for non-repeating calibration rows: "
            f"need at least {ceil(required_rows * row_bytes / (1024 * 1024))} MiB"
        )

    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    source = torch.empty((pool_rows, row_bytes), dtype=torch.uint8, device=device)
    destination = torch.empty_like(source)
    source.fill_(0x5A)
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream(device=device)
    rng = random.Random(arguments.seed)
    points: list[_Point] = []
    with JitPhaseProgram(program_path) as phases:
        warm_rows = 4096
        warm_source = (
            torch.arange(warm_rows, dtype=torch.int64, device=device) * row_bytes
            + source.data_ptr()
        )
        warm_destination = (
            torch.arange(warm_rows, dtype=torch.int64, device=device) * row_bytes
            + destination.data_ptr()
        )
        for _ in range(10):
            phases.compact_hbm_rows(
                warm_source, warm_destination, row_bytes, stream
            )
        stream.synchronize()

        for rows in row_counts:
            sample_count = rows * arguments.trials
            source_indices = torch.tensor(
                rng.sample(range(8192, pool_rows), sample_count),
                dtype=torch.int64,
                device=device,
            ).reshape(arguments.trials, rows)
            destination_indices = torch.tensor(
                rng.sample(range(8192, pool_rows), sample_count),
                dtype=torch.int64,
                device=device,
            ).reshape(arguments.trials, rows)
            source_addresses = source_indices * row_bytes + source.data_ptr()
            destination_addresses = (
                destination_indices * row_bytes + destination.data_ptr()
            )
            samples: list[float] = []
            for trial in range(arguments.trials):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                phases.compact_hbm_rows(
                    source_addresses[trial],
                    destination_addresses[trial],
                    row_bytes,
                    stream,
                )
                end.record(stream)
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1_000_000)
            points.append(
                _Point(
                    rows,
                    ceil(statistics.median(samples)),
                    _percentile(samples, 0.9),
                )
            )

    launch_ns, bandwidth_bps = _recommend(tuple(points))
    properties = torch.cuda.get_device_properties(arguments.gpu)
    report = {
        "schema": 1,
        "classification": "nta-hbm-compaction-calibration",
        "method": "nonrepeating-random-cold-hbm-service-curve",
        "recommendation_policy": (
            "median-small-p90-launch;fixed-intercept-large-median-fit;"
            "0.8-bandwidth-safety-factor"
        ),
        "gpu": {
            "ordinal": arguments.gpu,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
        "row_bytes": row_bytes,
        "pool_bytes": pool_rows * row_bytes,
        "trials": arguments.trials,
        "seed": arguments.seed,
        "points": [
            {
                "rows": point.rows,
                "exact_bytes": point.exact_bytes,
                "median_ns": point.median_ns,
                "p90_ns": point.p90_ns,
            }
            for point in points
        ],
        "recommended_serving_config": {
            "NTA_NVME_COMPACTION_LAUNCH_NS": launch_ns,
            "NTA_NVME_COMPACTION_BANDWIDTH_BPS": bandwidth_bps,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        atomic_write_text(arguments.output, encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
