"""Offline analysis for the all-or-nothing operator barrier."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import pathlib
import statistics
from collections.abc import Iterable
from typing import Any


@dataclasses.dataclass(frozen=True)
class TileArrival:
    request_id: str
    tile_id: int
    available_ns: int
    compute_ns: int
    logical_tile: int | None = None
    availability_source: str = "gpu_globaltimer"
    compute_source: str = "calibrated"

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TileArrival":
        tile = cls(
            str(value["request_id"]),
            int(value["tile_id"]),
            int(value["available_ns"]),
            int(value["compute_ns"]),
            None
            if value.get("logical_tile") is None
            else int(value["logical_tile"]),
            str(value.get("availability_source", "gpu_globaltimer")),
            str(value.get("compute_source", "calibrated")),
        )
        if tile.available_ns < 0 or tile.compute_ns <= 0:
            raise ValueError("tile arrival and compute costs must be positive")
        if tile.availability_source not in {
            "resident_at_launch",
            "gpu_globaltimer",
        }:
            raise ValueError("tile availability source is not measured")
        if tile.compute_source not in {"calibrated", "measured"}:
            raise ValueError("tile compute source is invalid")
        return tile

    def as_json(self) -> dict[str, int | str | None]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class OperatorArrival:
    batch_id: str
    layer: int
    tiles: tuple[TileArrival, ...]
    revision: str = ""
    engine: str = ""
    model: str = ""
    tier: str = ""
    observed_at_unix_ns: int = 0

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "OperatorArrival":
        tiles = tuple(TileArrival.from_json(tile) for tile in value["tiles"])
        if not tiles:
            raise ValueError("operator trace records need at least one tile")
        identities = {(tile.request_id, tile.tile_id) for tile in tiles}
        if len(identities) != len(tiles):
            raise ValueError("operator trace contains duplicate request/tile identity")
        observed_at = int(value.get("observed_at_unix_ns", 0))
        if observed_at < 0:
            raise ValueError("operator observation time cannot be negative")
        return cls(
            str(value["batch_id"]),
            int(value["layer"]),
            tiles,
            str(value.get("revision", "")),
            str(value.get("engine", "")),
            str(value.get("model", "")),
            str(value.get("tier", "")),
            observed_at,
        )

    def as_json(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["schema"] = 2
        return value


@dataclasses.dataclass(frozen=True)
class OpportunitySummary:
    operators: int
    tiles: int
    requests: int
    median_arrival_spread_ns: float
    p95_arrival_spread_ns: float
    available_before_atomic_launch: float
    blocked_compute_ns: int
    blocked_compute_area_ns2: int
    atomic_makespan_ns: int
    incremental_makespan_ns: int
    incremental_speedup: float

    def as_json(self) -> dict[str, int | float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TraceProvenance:
    records: int
    revisions: tuple[str, ...]
    engines: tuple[str, ...]
    models: tuple[str, ...]
    tiers: tuple[str, ...]
    gpu_timestamped_tiles: int
    resident_at_launch_tiles: int
    calibrated_compute_tiles: int

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _incremental_makespan(
    tiles: tuple[TileArrival, ...], launch_overhead_ns: int, grouping_window_ns: int
) -> int:
    remaining = sorted(tiles, key=lambda tile: (tile.available_ns, tile.tile_id))
    cursor = 0
    while remaining:
        first_arrival = max(cursor, remaining[0].available_ns)
        cutoff = first_arrival + grouping_window_ns
        wave_end = 0
        while wave_end < len(remaining) and remaining[wave_end].available_ns <= cutoff:
            wave_end += 1
        wave = remaining[:wave_end]
        del remaining[:wave_end]
        cursor = cutoff + launch_overhead_ns + sum(tile.compute_ns for tile in wave)
    return cursor


def summarize(
    records: Iterable[OperatorArrival],
    *,
    launch_overhead_ns: int = 0,
    grouping_window_ns: int = 0,
) -> OpportunitySummary:
    if launch_overhead_ns < 0 or grouping_window_ns < 0:
        raise ValueError("launch overhead and grouping window cannot be negative")

    materialized = tuple(records)
    if not materialized:
        raise ValueError("opportunity analysis needs at least one operator record")

    spreads: list[int] = []
    tile_count = 0
    request_count = 0
    available_before = 0
    blocked_compute = 0
    blocked_area = 0
    atomic_total = 0
    incremental_total = 0
    for record in materialized:
        tile_count += len(record.tiles)
        by_request: dict[str, list[int]] = {}
        for tile in record.tiles:
            by_request.setdefault(tile.request_id, []).append(tile.available_ns)
        request_count += len(by_request)
        for arrivals in by_request.values():
            spreads.append(max(arrivals) - int(statistics.median(arrivals)))

        atomic_launch = max(tile.available_ns for tile in record.tiles)
        compute = sum(tile.compute_ns for tile in record.tiles)
        atomic_total += atomic_launch + launch_overhead_ns + compute
        incremental_total += _incremental_makespan(
            record.tiles, launch_overhead_ns, grouping_window_ns
        )
        for tile in record.tiles:
            delay = atomic_launch - tile.available_ns
            if delay > 0:
                available_before += 1
                blocked_compute += tile.compute_ns
                blocked_area += delay * tile.compute_ns

    return OpportunitySummary(
        operators=len(materialized),
        tiles=tile_count,
        requests=request_count,
        median_arrival_spread_ns=float(statistics.median(spreads)),
        p95_arrival_spread_ns=_percentile(spreads, 0.95),
        available_before_atomic_launch=available_before / tile_count,
        blocked_compute_ns=blocked_compute,
        blocked_compute_area_ns2=blocked_area,
        atomic_makespan_ns=atomic_total,
        incremental_makespan_ns=incremental_total,
        incremental_speedup=(
            atomic_total / incremental_total if incremental_total != 0 else 0.0
        ),
    )


def summarize_provenance(records: Iterable[OperatorArrival]) -> TraceProvenance:
    materialized = tuple(records)
    if not materialized:
        raise ValueError("opportunity provenance needs at least one record")
    return TraceProvenance(
        records=len(materialized),
        revisions=tuple(sorted({record.revision for record in materialized})),
        engines=tuple(sorted({record.engine for record in materialized})),
        models=tuple(sorted({record.model for record in materialized})),
        tiers=tuple(sorted({record.tier for record in materialized})),
        gpu_timestamped_tiles=sum(
            tile.availability_source == "gpu_globaltimer"
            for record in materialized
            for tile in record.tiles
        ),
        resident_at_launch_tiles=sum(
            tile.availability_source == "resident_at_launch"
            for record in materialized
            for tile in record.tiles
        ),
        calibrated_compute_tiles=sum(
            tile.compute_source == "calibrated"
            for record in materialized
            for tile in record.tiles
        ),
    )


def load_json_lines(lines: Iterable[str]) -> tuple[OperatorArrival, ...]:
    records: list[OperatorArrival] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"trace line {line_number} is not a JSON object")
        records.append(OperatorArrival.from_json(value))
    return tuple(records)


def append_json_line(path: pathlib.Path, record: OperatorArrival) -> None:
    """Append one validated observation atomically across engine processes."""
    validated = OperatorArrival.from_json(record.as_json())
    if validated != record:
        raise ValueError("opportunity record changed during validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record.as_json(), sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short write while appending opportunity trace")
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
