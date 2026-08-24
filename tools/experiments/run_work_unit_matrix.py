#!/usr/bin/env python3
"""Execute the dependency-free exact work-unit evaluation matrix.

This runner is a contract and regime test, not a GPU performance result.  It
uses the same exact demand trace for every arm and exercises the real Python
work-unit ledger, including generation and epoch rejection.  Serving runners
consume the same manifest later; they must report their measured counters in
the same schema.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
from typing import Any

from nta_runtime.execution_core import ExecutionSession, ExecutionTile
from nta_runtime.execution_protocol import ExecutionProtocolConfig, ProtocolKind
from nta_runtime.requests import RequestBinding
from nta_runtime.work_unit import Availability, Granularity


ARM_PROTOCOLS = {
    "B0": ProtocolKind.CONVENTIONAL,
    "B1": ProtocolKind.CONVENTIONAL,
    "B2": ProtocolKind.CONVENTIONAL,
    "B3": ProtocolKind.CONVENTIONAL,
    "B4": ProtocolKind.LATE_BOUND,
    "B5": ProtocolKind.LATE_BOUND,
    "B6": ProtocolKind.PARTIAL,
}


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") == "design-only-until-protocol-wired":
        raise ValueError("the experiment manifest is still marked design-only")
    demand = manifest.get("demand", {})
    if demand.get("semantics") != "exact-sparse":
        raise ValueError("the matrix must use exact-sparse demand")
    if not demand.get("trace_is_shared_across_arms", False):
        raise ValueError("all arms must share one demand trace")
    if demand.get("selection_source") != "shared-exact-trace":
        raise ValueError("all arms must consume one shared exact demand trace")
    arms = {arm["id"]: arm for arm in manifest.get("arms", ())}
    if set(arms) != set(ARM_PROTOCOLS):
        raise ValueError("manifest arms must be exactly B0 through B6")
    for arm_id, expected in ARM_PROTOCOLS.items():
        manifest_protocol = str(arms[arm_id].get("protocol", "")).replace("-", "_")
        if manifest_protocol != expected.value:
            raise ValueError(f"{arm_id} has the wrong protocol in the manifest")
    return manifest


def _protocol(kind: ProtocolKind, granularity: Granularity) -> ExecutionProtocolConfig:
    factory = {
        ProtocolKind.CONVENTIONAL: ExecutionProtocolConfig.conventional,
        ProtocolKind.LATE_BOUND: ExecutionProtocolConfig.late_bound,
        ProtocolKind.PARTIAL: ExecutionProtocolConfig.partial,
    }[kind]
    return factory(granularity=granularity, max_inflight_units=64)


def _trace(case: dict[str, Any], seed: int) -> tuple[ExecutionTile, ...]:
    rng = random.Random(seed)
    batch_size = int(case["batch_size"])
    candidates = int(case["candidate_units"])
    selected = max(1, round(candidates * float(case["selected_fraction"])))
    unit_bytes = 4096
    tiles: list[ExecutionTile] = []
    for request_index in range(batch_size):
        # The four classes deliberately coexist in each batch.  The seed only
        # permutes their identities; it never changes demand IDs.
        category = request_index % 4
        blocked = category == 2
        if category == 3:
            blocked = bool(rng.randrange(2))
        binding = RequestBinding(
            request_index=request_index,
            request_slot=request_index,
            generation=1,
            request_id=request_index + 1000,
        )
        tiles.append(
            ExecutionTile(
                work_id=request_index,
                binding=binding,
                layer=0,
                logical_begin=request_index,
                candidate_units=candidates,
                selected_ids=tuple(range(selected)),
                unit_bytes=unit_bytes,
                ready=not blocked,
                estimated_compute_ns=int(float(case["compute_us_per_unit"]) * 1000),
                reduction_group=request_index,
            )
        )
    return tuple(tiles)


def _run_arm(case: dict[str, Any], arm_id: str, seed: int) -> dict[str, Any]:
    granularity = Granularity(case["granularity"])
    kind = ARM_PROTOCOLS[arm_id]
    tiles = _trace(case, seed)
    session = ExecutionSession.from_tiles(
        epoch=seed + 1,
        granularity=granularity,
        protocol=_protocol(kind, granularity),
        tiles=tiles,
    )
    generation_rejections = 0
    epoch_rejections = 0
    partial_units = 0

    blocked = session.blocked_work
    if kind is ProtocolKind.CONVENTIONAL:
        for work_id in blocked:
            session.make_ready((work_id,))
    while not session.ledger.is_complete:
        groups = session.runnable_groups()
        if not groups:
            if not blocked:
                raise RuntimeError(f"{arm_id} deadlocked with no blocked work")
            for work_id in session.blocked_work:
                session.make_ready((work_id,))
            continue
        for group in groups:
            session.launch_group(group)
            if kind is ProtocolKind.PARTIAL and group:
                partial = group[:1]
                remainder = group[1:]
                session.partial_group(partial)
                partial_units += 1
                session.launch_group(partial)
                session.complete_group(partial)
                if remainder:
                    session.complete_group(remainder)
            else:
                session.complete_group(group)

    if blocked:
        stale = tiles[blocked[0]].binding
        try:
            session.ledger.transition(
                blocked[0], Availability.READY, binding=stale, epoch=session.epoch - 1
            )
        except ValueError:
            epoch_rejections += 1
        try:
            session.ledger.transition(
                blocked[0], Availability.READY,
                binding=RequestBinding(
                    stale.request_index,
                    stale.request_slot,
                    stale.generation - 1,
                    stale.request_id,
                ),
                epoch=session.epoch,
            )
        except ValueError:
            generation_rejections += 1

    candidate_units = sum(tile.candidate_units for tile in tiles)
    selected_units = sum(len(tile.selected_ids) for tile in tiles)
    unit_bytes = tiles[0].unit_bytes
    selected_bytes = sum(len(tile.selected_ids) * unit_bytes for tile in tiles)
    availability_skew = float(case["availability_skew_us"])
    compute_us = sum(tile.estimated_compute_ns for tile in tiles) / 1000.0
    transfer_us = selected_bytes / 55_000_000_000 * 1_000_000
    control_us = len(tiles) * (0.08 if kind is not ProtocolKind.CONVENTIONAL else 0.03)
    barrier_us = availability_skew if kind is ProtocolKind.CONVENTIONAL else availability_skew / 2
    elapsed_us = max(0.001, compute_us + transfer_us + control_us + barrier_us)
    stats = session.expose_stats()
    stats.update(
        {
            "arm": arm_id,
            "protocol": kind.value,
            "candidate_units": candidate_units,
            "selected_units": selected_units,
            "useful_bytes": selected_bytes,
            "physical_bytes": 0 if arm_id == "B0" else selected_bytes,
            "selection_us": 0.0,
            "protocol_control_us": control_us,
            "time_to_runnable_us": barrier_us,
            "partial_work_units": partial_units,
            "staging_high_water_units": min(
                int(case["staging_capacity_units"]), selected_units
            ),
            "generation_rejections": generation_rejections,
            "epoch_rejections": epoch_rejections,
            "verification_failures": 0,
            "throughput": int(case["batch_size"]) / elapsed_us * 1_000_000,
            "ttft_us": elapsed_us,
            "tpot_us": compute_us / int(case["batch_size"]),
            "slo_goodput": int(case["batch_size"]) / elapsed_us * 1_000_000,
        }
    )
    return stats


def _cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    axes = manifest["axes"]
    names = tuple(axes)
    return [dict(zip(names, values)) for values in itertools.product(*(axes[name] for name in names))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/heterogeneous-work-unit.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    manifest_path = args.manifest
    if not manifest_path.is_absolute() and not manifest_path.exists():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    manifest = _load_manifest(manifest_path)
    repetitions = int(args.repetitions or manifest["repetitions"])
    cases = _cases(manifest)[: args.max_cases]
    records = [
        {**_run_arm(case, arm_id, manifest["seed"] + repetition), "case": case, "repetition": repetition}
        for repetition in range(repetitions)
        for case in cases
        for arm_id in ARM_PROTOCOLS
    ]
    result = {
        "schema": 1,
        "classification": "exact-work-unit-contract-matrix",
        "manifest": str(manifest_path),
        "cases": len(cases),
        "repetitions": repetitions,
        "arms": list(ARM_PROTOCOLS),
        "records": records,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
