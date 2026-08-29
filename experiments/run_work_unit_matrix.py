#!/usr/bin/env python3
"""Execute the dependency-free exact work-unit evaluation matrix.

This runner is a contract and regime test, not a GPU performance result.  It
uses the same exact demand trace for every diagnostic profile and exercises
the real Python work-unit ledger, including generation and epoch rejection.
Its D-prefixed profiles are deliberately separate from the measured A-prefixed
serving arms.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
from dataclasses import dataclass
import subprocess
from typing import Any

from nta_runtime.execution_core import ExecutionSession, ExecutionTile
from nta_runtime.execution_protocol import ExecutionProtocolConfig, ProtocolKind
from nta_runtime.requests import RequestBinding
from nta_runtime.work_unit import Availability, Granularity

if __package__:
    from .queueing import modeled_blocked_cohort_accounting
else:
    from queueing import modeled_blocked_cohort_accounting


@dataclass(frozen=True)
class DiagnosticProfile:
    """One synthetic component/regime profile; never a serving causal arm."""

    protocol: ProtocolKind
    demand_source: str
    readiness: str
    mapping: str
    bounded_staging: bool
    admission_feedback: bool


# These D-prefixed profiles belong only to the dependency-free diagnostic
# matrix.  Formal serving uses the independent A0--A3 mechanism arms in
# ``experiments.mechanism_arms``.  Keeping the namespaces distinct prevents a
# synthetic state-machine profile from masquerading as measured causal evidence.
DIAGNOSTIC_PROFILES = {
    "D0": DiagnosticProfile(
        ProtocolKind.CONVENTIONAL, "resident", "batch", "manual", False, False
    ),
    "D1": DiagnosticProfile(
        ProtocolKind.CONVENTIONAL, "host_promotion", "batch", "manual", False, False
    ),
    "D2": DiagnosticProfile(
        ProtocolKind.CONVENTIONAL, "host_demand", "batch", "manual", False, False
    ),
    "D3": DiagnosticProfile(
        ProtocolKind.CONVENTIONAL, "device_demand", "batch", "manual", False, False
    ),
    "D4": DiagnosticProfile(
        ProtocolKind.LATE_BOUND, "device_demand", "work_unit", "typed", True, False
    ),
    "D5": DiagnosticProfile(
        ProtocolKind.LATE_BOUND, "device_demand", "work_unit", "typed", True, True
    ),
    "D6": DiagnosticProfile(
        ProtocolKind.PARTIAL, "device_demand", "work_unit", "typed", True, True
    ),
}
PROFILE_PROTOCOLS = {
    profile: definition.protocol for profile, definition in DIAGNOSTIC_PROFILES.items()
}


@dataclass(frozen=True)
class AblationDefinition:
    """One declared mechanism boundary disabled for applicable profiles."""

    description: str
    target_profiles: frozenset[str]


ABLATIONS = {
    "full": AblationDefinition("all mechanism boundaries enabled", frozenset()),
    "host_demand": AblationDefinition(
        "materialize exact demand through the host control path",
        frozenset({"D3", "D4", "D5", "D6"}),
    ),
    "batch_readiness": AblationDefinition(
        "replace per-work-unit readiness with one batch barrier",
        frozenset({"D4", "D5", "D6"}),
    ),
    "coarse_granularity": AblationDefinition(
        "collapse the selected work into one coarse group",
        frozenset({"D4", "D5", "D6"}),
    ),
    "manual_mapping": AblationDefinition(
        "replace typed compiler coordinates with manual coordinates",
        frozenset({"D4", "D5", "D6"}),
    ),
    "shadow_generation_checks": AblationDefinition(
        "move generation checks off the hot path into shadow validation",
        frozenset({"D4", "D5", "D6"}),
    ),
    "admission_feedback": AblationDefinition(
        "disable engine admission feedback",
        frozenset({"D5", "D6"}),
    ),
    "unbounded_staging": AblationDefinition(
        "replace bounded staging with full promotion",
        frozenset({"D4", "D5", "D6"}),
    ),
}

TIER_PROFILES = {
    "hbm": (1_000_000_000_000, 0),
    "host_mem": (30_000_000_000, 2_000),
    "nvme": (7_000_000_000, 80_000),
    "dax": (20_000_000_000, 1_500),
}


def _git_metadata() -> dict[str, Any]:
    """Record provenance without making Git a runtime dependency."""

    revision = os.environ.get("NTA_REVISION", "")
    clean: bool | None = None
    try:
        if not revision:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        clean = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            ).returncode
            == 0
            and subprocess.run(
                ["git", "diff", "--cached", "--quiet", "--"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {"revision": revision or "unrecorded", "working_tree_clean": clean}


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 3:
        raise ValueError("diagnostic matrix manifest must use schema 3")
    if manifest.get("status") == "design-only-until-protocol-wired":
        raise ValueError("the experiment manifest is still marked design-only")
    demand = manifest.get("demand", {})
    if demand.get("semantics") != "exact-sparse":
        raise ValueError("the matrix must use exact-sparse demand")
    if not demand.get("trace_is_shared_across_profiles", False):
        raise ValueError("all diagnostic profiles must share one demand trace")
    if demand.get("selection_source") != "shared-exact-trace":
        raise ValueError("all diagnostic profiles must consume one exact trace")
    profiles = {
        profile["id"]: profile for profile in manifest.get("diagnostic_profiles", ())
    }
    if set(profiles) != set(PROFILE_PROTOCOLS):
        raise ValueError("manifest profiles must be exactly D0 through D6")
    for profile_id, expected in PROFILE_PROTOCOLS.items():
        manifest_protocol = str(profiles[profile_id].get("protocol", "")).replace(
            "-", "_"
        )
        if manifest_protocol != expected.value:
            raise ValueError(f"{profile_id} has the wrong protocol in the manifest")
    manifest_ablations = {
        item["id"]: frozenset(item.get("target_profiles", ()))
        for item in manifest.get("ablations", ())
    }
    if set(manifest_ablations) != set(ABLATIONS):
        raise ValueError("manifest ablations must describe every executable ablation")
    for ablation_id, definition in ABLATIONS.items():
        if manifest_ablations[ablation_id] != definition.target_profiles:
            raise ValueError(
                f"{ablation_id} has the wrong diagnostic-profile scope"
            )
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
        category_scale = (1.0, 1.5, 0.5, 0.75)[category]
        selected_for_request = max(1, min(candidates, round(selected * category_scale)))
        selected_ids = (
            ()
            if selected_for_request == candidates
            else tuple(
                (request_index * 13 + index) % candidates
                for index in range(selected_for_request)
            )
        )
        tiles.append(
            ExecutionTile(
                work_id=request_index,
                binding=binding,
                layer=0,
                logical_begin=request_index,
                candidate_units=candidates,
                selected_ids=selected_ids,
                unit_bytes=unit_bytes,
                ready=not blocked,
                estimated_compute_ns=int(float(case["compute_us_per_unit"]) * 1000),
                reduction_group=request_index,
            )
        )
    return tuple(tiles)


def _trace_hash(tiles: tuple[ExecutionTile, ...]) -> str:
    """Hash the complete exact trace so fairness is machine-checkable."""

    encoded = [
        {
            "work_id": tile.work_id,
            "request_index": tile.binding.request_index,
            "request_slot": tile.binding.request_slot,
            "generation": tile.binding.generation,
            "ready": tile.ready,
            "candidate_units": tile.candidate_units,
            "selected_units": tile.selected_units,
            "demand_semantics": tile.demand_semantics.value,
            "selected_ids": tile.selected_ids,
        }
        for tile in tiles
    ]
    payload = json.dumps(encoded, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _mode(
    profile_id: str, ablation_id: str
) -> tuple[DiagnosticProfile, dict[str, str | bool], bool]:
    definition = DIAGNOSTIC_PROFILES[profile_id]
    mode: dict[str, str | bool] = {
        "demand_source": definition.demand_source,
        "readiness": definition.readiness,
        "mapping": definition.mapping,
        "bounded_staging": definition.bounded_staging,
        "admission_feedback": definition.admission_feedback,
        "generation_checks": "hot",
        "granularity": "case",
    }
    ablation = ABLATIONS[ablation_id]
    applied = ablation_id != "full" and profile_id in ablation.target_profiles
    if applied:
        if ablation_id == "host_demand":
            mode["demand_source"] = "host_demand"
        elif ablation_id == "batch_readiness":
            mode["readiness"] = "batch"
        elif ablation_id == "coarse_granularity":
            mode["granularity"] = "coarse"
        elif ablation_id == "manual_mapping":
            mode["mapping"] = "manual"
        elif ablation_id == "shadow_generation_checks":
            mode["generation_checks"] = "shadow"
        elif ablation_id == "admission_feedback":
            mode["admission_feedback"] = False
        elif ablation_id == "unbounded_staging":
            mode["bounded_staging"] = False
    return definition, mode, applied


def _run_profile(
    case: dict[str, Any], profile_id: str, seed: int, ablation_id: str
) -> dict[str, Any]:
    definition, mode, ablation_applied = _mode(profile_id, ablation_id)
    granularity = Granularity(case["granularity"])
    kind = definition.protocol
    # A readiness ablation intentionally removes overlap.  D6 remains an
    # exact protocol record, but its partial-publication counter is zero in
    # this arm because a batch barrier prevents continuation from helping.
    session_kind = (
        ProtocolKind.CONVENTIONAL
        if mode["readiness"] == "batch" and kind is not ProtocolKind.CONVENTIONAL
        else kind
    )
    tiles = _trace(case, seed)
    session = ExecutionSession.from_tiles(
        epoch=seed + 1,
        granularity=granularity,
        protocol=_protocol(session_kind, granularity),
        tiles=tiles,
    )
    generation_rejections = 0
    epoch_rejections = 0
    partial_units = 0

    blocked = session.blocked_work
    if session_kind is ProtocolKind.CONVENTIONAL:
        for work_id in blocked:
            session.make_ready((work_id,))
    while not session.ledger.is_complete:
        groups = session.runnable_groups()
        if not groups:
            if not blocked:
                raise RuntimeError(f"{profile_id} deadlocked with no blocked work")
            for work_id in session.blocked_work:
                session.make_ready((work_id,))
            continue
        for group in groups:
            session.launch_group(group)
            if session_kind is ProtocolKind.PARTIAL and group:
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
                blocked[0],
                Availability.READY,
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
    selected_units = sum(tile.selected_units for tile in tiles)
    unit_bytes = tiles[0].unit_bytes
    selected_bytes = sum(tile.selected_units * unit_bytes for tile in tiles)
    availability_skew = float(case["availability_skew_us"])
    compute_us = sum(tile.estimated_compute_ns for tile in tiles) / 1000.0
    tier = str(case.get("tier", "host_mem"))
    try:
        bandwidth, tier_latency_ns = TIER_PROFILES[tier]
    except KeyError as error:
        raise ValueError(f"unknown experiment tier {tier}") from error
    if profile_id == "D0":
        physical_bytes = 0
    elif profile_id == "D1":
        physical_bytes = candidate_units * unit_bytes
    else:
        physical_bytes = selected_bytes
    transfer_us = physical_bytes / bandwidth * 1_000_000
    group_width = {
        "request": max(selected_units, 1),
        "layer": max(selected_units // 2, 1),
        "page_group": 8,
        "cta_tile": 4,
    }[case["granularity"]]
    if mode["granularity"] == "coarse":
        group_width = max(selected_units, 1)
    group_count = max(1, (selected_units + group_width - 1) // group_width)
    host_round_trips = (
        len(tiles) if mode["demand_source"] in ("host_promotion", "host_demand") else 0
    )
    host_demand_materializations = (
        len(tiles) if mode["demand_source"] in ("host_promotion", "host_demand") else 0
    )
    device_demand_discoveries = (
        len(tiles) if mode["demand_source"] == "device_demand" else 0
    )
    batch_barriers = 1 if mode["readiness"] == "batch" else 0
    work_unit_ready_events = len(tiles) if mode["readiness"] == "work_unit" else 0
    typed_sites = len(tiles) if mode["mapping"] == "typed" else 0
    manual_sites = len(tiles) if mode["mapping"] == "manual" else 0
    generation_checks_hot = len(tiles) if mode["generation_checks"] == "hot" else 0
    generation_checks_shadow = (
        len(tiles) if mode["generation_checks"] == "shadow" else 0
    )
    tier_ownership_bindings = (
        len(tiles)
        if mode["demand_source"] == "device_demand" and mode["mapping"] == "typed"
        else 0
    )
    admission_decisions = len(tiles) if mode["admission_feedback"] else 0
    bounded_decisions = 1 if mode["bounded_staging"] else 0
    selection_us = (
        host_demand_materializations * 0.12 + device_demand_discoveries * 0.02
    )
    control_us = (
        host_round_trips * 0.65
        + batch_barriers * (0.45 + availability_skew / 10_000)
        + work_unit_ready_events * 0.03
        + group_count * (0.08 if mode["mapping"] == "manual" else 0.04)
        + (0.02 * admission_decisions)
    )
    total_non_compute_us = transfer_us + selection_us + control_us
    compute_fraction = compute_us / max(compute_us + total_non_compute_us, 1.0e-9)
    load_ratio = (
        "control_dominated"
        if compute_fraction < 0.33
        else "compute_dominated"
        if compute_fraction > 0.66
        else "balanced"
    )
    availability_stratum = (
        "low"
        if availability_skew <= 50
        else "medium"
        if availability_skew <= 500
        else "high"
    )
    arrival = (
        "batch_release"
        if availability_skew == 0
        else "burst"
        if availability_skew <= 50
        else "calibrated_open_loop"
    )
    barrier_us = (
        availability_skew
        if mode["readiness"] == "batch"
        else availability_skew / max(group_count, 1)
    )
    elapsed_us = max(
        0.001,
        compute_us
        + transfer_us
        + selection_us
        + control_us
        + barrier_us
        + tier_latency_ns / 1_000,
    )
    initial_blocked = len(blocked)
    pending_accounting = modeled_blocked_cohort_accounting(
        initial_blocked, availability_skew
    )
    state_counts = {
        "resident": sum(index % 4 == 0 for index in range(len(tiles))),
        "external_ready": sum(index % 4 == 1 for index in range(len(tiles))),
        "external_blocked": sum(index % 4 == 2 for index in range(len(tiles))),
        "new_arrival": sum(index % 4 == 3 for index in range(len(tiles))),
    }
    stats = session.expose_stats()
    stats.update(
        {
            "diagnostic_profile": profile_id,
            "protocol": kind.value,
            "session_protocol": session_kind.value,
            "ablation": ablation_id,
            "ablation_applied": ablation_applied,
            "ablation_description": ABLATIONS[ablation_id].description,
            "demand_trace_hash": _trace_hash(tiles),
            "execution_mode": {
                "demand_source": mode["demand_source"],
                "readiness": mode["readiness"],
                "mapping": mode["mapping"],
                "granularity": mode["granularity"],
                "bounded_staging": mode["bounded_staging"],
                "admission_feedback": mode["admission_feedback"],
                "generation_checks": mode["generation_checks"],
            },
            "activation_counters": {
                "exact_demand_bindings": len(tiles),
                "typed_instrumentation_sites": typed_sites,
                "manual_mapping_sites": manual_sites,
                "request_generation_checks_hot": generation_checks_hot,
                "request_generation_checks_shadow": generation_checks_shadow,
                "tier_ownership_bindings": tier_ownership_bindings,
                "device_demand_discoveries": device_demand_discoveries,
                "host_demand_materializations": host_demand_materializations,
                "host_control_round_trips": host_round_trips,
                "batch_readiness_barriers": batch_barriers,
                "work_unit_ready_events": work_unit_ready_events,
                "work_unit_groups": group_count
                if mode["readiness"] == "work_unit"
                else 0,
                "bounded_staging_decisions": bounded_decisions,
                "admission_feedback_decisions": admission_decisions,
                "partial_publications": partial_units,
            },
            "candidate_units": candidate_units,
            "selected_units": selected_units,
            "useful_bytes": selected_bytes,
            "physical_bytes": physical_bytes,
            "tier": tier,
            "tier_bandwidth_bytes_per_second": bandwidth,
            "tier_latency_ns": tier_latency_ns,
            "stratum": {
                "request_state": "mixed",
                "granularity": case["granularity"],
                "tier": tier,
                "load_ratio": load_ratio,
                "availability_skew": availability_stratum,
                "availability_skew_us": availability_skew,
                "staging_pressure": (
                    "under_capacity"
                    if selected_units <= int(case["staging_capacity_units"]) * 0.8
                    else "near_capacity"
                    if selected_units <= int(case["staging_capacity_units"])
                    else "over_capacity"
                ),
                "arrival": arrival,
                "arrival_model": "synthetic_availability_skew",
                "request_state_counts": state_counts,
            },
            "ready_work_units": stats["work_ready"],
            "blocked_work_units": stats["work_blocked"],
            "selection_us": selection_us,
            "protocol_control_us": control_us,
            "time_to_runnable_us": barrier_us,
            "host_round_trips": host_round_trips,
            "device_demand_discoveries": device_demand_discoveries,
            "group_count": group_count,
            "partial_work_units": partial_units,
            "staging_high_water_units": (
                min(int(case["staging_capacity_units"]), selected_units)
                if mode["bounded_staging"]
                else selected_units
            ),
            "generation_rejections": generation_rejections,
            "epoch_rejections": epoch_rejections,
            "verification_failures": 0,
            "throughput": int(case["batch_size"]) / elapsed_us * 1_000_000,
            "ttft_us": elapsed_us,
            "tpot_us": compute_us / int(case["batch_size"]),
            "slo_goodput": int(case["batch_size"]) / elapsed_us * 1_000_000,
            "queueing_accounting_method": pending_accounting["method"],
            "queueing_release_process": pending_accounting["release_process"],
            "queueing_interpretation": pending_accounting["interpretation"],
            "initial_blocked_work_units": initial_blocked,
            "pending_release_count": pending_accounting["pending_release_count"],
            "pending_window_us": pending_accounting["pending_window_us"],
            "pending_area_unit_us": pending_accounting["pending_area_unit_us"],
            "pending_release_rate_per_second": pending_accounting[
                "release_rate_per_second"
            ],
            "mean_pending_units": pending_accounting["mean_pending_units"],
            "mean_pending_us": pending_accounting["mean_pending_us"],
            "measurement": {
                "kind": "synthetic_regime_contract",
                "serving_evidence": False,
                "timing_is_modeled": True,
            },
        }
    )
    return stats


def _cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    axes = manifest["axes"]
    names = tuple(axes)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(axes[name] for name in names))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/heterogeneous-work-unit.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument(
        "--ablation",
        choices=(*ABLATIONS, "all"),
        default="full",
        help="run one declared ablation or all ablations (default: full)",
    )
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    manifest_path = args.manifest
    if not manifest_path.is_absolute() and not manifest_path.exists():
        manifest_path = Path(__file__).resolve().parents[1] / manifest_path
    manifest = _load_manifest(manifest_path)
    repetitions = int(
        manifest["repetitions"] if args.repetitions is None else args.repetitions
    )
    if repetitions <= 0:
        parser.error("--repetitions or manifest repetitions must be positive")
    cases = _cases(manifest)[: args.max_cases]
    ablations = tuple(ABLATIONS) if args.ablation == "all" else (args.ablation,)
    records = [
        {
            **_run_profile(
                case,
                profile_id,
                manifest["seed"] + repetition,
                ablation_id,
            ),
            "case": case,
            "repetition": repetition,
        }
        for repetition in range(repetitions)
        for case in cases
        for profile_id in PROFILE_PROTOCOLS
        for ablation_id in ablations
    ]
    result = {
        "schema": 3,
        "classification": "exact-work-unit-diagnostic-matrix",
        "measurement": {
            "kind": "synthetic_regime_contract",
            "serving_evidence": False,
            "timing_is_modeled": True,
        },
        "manifest": str(manifest_path),
        "provenance": _git_metadata(),
        "cases": len(cases),
        "repetitions": repetitions,
        "diagnostic_profiles": list(PROFILE_PROTOCOLS),
        "ablations": list(ablations),
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
