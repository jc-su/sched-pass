#!/usr/bin/env python3
"""Generate the complete paired causal-mechanism evaluation specification.

The generator requires one concrete command for every arm.  It never invents
timing commands or silently drops an arm.  Command tokens may use
``{workload_manifest}``, ``{tier}``, ``{arm}``, ``{variant}``, and
``{stratum}`` substitutions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

try:
    from .mechanism_arms import (
        ARMS,
        ARM_DEFINITIONS,
        CAUSAL_PAIRS,
        FORMAL_SERVING_METRICS,
        arm_environment,
    )
    from .result_contracts import result_contract_names
    from .run_evaluation import validate_spec
    from .workload_scenario import (
        describe_workload_scenario,
        validate_workload_scenario,
    )
except ImportError:
    from mechanism_arms import (
        ARMS,
        ARM_DEFINITIONS,
        CAUSAL_PAIRS,
        FORMAL_SERVING_METRICS,
        arm_environment,
    )
    from result_contracts import result_contract_names
    from run_evaluation import validate_spec
    from workload_scenario import describe_workload_scenario, validate_workload_scenario

PAIRS = tuple((numerator, denominator) for numerator, denominator, _ in CAUSAL_PAIRS)
DEFAULT_METRICS = FORMAL_SERVING_METRICS
TIER_ENVIRONMENT = {
    "hbm": "hbm",
    "host_mem": "host_staged",
    "nvme": "nvme",
    "dax": "cxl_dax",
}


def _load_strata(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("strata file must contain a non-empty array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"stratum {index} is not an object")
        label = str(entry.get("id", f"stratum-{index}"))
        if re.fullmatch(r"[A-Za-z0-9_.-]+", label) is None or label in seen:
            raise ValueError(f"stratum id is invalid or duplicated: {label!r}")
        seen.add(label)
        manifest_value = entry.get("workload_manifest")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError(f"stratum {label!r} has no workload_manifest")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = path.resolve().parent / manifest_path
        manifest_path = manifest_path.resolve()
        result.append(
            {
                "id": label,
                "workload_manifest": str(manifest_path),
                "descriptor": describe_workload_scenario(label, manifest_path),
            }
        )
    return result


def _commands(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--arm-command must use ARM=COMMAND: {value!r}")
        arm, command = value.split("=", 1)
        if arm not in ARMS or not command.strip():
            raise ValueError(f"invalid arm command: {value!r}")
        if arm in result:
            raise ValueError(f"duplicate command for {arm}")
        result[arm] = command
    missing = [arm for arm in ARMS if arm not in result]
    if missing:
        raise ValueError(f"missing concrete commands for: {', '.join(missing)}")
    return result


def _result_contracts(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    supported = result_contract_names(formal_only=True)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--arm-result-contract must use ARM=CONTRACT: {value!r}")
        arm, contract = value.split("=", 1)
        if arm not in ARMS or contract not in supported:
            raise ValueError(f"invalid arm result contract: {value!r}")
        if arm in result:
            raise ValueError(f"duplicate result contract for {arm}")
        result[arm] = contract
    missing = [arm for arm in ARMS if arm not in result]
    if missing:
        raise ValueError("missing result contracts for: " + ", ".join(missing))
    return result


def _format_command_token(token: str, *, arm: str, values: Mapping[str, str]) -> str:
    """Substitute only the documented tokens; preserve literal command braces."""

    substitutions = {**values, "arm": arm, "variant": arm}
    formatted = token
    for name, value in substitutions.items():
        formatted = formatted.replace("{" + name + "}", value)
    return formatted


def build_spec(
    *,
    tier: str,
    arm_commands: Mapping[str, str],
    result_contracts: Mapping[str, str],
    strata: list[dict[str, Any]],
    repetitions: int = 10,
    seed: int = 20260824,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tier_qualification: Path | None = None,
) -> dict[str, Any]:
    if tier not in {"hbm", "host_mem", "nvme", "dax"}:
        raise ValueError(f"unsupported evaluation tier: {tier}")
    if repetitions < 5:
        raise ValueError("mechanism study requires at least five repetitions")
    if set(arm_commands) != set(ARMS):
        raise ValueError("complete canonical arm command set is required")
    if set(result_contracts) != set(ARMS):
        raise ValueError("complete canonical arm result contract set is required")
    if tier in {"nvme", "dax"} and tier_qualification is None:
        raise ValueError("NVMe/DAX spec requires --tier-qualification")
    experiments: list[dict[str, Any]] = []
    comparisons: list[dict[str, str]] = []
    for stratum in strata:
        label = stratum["id"]
        workload_manifest = Path(str(stratum["workload_manifest"])).resolve()
        descriptor = validate_workload_scenario(
            stratum["descriptor"], workload_manifest
        )
        name = f"mechanism-{label}"
        values = {
            "workload_manifest": str(workload_manifest.resolve()),
            "tier": tier,
            "stratum": label,
        }
        for arm in ARMS:
            command = [
                _format_command_token(token, arm=arm, values=values)
                for token in shlex.split(arm_commands[arm])
            ]
            experiments.append(
                {
                    "name": name,
                    "variant": arm,
                    "arm": arm,
                    "tier": tier,
                    "demand_semantics": "exact",
                    "consumer_kind": ARM_DEFINITIONS[arm]["consumer_kind"],
                    "mechanism_form": ARM_DEFINITIONS[arm]["name"],
                    "result_contract": result_contracts[arm],
                    "stratum": descriptor,
                    "workload_manifest": str(workload_manifest),
                    "command": command,
                    "environment": {
                        **arm_environment(arm),
                        "NTA_SERVING_TIER": TIER_ENVIRONMENT[tier],
                    },
                    "metrics": list(metrics),
                }
            )
        for numerator, denominator in PAIRS:
            comparisons.append(
                {
                    "name": (
                        f"{numerator.lower()}-vs-{denominator.lower()}-{label}-goodput"
                    ),
                    "experiment": name,
                    "numerator_variant": numerator,
                    "denominator_variant": denominator,
                    "metric": "slo_goodput_requests_per_second",
                }
            )
    spec: dict[str, Any] = {
        "schema": 1,
        "classification": "nta-paired-evaluation",
        "evaluation_profile": "mechanism-study",
        "generated_by": "experiments/make_evaluation_spec.py",
        "workload_manifests": sorted(
            {str(Path(stratum["workload_manifest"]).resolve()) for stratum in strata}
        ),
        "tier": tier,
        "repetitions": repetitions,
        "seed": seed,
        "experiments": experiments,
        "comparisons": comparisons,
    }
    if tier_qualification is not None:
        spec["tier_qualification"] = str(tier_qualification.resolve())
    validate_spec(
        spec,
        qualification_path=tier_qualification.resolve() if tier_qualification else None,
    )
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strata-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tier", choices=("hbm", "host_mem", "nvme", "dax"), default="host_mem"
    )
    parser.add_argument("--tier-qualification", type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--metric", action="append", default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--arm-command", action="append", required=True, metavar="ARM=COMMAND"
    )
    parser.add_argument(
        "--arm-result-contract",
        action="append",
        required=True,
        metavar="ARM=CONTRACT",
        help="validated command result schema, for example sglang-serving",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = tuple(dict.fromkeys(args.metric))
    if not metrics:
        parser.error("at least one metric is required")
    try:
        spec = build_spec(
            tier=args.tier,
            arm_commands=_commands(args.arm_command),
            result_contracts=_result_contracts(args.arm_result_contract),
            strata=_load_strata(args.strata_file),
            repetitions=args.repetitions,
            seed=args.seed,
            metrics=metrics,
            tier_qualification=args.tier_qualification,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
