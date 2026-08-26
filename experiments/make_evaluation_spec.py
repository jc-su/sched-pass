#!/usr/bin/env python3
"""Generate the complete paired B0--B6 evaluation specification.

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
    from .run_evaluation import validate_spec
    from .validate_workload import validate as validate_workload
except ImportError:
    from run_evaluation import validate_spec
    from validate_workload import validate as validate_workload


ARMS = tuple(f"B{index}" for index in range(7))
FORMAL_CONSUMER_KINDS = frozenset({"native_work_unit", "framework_reference"})
# Adjacent pairs expose each boundary. The two cross-boundary pairs are
# deliberate: B3/B1 seals the host-control round-trip effect, and B5/B3
# tests the complete device-demand-to-heterogeneous-execution jump.
PAIRS = (
    *tuple(zip(ARMS[1:], ARMS[:-1])),
    ("B3", "B1"),
    ("B5", "B3"),
)
DEFAULT_METRICS = (
    "slo_goodput",
    "p95_ttft_seconds",
    "p99_itl_seconds",
    "verification_failures",
)
STRATUM_FIELDS = (
    "request_state",
    "granularity",
    "load_ratio",
    "availability_skew",
    "staging_pressure",
    "arrival",
)


def _load_strata(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("strata file must contain a non-empty array")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"stratum {index} is not an object")
        missing = [
            field for field in STRATUM_FIELDS if not isinstance(entry.get(field), str)
        ]
        if missing:
            raise ValueError(f"stratum {index} lacks fields: {', '.join(missing)}")
        label = str(entry.get("id", f"stratum-{index}"))
        if re.fullmatch(r"[A-Za-z0-9_.-]+", label) is None or label in seen:
            raise ValueError(f"stratum id is invalid or duplicated: {label!r}")
        seen.add(label)
        result.append(
            {field: str(entry[field]) for field in STRATUM_FIELDS} | {"id": label}
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


def _consumer_kinds(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--arm-consumer-kind must use ARM=KIND: {value!r}")
        arm, kind = value.split("=", 1)
        if arm not in ARMS or kind not in FORMAL_CONSUMER_KINDS:
            raise ValueError(f"invalid arm consumer kind: {value!r}")
        if arm in result:
            raise ValueError(f"duplicate consumer kind for {arm}")
        result[arm] = kind
    missing = [arm for arm in ARMS if arm not in result]
    if missing:
        raise ValueError("missing consumer kinds for: " + ", ".join(missing))
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
    workload_manifest: Path,
    tier: str,
    arm_commands: Mapping[str, str],
    consumer_kinds: Mapping[str, str],
    strata: list[dict[str, str]],
    repetitions: int = 10,
    seed: int = 20260824,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tier_qualification: Path | None = None,
) -> dict[str, Any]:
    if tier not in {"hbm", "host_mem", "nvme", "dax"}:
        raise ValueError(f"unsupported evaluation tier: {tier}")
    if repetitions < 5:
        raise ValueError("OSDI evaluation requires at least five repetitions")
    if set(arm_commands) != set(ARMS):
        raise ValueError("complete B0--B6 arm command set is required")
    if set(consumer_kinds) != set(ARMS):
        raise ValueError("complete B0--B6 consumer kind set is required")
    if any(kind not in FORMAL_CONSUMER_KINDS for kind in consumer_kinds.values()):
        raise ValueError("formal arms require a numerical consumer kind")
    validate_workload(workload_manifest.resolve())
    if tier in {"nvme", "dax"} and tier_qualification is None:
        raise ValueError("NVMe/DAX spec requires --tier-qualification")
    experiments: list[dict[str, Any]] = []
    comparisons: list[dict[str, str]] = []
    for numerator, denominator in PAIRS:
        for stratum in strata:
            label = stratum["id"]
            name = f"{numerator.lower()}-vs-{denominator.lower()}-{label}"
            fields = {field: stratum[field] for field in STRATUM_FIELDS}
            values = {
                "workload_manifest": str(workload_manifest.resolve()),
                "tier": tier,
                "stratum": label,
            }
            for arm in (denominator, numerator):
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
                        "consumer_kind": consumer_kinds[arm],
                        "stratum": fields,
                        "command": command,
                        "metrics": list(metrics),
                    }
                )
            comparisons.append(
                {
                    "name": f"{name}-goodput",
                    "experiment": name,
                    "numerator_variant": numerator,
                    "denominator_variant": denominator,
                    "metric": "slo_goodput",
                }
            )
    spec: dict[str, Any] = {
        "schema": 1,
        "classification": "nta-paired-evaluation",
        "evaluation_profile": "osdi-complete",
        "generated_by": "experiments/make_evaluation_spec.py",
        "workload_manifest": str(workload_manifest.resolve()),
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
        workload_manifest.resolve(),
        qualification_path=tier_qualification.resolve() if tier_qualification else None,
    )
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument(
        "--strata-file",
        type=Path,
        default=Path(__file__).with_name("strata.example.json"),
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
        "--arm-consumer-kind",
        action="append",
        required=True,
        metavar="ARM=KIND",
        help="formal numerical consumer: native_work_unit or framework_reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = tuple(dict.fromkeys(args.metric))
    if not metrics:
        parser.error("at least one metric is required")
    try:
        spec = build_spec(
            workload_manifest=args.workload_manifest,
            tier=args.tier,
            arm_commands=_commands(args.arm_command),
            consumer_kinds=_consumer_kinds(args.arm_consumer_kind),
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
