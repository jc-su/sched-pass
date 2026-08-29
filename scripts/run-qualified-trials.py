#!/usr/bin/env python3
"""Run randomized qualification trials with revision and raw-log provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.validate_tier_qualification import (  # noqa: E402
    validate_file as validate_tier_qualification,
)
from experiments.atomic_io import atomic_write_json, atomic_write_text  # noqa: E402
from experiments.hardware import nvme_controllers, platform_identity  # noqa: E402
from experiments.mechanism_arms import validate_arm_result  # noqa: E402
from experiments.result_contracts import (  # noqa: E402
    extract_trial_metrics,
    result_demand_digest,
    result_contract_names,
    validate_trial_result,
)
from experiments.validate_workload import validate as validate_workload  # noqa: E402

T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
    21: 2.086,
    22: 2.080,
    23: 2.074,
    24: 2.069,
    25: 2.064,
    26: 2.060,
    27: 2.056,
    28: 2.052,
    29: 2.048,
    30: 2.045,
}


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def machine_metadata() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "physical_identity": platform_identity(),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
        "gpu_clocks": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.current.sm,clocks.current.memory,"
                "clocks.applications.graphics,clocks.applications.memory,power.limit",
                "--format=csv,noheader",
            ]
        ),
        "nvme": {
            "source": "read_only_sysfs",
            "controllers": nvme_controllers(),
        },
        "iommu_groups": command_output(
            ["find", "/sys/kernel/iommu_groups", "-type", "l"]
        ),
    }


def read_spec(path: pathlib.Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read trial specification: {error}") from error
    if not isinstance(spec, dict) or spec.get("schema") != 1:
        raise ValueError("trial specification must use schema 1")
    repetitions = spec.get("repetitions")
    experiments = spec.get("experiments")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("trial repetitions must be positive")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("trial specification needs experiments")
    names: set[tuple[str, str]] = set()
    formal = spec.get("evaluation_profile") == "osdi-complete"
    supported_contracts = result_contract_names(formal_only=formal)
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("experiment entries must be objects")
        name = experiment.get("name")
        variant = experiment.get("variant")
        command = experiment.get("command")
        metrics = experiment.get("metrics", [])
        environment = experiment.get("environment", {})
        result_contract = experiment.get("result_contract")
        workload_manifest = experiment.get("workload_manifest")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(variant, str)
            or not variant
            or not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) and value for value in command)
            or not isinstance(metrics, list)
            or not all(isinstance(value, str) and value for value in metrics)
            or not isinstance(environment, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()
            )
            or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
            or re.fullmatch(r"[A-Za-z0-9_.-]+", variant) is None
            or (formal and result_contract not in supported_contracts)
            or (
                workload_manifest is not None
                and (not isinstance(workload_manifest, str) or not workload_manifest)
            )
            or (formal and not isinstance(workload_manifest, str))
            or (
                not formal
                and result_contract is not None
                and result_contract not in supported_contracts
            )
        ):
            raise ValueError(f"invalid experiment definition: {experiment}")
        static_output = any(
            (
                token == "--output"
                and (
                    index + 1 >= len(command)
                    or "{trial_output}" not in command[index + 1]
                )
            )
            or (token.startswith("--output=") and "{trial_output}" not in token)
            for index, token in enumerate(command)
        )
        if formal and static_output:
            raise ValueError(
                "formal trial commands with --output must use the unique "
                "{trial_output} placeholder"
            )
        identity = (name, variant)
        if identity in names:
            raise ValueError(f"duplicate experiment variant: {name}/{variant}")
        names.add(identity)
    comparisons = spec.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("trial comparisons must be a list")
    comparison_names: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("comparison entries must be objects")
        name = comparison.get("name")
        experiment = comparison.get("experiment")
        numerator = comparison.get("numerator_variant")
        denominator = comparison.get("denominator_variant")
        metric = comparison.get("metric")
        if (
            not all(
                isinstance(value, str) and value
                for value in (name, experiment, numerator, denominator, metric)
            )
            or (experiment, numerator) not in names
            or (experiment, denominator) not in names
            or numerator == denominator
            or name in comparison_names
        ):
            raise ValueError(f"invalid comparison definition: {comparison}")
        comparison_names.add(name)
        for variant in (numerator, denominator):
            matching = next(
                entry
                for entry in experiments
                if entry["name"] == experiment and entry["variant"] == variant
            )
            if metric not in matching.get("metrics", []):
                raise ValueError(
                    f"comparison metric {metric} is not collected for "
                    f"{experiment}/{variant}"
                )
    # Specification paths are relative to the specification itself, not to the
    # source checkout.  Normalize them once before provenance capture and
    # execution.  Replacing the declared path inside command tokens keeps a
    # copied artifact relocatable without asking each framework harness to
    # implement a second path-resolution convention.
    for experiment in experiments:
        raw_workload = experiment.get("workload_manifest")
        if not isinstance(raw_workload, str):
            continue
        workload_path = pathlib.Path(raw_workload)
        if not workload_path.is_absolute():
            workload_path = path.parent / workload_path
        resolved_workload = str(workload_path.resolve())
        experiment["command"] = [
            token.replace(raw_workload, resolved_workload)
            for token in experiment["command"]
        ]
        experiment["workload_manifest"] = resolved_workload
    declared_workloads = spec.get("workload_manifests")
    if isinstance(declared_workloads, list):
        spec["workload_manifests"] = [
            str(
                (
                    pathlib.Path(value)
                    if pathlib.Path(value).is_absolute()
                    else path.parent / pathlib.Path(value)
                ).resolve()
            )
            for value in declared_workloads
            if isinstance(value, str)
        ]
    qualification = spec.get("tier_qualification")
    if isinstance(qualification, str):
        qualification_path = pathlib.Path(qualification)
        if not qualification_path.is_absolute():
            qualification_path = path.parent / qualification_path
        spec["tier_qualification"] = str(qualification_path.resolve())
    return spec


def workload_provenance(value: str | pathlib.Path) -> dict[str, Any]:
    """Return immutable identity for one experiment-owned workload."""

    path = pathlib.Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"workload manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"workload manifest cannot be read: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("workload manifest must be a JSON object")
    try:
        validate_workload(path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"workload manifest failed validation: {error}") from error
    demand_digest = manifest.get("demand_trace_digest")
    if not isinstance(demand_digest, str) or not demand_digest:
        raise ValueError("workload manifest has no exact demand digest")
    return {
        "workload_manifest": str(path),
        "workload_manifest_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
        "workload_demand_digest": demand_digest,
    }


def workload_provenances(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate every distinct workload before launching any trial."""

    identities: dict[str, dict[str, Any]] = {}
    for experiment in spec.get("experiments", []):
        value = experiment.get("workload_manifest")
        if value is None:
            continue
        identity = workload_provenance(value)
        identities[identity["workload_manifest"]] = identity
    return identities


def tier_qualification_provenance(
    spec: dict[str, Any], *, revision: str
) -> dict[str, Any]:
    """Attach the physical-tier admission artifact to every trial record."""

    value = spec.get("tier_qualification")
    if not isinstance(value, str) or not value:
        return {}
    path = pathlib.Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"tier qualification artifact is missing: {path}")
    required_tiers = tuple(
        dict.fromkeys(
            str(experiment["tier"])
            for experiment in spec.get("experiments", [])
            if isinstance(experiment, dict)
            and experiment.get("tier") in {"hbm", "host_mem", "nvme", "dax"}
        )
    ) or ("hbm", "host_mem")
    try:
        document = validate_tier_qualification(path, required_tiers=required_tiers)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(
            f"tier qualification artifact failed validation: {error}"
        ) from error
    current_platform = platform_identity()
    for entry in document["entries"]:
        if entry.get("tier") != "nvme":
            continue
        report = entry["report"]
        if report.get("revision") != revision:
            raise ValueError(
                "NVMe qualification revision does not match the trial revision"
            )
        if report.get("platform_identity") != current_platform:
            raise ValueError(
                "NVMe qualification belongs to a different boot, kernel, or driver"
            )
    return {
        "tier_qualification": str(path),
        "tier_qualification_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
        "qualification_platform_identity": current_platform,
    }


def final_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("trial command did not emit a JSON object")


def materialize_trial_command(
    command: list[str], trial_output: pathlib.Path
) -> tuple[list[str], bool]:
    """Bind a command's optional result token to one non-overwriting path."""

    used = any("{trial_output}" in token for token in command)
    return (
        [token.replace("{trial_output}", str(trial_output)) for token in command],
        used,
    )


def interval(values: list[float]) -> dict[str, float | int]:
    count = len(values)
    mean = statistics.fmean(values)
    if count == 1:
        half = math.nan
    else:
        critical = T95.get(count, 1.96)
        half = critical * statistics.stdev(values) / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "ci95_half_width": half,
    }


def summarize(records: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for experiment in spec["experiments"]:
        selected = [
            record
            for record in records
            if record["experiment"] == experiment["name"]
            and record["variant"] == experiment["variant"]
        ]
        metrics: dict[str, Any] = {}
        for metric in experiment.get("metrics", []):
            values = [record["metrics"].get(metric) for record in selected]
            if len(values) != len(selected) or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in values
            ):
                raise ValueError(
                    f"metric {metric} is missing or non-finite in "
                    f"{experiment['name']}/{experiment['variant']}"
                )
            metrics[metric] = interval([float(value) for value in values])
        summaries.append(
            {
                "experiment": experiment["name"],
                "variant": experiment["variant"],
                "metrics": metrics,
            }
        )
    comparisons: list[dict[str, Any]] = []
    for comparison in spec.get("comparisons", []):
        ratios: list[float] = []
        for repetition in range(spec["repetitions"]):
            values: dict[str, float] = {}
            for record in records:
                if (
                    record["repetition"] == repetition
                    and record["experiment"] == comparison["experiment"]
                    and record["variant"]
                    in (
                        comparison["numerator_variant"],
                        comparison["denominator_variant"],
                    )
                ):
                    value = record["metrics"].get(comparison["metric"])
                    if not isinstance(value, (int, float)) or not math.isfinite(
                        float(value)
                    ):
                        raise ValueError(
                            f"comparison metric {comparison['metric']} is invalid"
                        )
                    values[record["variant"]] = float(value)
            if len(values) != 2 or values[comparison["denominator_variant"]] == 0:
                raise ValueError(
                    f"comparison {comparison['name']} lacks a complete trial pair"
                )
            ratios.append(
                values[comparison["numerator_variant"]]
                / values[comparison["denominator_variant"]]
            )
        comparisons.append(
            {
                "name": comparison["name"],
                "experiment": comparison["experiment"],
                "metric": comparison["metric"],
                "ratio": (
                    f"{comparison['numerator_variant']}/"
                    f"{comparison['denominator_variant']}"
                ),
                "interval": interval(ratios),
            }
        )
    return {"schema": 1, "summaries": summaries, "comparisons": comparisons}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = read_spec(args.spec)
    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise RuntimeError("qualification trials require a clean worktree")
    expected_revision = spec.get("revision")
    if expected_revision is not None and expected_revision != revision:
        raise RuntimeError("trial specification revision does not match HEAD")

    output = args.output_dir.resolve()
    if output == ROOT or ROOT in output.parents:
        raise RuntimeError("qualification output must be outside the source tree")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"qualification output is not empty: {output}")
    logs = output / "logs"
    structured_results = output / "results"
    logs.mkdir(parents=True, exist_ok=True)
    structured_results.mkdir(parents=True, exist_ok=True)
    generator = random.Random(int(spec.get("seed", 1)))
    repetitions = list(range(spec["repetitions"]))
    generator.shuffle(repetitions)
    jobs: list[tuple[int, dict[str, Any]]] = []
    for repetition in repetitions:
        block = list(spec["experiments"])
        generator.shuffle(block)
        jobs.extend((repetition, experiment) for experiment in block)
    metadata = {
        "schema": 1,
        "revision": revision,
        "dirty": dirty,
        "spec": spec,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "machine": machine_metadata(),
    }
    workload_identities = workload_provenances(spec)
    tier_identity = tier_qualification_provenance(spec, revision=revision)
    metadata["workloads"] = [
        workload_identities[path] for path in sorted(workload_identities)
    ]
    metadata.update(tier_identity)
    atomic_write_json(output / "metadata.json", metadata)

    records: list[dict[str, Any]] = []
    trial_path = output / "trials.jsonl"
    with trial_path.open("w", encoding="utf-8") as trial_file:
        for sequence, (repetition, experiment) in enumerate(jobs):
            workload_value = experiment.get("workload_manifest")
            workload_identity = (
                workload_identities[str(pathlib.Path(workload_value).resolve())]
                if workload_value is not None
                else {}
            )
            environment = os.environ.copy()
            environment.update(experiment.get("environment", {}))
            result_name = (
                f"{sequence:04d}-{experiment['name']}-{experiment['variant']}-"
                f"{repetition:03d}.json"
            )
            structured_result_path = structured_results / result_name
            command, writes_structured_result = materialize_trial_command(
                experiment["command"], structured_result_path
            )
            environment["NTA_TRIAL_OUTPUT"] = str(structured_result_path)
            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            started = time.monotonic()
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            duration = time.monotonic() - started
            log_name = (
                f"{sequence:04d}-{experiment['name']}-{experiment['variant']}-"
                f"{repetition:03d}.log"
            )
            log_path = logs / log_name
            atomic_write_text(log_path, result.stdout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"trial failed with status {result.returncode}: {log_path}"
                )
            parsed = final_json(result.stdout)
            structured_result_digest = None
            if writes_structured_result:
                if not structured_result_path.is_file():
                    raise RuntimeError(
                        "trial command consumed {trial_output} but did not write "
                        f"{structured_result_path}"
                    )
                try:
                    file_result = json.loads(
                        structured_result_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"trial structured result is invalid: {error}"
                    ) from error
                if file_result != parsed:
                    raise RuntimeError("trial stdout and structured result disagree")
                structured_result_digest = hashlib.sha256(
                    structured_result_path.read_bytes()
                ).hexdigest()
            formal = spec.get("evaluation_profile") == "osdi-complete"
            validate_trial_result(
                parsed,
                expected_contract=experiment.get("result_contract"),
                formal=formal,
            )
            metric_values = extract_trial_metrics(
                parsed,
                experiment.get("metrics", []),
                expected_contract=experiment.get("result_contract"),
                formal=formal,
            )
            observed_demand_digest = None
            arm_activation = None
            if formal:
                arm_activation = validate_arm_result(parsed, experiment["arm"])
                observed_demand_digest = result_demand_digest(parsed)
                expected_demand_digest = workload_identity.get("workload_demand_digest")
                if observed_demand_digest != expected_demand_digest:
                    raise RuntimeError(
                        "trial consumed a different demand trace than its "
                        "qualified workload manifest"
                    )
            record = {
                "schema": 1,
                "revision": revision,
                "sequence": sequence,
                "repetition": repetition,
                "experiment": experiment["name"],
                "variant": experiment["variant"],
                "command": command,
                "environment": experiment.get("environment", {}),
                "arm": experiment.get("arm"),
                "tier": experiment.get("tier"),
                "stratum": experiment.get("stratum"),
                "demand_semantics": experiment.get("demand_semantics"),
                "result_contract": experiment.get("result_contract"),
                "workload_manifest": workload_identity.get("workload_manifest"),
                "observed_workload_demand_digest": observed_demand_digest,
                "started_at": started_at,
                "duration_seconds": duration,
                "log": str(log_path.relative_to(ROOT))
                if log_path.is_relative_to(ROOT)
                else str(log_path),
                "result": parsed,
                "structured_result": (
                    str(structured_result_path) if writes_structured_result else None
                ),
                "structured_result_digest": structured_result_digest,
                "metrics": metric_values,
                "arm_activation": arm_activation,
            }
            record.update(workload_identity)
            record.update(tier_identity)
            records.append(record)
            trial_file.write(json.dumps(record, sort_keys=True) + "\n")
            trial_file.flush()

    summary = summarize(records, spec)
    summary.update(
        {
            "revision": revision,
            "independent_trials": spec["repetitions"],
            "randomized_order": True,
            "randomization": "randomized_complete_blocks",
            "controlled_clocks": spec.get("controlled_clocks") is True,
        }
    )
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps({"trials": len(records), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
