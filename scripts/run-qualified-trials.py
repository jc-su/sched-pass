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
        "nvme": command_output(["nvme", "list", "-o", "json"]),
        "iommu_groups": command_output(["find", "/sys/kernel/iommu_groups", "-type", "l"]),
    }


def read_spec(path: pathlib.Path) -> dict[str, Any]:
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
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("experiment entries must be objects")
        name = experiment.get("name")
        variant = experiment.get("variant")
        command = experiment.get("command")
        metrics = experiment.get("metrics", [])
        environment = experiment.get("environment", {})
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
        ):
            raise ValueError(f"invalid experiment definition: {experiment}")
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
    return spec


def final_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("trial command did not emit a JSON object")


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
            values = [record["result"].get(metric) for record in selected]
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
                    value = record["result"].get(comparison["metric"])
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
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
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
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "machine": machine_metadata(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    trial_path = output / "trials.jsonl"
    with trial_path.open("w", encoding="utf-8") as trial_file:
        for sequence, (repetition, experiment) in enumerate(jobs):
            environment = os.environ.copy()
            environment.update(experiment.get("environment", {}))
            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            started = time.monotonic()
            result = subprocess.run(
                experiment["command"],
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
            log_path.write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"trial failed with status {result.returncode}: {log_path}"
                )
            parsed = final_json(result.stdout)
            record = {
                "schema": 1,
                "revision": revision,
                "sequence": sequence,
                "repetition": repetition,
                "experiment": experiment["name"],
                "variant": experiment["variant"],
                "command": experiment["command"],
                "environment": experiment.get("environment", {}),
                "started_at": started_at,
                "duration_seconds": duration,
                "log": str(log_path.relative_to(ROOT))
                if log_path.is_relative_to(ROOT)
                else str(log_path),
                "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                "result": parsed,
            }
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
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"trials": len(records), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
