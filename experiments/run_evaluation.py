#!/usr/bin/env python3
"""Run the canonical paired-trial evaluator with exact-tier metadata.

The existing qualified-trial engine owns subprocess execution, randomized
complete blocks, raw logs, and confidence intervals.  This wrapper owns the
OSDI-specific contract: a normalized workload, explicit arm/tier/stratum
identity, exact demand semantics, and an external non-overwriting artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

try:
    from .validate_workload import validate as validate_workload
    from .analyze_evaluation import analyze as analyze_evaluation
    from .validate_tier_qualification import (
        validate_file as validate_tier_qualification,
    )
    from .validate_evaluation import validate as validate_evaluation_contract
    from .validate_evaluation_artifact import validate as validate_evaluation_artifact
except ImportError:
    from validate_workload import validate as validate_workload
    from analyze_evaluation import analyze as analyze_evaluation
    from validate_tier_qualification import validate_file as validate_tier_qualification
    from validate_evaluation import validate as validate_evaluation_contract
    from validate_evaluation_artifact import validate as validate_evaluation_artifact


ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_RUNNER = ROOT / "scripts" / "run-qualified-trials.py"
EVALUATION_MANIFEST = ROOT / "experiments" / "evaluation-manifest.json"
REQUIRED_STRATA = {
    "request_state",
    "granularity",
    "load_ratio",
    "availability_skew",
    "staging_pressure",
    "arrival",
}
EVALUATION_PROFILES = {"contract", "osdi-complete"}
CANONICAL_ARMS = {f"B{index}" for index in range(7)}
FORMAL_CONSUMER_KINDS = {"native_work_unit", "framework_reference"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_qualification_tiers(spec: dict[str, Any]) -> set[str]:
    return {
        str(trial["tier"])
        for trial in spec.get("experiments", [])
        if trial.get("tier") in {"nvme", "dax"}
    }


def validate_spec(
    spec: dict[str, Any],
    workload_path: Path,
    *,
    qualification_path: Path | None = None,
) -> dict[str, Any]:
    if spec.get("schema") != 1 or spec.get("classification") != "nta-paired-evaluation":
        raise ValueError(
            "evaluation trial spec must use nta-paired-evaluation schema 1"
        )
    repetitions = spec.get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 5:
        raise ValueError("OSDI evaluation requires at least five repetitions")
    trials = spec.get("experiments")
    if not isinstance(trials, list) or not trials:
        raise ValueError("evaluation trial spec contains no experiments")
    evaluation_profile = spec.get("evaluation_profile", "contract")
    if evaluation_profile not in EVALUATION_PROFILES:
        raise ValueError("evaluation_profile must be contract or osdi-complete")
    contract = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    validate_evaluation_contract(contract)
    tiers = {tier["id"] for tier in contract["tiers"]}
    arms = {arm["id"] for arm in contract["arms"]}
    allowed_strata = contract["strata"]
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    consumer_kinds_by_arm: dict[str, set[str]] = {}
    for trial in trials:
        if not isinstance(trial, dict) or not trial.get("command"):
            raise ValueError("each evaluation trial needs a command")
        if not isinstance(trial.get("name"), str) or not isinstance(
            trial.get("variant"), str
        ):
            raise ValueError("each evaluation trial needs a name and variant")
        if trial.get("arm") not in arms:
            raise ValueError(f"trial uses an undeclared arm: {trial.get('arm')}")
        if trial.get("tier") not in tiers:
            raise ValueError(f"trial uses an undeclared tier: {trial.get('tier')}")
        if trial.get("demand_semantics") != "exact":
            raise ValueError("evaluation trials must declare exact demand")
        if evaluation_profile == "osdi-complete":
            consumer_kind = trial.get("consumer_kind")
            if consumer_kind not in FORMAL_CONSUMER_KINDS:
                raise ValueError(
                    "osdi-complete trials must declare a numerical consumer_kind "
                    "of native_work_unit or framework_reference"
                )
            consumer_kinds_by_arm.setdefault(trial["arm"], set()).add(
                consumer_kind
            )
        if not isinstance(trial.get("stratum"), dict) or not REQUIRED_STRATA <= set(
            trial["stratum"]
        ):
            raise ValueError(
                "each trial needs the complete request/tier/load/arrival strata"
            )
        for stratum_name in REQUIRED_STRATA:
            if trial["stratum"][stratum_name] not in allowed_strata[stratum_name]:
                raise ValueError(
                    f"trial uses an undeclared {stratum_name} stratum: "
                    f"{trial['stratum'][stratum_name]!r}"
                )
        metrics = trial.get("metrics")
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(metric, str) and metric for metric in metrics)
        ):
            raise ValueError("each evaluation trial needs a non-empty metric contract")
        identity = (trial["name"], trial["variant"])
        if identity in identities:
            raise ValueError(
                f"duplicate evaluation variant: {identity[0]}/{identity[1]}"
            )
        identities[identity] = trial
    by_name: dict[str, list[dict[str, Any]]] = {}
    for trial in identities.values():
        by_name.setdefault(trial["name"], []).append(trial)
    if not by_name or any(len(variants) < 2 for variants in by_name.values()):
        raise ValueError("each experiment needs at least two paired variants")
    for name, variants in by_name.items():
        reference = variants[0]
        reference_key = (
            reference["tier"],
            reference["demand_semantics"],
            json.dumps(reference["stratum"], sort_keys=True, separators=(",", ":")),
        )
        for variant in variants[1:]:
            key = (
                variant["tier"],
                variant["demand_semantics"],
                json.dumps(variant["stratum"], sort_keys=True, separators=(",", ":")),
            )
            if key != reference_key:
                raise ValueError(
                    f"paired variants in {name} do not share tier/demand/stratum"
                )
    comparisons = spec.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("evaluation specification needs causal comparisons")
    comparison_names: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("causal comparison is not an object")
        name = comparison.get("name")
        experiment = comparison.get("experiment")
        numerator = comparison.get("numerator_variant")
        denominator = comparison.get("denominator_variant")
        metric = comparison.get("metric")
        if (
            not isinstance(name, str)
            or not name
            or name in comparison_names
            or not isinstance(experiment, str)
            or not isinstance(numerator, str)
            or not isinstance(denominator, str)
            or numerator == denominator
            or not isinstance(metric, str)
        ):
            raise ValueError("causal comparison has invalid identity")
        variants = {variant["variant"] for variant in by_name.get(experiment, [])}
        if numerator not in variants or denominator not in variants:
            raise ValueError(
                f"causal comparison {name} references an undeclared variant"
            )
        declared_metrics = {
            metric_name
            for variant in by_name[experiment]
            for metric_name in variant.get("metrics", [])
        }
        if metric not in declared_metrics:
            raise ValueError(
                f"causal comparison {name} references an unmeasured metric"
            )
        comparison_names.add(name)
    if evaluation_profile == "osdi-complete":
        declared_arms = {trial["arm"] for trial in identities.values()}
        if declared_arms != CANONICAL_ARMS:
            raise ValueError("osdi-complete evaluation must contain exactly B0-B6")
        if any(
            consumer_kinds_by_arm.get(arm, set())
            and len(consumer_kinds_by_arm[arm]) != 1
            for arm in CANONICAL_ARMS
        ):
            raise ValueError(
                "osdi-complete arms must use one declared consumer_kind across "
                "all strata"
            )
        if any(
            len(consumer_kinds_by_arm.get(arm, set())) != 1
            for arm in CANONICAL_ARMS
        ):
            raise ValueError(
                "osdi-complete evaluation must declare a consumer_kind for every "
                "canonical arm"
            )
        declared_tiers = {trial["tier"] for trial in identities.values()}
        if len(declared_tiers) != 1:
            raise ValueError(
                "osdi-complete evaluation must measure one tier per paired spec"
            )

        def stratum_key(value: Mapping[str, Any]) -> str:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))

        arm_strata = {
            arm: {
                stratum_key(trial["stratum"])
                for trial in identities.values()
                if trial["arm"] == arm
            }
            for arm in CANONICAL_ARMS
        }
        if len({frozenset(value) for value in arm_strata.values()}) != 1:
            raise ValueError("osdi-complete arms do not cover the same strata")
        stratum_keys = next(iter(arm_strata.values()))
        if len(stratum_keys) < 6:
            raise ValueError(
                "osdi-complete evaluation needs at least six workload strata"
            )

        actual_pairs: set[tuple[str, str, str]] = set()
        for comparison in comparisons:
            variants = by_name[comparison["experiment"]]
            numerator_trials = [
                trial
                for trial in variants
                if trial["variant"] == comparison["numerator_variant"]
            ]
            if len(numerator_trials) != 1:
                raise ValueError(
                    "osdi-complete causal comparisons must identify one stratum"
                )
            actual_pairs.add(
                (
                    comparison["numerator_variant"],
                    comparison["denominator_variant"],
                    stratum_key(numerator_trials[0]["stratum"]),
                )
            )
        if len(actual_pairs) != len(comparisons):
            raise ValueError("osdi-complete causal comparisons contain duplicates")
        expected_pairs = {
            (pair["numerator"], pair["denominator"], stratum)
            for pair in contract["causal_pairs"]
            for stratum in stratum_keys
        }
        if actual_pairs != expected_pairs:
            raise ValueError(
                "osdi-complete causal comparisons must cover every canonical "
                "boundary in every declared stratum"
            )
    required_qualification_tiers = _required_qualification_tiers(spec)
    if required_qualification_tiers and qualification_path is None:
        raise ValueError(
            "NVMe/DAX evaluation requires tier_qualification with a qualified "
            "physical-tier artifact"
        )
    if qualification_path is not None:
        validate_tier_qualification(
            qualification_path,
            required_tiers=required_qualification_tiers or {"hbm", "host_mem"},
        )
    return validate_workload(workload_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve()
    workload_path = None
    qualification_path = None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("evaluation trial spec is not an object")
        workload_path = Path(spec.get("workload_manifest", "")).resolve()
        qualification_value = spec.get("tier_qualification")
        if qualification_value is not None:
            qualification_path = Path(str(qualification_value)).resolve()
        workload_manifest = validate_spec(
            spec,
            workload_path,
            qualification_path=qualification_path,
        )
        output = args.output_dir.resolve()
        if output == ROOT or ROOT in output.parents:
            raise ValueError("evaluation output must be outside the source tree")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"evaluation output is not empty: {output}")
        command = [
            sys.executable,
            str(QUALIFIED_RUNNER),
            "--spec",
            str(spec_path),
            "--output-dir",
            str(output),
        ]
        if args.allow_dirty:
            command.append("--allow-dirty")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            return completed.returncode
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation-contract.json").write_bytes(
            EVALUATION_MANIFEST.read_bytes()
        )
        (output / "evaluation-metadata.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "classification": "nta-osdi-paired-evaluation",
                    "evaluation_profile": spec.get("evaluation_profile", "contract"),
                    "spec": str(spec_path),
                    "spec_digest": _digest(spec_path),
                    "workload_manifest": str(workload_path),
                    "workload_manifest_digest": _digest(workload_path),
                    "workload_demand_digest": workload_manifest["demand_trace_digest"],
                    "tier_qualification": str(qualification_path)
                    if qualification_path
                    else None,
                    "tier_qualification_digest": (
                        _digest(qualification_path) if qualification_path else None
                    ),
                    "qualified_tiers": sorted(_required_qualification_tiers(spec)),
                    "evaluation_contract_digest": _digest(EVALUATION_MANIFEST),
                    "repetitions": spec["repetitions"],
                    "randomized_order": True,
                    "arm_set": sorted({trial["arm"] for trial in spec["experiments"]}),
                    "consumer_kinds": {
                        arm: sorted(
                            {
                                trial["consumer_kind"]
                                for trial in spec["experiments"]
                                if trial["arm"] == arm
                                and isinstance(trial.get("consumer_kind"), str)
                            }
                        )[0]
                        for arm in sorted(
                            {trial["arm"] for trial in spec["experiments"]}
                        )
                        if any(
                            isinstance(trial.get("consumer_kind"), str)
                            for trial in spec["experiments"]
                            if trial["arm"] == arm
                        )
                    },
                    "tier_set": sorted({trial["tier"] for trial in spec["experiments"]}),
                    "causal_pairs": sorted(
                        {
                            f"{comparison['numerator_variant']}>{comparison['denominator_variant']}"
                            for comparison in spec["comparisons"]
                        }
                    ),
                    "strata_count": len(
                        {
                            json.dumps(trial["stratum"], sort_keys=True)
                            for trial in spec["experiments"]
                        }
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        analyze_evaluation(output)
        # Keep the direct evaluator fail-closed as well as the outer artifact
        # packager.  A caller must never mistake a generated-but-incomplete
        # report for a qualified result simply because it skipped the second
        # command-line validator.
        validate_evaluation_artifact(output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"run_evaluation refused: {error}", file=sys.stderr)
        return 2
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
