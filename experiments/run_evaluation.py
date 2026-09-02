#!/usr/bin/env python3
"""Run the canonical paired-trial evaluator with exact-tier metadata.

The existing qualified-trial engine owns subprocess execution, randomized
complete blocks, raw logs, and confidence intervals.  This wrapper owns the
mechanism-study contract: a normalized workload, explicit arm/tier/stratum
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
    from .mechanism_arms import (
        ARMS,
        ARM_DEFINITIONS,
        FORMAL_SERVING_METRICS,
        arm_environment,
    )
    from .analyze_evaluation import analyze as analyze_evaluation
    from .validate_tier_qualification import (
        validate_file as validate_tier_qualification,
    )
    from .validate_evaluation import validate as validate_evaluation_contract
    from .validate_evaluation_artifact import validate as validate_evaluation_artifact
    from .result_contracts import result_contract_names
    from .workload_scenario import validate_workload_scenario
except ImportError:
    from mechanism_arms import (
        ARMS,
        ARM_DEFINITIONS,
        FORMAL_SERVING_METRICS,
        arm_environment,
    )
    from analyze_evaluation import analyze as analyze_evaluation
    from validate_tier_qualification import validate_file as validate_tier_qualification
    from validate_evaluation import validate as validate_evaluation_contract
    from validate_evaluation_artifact import validate as validate_evaluation_artifact
    from result_contracts import result_contract_names
    from workload_scenario import validate_workload_scenario


ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_RUNNER = ROOT / "scripts" / "run-qualified-trials.py"
EVALUATION_MANIFEST = ROOT / "experiments" / "evaluation-manifest.json"
EVALUATION_PROFILES = {"contract", "mechanism-study"}
CANONICAL_ARMS = set(ARMS)
FORMAL_CONSUMER_KINDS = {"native_work_unit", "framework_reference"}
TIER_ENVIRONMENT = {
    "hbm": "hbm",
    "host_mem": "host_staged",
    "nvme": "nvme",
    "dax": "cxl_dax",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_spec_path(value: str, base_dir: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _required_qualification_tiers(spec: dict[str, Any]) -> set[str]:
    return {
        str(trial["tier"])
        for trial in spec.get("experiments", [])
        if trial.get("tier") in {"nvme", "dax"}
    }


def validate_spec(
    spec: dict[str, Any],
    *,
    qualification_path: Path | None = None,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if spec.get("schema") != 1 or spec.get("classification") != "nta-paired-evaluation":
        raise ValueError(
            "evaluation trial spec must use nta-paired-evaluation schema 1"
        )
    repetitions = spec.get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 5:
        raise ValueError("paired evaluation requires at least five repetitions")
    trials = spec.get("experiments")
    if not isinstance(trials, list) or not trials:
        raise ValueError("evaluation trial spec contains no experiments")
    evaluation_profile = spec.get("evaluation_profile", "contract")
    if evaluation_profile not in EVALUATION_PROFILES:
        raise ValueError("evaluation_profile must be contract or mechanism-study")
    contract = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    validate_evaluation_contract(contract)
    tiers = {tier["id"] for tier in contract["tiers"]}
    mechanism_contract = contract["mechanism_study"]
    arms = {arm["id"] for arm in mechanism_contract["arms"]}
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    consumer_kinds_by_arm: dict[str, set[str]] = {}
    workloads: dict[str, dict[str, Any]] = {}
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
        if evaluation_profile == "mechanism-study":
            consumer_kind = trial.get("consumer_kind")
            if consumer_kind not in FORMAL_CONSUMER_KINDS:
                raise ValueError(
                    "mechanism-study trials must declare a numerical consumer_kind "
                    "of native_work_unit or framework_reference"
                )
            consumer_kinds_by_arm.setdefault(trial["arm"], set()).add(consumer_kind)
            definition = ARM_DEFINITIONS[trial["arm"]]
            if consumer_kind != definition["consumer_kind"]:
                raise ValueError(
                    f"{trial['arm']} consumer_kind diverges from its causal contract"
                )
            if trial.get("mechanism_form") != definition["name"]:
                raise ValueError(
                    f"{trial['arm']} mechanism_form diverges from its causal contract"
                )
            environment = trial.get("environment")
            expected_environment = arm_environment(trial["arm"])
            if not isinstance(environment, dict) or any(
                environment.get(name) != value
                for name, value in expected_environment.items()
            ) or environment.get("NTA_SERVING_TIER") != TIER_ENVIRONMENT.get(
                trial["tier"]
            ):
                raise ValueError(
                    f"{trial['arm']} environment does not select its causal form"
                )
            if trial.get("result_contract") not in result_contract_names(
                formal_only=True
            ):
                raise ValueError(
                    "mechanism-study trials must declare a supported result_contract"
                )
        elif trial.get("result_contract") is not None and trial.get(
            "result_contract"
        ) not in result_contract_names():
            raise ValueError("evaluation trial declares an unknown result_contract")
        workload_value = trial.get("workload_manifest")
        if not isinstance(workload_value, str) or not workload_value:
            raise ValueError("each evaluation trial needs a workload_manifest")
        workload_path = _resolve_spec_path(workload_value, base_dir)
        stratum = trial.get("stratum")
        if not isinstance(stratum, dict):
            raise ValueError("each evaluation trial needs a workload descriptor")
        descriptor = validate_workload_scenario(stratum, workload_path)
        prior_descriptor = workloads.get(str(workload_path))
        if prior_descriptor is not None and prior_descriptor != descriptor:
            raise ValueError(
                "one workload manifest was assigned multiple scenario identities"
            )
        workloads[str(workload_path)] = descriptor
        metrics = trial.get("metrics")
        if (
            not isinstance(metrics, list)
            or not metrics
            or not all(isinstance(metric, str) and metric for metric in metrics)
        ):
            raise ValueError("each evaluation trial needs a non-empty metric contract")
        if evaluation_profile == "mechanism-study" and not set(
            FORMAL_SERVING_METRICS
        ).issubset(metrics):
            raise ValueError(
                "mechanism-study trials must report TTFT, TPOT, ITL, throughput, "
                "admission delay, SLO goodput, and correctness"
            )
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
            str(_resolve_spec_path(reference["workload_manifest"], base_dir)),
            json.dumps(reference["stratum"], sort_keys=True, separators=(",", ":")),
        )
        for variant in variants[1:]:
            key = (
                variant["tier"],
                variant["demand_semantics"],
                str(_resolve_spec_path(variant["workload_manifest"], base_dir)),
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
    if evaluation_profile == "mechanism-study":
        declared_arms = {trial["arm"] for trial in identities.values()}
        if declared_arms != CANONICAL_ARMS:
            raise ValueError("mechanism-study evaluation must contain exactly A0-A3")
        if any(
            consumer_kinds_by_arm.get(arm, set())
            and len(consumer_kinds_by_arm[arm]) != 1
            for arm in CANONICAL_ARMS
        ):
            raise ValueError(
                "mechanism-study arms must use one declared consumer_kind across "
                "all strata"
            )
        if any(
            len(consumer_kinds_by_arm.get(arm, set())) != 1 for arm in CANONICAL_ARMS
        ):
            raise ValueError(
                "mechanism-study evaluation must declare a consumer_kind for every "
                "canonical arm"
            )
        declared_tiers = {trial["tier"] for trial in identities.values()}
        if len(declared_tiers) != 1:
            raise ValueError(
                "mechanism-study evaluation must measure one tier per paired spec"
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
            raise ValueError("mechanism-study arms do not cover the same strata")
        stratum_keys = next(iter(arm_strata.values()))
        if len(stratum_keys) < 6:
            raise ValueError(
                "mechanism-study evaluation needs at least six workload strata"
            )
        demand_identities = {
            str(trial["stratum"]["demand_trace_digest"])
            for trial in identities.values()
        }
        if len(demand_identities) < 6:
            raise ValueError(
                "mechanism-study evaluation needs at least six distinct consumed "
                "workload identities"
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
                    "mechanism-study causal comparisons must identify one stratum"
                )
            actual_pairs.add(
                (
                    comparison["numerator_variant"],
                    comparison["denominator_variant"],
                    stratum_key(numerator_trials[0]["stratum"]),
                )
            )
        if len(actual_pairs) != len(comparisons):
            raise ValueError("mechanism-study causal comparisons contain duplicates")
        expected_pairs = {
            (pair["numerator"], pair["denominator"], stratum)
            for pair in mechanism_contract["causal_pairs"]
            for stratum in stratum_keys
        }
        if actual_pairs != expected_pairs:
            raise ValueError(
                "mechanism-study causal comparisons must cover every canonical "
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
    declared_workloads = spec.get("workload_manifests")
    if (
        not isinstance(declared_workloads, list)
        or not all(isinstance(value, str) and value for value in declared_workloads)
        or sorted(_resolve_spec_path(value, base_dir).as_posix() for value in declared_workloads)
        != sorted(workloads)
    ):
        raise ValueError(
            "evaluation workload_manifests do not match trial-owned scenarios"
        )
    return workloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve()
    qualification_path = None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("evaluation trial spec is not an object")
        qualification_value = spec.get("tier_qualification")
        if qualification_value is not None:
            qualification_path = _resolve_spec_path(
                str(qualification_value), spec_path.parent
            )
        workload_manifests = validate_spec(
            spec,
            qualification_path=qualification_path,
            base_dir=spec_path.parent,
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
                    "workloads": [
                        {
                            "manifest": path,
                            "manifest_digest": _digest(Path(path)),
                            "demand_trace_digest": descriptor[
                                "demand_trace_digest"
                            ],
                            "scenario": descriptor,
                        }
                        for path, descriptor in sorted(workload_manifests.items())
                    ],
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
                    "tier_set": sorted(
                        {trial["tier"] for trial in spec["experiments"]}
                    ),
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
