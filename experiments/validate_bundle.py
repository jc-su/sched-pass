#!/usr/bin/env python3
"""Validate the structure and provenance of a reproduction bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .artifact import file_digest
    from .hardware import validate as validate_hardware
    from .validate_evaluation import validate as validate_evaluation
    from .validate_evaluation_artifact import validate as validate_evaluation_artifact
    from .validate_matrix_artifact import validate as validate_matrix
    from .validate_serving_report import validate as validate_serving_report
    from .validate_tier_qualification import validate_file as validate_tier_qualification
    from .validate_tier_catalog import validate as validate_tier_catalog
    from .validate_workload import validate as validate_workload
except ImportError:  # Direct script execution.
    from artifact import file_digest
    from hardware import validate as validate_hardware
    from validate_evaluation import validate as validate_evaluation
    from validate_evaluation_artifact import validate as validate_evaluation_artifact
    from validate_matrix_artifact import validate as validate_matrix
    from validate_serving_report import validate as validate_serving_report
    from validate_tier_qualification import validate_file as validate_tier_qualification
    from validate_tier_catalog import validate as validate_tier_catalog
    from validate_workload import validate as validate_workload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_bundle(bundle: Path) -> dict[str, Any]:
    metadata_path = bundle / "metadata.json"
    manifest_path = bundle / "artifact-manifest.json"
    commands_path = bundle / "commands.json"
    evaluation_path = bundle / "evaluation-manifest.json"
    _require(metadata_path.is_file(), "bundle has no metadata.json")
    _require(manifest_path.is_file(), "bundle has no artifact-manifest.json")
    _require(commands_path.is_file(), "bundle has no commands.json")
    _require(evaluation_path.is_file(), "bundle has no evaluation manifest")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(metadata.get("schema") == 1, "unsupported artifact metadata schema")
    _require(metadata.get("status") == "complete", "artifact did not complete")
    _require(
        file_digest(manifest_path) == metadata.get("artifact_manifest_digest"),
        "artifact manifest digest does not match metadata",
    )
    _require(
        file_digest(evaluation_path) == metadata.get("evaluation_manifest_digest"),
        "evaluation manifest digest does not match metadata",
    )
    validate_evaluation(json.loads(evaluation_path.read_text(encoding="utf-8")))
    commands = metadata.get("commands", [])
    _require(
        commands == json.loads(commands_path.read_text(encoding="utf-8")),
        "commands.json does not match metadata",
    )
    for command in commands:
        _require(command.get("returncode") == 0, "bundle contains a failed command")
        log = Path(command.get("log", ""))
        _require(not log.is_absolute() and ".." not in log.parts, "unsafe log path")
        _require((bundle / log).is_file(), f"command log is missing: {log}")
    if metadata.get("profile") == "serving":
        _require(bool(metadata.get("result")), "serving bundle has no structured result")
        if metadata.get("tier_catalog"):
            catalog_name = Path(str(metadata["tier_catalog"]))
            _require(
                not catalog_name.is_absolute() and ".." not in catalog_name.parts,
                "unsafe tier catalog path",
            )
            catalog_path = bundle / catalog_name
            _require(catalog_path.is_file(), "serving tier catalog is missing")
            _require(
                file_digest(catalog_path) == metadata.get("tier_catalog_digest"),
                "serving tier catalog digest does not match metadata",
            )
            validate_tier_catalog(catalog_path, str(metadata.get("serving_tier")))
    if metadata.get("profile") == "hardware":
        inventory_name = metadata.get("hardware_inventory")
        _require(isinstance(inventory_name, str) and bool(inventory_name), "hardware bundle has no inventory")
        inventory_path = Path(inventory_name)
        _require(not inventory_path.is_absolute() and ".." not in inventory_path.parts, "unsafe hardware inventory path")
        inventory_file = bundle / inventory_path
        _require(inventory_file.is_file(), "hardware inventory is missing")
        _require(
            file_digest(inventory_file) == metadata.get("hardware_inventory_digest"),
            "hardware inventory digest does not match metadata",
        )
        validate_hardware(json.loads(inventory_file.read_text(encoding="utf-8")))
    if metadata.get("result"):
        result = Path(metadata["result"])
        _require(not result.is_absolute() and ".." not in result.parts, "unsafe result path")
        result_path = bundle / result
        _require(result_path.is_file(), f"result file is missing: {result}")
        _require(
            file_digest(result_path) == metadata.get("result_digest"),
            "result digest does not match metadata",
        )
        if metadata.get("profile") == "serving":
            serving_report = json.loads(result_path.read_text(encoding="utf-8"))
            validate_serving_report(serving_report)
            if metadata.get("workload_replay_manifest"):
                workload_file = bundle / Path(metadata["workload_replay_manifest"])
                workload_records = workload_file.parent / "records.jsonl"
                reports = [serving_report]
                if serving_report.get("classification") == "sglang-hicache-load-comparison":
                    reports.extend(
                        [serving_report["stock"], serving_report["nta"]]
                    )
                for report in reports:
                    workload = report.get("workload")
                    if workload is None:
                        continue
                    _require(
                        workload.get("manifest_digest") == file_digest(workload_file),
                        "serving report manifest digest is not bundled workload",
                    )
                    _require(
                        workload.get("records_digest") == file_digest(workload_records),
                        "serving report records digest is not bundled workload",
                    )
    if metadata.get("workload_replay_manifest"):
        workload_manifest = Path(metadata["workload_replay_manifest"])
        _require(
            not workload_manifest.is_absolute() and ".." not in workload_manifest.parts,
            "unsafe workload manifest path",
        )
        workload_path = bundle / workload_manifest
        _require(workload_path.is_file(), "workload replay manifest is missing")
        _require(
            file_digest(workload_path) == metadata.get("workload_replay_manifest_digest"),
            "workload replay manifest digest does not match metadata",
        )
        records_path = workload_path.parent / "records.jsonl"
        _require(records_path.is_file(), "workload replay records are missing")
        _require(
            file_digest(records_path) == metadata.get("workload_replay_records_digest"),
            "workload replay records digest does not match metadata",
        )
        validate_workload(workload_path)
    if metadata.get("profile") in ("core", "matrix"):
        workload_path = bundle / "workload-manifest.json"
        matrix_path = bundle / "matrix.json"
        _require(workload_path.is_file(), "matrix bundle has no workload manifest")
        _require(matrix_path.is_file(), "matrix bundle has no matrix artifact")
        _require(
            file_digest(workload_path) == metadata.get("workload_manifest_digest"),
            "workload manifest digest does not match metadata",
        )
        validate_matrix(
            json.loads(matrix_path.read_text(encoding="utf-8")),
            require_all_ablations=True,
        )
    if metadata.get("profile") == "evaluation":
        rq0_path = bundle / str(metadata.get("rq0_opportunity", ""))
        _require(rq0_path.is_file(), "evaluation bundle has no RQ0 opportunity report")
        _require(
            file_digest(rq0_path) == metadata.get("rq0_opportunity_digest"),
            "RQ0 opportunity report digest does not match metadata",
        )
        rq0 = json.loads(rq0_path.read_text(encoding="utf-8"))
        _require(rq0.get("classification") == "bailian-rq0-opportunity-report", "invalid RQ0 opportunity report")
        evaluation_output = bundle / "evaluation"
        _require(evaluation_output.is_dir(), "evaluation bundle has no evaluation output")
        validate_evaluation_artifact(evaluation_output)
        spec_path = bundle / "evaluation-spec.json"
        _require(spec_path.is_file(), "evaluation bundle has no copied trial spec")
        _require(
            file_digest(spec_path) == metadata.get("evaluation_spec_digest"),
            "evaluation spec digest does not match metadata",
        )
        evaluation_metadata_path = evaluation_output / "evaluation-metadata.json"
        _require(evaluation_metadata_path.is_file(), "evaluation metadata is missing")
        evaluation_metadata = json.loads(
            evaluation_metadata_path.read_text(encoding="utf-8")
        )
        _require(
            evaluation_metadata.get("workload_manifest_digest")
            == metadata.get("workload_replay_manifest_digest")
            or evaluation_metadata.get("workload_manifest_digest")
            == file_digest(bundle / "workload/manifest.json"),
            "evaluation workload digest is not the bundled workload",
        )
        copied_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        physical_tiers = {
            str(trial["tier"])
            for trial in copied_spec.get("experiments", [])
            if trial.get("tier") in {"nvme", "dax"}
        }
        qualification_name = metadata.get("tier_qualification_manifest")
        _require(
            not physical_tiers or bool(qualification_name),
            "physical-tier evaluation bundle has no tier qualification",
        )
        if qualification_name:
            qualification_path = Path(str(qualification_name))
            _require(
                not qualification_path.is_absolute()
                and ".." not in qualification_path.parts,
                "unsafe tier qualification path",
            )
            qualification_file = bundle / qualification_path
            _require(qualification_file.is_file(), "tier qualification is missing")
            _require(
                file_digest(qualification_file)
                == metadata.get("tier_qualification_manifest_digest"),
                "tier qualification digest does not match metadata",
            )
            _require(
                evaluation_metadata.get("tier_qualification_digest")
                == metadata.get("tier_qualification_manifest_digest"),
                "evaluation qualification digest does not match bundle",
            )
            validate_tier_qualification(
                qualification_file,
                required_tiers=physical_tiers or {"hbm", "host_mem"},
            )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    validate_bundle(args.bundle.resolve())
    print("artifact_bundle=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
