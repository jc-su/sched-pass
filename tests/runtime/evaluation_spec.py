#!/usr/bin/env python3
"""Contract tests for complete paired evaluation-spec generation."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.bailian import normalize, write_workload  # noqa: E402
from experiments.make_evaluation_spec import (  # noqa: E402
    ARMS,
    PAIRS,
    _format_command_token,
    _load_strata,
    build_spec,
)
from experiments.analyze_evaluation import (  # noqa: E402
    _validate_result,
    _workload_digest,
)
from experiments.run_evaluation import validate_spec  # noqa: E402
from experiments.hardware import collect, validate  # noqa: E402


def _fake_command() -> str:
    payload = "print('evaluation-fixture')"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(payload)}"


def main() -> None:
    assert (
        _format_command_token(
            'python -c \'{"arm": "{arm}"}\'',
            arm="B5",
            values={"tier": "host_mem"},
        )
        == 'python -c \'{"arm": "B5"}\''
    )
    with tempfile.TemporaryDirectory(prefix="nta-evaluation-contract-") as directory:
        root = Path(directory)
        workload_manifest, rows = normalize(
            [
                {
                    "request_id": "a",
                    "input_length": 32,
                    "output_length": 4,
                    "hash_ids": ["p0", "p1"],
                },
                {
                    "request_id": "b",
                    "input_length": 48,
                    "output_length": 5,
                    "hash_ids": ["p0", "p1", "p2"],
                },
            ],
            arrival_mode="batch_release",
            synthesize_prompts=True,
        )
        manifest_path = root / "manifest.json"
        write_workload(
            root / "manifest.json", root / "records.jsonl", workload_manifest, rows
        )
        strata = _load_strata(ROOT / "experiments" / "strata.example.json")
        spec = build_spec(
            workload_manifest=manifest_path,
            tier="host_mem",
            arm_commands={arm: _fake_command() for arm in ARMS},
            consumer_kinds={
                arm: ("framework_reference" if arm == "B0" else "native_work_unit")
                for arm in ARMS
            },
            strata=strata,
            repetitions=5,
        )
        validated_manifest = validate_spec(spec, manifest_path)
        assert (
            validated_manifest["demand_trace_digest"]
            == workload_manifest["demand_trace_digest"]
        )
        assert (
            _workload_digest(
                {"result": {}},
                {
                    "workload_demand_digest": "exact-demand",
                    "workload_manifest_digest": "file",
                },
            )
            == "exact-demand"
        )
        assert (
            _workload_digest({"result": {}}, {"workload_manifest_digest": "file"})
            is None
        )
        assert len(spec["experiments"]) == len(PAIRS) * len(strata) * 2
        assert len(spec["comparisons"]) == len(PAIRS) * len(strata)
        assert spec["evaluation_profile"] == "osdi-complete"
        assert {trial["arm"] for trial in spec["experiments"]} == set(ARMS)
        assert all(
            trial["demand_semantics"] == "exact" for trial in spec["experiments"]
        )
        assert all(
            set(trial["stratum"])
            == {
                "request_state",
                "granularity",
                "load_ratio",
                "availability_skew",
                "staging_pressure",
                "arrival",
            }
            for trial in spec["experiments"]
        )
        try:
            _validate_result(
                {
                    "experiment": "missing-evidence",
                    "variant": "B5",
                    "result": {
                        "classification": "fixture",
                        "verification_failures": 0,
                    },
                },
                required_consumer_kind="native_work_unit",
            )
        except ValueError as error:
            assert "consumer contract" in str(error)
        else:
            raise AssertionError(
                "formal trial without a consumer contract was accepted"
            )
        native_contract = {
            "schema": 1,
            "engine": "fixture",
            "backend": "nta_flashinfer",
            "kind": "native_work_unit",
            "exact_demand": True,
            "typed_work_plan": True,
            "native_submission": True,
            "numerical_consumer": True,
            "engine_version": "fixture",
        }
        assert (
            _validate_result(
                {
                    "experiment": "with-evidence",
                    "variant": "B5",
                    "result": {
                        "classification": "fixture",
                        "verification_failures": 0,
                        "consumer_contract": native_contract,
                    },
                },
                required_consumer_kind="native_work_unit",
            )[0]["kind"]
            == "native_work_unit"
        )
        cli_output = root / "cli-spec.json"
        command = [
            sys.executable,
            str(ROOT / "experiments" / "make_evaluation_spec.py"),
            "--workload-manifest",
            str(manifest_path),
            "--strata-file",
            str(ROOT / "experiments" / "strata.example.json"),
            "--tier",
            "host_mem",
            "--repetitions",
            "5",
            "--output",
            str(cli_output),
        ]
        for arm in ARMS:
            command.extend(("--arm-command", f"{arm}={_fake_command()}"))
            command.extend(
                (
                    "--arm-consumer-kind",
                    f"{arm}={'framework_reference' if arm == 'B0' else 'native_work_unit'}",
                )
            )
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        cli_spec = json.loads(cli_output.read_text(encoding="utf-8"))
        assert len(cli_spec["comparisons"]) == len(PAIRS) * len(strata)
        assert cli_spec["evaluation_profile"] == "osdi-complete"
        assert cli_spec["experiments"][0]["consumer_kind"] in {
            "native_work_unit",
            "framework_reference",
        }

        incomplete = json.loads(json.dumps(spec))
        incomplete["experiments"] = [
            trial for trial in incomplete["experiments"] if trial["arm"] != "B6"
        ]
        try:
            validate_spec(incomplete, manifest_path)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete osdi evaluation was accepted")

        missing_consumer_kind = json.loads(json.dumps(spec))
        missing_consumer_kind["experiments"][0].pop("consumer_kind")
        try:
            validate_spec(missing_consumer_kind, manifest_path)
        except ValueError as error:
            assert "consumer_kind" in str(error)
        else:
            raise AssertionError(
                "formal evaluation without consumer evidence was accepted"
            )

        sysfs = root / "sysfs"
        dev = root / "dev"
        pci = sysfs / "0000:01:00.0"
        (pci / "nvme" / "nvme0").mkdir(parents=True)
        (pci / "class").write_text("0x010802\n", encoding="utf-8")
        (pci / "nvme" / "nvme0" / "nvme0n1").mkdir()
        (dev / "nvme0n1").parent.mkdir(parents=True)
        (dev / "nvme0n1").touch()
        drivers = sysfs / "drivers"
        (drivers / "nvme").mkdir(parents=True)
        (pci / "driver").symlink_to(drivers / "nvme", target_is_directory=True)
        iommu = sysfs / "iommu" / "7"
        iommu.mkdir(parents=True)
        (pci / "iommu_group").symlink_to(iommu, target_is_directory=True)
        cxl = root / "cxl"
        (cxl / "root0").mkdir(parents=True)
        decoder = cxl / "decoder0.0"
        decoder.mkdir()
        (decoder / "size").write_text("0x100000000\n", encoding="utf-8")
        (decoder / "target_list").write_text("3\n", encoding="utf-8")
        inventory = collect(sysfs_root=sysfs, dev_root=dev, cxl_sysfs_root=cxl)
        validate(inventory)
        assert inventory["nvme"]["controllers"][0]["driver"] == "nvme"
        assert inventory["nvme"]["controllers"][0]["namespaces"] == ["nvme0n1"]
        assert inventory["cxl"]["status"] == "root_decoder_only"
        assert inventory["cxl"]["decoders"][0]["target_list"] == "3"
        assert inventory["safety"]["qualification_performed"] is False
    print("evaluation_spec=pass")


if __name__ == "__main__":
    main()
