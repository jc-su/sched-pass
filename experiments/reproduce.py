#!/usr/bin/env python3
"""Produce a self-contained, provenance-recorded NTA artifact bundle.

Examples from a clean checkout::

    python experiments/reproduce.py --profile core --output /tmp/nta-core
    python experiments/reproduce.py --profile matrix --output /tmp/nta-matrix
    python experiments/reproduce.py --profile test --output /tmp/nta-test \
        --build-dir /tmp/nta-build
    python experiments/reproduce.py --profile evaluation \
        --spec /path/to/paired-evaluation.json \
        --output /tmp/nta-evaluation --allow-dirty
    python experiments/reproduce.py --profile serving --output /tmp/nta-serving \
        -- python benchmarks/serving/CompareSglangHiCacheLoad.py ...

    The serving profile deliberately accepts a command instead of guessing a
model, framework version, or dataset.  The exact command and raw output are
part of the artifact bundle.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

try:
    from .artifact import ArtifactRun, ROOT, file_digest, git_metadata
    from .hardware import validate as validate_hardware
    from .validate_workload import validate as validate_workload
    from .validate_tier_qualification import validate_file as validate_tier_qualification
    from .validate_tier_catalog import validate as validate_tier_catalog
except ImportError:  # Direct ``python experiments/reproduce.py`` execution.
    from artifact import ArtifactRun, ROOT, file_digest, git_metadata
    from hardware import validate as validate_hardware
    from validate_workload import validate as validate_workload
    from validate_tier_qualification import validate_file as validate_tier_qualification
    from validate_tier_catalog import validate as validate_tier_catalog


MANIFEST = ROOT / "experiments" / "artifact-manifest.json"
MATRIX_MANIFEST = "experiments/heterogeneous-work-unit.json"
EVALUATION_MANIFEST = "experiments/evaluation-manifest.json"


def _cmake_cache(build: Path) -> dict[str, str]:
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or ":" not in line:
            continue
        key, encoded = line.split("=", 1)
        name, _, _kind = key.partition(":")
        values[name] = encoded
    return values


def _tool_version(path: str | None) -> str | None:
    if not path:
        return None
    try:
        completed = subprocess.run(
            [path, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _require_clean(allow_dirty: bool) -> dict[str, object]:
    metadata = git_metadata()
    if metadata["dirty"] and not allow_dirty:
        raise RuntimeError(
            "artifact reproduction requires a clean checkout; use "
            "--allow-dirty only for local debugging"
        )
    return metadata


def _configure_and_build(
    run: ArtifactRun,
    args: argparse.Namespace,
    *,
    build: Path,
    environment: dict[str, str],
) -> None:
    try:
        build.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"artifact build directory must be outside the source tree: {build}")
    if build.exists() and any(build.iterdir()):
        raise RuntimeError(
            f"artifact build directory is not empty; use a fresh directory: {build}"
        )
    cuda = args.cuda
    if cuda == "auto":
        cuda = "on" if shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists() else "off"
    configure = [
        "cmake",
        "-S",
        str(ROOT),
        "-B",
        str(build),
        "-G",
        args.generator,
        f"-DNTA_ENABLE_CUDA={'ON' if cuda == 'on' else 'OFF'}",
    ]
    configure.extend(args.cmake_arg)
    run.command(configure, name="cmake-configure", environment=environment)
    run.command(
        ["cmake", "--build", str(build), "--parallel", str(args.jobs)],
        name="cmake-build",
        environment=environment,
    )
    cache = _cmake_cache(build)
    llvm_tools = cache.get("LLVM_TOOLS_BINARY_DIR")
    if not llvm_tools and cache.get("NTA_CLANG_CUDA"):
        llvm_tools = str(Path(cache["NTA_CLANG_CUDA"]).parent)
    selected_tools = {
        name: str(Path(llvm_tools) / name)
        for name in ("clang++", "opt", "llc", "llvm-dis")
    } if llvm_tools else {}
    selected_tools.update(
        {
            "cuda_frontend": cache.get("NTA_CLANG_CUDA", ""),
            "ptxas": cache.get("NTA_PTXAS", ""),
        }
    )
    run.update(
        build_toolchain={
            "llvm_dir": cache.get("LLVM_DIR"),
            "llvm_tools_binary_dir": llvm_tools,
            "tools": {
                name: {
                    "path": path,
                    "version": _tool_version(path),
                }
                for name, path in selected_tools.items()
                if path
            },
            "cuda_frontend_opt_level": cache.get("NTA_CUDA_FRONTEND_OPT_LEVEL"),
            "llc_opt_level": cache.get("NTA_CUDA_LLC_OPT_LEVEL"),
        }
    )


def _run_matrix(
    run: ArtifactRun,
    args: argparse.Namespace,
    *,
    full: bool,
    environment: dict[str, str],
) -> None:
    output = run.output / "matrix.json"
    command = [
        sys.executable,
        "experiments/run_work_unit_matrix.py",
        "--manifest",
        MATRIX_MANIFEST,
        "--output",
        str(output),
        "--ablation",
        "all",
        "--max-cases",
        str(args.max_cases if full else min(args.max_cases, 2)),
    ]
    if args.repetitions is not None:
        command.extend(("--repetitions", str(args.repetitions)))
    run.command(
        command,
        name="matrix-run",
        environment=environment,
    )
    run.command(
        [
            sys.executable,
            "experiments/validate_matrix_artifact.py",
            str(output),
            "--require-all-ablations",
        ],
        name="matrix-validate",
        environment=environment,
    )


def _replace_path(value: object, source: Path, destination: Path) -> object:
    if isinstance(value, str):
        return value.replace(str(source), str(destination))
    if isinstance(value, list):
        return [_replace_path(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_path(item, source, destination)
            for key, item in value.items()
        }
    return value


def _copy_workload(run: ArtifactRun, source: Path) -> Path:
    manifest = validate_workload(source)
    source_records = source.parent / str(manifest["records_file"])
    destination = run.output / "workload"
    destination.mkdir()
    shutil.copy2(source_records, destination / "records.jsonl")
    copied_manifest = dict(manifest)
    copied_manifest["records_file"] = "records.jsonl"
    copied_manifest_path = destination / "manifest.json"
    copied_manifest_path.write_text(
        json.dumps(copied_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_workload(copied_manifest_path)
    run.update(
        workload_replay_manifest="workload/manifest.json",
        workload_replay_manifest_digest=file_digest(copied_manifest_path),
        workload_replay_records_digest=file_digest(destination / "records.jsonl"),
    )
    return copied_manifest_path


def _run_evaluation(
    run: ArtifactRun,
    args: argparse.Namespace,
    *,
    environment: dict[str, str],
) -> None:
    if args.spec is None:
        raise RuntimeError("--profile evaluation requires --spec")
    spec = args.spec.resolve()
    if not spec.is_file():
        raise RuntimeError(f"evaluation specification does not exist: {spec}")
    destination = run.output / "evaluation"
    try:
        spec_document = json.loads(spec.read_text(encoding="utf-8"))
        workload = Path(spec_document["workload_manifest"]).resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"evaluation specification has no readable workload manifest: {error}") from error
    copied_workload = _copy_workload(run, workload)
    rewritten_spec = _replace_path(spec_document, workload, copied_workload)
    qualification = spec_document.get("tier_qualification")
    copied_qualification: Path | None = None
    if qualification is not None:
        qualification_path = Path(str(qualification)).resolve()
        required_tiers = {
            str(trial["tier"])
            for trial in spec_document.get("experiments", [])
            if trial.get("tier") in {"nvme", "dax"}
        }
        validate_tier_qualification(
            qualification_path,
            required_tiers=required_tiers or {"hbm", "host_mem"},
        )
        copied_qualification = run.output / "tier-qualification.json"
        shutil.copy2(qualification_path, copied_qualification)
        rewritten_spec = _replace_path(
            rewritten_spec, qualification_path, copied_qualification
        )
        run.update(
            tier_qualification_manifest="tier-qualification.json",
            tier_qualification_manifest_digest=file_digest(copied_qualification),
        )
    spec_copy = run.output / "evaluation-spec.json"
    spec_copy.write_text(
        json.dumps(rewritten_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rq0 = run.output / "rq0-opportunity.json"
    run.command(
        [
            sys.executable,
            "experiments/analyze_workload.py",
            str(copied_workload),
            "--output",
            str(rq0),
        ],
        name="workload-rq0",
        environment=environment,
    )
    command = [
        sys.executable,
        "experiments/run_evaluation.py",
        "--spec",
        str(spec_copy),
        "--output-dir",
        str(destination),
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    run.command(command, name="evaluation-run", environment=environment)
    run.command(
        [
            sys.executable,
            "experiments/validate_evaluation_artifact.py",
            str(destination),
        ],
        name="evaluation-validate",
        environment=environment,
    )
    run.update(
        evaluation_spec="evaluation-spec.json",
        evaluation_spec_digest=file_digest(spec_copy),
        rq0_opportunity="rq0-opportunity.json",
        rq0_opportunity_digest=file_digest(rq0),
        evaluation_output="evaluation",
    )


def _run_hardware(run: ArtifactRun, *, environment: dict[str, str]) -> None:
    output = run.output / "hardware-inventory.json"
    run.command(
        [
            sys.executable,
            "experiments/inspect_hardware.py",
            "--output",
            str(output),
        ],
        name="hardware-inventory",
        environment=environment,
    )
    validate_hardware(json.loads(output.read_text(encoding="utf-8")))
    run.update(
        hardware_inventory="hardware-inventory.json",
        hardware_inventory_digest=file_digest(output),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("core", "matrix", "build", "test", "evaluation", "hardware", "serving"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="external CMake build directory (default: next to the artifact)",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--generator", default="Ninja")
    parser.add_argument("--cuda", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="explicit environment override recorded with the command",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="serving result file to copy into the artifact bundle",
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        help="normalized workload manifest to copy into a serving artifact",
    )
    parser.add_argument(
        "--tier-catalog",
        type=Path,
        help="exact physical-tier catalog to copy into a serving artifact",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="paired evaluation specification for --profile evaluation",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.max_cases <= 0 or args.jobs <= 0:
        parser.error("--max-cases and --jobs must be positive")
    if args.repetitions is not None and args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.profile == "serving" and not args.command:
        parser.error("--profile serving requires a command after --")
    if args.profile == "serving" and args.result is None:
        parser.error("--profile serving requires --result for a structured report")
    if args.profile != "serving" and args.command:
        parser.error("a command is only valid after --profile serving")
    if args.workload_manifest is not None and args.profile != "serving":
        parser.error("--workload-manifest is only valid for --profile serving")
    if args.tier_catalog is not None and args.profile != "serving":
        parser.error("--tier-catalog is only valid for --profile serving")
    if args.spec is not None and args.profile != "evaluation":
        parser.error("--spec is only valid for --profile evaluation")
    if args.result is not None and args.profile != "serving":
        parser.error("--result is only valid for --profile serving")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    environment: dict[str, str] = {}
    for item in args.env:
        if "=" not in item or not item.split("=", 1)[0]:
            parser.error(f"--env must use KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        environment[key] = value
    args.environment = environment
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repository = _require_clean(args.allow_dirty)
    except RuntimeError as error:
        print(f"artifact reproduction refused: {error}", file=sys.stderr)
        return 2
    try:
        run = ArtifactRun(args.output, profile=args.profile, arguments=sys.argv[1:])
    except (OSError, ValueError) as error:
        print(f"artifact reproduction refused: {error}", file=sys.stderr)
        return 2
    workload_manifest = ROOT / MATRIX_MANIFEST
    (run.output / "artifact-manifest.json").write_bytes(MANIFEST.read_bytes())
    (run.output / "workload-manifest.json").write_bytes(workload_manifest.read_bytes())
    evaluation_manifest = ROOT / EVALUATION_MANIFEST
    (run.output / "evaluation-manifest.json").write_bytes(evaluation_manifest.read_bytes())
    run.update(
        artifact_manifest=str(MANIFEST.relative_to(ROOT)),
        artifact_manifest_digest=file_digest(MANIFEST),
        workload_manifest=MATRIX_MANIFEST,
        workload_manifest_digest=file_digest(ROOT / MATRIX_MANIFEST),
        evaluation_manifest=EVALUATION_MANIFEST,
        evaluation_manifest_digest=file_digest(evaluation_manifest),
        repository_at_start=repository,
    )
    environment = {"PYTHONPATH": str(ROOT / "python"), **args.environment}
    try:
        if args.profile == "serving":
            environment.setdefault(
                "NTA_SERVING_WORKSPACE_ROOT",
                str(run.output / "serving-workspace"),
            )
            if args.tier_catalog is not None:
                source_catalog = args.tier_catalog.resolve()
                if not source_catalog.is_file():
                    raise RuntimeError(f"tier catalog does not exist: {source_catalog}")
                selected_tier = environment.get(
                    "NTA_SERVING_TIER",
                    os.environ.get("NTA_SERVING_TIER", "host_staged"),
                )
                selected_tier = {"host": "host_staged", "cxl": "cxl_dax"}.get(
                    selected_tier, selected_tier
                )
                if selected_tier not in {"nvme", "cxl_dax"}:
                    raise RuntimeError(
                        "--tier-catalog requires NTA_SERVING_TIER=nvme or cxl_dax"
                    )
                validate_tier_catalog(source_catalog, selected_tier)
                catalog_destination = run.output / "tier" / "catalog.json"
                catalog_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_catalog, catalog_destination)
                environment["NTA_TIER_CATALOG"] = str(catalog_destination)
                run.update(
                    tier_catalog="tier/catalog.json",
                    tier_catalog_digest=file_digest(catalog_destination),
                    serving_tier=selected_tier,
                )
    except (OSError, ValueError, RuntimeError) as error:
        run.finish(status="failed", error=str(error))
        print(f"artifact reproduction failed: {error}", file=sys.stderr)
        return 2
    try:
        if args.profile == "core":
            _run_matrix(run, args, full=False, environment=environment)
        elif args.profile == "matrix":
            _run_matrix(run, args, full=True, environment=environment)
        elif args.profile in ("build", "test"):
            build = (
                args.build_dir.resolve()
                if args.build_dir is not None
                else run.output.parent / f"{run.output.name}.build"
            )
            _configure_and_build(
                run, args, build=build, environment=environment
            )
            if args.profile == "test":
                run.command(
                    ["ctest", "--test-dir", str(build), "--output-on-failure"],
                    name="ctest",
                    environment=environment,
                )
        elif args.profile == "evaluation":
            _run_evaluation(run, args, environment=environment)
        elif args.profile == "hardware":
            _run_hardware(run, environment=environment)
        else:
            run.command(args.command, name="serving", environment=environment)
            if args.workload_manifest is not None:
                source_manifest = args.workload_manifest.resolve()
                if not source_manifest.is_file():
                    raise RuntimeError(f"workload manifest does not exist: {source_manifest}")
                validate_workload(source_manifest)
                source_records = source_manifest.parent / json.loads(
                    source_manifest.read_text(encoding="utf-8")
                )["records_file"]
                destination = run.output / "workload"
                destination.mkdir()
                shutil.copy2(source_manifest, destination / "manifest.json")
                shutil.copy2(source_records, destination / "records.jsonl")
                run.update(
                    workload_replay_manifest="workload/manifest.json",
                    workload_replay_manifest_digest=file_digest(destination / "manifest.json"),
                    workload_replay_records_digest=file_digest(destination / "records.jsonl"),
                )
            if args.result is not None:
                result = args.result.resolve()
                if not result.is_file():
                    raise RuntimeError(f"serving result file does not exist: {result}")
                destination = run.output / result.name
                if result != destination.resolve():
                    shutil.copy2(result, destination)
                run.update(
                    result=str(destination.name),
                    result_digest=file_digest(destination),
                )
                run.command(
                    [
                        sys.executable,
                        "experiments/validate_serving_report.py",
                        str(destination),
                    ],
                    name="serving-validate",
                    environment=environment,
                )
        finish_fields = {}
        if args.profile in ("build", "test"):
            finish_fields["build_dir"] = str(build)
        run.finish(status="complete", **finish_fields)
    except Exception as error:
        run.finish(status="failed", error=str(error))
        print(f"artifact reproduction failed: {error}", file=sys.stderr)
        print(f"partial artifact: {run.output}", file=sys.stderr)
        return 1
    print(run.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
