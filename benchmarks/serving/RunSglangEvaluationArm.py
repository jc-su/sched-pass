#!/usr/bin/env python3
"""Run one canonical SGLang causal arm with exclusive-GPU evidence.

Arguments after ``--`` are forwarded to ``SglangHiCacheLoad.py``.  The
wrapper owns the arm-to-execution-form mapping, JIT activation, raw log, GPU
co-tenant sampler, and result-derived activation proof.  It emits one serving
report, never a nested stock/NTA comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.serving.gpu_trial import (  # noqa: E402
    CotenantSampler,
    TRIAL_OWNER_ENV,
    wait_for_free_gpu,
)
from experiments.atomic_io import atomic_write_json, atomic_write_text  # noqa: E402
from experiments.mechanism_arms import (  # noqa: E402
    ARMS,
    arm_backend,
    arm_environment,
    validate_arm_result,
)


_OWNED_EXECUTION_ENV = {
    "NTA_EXECUTION_PROTOCOL",
    "NTA_EXECUTION_HOST_FORM",
    "NTA_EXECUTION_CALIBRATION_PROBES",
    "NTA_EXECUTION_MAX_ROUNDS",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=600.0)
    parser.add_argument("load_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    load_args = list(args.load_args)
    if load_args and load_args[0] == "--":
        load_args.pop(0)
    if not load_args:
        parser.error("arguments for SglangHiCacheLoad.py are required after --")
    forbidden = {"--attention-backend", "--flashinfer-workspace-base", "--output"}
    conflicts = sorted(
        {
            token.split("=", 1)[0]
            for token in load_args
            if token.split("=", 1)[0] in forbidden
        }
    )
    if conflicts:
        parser.error("wrapper-owned load arguments were supplied: " + ", ".join(conflicts))
    if args.gpu_wait_timeout_seconds <= 0:
        parser.error("GPU wait timeout must be positive")
    return args, load_args


def _report(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("classification") == "sglang-hicache-load":
            return value
    raise RuntimeError("SGLang arm emitted no structured load report")


def main() -> int:
    args, load_args = _parse_args()
    output = args.output.resolve()
    workspace = args.workspace_root.resolve() / args.arm.lower()
    raw_report = output.with_suffix(output.suffix + ".raw.json")
    command = [
        str(ROOT / "tools" / "jit" / "activate.py"),
        "--build-dir",
        args.build_dir,
        "--cache-root",
        str(workspace),
        "--flashinfer-hook",
        "--",
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "SglangHiCacheLoad.py"),
        "--attention-backend",
        arm_backend(args.arm),
        "--flashinfer-workspace-base",
        str(workspace / "flashinfer"),
        "--output",
        str(raw_report),
        *load_args,
    ]
    environment = os.environ.copy()
    for name in _OWNED_EXECUTION_ENV:
        environment.pop(name, None)
    environment.update(arm_environment(args.arm))
    environment["NTA_EVALUATION_ARM"] = args.arm
    owner_token = f"{os.getpid()}:{time.monotonic_ns()}:{args.arm}"
    environment[TRIAL_OWNER_ENV] = owner_token

    wait_for_free_gpu(timeout_seconds=args.gpu_wait_timeout_seconds)
    with CotenantSampler(owner_token) as sampler:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_path = output.with_suffix(output.suffix + ".stdout.log")
    atomic_write_text(log_path, completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"SGLang {args.arm} failed with status {completed.returncode}; "
            f"see {log_path}"
        )
    if not sampler.complete:
        raise RuntimeError("GPU co-tenant sampler did not terminate")
    report = _report(completed.stdout)
    report.update(
        {
            "gpu_samples": sampler.samples,
            "gpu_sampling_errors": sampler.sampling_errors,
            "gpu_sampling_complete": sampler.complete,
            "cotenant_gpu_samples": sampler.foreign_samples,
            "cotenant_pids_seen": sorted(sampler.foreign_pids),
            "evaluation_arm": args.arm,
        }
    )
    report["evaluation_arm_activation"] = validate_arm_result(report, args.arm)
    atomic_write_json(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
