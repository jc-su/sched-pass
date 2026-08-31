#!/usr/bin/env python3
"""Run and validate a paired stock/NTA vLLM serving integration gate.

This gate proves native worker execution and compares every native attention
result with stock FlashInfer on the same tensors.  A clean stock process also
checks the first generated batch.  Timing is diagnostic only: each arm builds
an independent offline engine process.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument(
        "--nta-iterations",
        type=int,
        default=3,
        help="long-lived native batches checked in-process against stock",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="set both arms to the same vLLM scheduling mode",
    )
    parser.add_argument("--cache-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.max_num_seqs < args.requests:
        parser.error("--max-num-seqs must cover --requests")
    if args.nta_iterations <= 0:
        parser.error("--nta-iterations must be positive")
    return args


def clean_child_environment() -> dict[str, str]:
    """Return an environment that cannot leak an activated NTA arm to stock."""

    environment = dict(os.environ)
    for name in (
        "FLASHINFER_NVCC",
        "FLASHINFER_WORKSPACE_BASE",
        "NTA_ENGINE_STATS_FILE",
        "NTA_FLASHINFER_HOOK",
        "NTA_FLASHINFER_OVERLAY",
        "NTA_PLUGIN",
        "NTA_RUNTIME_LIBRARY",
        "NTA_TRANSPORT_PROGRAM",
        "NTA_VLLM_ALLOW_STOCK_FALLBACK",
        "NTA_VLLM_COMPARE_STOCK",
        "NTA_VLLM_EVIDENCE_TOKEN",
        "NTA_VLLM_NATIVE",
        "NTA_VLLM_VERIFY_TRANSFER",
        "NTA_VERIFY_EXECUTION",
        "NTA_SERVING_TIER",
    ):
        environment.pop(name, None)
    python_paths = [
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and "nta-flashinfer-overlay" not in entry
    ]
    if python_paths:
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    else:
        environment.pop("PYTHONPATH", None)
    return environment


def run_arm(
    args: argparse.Namespace,
    backend: str,
    *,
    iterations: int,
    compare_stock: bool = False,
) -> dict[str, Any]:
    arm_root = args.cache_root.resolve() / backend
    arm_root.mkdir(parents=True, exist_ok=True)
    report_path = arm_root / "report.json"
    command = [
        sys.executable,
        str(ROOT / "benchmarks" / "serving" / "VllmSmoke.py"),
        "--model",
        str(args.model.resolve()),
        "--backend",
        backend,
        "--serving-tier",
        "hbm",
        "--requests",
        str(args.requests),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--iterations",
        str(iterations),
        "--warmup-iterations",
        "0",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--flashinfer-workspace-base",
        str(arm_root / "flashinfer"),
        "--output",
        str(report_path),
    ]
    if compare_stock:
        command.append("--compare-stock")
    command.append("--prefix-caching" if args.prefix_caching else "--no-prefix-caching")
    if args.async_scheduling is not None:
        command.append(
            "--async-scheduling"
            if args.async_scheduling
            else "--no-async-scheduling"
        )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=clean_child_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"vLLM {backend} arm failed:\n{tail}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"vLLM {backend} arm emitted no valid report") from error
    if not isinstance(report, dict):
        raise RuntimeError(f"vLLM {backend} report is not an object")
    return report


def require_contract(report: dict[str, Any], *, native: bool) -> None:
    contract = report.get("consumer_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("vLLM report has no consumer contract")
    expected_kind = "native_work_unit" if native else "framework_reference"
    if contract.get("kind") != expected_kind:
        raise RuntimeError(
            f"vLLM contract is {contract.get('kind')!r}, expected {expected_kind!r}"
        )
    if native:
        required = (
            "exact_demand",
            "typed_work_plan",
            "native_submission",
            "numerical_consumer",
        )
        missing = [name for name in required if contract.get(name) is not True]
        if missing:
            raise RuntimeError(
                "vLLM native contract is incomplete: " + ", ".join(missing)
            )
        if not report.get("native_execution_verified"):
            raise RuntimeError("vLLM NTA arm has no worker-native evidence")
        if int(report.get("native_launches", 0)) <= 0:
            raise RuntimeError("vLLM NTA arm executed no native launches")
        if int(report.get("reference_fallback_launches", 0)) != 0:
            raise RuntimeError("vLLM NTA arm used reference fallback")
        if report.get("stock_fallback_enabled"):
            raise RuntimeError("vLLM NTA arm enabled stock fallback")
        if report.get("differential_verification") is not True:
            raise RuntimeError("vLLM NTA arm skipped in-process stock comparison")
        if int(report.get("native_stock_comparisons", 0)) <= 0:
            raise RuntimeError("vLLM NTA arm emitted no differential comparisons")


def require_same_workload(stock: dict[str, Any], nta: dict[str, Any]) -> None:
    identity_fields = (
        "model",
        "engine_version",
        "flashinfer_version",
        "torch_version",
        "cuda_version",
        "async_scheduling",
        "prefix_caching",
        "requests",
        "max_new_tokens",
        "max_model_len",
        "max_num_seqs",
        "prompt_profile",
        "long_prefix_repetitions",
    )
    mismatches = [name for name in identity_fields if stock.get(name) != nta.get(name)]
    nta_batches = nta.get("measured_generated_token_id_batches")
    first_nta = nta_batches[0] if isinstance(nta_batches, list) and nta_batches else None
    stock_tokens = stock.get("generated_token_id_sequences")
    if mismatches or stock_tokens != first_nta:
        raise RuntimeError(
            "paired vLLM arms differ in workload or output: "
            + ", ".join(mismatches or ("first_batch_tokens",))
            + f"; token_sequences={{'stock': {stock_tokens!r}, "
            + f"'nta_first': {first_nta!r}}}"
        )


def main() -> int:
    args = parse_args()
    stock = run_arm(args, "stock", iterations=1)
    nta = run_arm(
        args,
        "nta",
        iterations=args.nta_iterations,
        compare_stock=True,
    )
    require_contract(stock, native=False)
    require_contract(nta, native=True)
    require_same_workload(stock, nta)
    report = {
        "schema": 1,
        "classification": "vllm-serving-integration-pair",
        "performance_claim": False,
        "numerically_equivalent": True,
        "native_execution_verified": True,
        "first_batch_generated_token_ids": stock["generated_token_id_sequences"],
        "native_stock_comparisons": nta["native_stock_comparisons"],
        "native_stock_diff_max_e9": nta["native_stock_diff_max_e9"],
        "configuration": {
            "model": nta["model"],
            "requests": nta["requests"],
            "max_new_tokens": nta["max_new_tokens"],
            "max_model_len": nta["max_model_len"],
            "max_num_seqs": nta["max_num_seqs"],
            "gpu_memory_utilization": nta["gpu_memory_utilization"],
            "async_scheduling": nta["async_scheduling"],
            "prefix_caching": nta["prefix_caching"],
            "nta_iterations": args.nta_iterations,
        },
        "stock": stock,
        "nta": nta,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True)
    destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
