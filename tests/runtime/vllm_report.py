#!/usr/bin/env python3
"""Test the vLLM integration-report and artifact-dispatch contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from experiments.validate_serving_result import validate as validate_serving_result  # noqa: E402
from experiments.validate_vllm_report import validate  # noqa: E402
from nta_runtime.adapters.base import ConsumerContract  # noqa: E402
from nta_runtime.engines.vllm import (  # noqa: E402
    VLLM_STATS,
    consumer_contract as live_consumer_contract,
)


def _contract(kind: str) -> dict[str, object]:
    if kind == "native_work_unit":
        return ConsumerContract.native_work_unit(
            engine="vllm",
            backend="nta_flashinfer",
            engine_version="0.26.0",
        ).as_dict()
    return ConsumerContract.framework_reference(
        engine="vllm",
        backend="flashinfer",
        engine_version="0.26.0",
    ).as_dict()


def report(backend: str) -> dict[str, object]:
    native = backend == "nta"
    evidence = (
        [
            {
                "engine": "vllm",
                "backend": "nta_flashinfer",
                "native_enabled": True,
                "stock_fallback_enabled": False,
                "consumer_contract": _contract("native_work_unit"),
                "stats": {
                    "native_decode_launches": 2,
                    "native_prefill_launches": 1,
                    "reference_fallback_launches": 0,
                },
            }
        ]
        if native
        else []
    )
    return {
        "schema": 1,
        "classification": "vllm-serving-integration-smoke",
        "backend": backend,
        "backend_selected": True,
        "native_execution_verified": native,
        "native_launches": 3 if native else 0,
        "reference_fallback_launches": 0,
        "stock_fallback_enabled": False,
        "engine": "vllm",
        "engine_version": "0.26.0",
        "flashinfer_version": "0.6.14",
        "torch_version": "2.11.0+cu130",
        "cuda_version": "13.0",
        "model": "/models/test",
        "revision": "test-revision",
        "dirty": False,
        "machine": {"hostname": "test", "kernel": "test-kernel"},
        "requests": 2,
        "max_new_tokens": 2,
        "iterations": 3,
        "warmup_iterations": 1,
        "generated_tokens": 4,
        "generated_text_sha256": "a" * 64,
        "generated_token_ids_sha256": "b" * 64,
        "load_seconds": 1.0,
        "batch_seconds_samples": [0.2, 0.4, 0.3],
        "median_batch_seconds": 0.3,
        "requests_per_second": 2 / 0.3,
        "generated_tokens_per_second": 4 / 0.3,
        "evidence": evidence,
        "consumer_contract": _contract(
            "native_work_unit" if native else "framework_reference"
        ),
        "flashinfer_workspace_base": "/tmp/nta-vllm-test",
    }


def main() -> None:
    saved_stats = dict(VLLM_STATS)
    try:
        VLLM_STATS["native_decode_launches"] = 1
        VLLM_STATS["reference_fallback_launches"] = 1
        assert live_consumer_contract()["kind"] == "framework_reference"
    finally:
        VLLM_STATS.clear()
        VLLM_STATS.update(saved_stats)
    stock = report("stock")
    nta = report("nta")
    validate(stock)
    validate(nta)
    assert validate_serving_result(nta)["classification"] == (
        "vllm-serving-integration-smoke"
    )
    try:
        validate_serving_result({"classification": "unknown-serving-report"})
    except ValueError as error:
        assert "unsupported serving result classification" in str(error)
    else:
        raise AssertionError("unknown serving result classification was dispatched")

    invalid = copy.deepcopy(nta)
    invalid["stock_fallback_enabled"] = True
    try:
        validate(invalid)
    except ValueError as error:
        assert "fallback" in str(error)
    else:
        raise AssertionError("vLLM report with fallback enabled was accepted")

    invalid = copy.deepcopy(nta)
    invalid["native_launches"] = 2
    try:
        validate(invalid)
    except ValueError as error:
        assert "accounting" in str(error)
    else:
        raise AssertionError("inconsistent native launch accounting was accepted")

    with tempfile.TemporaryDirectory(prefix="nta-vllm-artifact-") as directory:
        root = Path(directory)
        result = root / "vllm.json"
        result.write_text(json.dumps(nta) + "\n", encoding="utf-8")
        bundle = root / "bundle"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "reproduce.py"),
                "--profile",
                "serving",
                "--output",
                str(bundle),
                "--result",
                str(result),
                "--allow-dirty",
                "--",
                sys.executable,
                "-c",
                "print('vllm-fixture')",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "validate_bundle.py"),
                str(bundle),
            ],
            cwd=ROOT,
            check=True,
        )
    print("vllm_report=pass")


if __name__ == "__main__":
    main()
