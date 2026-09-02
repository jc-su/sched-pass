#!/usr/bin/env python3
"""Regression tests for the paper-level campaign gate."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.validate_osdi_campaign import validate  # noqa: E402


METRICS = [
    "ttft_p50_p95_p99",
    "tpot_p50_p95_p99",
    "itl_p99",
    "slo_goodput",
    "request_throughput",
    "output_token_throughput",
]


def _campaign() -> dict[str, object]:
    systems = [
        {"id": "nta-full"},
        {"id": "sglang-hicache-kernel"},
        {"id": "vllm-lmcache"},
        {"id": "gpu-only-reference"},
    ]
    models = [
        {
            "id": "qwen-long",
            "family": "qwen",
            "context_window_tokens": 131072,
            "long_context": True,
        },
        {
            "id": "llama-long",
            "family": "llama",
            "context_window_tokens": 131072,
            "long_context": True,
        },
        {
            "id": "qwen-small",
            "family": "qwen",
            "context_window_tokens": 32768,
            "long_context": False,
        },
    ]
    workloads = [
        {
            "id": "bailian-chat",
            "family": "chat",
            "provenance": "natural_trace",
            "statistical_independence": "source_request_identity",
        },
        {
            "id": "bailian-coder",
            "family": "code",
            "provenance": "natural_trace",
            "statistical_independence": "source_request_identity",
        },
        {
            "id": "long-document",
            "family": "long_document_qa",
            "provenance": "public_dataset",
            "statistical_independence": "source_request_identity",
        },
        {
            "id": "short-chat-control",
            "family": "short_chat",
            "provenance": "public_dataset",
            "statistical_independence": "source_request_identity",
            "short_context_control": True,
        },
    ]
    curve_inputs = [
        ("qwen-long", "bailian-chat", "warm"),
        ("llama-long", "bailian-coder", "cold"),
        ("qwen-long", "long-document", "cold"),
        ("qwen-small", "short-chat-control", "warm"),
    ]
    curves = []
    for curve_index, (model, workload, cache_state) in enumerate(curve_inputs):
        curves.append(
            {
                "id": f"curve-{curve_index}",
                "model": model,
                "workload": workload,
                "systems": [entry["id"] for entry in systems],
                "cache_state": cache_state,
                "metrics": METRICS,
                "load_points": [
                    {
                        "stock_knee_fraction": (0.5, 0.75, 1.0, 1.25)[level],
                        "independent_requests": 1000,
                        "request_identity_set": f"curve-{curve_index}-load-{level}",
                        "repetitions": 5,
                    }
                    for level in range(4)
                ],
            }
        )
    return {
        "schema": 1,
        "classification": "nta-osdi-campaign",
        "status": "registered",
        "correctness_gate": {
            "role": "validity_gate_not_research_question",
            "status": "pending",
        },
        "models": models,
        "systems": systems,
        "load_selection": {
            "policy": "stock_only_frozen_knee",
            "nta_observed_during_selection": False,
            "fractions_of_stock_knee": [0.5, 0.75, 1.0, 1.25],
        },
        "workloads": workloads,
        "headline_curves": curves,
        "mechanism_studies": [
            {
                "profile": "mechanism-study",
                "arms": ["A0", "A1", "A1P", "A2", "A3"],
                "causal_pairs": [
                    "A1>A0",
                    "A1P>A1",
                    "A2>A1",
                    "A3>A2",
                ],
                "scenarios": 6,
                "result_emitted_activation": True,
            }
        ],
        "opportunity_sweeps": [
            {"axis": axis, "levels": ["low", "medium", "high"]}
            for axis in ("context", "locality", "load", "tier")
        ],
        "deployment_studies": [
            {
                "kind": kind,
                "repetitions": 5,
                "metrics": ["latency", "throughput"],
            }
            for kind in (
                "short_context_control",
                "resource_profile",
                "tenant_interference",
            )
        ],
    }


def _attach_artifact(entry: dict[str, object], artifact: Path, digest: str) -> None:
    entry["artifact"] = str(artifact)
    entry["artifact_sha256"] = digest


def main() -> int:
    registered = _campaign()
    summary = validate(registered, base_dir=ROOT)
    assert summary["models"] == 3
    assert summary["systems"] == 4
    assert summary["load_points"] == 16
    assert summary["independent_requests"] == 16000
    try:
        validate(registered, base_dir=ROOT, require_complete=True)
    except ValueError as error:
        assert "not complete" in str(error)
    else:
        raise AssertionError("registered campaign passed the complete evidence gate")

    with tempfile.TemporaryDirectory() as directory:
        artifact = Path(directory) / "evidence.json"
        artifact.write_text('{"status":"pass"}\n', encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        complete = copy.deepcopy(registered)
        complete["status"] = "complete"
        correctness = complete["correctness_gate"]
        assert isinstance(correctness, dict)
        correctness["status"] = "pass"
        _attach_artifact(correctness, artifact, digest)
        _attach_artifact(complete["load_selection"], artifact, digest)
        for curve in complete["headline_curves"]:
            for point in curve["load_points"]:
                point["offered_requests_per_second"] = 1.0 / point[
                    "stock_knee_fraction"
                ]
                _attach_artifact(point, artifact, digest)
        for section in (
            "mechanism_studies",
            "opportunity_sweeps",
            "deployment_studies",
        ):
            for entry in complete[section]:
                _attach_artifact(entry, artifact, digest)
        completed = validate(complete, base_dir=ROOT, require_complete=True)
        assert completed["status"] == "complete"

        corrupted = copy.deepcopy(complete)
        corrupted["headline_curves"][0]["load_points"][0][
            "artifact_sha256"
        ] = "0" * 64
        try:
            validate(corrupted, base_dir=ROOT, require_complete=True)
        except ValueError as error:
            assert "digest mismatch" in str(error)
        else:
            raise AssertionError("campaign accepted corrupted performance evidence")

    print("osdi_campaign=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
