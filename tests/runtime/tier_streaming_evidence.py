#!/usr/bin/env python3
"""Exercise the bounded-HBM mechanism evidence gate without CUDA."""

from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_validator():
    path = ROOT / "scripts" / "validate-tier-streaming-results.py"
    spec = importlib.util.spec_from_file_location("nta_tier_streaming_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tier-streaming evidence validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(*, heterogeneous: bool) -> dict:
    requests = [
        {
            "request_id": index,
            "context_tokens": 65_536 - index * 16_384 if heterogeneous else 65_536,
            "query_tokens": 128 + index * 128 if heterogeneous else 256,
            "resident_tokens": index * 16_384,
        }
        for index in range(4)
    ]
    return {
        "schema": 2,
        "classification": "flashinfer-request-aware-tier-streaming",
        "revision": "revision",
        "dirty": False,
        "gpu": "gpu",
        "flashinfer_version": "0.6.14",
        "torch_version": "2.11",
        "cuda_version": "13.0",
        "real_flashinfer_attention": True,
        "real_flashinfer_online_softmax_merge": True,
        "custom_attention_kernel": False,
        "compiler_transformed_attention": False,
        "graph_replay_verified": True,
        "graph_dynamic_source_verified": True,
        "graph_dynamic_max_abs_error": 0.0,
        "generation_reuse_verified": True,
        "cancellation_isolation_verified": True,
        "output_parity": True,
        "request_semantics_retained": True,
        "requests": requests,
        "request_completion_us": {str(index): index + 1 for index in range(4)},
        "request_completion_samples_us": {
            str(index): [index + 1] * 10 for index in range(4)
        },
        "completion_observed_streaming_us": {"samples": [10] * 10},
        "streaming_speedup_over_atomic": 1.18 if not heterogeneous else 1.12,
        "streaming_speedup_95ci": {
            "confidence": 0.95,
            "lower": 1.1,
            "upper": 1.2,
        },
        "staging_capacity_reduction": 6.0,
        "streaming_us": {"samples": list(range(10))},
    }


def main() -> int:
    validator = load_validator()
    headline = fixture(heterogeneous=False)
    heterogeneous = fixture(heterogeneous=True)
    report = validator.validate_results(headline, heterogeneous)
    assert report["passed"] is True

    headline["custom_attention_kernel"] = True
    report = validator.validate_results(headline, heterogeneous)
    assert report["passed"] is False
    assert (
        next(
            check
            for check in report["checks"]
            if check["name"] == "canonical FlashInfer numerical path"
        )["passed"]
        is False
    )

    headline["custom_attention_kernel"] = False
    headline["graph_dynamic_source_verified"] = False
    report = validator.validate_results(headline, heterogeneous)
    assert report["passed"] is False
    headline["graph_dynamic_source_verified"] = True
    headline["streaming_speedup_95ci"]["lower"] = 0.99
    report = validator.validate_results(headline, heterogeneous)
    assert report["passed"] is False

    headline = fixture(heterogeneous=False)
    headline["compiler_transformed_attention"] = True
    headline["compiler_runtime_protocol_active"] = True
    headline["compiler_operator_plan"] = {
        "supported_forms": 6,
        "partial_state": "online_softmax_value_lse",
        "reduction": "ordered_merge_state",
        "plan_fingerprint": "plan",
    }
    heterogeneous = fixture(heterogeneous=True)
    heterogeneous.update(
        {
            "compiler_transformed_attention": True,
            "compiler_runtime_protocol_active": True,
            "compiler_operator_plan": headline["compiler_operator_plan"],
        }
    )
    report = validator.validate_results(
        headline, heterogeneous, require_compiler_transform=True
    )
    assert report["passed"] is True
    heterogeneous["cancellation_isolation_verified"] = False
    report = validator.validate_results(
        headline, heterogeneous, require_compiler_transform=True
    )
    assert report["passed"] is False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
