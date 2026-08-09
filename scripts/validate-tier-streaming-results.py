#!/usr/bin/env python3
"""Validate the bounded-HBM FlashInfer mechanism and heterogeneity evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline", type=pathlib.Path, required=True)
    parser.add_argument("--heterogeneous", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--require-compiler-transform", action="store_true")
    return parser.parse_args()


def read_result(path: pathlib.Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read tier-streaming result {path}: {error}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError(f"tier-streaming result must be an object: {path}")
    return result


def _canonical_and_exact(result: dict[str, Any]) -> bool:
    return (
        result.get("schema") == 1
        and result.get("classification") == "flashinfer-request-aware-tier-streaming"
        and result.get("real_flashinfer_attention") is True
        and result.get("real_flashinfer_online_softmax_merge") is True
        and result.get("custom_attention_kernel") is False
        and isinstance(result.get("compiler_transformed_attention"), bool)
        and result.get("output_parity") is True
        and result.get("request_semantics_retained") is True
    )


def _request_field_count(result: dict[str, Any], field: str) -> int:
    requests = result.get("requests")
    if not isinstance(requests, list):
        return 0
    return len(
        {
            request.get(field)
            for request in requests
            if isinstance(request, dict)
            and isinstance(request.get(field), (int, float))
        }
    )


def _sample_count(result: dict[str, Any]) -> int:
    streaming = result.get("streaming_us")
    samples = streaming.get("samples") if isinstance(streaming, dict) else None
    return len(samples) if isinstance(samples, list) else 0


def validate_results(
    headline: dict[str, Any],
    heterogeneous: dict[str, Any],
    *,
    require_compiler_transform: bool = False,
) -> dict[str, Any]:
    headline_ci = headline.get("streaming_speedup_95ci", {})
    heterogeneous_ci = heterogeneous.get("streaming_speedup_95ci", {})
    checks = [
        {
            "name": "canonical FlashInfer numerical path",
            "passed": _canonical_and_exact(headline)
            and _canonical_and_exact(heterogeneous),
        },
        {
            "name": "typed compiler execution plan",
            "passed": not require_compiler_transform
            or all(
                result.get("compiler_transformed_attention") is True
                and result.get("compiler_runtime_protocol_active") is True
                and isinstance(result.get("compiler_operator_plan"), dict)
                and result["compiler_operator_plan"].get("supported_forms") == 6
                and result["compiler_operator_plan"].get("partial_state")
                == "online_softmax_value_lse"
                and result["compiler_operator_plan"].get("reduction")
                == "ordered_merge_state"
                and bool(result["compiler_operator_plan"].get("plan_fingerprint"))
                for result in (headline, heterogeneous)
            ),
        },
        {
            "name": "matched hardware and software",
            "passed": all(
                headline.get(field) == heterogeneous.get(field)
                for field in (
                    "gpu",
                    "flashinfer_version",
                    "torch_version",
                    "cuda_version",
                )
            ),
        },
        {
            "name": "dynamic-source graph replay",
            "passed": headline.get("graph_replay_verified") is True
            and headline.get("graph_dynamic_source_verified") is True
            and isinstance(headline.get("graph_dynamic_max_abs_error"), (int, float))
            and heterogeneous.get("graph_replay_verified") is True
            and heterogeneous.get("graph_dynamic_source_verified") is True
            and isinstance(
                heterogeneous.get("graph_dynamic_max_abs_error"), (int, float)
            ),
        },
        {
            "name": "generation reuse and cancellation isolation",
            "passed": not require_compiler_transform
            or all(
                result.get("generation_reuse_verified") is True
                and result.get("cancellation_isolation_verified") is True
                for result in (headline, heterogeneous)
            ),
        },
        {
            "name": "headline effect size",
            "passed": isinstance(
                headline.get("streaming_speedup_over_atomic"), (int, float)
            )
            and headline["streaming_speedup_over_atomic"] >= 1.15,
        },
        {
            "name": "headline confidence",
            "passed": _sample_count(headline) >= 10
            and isinstance(headline_ci, dict)
            and isinstance(headline_ci.get("lower"), (int, float))
            and headline_ci["lower"] > 1.0
            and headline_ci.get("confidence") == 0.95,
        },
        {
            "name": "bounded HBM capacity",
            "passed": isinstance(
                headline.get("staging_capacity_reduction"), (int, float)
            )
            and headline["staging_capacity_reduction"] >= 4.0
            and isinstance(
                heterogeneous.get("staging_capacity_reduction"), (int, float)
            )
            and heterogeneous["staging_capacity_reduction"] >= 4.0,
        },
        {
            "name": "per-request placement and completion",
            "passed": _request_field_count(headline, "resident_tokens") >= 3
            and isinstance(headline.get("request_completion_us"), dict)
            and len(set(headline["request_completion_us"].values()))
            == len(headline.get("requests", [])),
        },
        {
            "name": "heterogeneous request shapes",
            "passed": _request_field_count(heterogeneous, "context_tokens") >= 3
            and _request_field_count(heterogeneous, "query_tokens") >= 3
            and _request_field_count(heterogeneous, "resident_tokens") >= 3,
        },
        {
            "name": "heterogeneous benefit confidence",
            "passed": isinstance(heterogeneous_ci, dict)
            and isinstance(heterogeneous_ci.get("lower"), (int, float))
            and heterogeneous_ci["lower"] > 1.0
            and heterogeneous_ci.get("confidence") == 0.95,
        },
    ]
    return {
        "schema": 1,
        "classification": "tier-streaming-mechanism-qualification",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "headline_revision": headline.get("revision"),
        "headline_dirty": headline.get("dirty"),
        "headline_speedup": headline.get("streaming_speedup_over_atomic"),
        "headline_speedup_95ci": headline_ci,
        "headline_staging_capacity_reduction": headline.get(
            "staging_capacity_reduction"
        ),
        "compiler_transformed_attention": headline.get(
            "compiler_transformed_attention"
        ),
        "compiler_transform_required": require_compiler_transform,
        "heterogeneous_speedup": heterogeneous.get("streaming_speedup_over_atomic"),
        "heterogeneous_speedup_95ci": heterogeneous_ci,
    }


def main() -> int:
    arguments = parse_args()
    report = validate_results(
        read_result(arguments.headline),
        read_result(arguments.heterogeneous),
        require_compiler_transform=arguments.require_compiler_transform,
    )
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
