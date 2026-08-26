#!/usr/bin/env python3
"""Validate the standalone vLLM integration-smoke report.

vLLM's resident integration smoke has a deliberately smaller contract than
the SGLang serving comparison: it proves that the real vLLM consumer reached
the NTA attention implementation and that stock and NTA runs can be compared
by output digests.  It is not a remote-tier throughput report.  Keeping this
validator separate prevents a vLLM integration result from being accepted as
an SGLang load-comparison report, or vice versa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from .consumer_contract import validate_consumer_contract
except ImportError:  # Direct script execution.
    from consumer_contract import validate_consumer_contract


_DIGEST_SIZE = hashlib.sha256().digest_size * 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"vLLM report has no finite {name}",
    )
    result = float(value)
    _require(
        result > 0 if positive else result >= 0,
        f"vLLM report has invalid {name}",
    )
    return result


def _positive_integer(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"vLLM report has invalid {name}",
    )
    return value


def _digest(value: Any, name: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == _DIGEST_SIZE
        and all(character in "0123456789abcdef" for character in value),
        f"vLLM report has invalid {name}",
    )


def _nonempty_string(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(value), f"vLLM report has no {name}")


def _validate_evidence(report: dict[str, Any], *, native: bool) -> None:
    evidence = report.get("evidence")
    _require(isinstance(evidence, list), "vLLM report evidence is not a list")
    contract = report.get("consumer_contract")
    expected_kind = "native_work_unit" if native else "framework_reference"
    validate_consumer_contract(
        contract,
        expected_engine="vllm",
        expected_kind=expected_kind,
        require_formal_execution=True,
    )

    if not native:
        _require(not evidence, "stock vLLM report unexpectedly contains NTA evidence")
        _require(
            report.get("native_execution_verified") is False,
            "stock vLLM report claims native execution",
        )
        _require(
            report.get("native_launches") == 0,
            "stock vLLM report has native launches",
        )
        return

    _require(evidence, "NTA vLLM report has no worker evidence")
    _require(
        report.get("native_execution_verified") is True,
        "NTA vLLM report does not verify native execution",
    )
    _require(
        report.get("stock_fallback_enabled") is False,
        "NTA vLLM report permits stock fallback",
    )
    observed_launches = 0
    observed_fallbacks = 0
    for index, entry in enumerate(evidence):
        _require(isinstance(entry, dict), f"vLLM evidence {index} is not an object")
        _require(
            entry.get("engine") == "vllm",
            f"vLLM evidence {index} has wrong engine",
        )
        _require(
            entry.get("backend") == "nta_flashinfer",
            f"vLLM evidence {index} has wrong backend",
        )
        _require(
            entry.get("native_enabled") is True,
            f"vLLM evidence {index} did not enable native mode",
        )
        _require(
            entry.get("stock_fallback_enabled") is False,
            f"vLLM evidence {index} permits stock fallback",
        )
        validate_consumer_contract(
            entry.get("consumer_contract"),
            expected_engine="vllm",
            expected_backend="nta_flashinfer",
            expected_kind="native_work_unit",
            require_formal_execution=True,
        )
        stats = entry.get("stats")
        _require(isinstance(stats, dict), f"vLLM evidence {index} has no statistics")
        for field in ("native_decode_launches", "native_prefill_launches"):
            value = stats.get(field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"vLLM evidence {index} has invalid {field}",
            )
        fallback_value = stats.get("reference_fallback_launches", 0)
        _require(
            isinstance(fallback_value, int)
            and not isinstance(fallback_value, bool)
            and fallback_value >= 0,
            f"vLLM evidence {index} has invalid reference_fallback_launches",
        )
        observed_launches += (
            stats["native_decode_launches"] + stats["native_prefill_launches"]
        )
        observed_fallbacks += fallback_value
    _require(
        report.get("native_launches") == observed_launches > 0,
        "vLLM native launch accounting is inconsistent",
    )
    _require(
        report.get("reference_fallback_launches") == observed_fallbacks == 0,
        "vLLM native report contains reference fallback launches",
    )


def validate(report: dict[str, Any]) -> dict[str, Any]:
    """Validate and return one vLLM integration-smoke report."""

    _require(isinstance(report, dict), "vLLM report is not an object")
    _require(report.get("schema") == 1, "unsupported vLLM report schema")
    _require(
        report.get("classification") == "vllm-serving-integration-smoke",
        "report is not a vLLM integration smoke",
    )
    backend = report.get("backend")
    _require(backend in {"stock", "nta"}, "vLLM report has an unknown backend")
    _require(report.get("backend_selected") is True, "vLLM backend was not selected")
    _require(report.get("engine") == "vllm", "vLLM report has the wrong engine")
    for field in (
        "engine_version",
        "flashinfer_version",
        "torch_version",
        "cuda_version",
        "model",
        "revision",
        "flashinfer_workspace_base",
    ):
        _nonempty_string(report.get(field), field)
    _require(
        isinstance(report.get("machine"), dict) and report["machine"],
        "vLLM report has no machine metadata",
    )
    _require(
        isinstance(report.get("dirty"), bool), "vLLM report has invalid dirty state"
    )
    requests = _positive_integer(report.get("requests"), "request count")
    _positive_integer(report.get("max_new_tokens"), "maximum new token count")
    iterations = _positive_integer(report.get("iterations"), "iteration count")
    _positive_integer(report.get("warmup_iterations"), "warmup iteration count")
    generated_tokens = _positive_integer(
        report.get("generated_tokens"), "generated token count"
    )
    _digest(report.get("generated_text_sha256"), "generated text digest")
    _digest(report.get("generated_token_ids_sha256"), "generated token-ID digest")
    _finite(report.get("load_seconds"), "load time")

    samples = report.get("batch_seconds_samples")
    _require(
        isinstance(samples, list) and len(samples) == iterations and samples,
        "vLLM report has an incomplete timing sample list",
    )
    normalized_samples = [
        _finite(value, f"batch sample {index}", positive=True)
        for index, value in enumerate(samples)
    ]
    median = _finite(
        report.get("median_batch_seconds"), "median batch time", positive=True
    )
    _require(
        math.isclose(
            median,
            statistics.median(normalized_samples),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "vLLM report median does not match timing samples",
    )
    request_rate = _finite(
        report.get("requests_per_second"), "request throughput", positive=True
    )
    _require(
        math.isclose(
            request_rate,
            requests / median,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "vLLM request throughput is inconsistent with the median",
    )
    _require(
        math.isclose(
            _finite(
                report.get("generated_tokens_per_second"),
                "generated-token throughput",
                positive=True,
            ),
            generated_tokens / median,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        "vLLM token throughput is inconsistent with the median",
    )
    _require(
        isinstance(report.get("stock_fallback_enabled"), bool),
        "vLLM report has invalid fallback state",
    )
    _validate_evidence(report, native=backend == "nta")
    if backend == "stock":
        _require(
            report["stock_fallback_enabled"] is False,
            "stock vLLM report has an NTA fallback state",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.report.resolve().read_text(encoding="utf-8")))
    print("vllm_report=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
