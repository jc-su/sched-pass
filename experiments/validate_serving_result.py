#!/usr/bin/env python3
"""Dispatch validation for structured serving results.

SGLang load comparisons and vLLM integration smokes intentionally have
different evidence contracts.  This small boundary keeps artifact assembly
framework-neutral without weakening either validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .validate_serving_report import validate as validate_sglang
    from .validate_vllm_report import validate as validate_vllm
except ImportError:  # Direct script execution.
    from validate_serving_report import validate as validate_sglang
    from validate_vllm_report import validate as validate_vllm


def validate(report: dict[str, Any]) -> dict[str, Any]:
    """Validate one supported framework result by its explicit classification."""

    classification = report.get("classification")
    if classification == "vllm-serving-integration-smoke":
        return validate_vllm(report)
    return validate_sglang(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    validate(report)
    print("serving_result=valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
