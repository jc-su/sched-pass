#!/usr/bin/env python3
"""Compose a validated, provenance-complete performance evidence bundle.

Profiler execution stays in :mod:`profile`; this command is the single
composition boundary for the four files consumed by an artifact evaluation.
It copies only the profiler files named by ``profile.json``, computes the
machine-specific regression comparison, records every input digest, and
refuses to turn a failed comparison into a passing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

try:
    from .check_regression import compare
    from .validate_performance_artifact import validate
except ImportError:  # Direct script execution.
    from check_regression import compare
    from validate_performance_artifact import validate


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _outside_source(path: Path, name: str) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{name} must be outside the source tree: {resolved}")
    return resolved


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_profile(source: Path, output: Path) -> dict[str, Any]:
    profile_path = source / "profile.json"
    profile = _load_object(profile_path, "profiler metadata")
    outputs = profile.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("profiler metadata names no raw output files")
    names = ["profile.json", "stdout.log"]
    names.extend(str(value) for value in outputs)
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"profiler output escapes its source directory: {name}")
        input_path = source / relative
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise ValueError(f"profiler output is missing or empty: {input_path}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, destination)
    return profile


def compose(
    *, profile_source: Path, baseline_source: Path, measured_source: Path, output: Path
) -> dict[str, Any]:
    profile_source = _outside_source(profile_source, "profiler artifact")
    baseline_source = _outside_source(baseline_source, "baseline report")
    measured_source = _outside_source(measured_source, "measured report")
    output = _outside_source(output, "performance output")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"performance output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    profile = _copy_profile(profile_source, output)
    baseline = _load_object(baseline_source, "baseline report")
    measured = _load_object(measured_source, "measured report")
    _write_json(output / "baseline.json", baseline)
    _write_json(output / "measured.json", measured)
    regression = compare(baseline, measured)
    _write_json(output / "regression.json", regression)
    capture = {
        "schema": 1,
        "classification": "nta-performance-capture",
        "profile_source": profile_source.name,
        "baseline_source": baseline_source.name,
        "measured_source": measured_source.name,
        "profile_digest": _digest(output / "profile.json"),
        "baseline_digest": _digest(output / "baseline.json"),
        "measured_digest": _digest(output / "measured.json"),
        "regression_digest": _digest(output / "regression.json"),
        "machine": measured.get("machine"),
        "revision": measured.get("revision"),
    }
    _write_json(output / "capture.json", capture)
    # ``validate`` also checks that the regression is passing.  Thus a failed
    # comparison is left as an auditable directory but can never be consumed
    # by reproduce.py as a complete performance artifact.
    validation = validate(output) if regression.get("pass") is True else None
    return {
        "capture": capture,
        "profile": profile,
        "regression": regression,
        "validation": validation,
        "valid": validation is not None and validation.get("pass") is True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-artifact", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compose(
            profile_source=args.profile_artifact,
            baseline_source=args.baseline,
            measured_source=args.measured,
            output=args.output,
        )
    except (OSError, ValueError) as error:
        print(f"capture_performance refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
