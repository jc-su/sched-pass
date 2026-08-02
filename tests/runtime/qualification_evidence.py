#!/usr/bin/env python3
"""Ensure release gates reject evidence for the superseded mechanism."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_qualifier():
    path = ROOT / "scripts" / "qualify-release.py"
    spec = importlib.util.spec_from_file_location("nta_qualify_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load release qualifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    qualifier = load_qualifier()
    with tempfile.TemporaryDirectory() as temporary:
        evidence = pathlib.Path(temporary)
        old = {"schema": 1, "revision": "revision"}
        (evidence / "production-evidence.json").write_text(
            json.dumps(old), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        assert len(production) == 1 and not production[0]["passed"]
        assert "schema 2" in production[0]["detail"]

        current = {"schema": 2, "revision": "revision", "artifacts": []}
        (evidence / "production-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        production = qualifier.production_checks(evidence, "revision")
        production_names = {item["name"] for item in production}
        assert {
            "serving graph path",
            "serving performance bounds",
            "serving tier coverage",
        }.issubset(production_names)
        assert not any(item["passed"] for item in production)

        (evidence / "osdi-evidence.json").write_text(
            json.dumps(current), encoding="utf-8"
        )
        osdi = qualifier.osdi_checks(evidence, "revision")
        osdi_names = {item["name"] for item in osdi}
        assert {
            "measured dense opportunity",
            "compiler-generated forms",
            "real FlashInfer incremental execution",
            "unified scheduler and engine feedback",
            "real GPU-selected sparse stress",
            "mechanism performance bounds",
        }.issubset(osdi_names)
        assert not any(item["passed"] for item in osdi)

    print("qualification_evidence=pass")


if __name__ == "__main__":
    main()
