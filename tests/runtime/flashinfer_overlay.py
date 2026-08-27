#!/usr/bin/env python3
"""Check that overlay source hashing ignores only editor backup artifacts."""

from __future__ import annotations

import json
import pathlib
import tempfile

from tools.flashinfer.prepare_overlay import sha256, tree_hash, validate_existing


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-overlay-hash-") as value:
        root = pathlib.Path(value)
        source = root / "flashinfer" / "attention" / "prefill.cuh"
        source.parent.mkdir(parents=True)
        source.write_text("validated source\n", encoding="utf-8")
        baseline = tree_hash(root / "flashinfer")
        (source.with_suffix(".cuh.sched_bak")).write_text(
            "editor backup\n", encoding="utf-8"
        )
        assert tree_hash(root / "flashinfer") == baseline
        source.write_text("changed source\n", encoding="utf-8")
        assert tree_hash(root / "flashinfer") != baseline

        output = root / "overlay"
        overlay_source = output / "flashinfer" / "attention" / "prefill.cuh"
        overlay_source.parent.mkdir(parents=True)
        overlay_source.write_text("generated overlay\n", encoding="utf-8")
        identity = {
            "flashinfer_version": "test",
            "source_hashes": {"attention/prefill.cuh": "source"},
            "source_tree_hash": "source-tree",
            "hooks": ["test-hook"],
        }
        manifest = {
            **identity,
            # Installation location is provenance and may differ across
            # byte-identical Python environments sharing one cache.
            "source_include": "/environment/one/include",
            "overlay_hashes": {
                "attention/prefill.cuh": sha256(overlay_source),
            },
            "overlay_tree_hash": tree_hash(output / "flashinfer"),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        assert (
            validate_existing(
                output,
                identity,
                {"attention/prefill.cuh": "validated-source-hash"},
            )
            == manifest
        )


if __name__ == "__main__":
    main()
