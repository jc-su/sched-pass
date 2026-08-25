#!/usr/bin/env python3
"""Check that overlay source hashing ignores only editor backup artifacts."""

from __future__ import annotations

import pathlib
import tempfile

from tools.flashinfer.prepare_overlay import tree_hash


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


if __name__ == "__main__":
    main()
