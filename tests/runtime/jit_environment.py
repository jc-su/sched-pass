#!/usr/bin/env python3
"""Verify idempotent serving JIT workspace activation."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))

from cuda_environment import _activation_cache_root  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        cache_root = Path(temporary).resolve()
        assert _activation_cache_root(cache_root) == cache_root

        activated = cache_root / f"uid-{os.getuid()}" / "nta-abi35-fi-test-deadbeef"
        manifest = activated / "nta-flashinfer-overlay" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        assert _activation_cache_root(activated) == cache_root

        lookalike = cache_root / f"uid-{os.getuid()}" / "nta-abi35-no-manifest"
        lookalike.mkdir(parents=True)
        assert _activation_cache_root(lookalike) == lookalike

    print("jit_environment=pass")


if __name__ == "__main__":
    main()
