#!/usr/bin/env python3
"""Verify GPU plan production and acquisition lowering in the real MoE IR."""

from __future__ import annotations

import pathlib
import re
import sys


def kernel_body(module: str, name: str) -> str:
    match = re.search(
        rf"define[^@]*@{re.escape(name)}\([^)]*\)[^{{]*\{{(?P<body>.*?)\n\}}",
        module,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing kernel {name}")
    return match.group("body")


def main() -> None:
    module = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    route = kernel_body(module, "nta_moe_route_kernel")
    assert "store i32" in route
    assert "store i64" in route
    assert "nta.acquire-set" not in route

    for name in ("nta_moe_tile_kernel", "nta_moe_ready_kernel"):
        consumer = kernel_body(module, name)
        assert "@_ZZ20nta_acquire_set_slowE8ctaReady" in consumer
        assert "llvm.nvvm.barrier.cta.sync" in consumer
        assert "call void @nta_defer" in consumer
        assert "!nta.acquire" in consumer
        assert "__nta_acquire_set_marker" not in consumer

    print("moe_lowering=pass")


if __name__ == "__main__":
    main()
