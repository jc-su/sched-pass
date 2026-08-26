#!/usr/bin/env python3
"""Ensure semantic imports do not initialize the native runtime."""

from __future__ import annotations

import ctypes
import importlib
import sys


def main() -> None:
    calls: list[tuple[object, ...]] = []

    def forbidden_cdll(*args: object, **kwargs: object) -> object:
        calls.append(args)
        raise AssertionError("semantic import attempted to load a native library")

    original_cdll = ctypes.CDLL
    ctypes.CDLL = forbidden_cdll  # type: ignore[assignment]
    try:
        importlib.import_module("nta_runtime.work_unit")
        importlib.import_module("nta_runtime.execution_protocol")
        importlib.import_module("nta_runtime.adapters.base")
        importlib.import_module("nta_runtime.runtime_resources")
        from nta_runtime import RequestSpec
        from nta_runtime.request_contract import RequestSpec as ContractRequestSpec
    finally:
        ctypes.CDLL = original_cdll

    assert not calls
    assert "nta_runtime.runtime" not in sys.modules
    assert RequestSpec is ContractRequestSpec
    assert RequestSpec(0, 17, 1).native().request_id == 17
    print("import_boundaries=pass")


if __name__ == "__main__":
    main()
