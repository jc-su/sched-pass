#!/usr/bin/env python3
"""Ensure the transport artifact owns its CUDA runtime dependency."""

from __future__ import annotations

import ctypes
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: transport_program_standalone.py <program.so>")
    path = pathlib.Path(sys.argv[1]).resolve(strict=True)
    # This process intentionally imports neither torch nor nta-runtime.  A
    # transport artifact that relies on either one to preload libcudart has an
    # implicit framework/load-order dependency and must fail this gate.
    ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
