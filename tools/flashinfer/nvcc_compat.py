#!/usr/bin/env python3
"""NVCC entry point for FlashInfer JIT on hosts newer than CUDA's GCC gate."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def main() -> int:
    nvcc = os.environ.get("NTA_REAL_NVCC", "/usr/local/cuda/bin/nvcc")
    arguments = sys.argv[1:]
    requested_host = os.environ.get("NTA_NVCC_HOST_COMPILER")
    host_compiler = (
        shutil.which(requested_host)
        if requested_host
        else next(
            (path for name in ("g++-14", "g++-13", "g++-12")
             if (path := shutil.which(name)) is not None),
            None,
        )
    )
    if host_compiler and not any(
        argument == "-ccbin" or argument.startswith("-ccbin=")
        for argument in arguments
    ):
        arguments = [f"-ccbin={host_compiler}", *arguments]
    return subprocess.call([nvcc, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
