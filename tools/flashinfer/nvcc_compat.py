#!/usr/bin/env python3
"""NVCC entry point for FlashInfer JIT on hosts newer than CUDA's GCC gate."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _default_nvcc() -> str:
    configured_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured_home:
        candidate = Path(configured_home) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    discovered = shutil.which("nvcc")
    if discovered:
        return discovered
    return "/usr/local/cuda/bin/nvcc"


def main() -> int:
    nvcc = os.environ.get("NTA_REAL_NVCC", _default_nvcc())
    arguments = sys.argv[1:]
    requested_host = os.environ.get("NTA_NVCC_HOST_COMPILER")
    host_compiler = (
        shutil.which(requested_host)
        if requested_host
        else next(
            (
                path
                for name in ("g++-14", "g++-13", "g++-12")
                if (path := shutil.which(name)) is not None
            ),
            None,
        )
    )
    if host_compiler and not any(
        argument == "-ccbin" or argument.startswith("-ccbin=") for argument in arguments
    ):
        arguments = [f"-ccbin={host_compiler}", *arguments]
    return subprocess.call([nvcc, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
