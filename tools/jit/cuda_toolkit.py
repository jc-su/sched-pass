#!/usr/bin/env python3
"""Resolve one CUDA toolkit for every JIT producer in the project.

Framework wheels can be built against a different CUDA release than the
system default symlink.  Letting the launcher and the framework independently
discover CUDA is unsafe: the generated command then contains headers from one
toolkit while the compiler driver uses another.  This module keeps the
resolution policy in one small, dependency-free place.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
from typing import Iterable


def _candidate(value: str | pathlib.Path | None) -> pathlib.Path | None:
    if value is None:
        return None
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return path.resolve()


def _torch_cuda_home() -> pathlib.Path | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    version = str(getattr(torch.version, "cuda", "") or "").split(".")
    if len(version) < 2:
        return None
    return _candidate(f"/usr/local/cuda-{version[0]}.{version[1]}")


def _valid(path: pathlib.Path | None) -> pathlib.Path | None:
    if path is None:
        return None
    if (path / "bin" / "nvcc").is_file():
        return path
    return None


def resolve_cuda_home(requested: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return a CUDA root whose nvcc is present.

    Explicit configuration wins.  Otherwise the framework's CUDA ABI is the
    next source of truth, followed by PATH and the system alternatives link.
    """

    configured = requested or os.environ.get("NTA_CUDA_PATH")
    if configured is None:
        configured = (
            os.environ.get("NTA_CUDA_ROOT")
            or os.environ.get("CUDAToolkit_ROOT")
            or os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
        )
    candidates: list[pathlib.Path | None] = [_candidate(configured)]
    if configured is None:
        candidates.append(_torch_cuda_home())
        discovered = shutil.which("nvcc")
        if discovered:
            candidates.append(_candidate(pathlib.Path(discovered).parent.parent))
        candidates.append(_candidate("/usr/local/cuda"))
    for candidate in candidates:
        valid = _valid(candidate)
        if valid is not None:
            return valid
    rendered = ", ".join(str(value) for value in candidates if value is not None)
    raise RuntimeError(
        "CUDA toolkit with nvcc was not found; checked "
        + (rendered or "the configured environment")
    )


def nvcc_path(cuda_home: pathlib.Path) -> pathlib.Path:
    path = cuda_home / "bin" / "nvcc"
    if not path.is_file():
        raise RuntimeError(f"CUDA toolkit has no nvcc: {path}")
    return path


def cuda_include_dirs(cuda_home: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return the toolkit include roots needed by Clang and nvcc.

    CUDA 12.x exposes headers directly below ``<root>/include``.  CUDA 13.x
    keeps the CUDA C++ headers in the Linux target sysroot instead.  Passing
    only the former makes a valid CUDA 13 toolkit look incomplete (notably
    ``<cuda/barrier>``).  Keep the roots ordered and deduplicated so callers
    can inject the same ABI view into every compiler frontend.
    """

    candidates: list[pathlib.Path] = [cuda_home / "include"]
    for target_include in sorted(cuda_home.glob("targets/*-linux/include")):
        candidates.append(target_include)
        candidates.append(target_include / "cccl")
    return tuple(dict.fromkeys(path for path in candidates if path.is_dir()))


def cuda_release(cuda_home: pathlib.Path) -> tuple[int, int]:
    """Return the release major/minor reported by the selected nvcc."""

    try:
        result = subprocess.run(
            [str(nvcc_path(cuda_home)), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot query CUDA toolkit version: {cuda_home}") from error
    match = re.search(r"release\s+(\d+)\.(\d+)", result)
    if match is None:
        raise RuntimeError(
            f"cannot parse CUDA toolkit version from {cuda_home / 'bin' / 'nvcc'}"
        )
    return int(match.group(1)), int(match.group(2))


def is_cuda_include(path: str | pathlib.Path) -> bool:
    """Identify a toolkit include flag emitted by an external JIT builder."""

    value = pathlib.Path(path).expanduser()
    try:
        resolved = value.resolve()
    except OSError:
        resolved = value
    for candidate in (value, resolved):
        for include in (candidate, *candidate.parents):
            if include.name != "include":
                continue
            parent = include.parent.name
            grandparent = include.parent.parent.name
            if (
                parent == "cuda"
                or re.fullmatch(r"cuda-\d+(?:\.\d+)*", parent)
                or (grandparent == "targets" and parent.endswith("-linux"))
            ):
                return True
    return False


def filter_cuda_include_args(arguments: Iterable[str]) -> list[str]:
    """Drop external toolkit include flags; the selected root is injected once."""

    result: list[str] = []
    values = list(arguments)
    index = 0
    while index < len(values):
        argument = values[index]
        if argument in ("-I", "-isystem") and index + 1 < len(values):
            if is_cuda_include(values[index + 1]):
                index += 2
                continue
        if argument.startswith("-I") and is_cuda_include(argument[2:]):
            index += 1
            continue
        if argument.startswith("-isystem") and is_cuda_include(argument[8:]):
            index += 1
            continue
        result.append(argument)
        index += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-path",
        help="explicit CUDA toolkit root; otherwise use the shared resolver",
    )
    parser.add_argument(
        "--print-root", action="store_true", help="print the resolved toolkit root"
    )
    parser.add_argument(
        "--print-release", action="store_true", help="print the nvcc release"
    )
    args = parser.parse_args()
    if not args.print_root and not args.print_release:
        parser.error("select --print-root or --print-release")
    home = resolve_cuda_home(args.cuda_path)
    if args.print_root:
        print(home)
    if args.print_release:
        major, minor = cuda_release(home)
        print(f"{major}.{minor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
