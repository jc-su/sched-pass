"""Shared CUDA/JIT environment setup for serving experiments.

The serving harnesses use two independent JIT builders: FlashInfer's launcher
and SGLang's tvm-ffi builder.  This module gives both the same toolkit,
compiler, and glibc compatibility contract so an experiment cannot silently
compile different kernels for its baseline and mechanism arms.
"""

from __future__ import annotations

import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys


def _resolve_host_cxx(requested: pathlib.Path | None) -> pathlib.Path:
    candidate = requested
    if candidate is None:
        discovered = next(
            (shutil.which(name) for name in ("g++-13", "g++-14", "g++-12")),
            None,
        )
        candidate = pathlib.Path(discovered) if discovered else None
    if candidate is None:
        raise RuntimeError("CUDA-compatible host compiler not found")
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise RuntimeError(f"CUDA host compiler does not exist: {candidate}")
    return candidate


def _resolve_cuda_home(requested: pathlib.Path | None) -> pathlib.Path:
    candidate = requested
    if candidate is None:
        configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        candidate = pathlib.Path(configured) if configured else None
    if candidate is None:
        import torch

        version = str(torch.version.cuda or "").split(".")
        candidates: list[pathlib.Path] = []
        if len(version) >= 2:
            candidates.extend(
                (
                    pathlib.Path(f"/usr/local/cuda-{version[0]}.{version[1]}"),
                    pathlib.Path(f"/usr/local/cuda-{version[0]}"),
                )
            )
        candidates.append(pathlib.Path("/usr/local/cuda"))
        candidate = next(
            (item for item in candidates if (item / "bin" / "nvcc").is_file()),
            None,
        )
    if candidate is None:
        discovered = shutil.which("nvcc")
        candidate = pathlib.Path(discovered).parent.parent if discovered else None
    if candidate is None:
        raise RuntimeError(
            "CUDA toolkit matching torch.version.cuda was not found; "
            "pass an explicit CUDA home"
        )
    candidate = candidate.resolve()
    nvcc = candidate / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA toolkit has no nvcc: {nvcc}")
    return candidate


def _cuda_major(cuda_home: pathlib.Path) -> int:
    nvcc = cuda_home / "bin" / "nvcc"
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot query CUDA toolkit version: {nvcc}") from error
    match = re.search(r"release\s+(\d+)", result)
    if match is None:
        raise RuntimeError(f"cannot parse CUDA toolkit version from {nvcc}")
    return int(match.group(1))


def configure_jit_environment(
    *,
    root: pathlib.Path,
    workspace: pathlib.Path,
    host_cxx: pathlib.Path | None = None,
    cuda_home: pathlib.Path | None = None,
    revision: str | None = None,
    instrumented: bool = False,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Configure serving JIT builders and return host/toolkit/workspace.

    An NTA attention backend must use the complete launcher environment from
    ``tools/jit/activate.py``.  Configuring only FlashInfer's stock nvcc
    wrapper is unsafe: the resulting module can have an NTA name and overlay
    source while missing the exported phase ABI.  ``instrumented=True``
    therefore activates the pass, overlay, ABI-tagged cache, and runtime
    library before any framework imports or JIT compilation occur.
    """

    resolved_host_cxx = _resolve_host_cxx(host_cxx)
    resolved_cuda_home = _resolve_cuda_home(cuda_home)
    launcher = pathlib.Path(
        os.environ.get(
            "FLASHINFER_NVCC", root / "tools" / "flashinfer" / "nvcc_compat.py"
        )
    ).resolve()
    if not launcher.is_file():
        raise RuntimeError(f"FlashInfer NVCC launcher does not exist: {launcher}")

    resolved_workspace = pathlib.Path(
        os.environ.get("FLASHINFER_WORKSPACE_BASE", workspace)
    ).resolve()
    resolved_workspace.mkdir(parents=True, exist_ok=True)
    for stale in resolved_workspace.glob("nta-engine.*.json"):
        stale.unlink()

    host_cc = os.environ.get("CC") or shutil.which(
        resolved_host_cxx.name.replace("g++", "gcc")
    )
    if host_cc is None:
        raise RuntimeError(
            f"no CUDA-compatible C compiler matches {resolved_host_cxx.name}; "
            "set CC explicitly"
        )
    compat_header = root / "benchmarks" / "serving" / "pthread_clock_compat.h"
    if not compat_header.is_file():
        raise RuntimeError(f"serving CUDA compatibility header is missing: {compat_header}")
    nvcc_flags = [f"-ccbin={resolved_host_cxx}"]
    if _cuda_major(resolved_cuda_home) < 13:
        nvcc_flags.extend(
            (
                "-U_GNU_SOURCE",
                "-D_DEFAULT_SOURCE",
                "-include",
                str(compat_header),
            )
        )
    os.environ.update(
        {
            "CC": str(host_cc),
            "CXX": str(resolved_host_cxx),
            "CUDAHOSTCXX": str(resolved_host_cxx),
            "NTA_NVCC_HOST_COMPILER": str(resolved_host_cxx),
            # tvm-ffi emits a direct nvcc rule and ignores CUDAHOSTCXX.  The
            # glibc feature flags avoid the CUDA 12.x rsqrt declaration clash;
            # the forced header restores pthread declarations hidden by the
            # feature-macro workaround.
            "NVCC_PREPEND_FLAGS": shlex.join(nvcc_flags),
            "CUDA_HOME": str(resolved_cuda_home),
            "CUDA_PATH": str(resolved_cuda_home),
            "FLASHINFER_NVCC": str(launcher),
            "FLASHINFER_WORKSPACE_BASE": str(resolved_workspace),
            "NTA_ENGINE_STATS_FILE": str(resolved_workspace / "nta-engine.json"),
        }
    )
    if instrumented:
        activation = root / "tools" / "jit" / "activate.py"
        if not activation.is_file():
            raise RuntimeError(f"NTA JIT activation script is missing: {activation}")
        activation_result = subprocess.run(
            [
                sys.executable,
                str(activation),
                "--build-dir",
                str(root / "build"),
                "--cache-root",
                str(workspace.resolve()),
                "--cuda-path",
                str(resolved_cuda_home),
                "--flashinfer-hook",
                "--print-env",
            ],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if activation_result.returncode != 0:
            detail = activation_result.stderr.strip()
            raise RuntimeError(
                "NTA JIT activation failed"
                + (f": {detail}" if detail else "")
            )
        activated: dict[str, str] = {}
        for line in activation_result.stdout.splitlines():
            if not line.startswith("export "):
                continue
            name, value = line[7:].split("=", 1)
            parsed = shlex.split(value)
            if len(parsed) != 1:
                raise RuntimeError(f"invalid NTA activation export: {line!r}")
            activated[name] = parsed[0]
        required = {
            "FLASHINFER_NVCC",
            "FLASHINFER_WORKSPACE_BASE",
            "NTA_PLUGIN",
            "NTA_RUNTIME_LIBRARY",
            "NTA_FLASHINFER_HOOK",
            "NTA_FLASHINFER_OVERLAY",
        }
        missing = sorted(required - activated.keys())
        if missing:
            raise RuntimeError(
                "NTA JIT activation did not provide required settings: "
                + ", ".join(missing)
            )
        os.environ.update(activated)
        resolved_workspace = pathlib.Path(
            activated["FLASHINFER_WORKSPACE_BASE"]
        ).resolve()
        resolved_workspace.mkdir(parents=True, exist_ok=True)
        for stale in resolved_workspace.glob("nta-engine.*.json"):
            stale.unlink()
    os.environ["NTA_ENGINE_STATS_FILE"] = str(resolved_workspace / "nta-engine.json")
    if revision is not None:
        os.environ["NTA_REVISION"] = revision
    return resolved_host_cxx, resolved_cuda_home, resolved_workspace
