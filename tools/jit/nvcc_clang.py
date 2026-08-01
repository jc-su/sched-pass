#!/usr/bin/env python3
"""Translate an nvcc-style JIT command to clang CUDA and load NtaPass."""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(
    os.environ.get("NTA_PROJECT_ROOT", pathlib.Path(__file__).parents[2])
).resolve()
CLANG = os.environ.get("NTA_CLANG", "clang++-22")
PLUGIN = os.environ.get("NTA_PLUGIN", "")
CUDA_PATH = os.environ.get("NTA_CUDA_PATH", "/usr/local/cuda-12.9")
LOG = os.environ.get("NTA_JIT_LOG", "")
PRELUDE = pathlib.Path(__file__).with_name("clang_cuda_prelude.h")


def clang_arch(code: str) -> str:
    if os.environ.get("NTA_STRIP_ARCH"):
        match = re.match(r"(sm_\d+)", code)
        return match.group(1) if match else code
    match = re.match(r"(sm_\d+)f$", code)
    return f"{match.group(1)}a" if match else code


def should_instrument(arguments: list[str]) -> bool:
    filters = os.environ.get("NTA_JIT_ONLY", "")
    if not filters:
        return True
    command = " ".join(arguments)
    return any(token.strip() in command for token in filters.split(",")
               if token.strip())


def matches_filter(arguments: list[str], variable: str) -> bool:
    filters = os.environ.get(variable, "")
    command = " ".join(arguments)
    return any(token.strip() in command for token in filters.split(",")
               if token.strip())


def translate(arguments: list[str], instrument: bool) -> list[str]:
    command = [
        CLANG,
        "-x",
        "cuda",
        f"--cuda-path={CUDA_PATH}",
        "-Wno-unknown-cuda-version",
        "-isystem",
        str(pathlib.Path(CUDA_PATH) / "include"),
        "-I",
        str(ROOT / "include"),
        "-I",
        str(ROOT),
    ]
    overlay = os.environ.get("NTA_FLASHINFER_OVERLAY", "")
    if overlay:
        command[1:1] = ["-I", overlay]
    command.extend(["-include", str(PRELUDE)])
    toolkit = re.search(r"cuda-(\d+)\.(\d+)", CUDA_PATH)
    major, minor = toolkit.groups() if toolkit else ("12", "0")
    command.extend([
        f"-D__CUDACC_VER_MAJOR__={major}",
        f"-D__CUDACC_VER_MINOR__={minor}",
        "-D__CUDACC_VER_BUILD__=0",
    ])
    if instrument:
        if not PLUGIN or not pathlib.Path(PLUGIN).is_file():
            raise RuntimeError(
                "NTA_PLUGIN must identify the built libNtaPass.so")
        command.append(f"-fpass-plugin={PLUGIN}")
        if os.environ.get("NTA_FLASHINFER_HOOK"):
            phase_source = matches_filter(arguments, "NTA_JIT_PHASE_SOURCE")
            command.extend([
                f"-DNTA_DEVICE_PHASE_KERNELS={1 if phase_source else 0}",
                "-include",
                str(
                    ROOT / (
                        "runtime/device/JitRuntime.cuh"
                        if phase_source else "runtime/device/Acquire.cuh"
                    )
                ),
            ])

    have_standard = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument.startswith("-gencode="):
            match = re.search(r",code=([A-Za-z0-9_]+)", argument)
            if match:
                command.append(f"--cuda-gpu-arch={clang_arch(match.group(1))}")
            index += 1
            continue
        if argument == "-arch" and index + 1 < len(arguments):
            command.append(
                f"--cuda-gpu-arch={clang_arch(arguments[index + 1])}")
            index += 2
            continue
        if argument.startswith("--compiler-options="):
            command.append(argument.split("=", 1)[1])
            index += 1
            continue
        if argument in ("-Xcompiler", "--compiler-options") and index + 1 < len(arguments):
            command.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("-Xfatbin") or argument.startswith("--fatbin-options"):
            index += 2 if argument in ("-Xfatbin", "--fatbin-options") else 1
            continue
        if argument == "-Xptxas" and index + 1 < len(arguments):
            command.extend(["-Xcuda-ptxas", arguments[index + 1]])
            index += 2
            continue
        if argument.startswith("-Xptxas="):
            command.extend(["-Xcuda-ptxas", argument.split("=", 1)[1]])
            index += 1
            continue
        if argument == "--generate-dependencies-with-compile":
            index += 1
            continue
        if argument == "--dependency-output" and index + 1 < len(arguments):
            command.extend(["-MMD", "-MF", arguments[index + 1]])
            index += 2
            continue
        if argument in (
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-static-global-template-stub=false",
            "-static-global-template-stub=true",
            "-forward-unknown-to-host-compiler",
        ):
            index += 1
            continue
        if argument in ("--use_fast_math", "-use_fast_math"):
            command.append("-ffast-math")
            index += 1
            continue
        if argument in ("--generate-line-info", "-lineinfo"):
            command.append("-gline-tables-only")
            index += 1
            continue
        if argument.startswith("--threads") or argument.startswith(
            "-static-global-template-stub"
        ):
            index += 2 if argument == "--threads" else 1
            continue
        if argument == "-ccbin" and index + 1 < len(arguments):
            index += 2
            continue
        if argument.startswith("-ccbin="):
            index += 1
            continue
        if argument.startswith("-std="):
            have_standard = True
        command.append(argument)
        index += 1

    if not have_standard:
        command.append("-std=c++20")
    command.append("-fPIC")
    return command


def main() -> int:
    arguments = sys.argv[1:]
    if "--version" in arguments:
        toolkit = re.search(r"cuda-(\d+\.\d+)", CUDA_PATH)
        release = toolkit.group(1) if toolkit else "12.9"
        print("nvcc: NVIDIA (R) Cuda compiler driver (NTA clang JIT shim)")
        print(f"Cuda compilation tools, release {release}, V{release}.0")
        return 0

    instrument = should_instrument(arguments)
    if not instrument:
        real_nvcc = os.environ.get("NTA_REAL_NVCC", "")
        if real_nvcc and pathlib.Path(real_nvcc).is_file():
            os.execv(real_nvcc, [real_nvcc, *arguments])

    cache_tag = os.environ.get("NTA_JIT_CACHE_TAG", "")
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if instrument and cache_tag and cache_tag not in workspace:
        raise RuntimeError(
            "instrumented FlashInfer JIT requires an NTA-tagged workspace")

    command = translate(arguments, instrument)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as output:
            output.write(" ".join(command) + "\n")
    return subprocess.call(command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"nta-jit: {error}", file=sys.stderr)
        raise SystemExit(2) from error
