#!/usr/bin/env python3
"""Run a FlashInfer process with NTA's clang JIT and isolated cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys


def fingerprint(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def runtime_abi_version(header: pathlib.Path) -> int:
    match = re.search(
        r"inline constexpr std::uint32_t Version = (\d+);",
        header.read_text(encoding="utf-8"),
    )
    if match is None:
        raise RuntimeError(f"cannot read the runtime ABI version from {header}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--cache-root")
    parser.add_argument("--clang", default="/usr/bin/clang++-22")
    parser.add_argument("--cuda-path", default="/usr/local/cuda-12.9")
    parser.add_argument("--flashinfer-hook", action="store_true")
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    build = (root / options.build_dir).resolve()
    plugin = build / "libNtaPass.so"
    shim = root / "tools" / "jit" / "nvcc_clang.py"
    if not plugin.is_file():
        parser.error(f"pass plugin not found: {plugin}")
    abi_header = root / "include/nta/RuntimeABI.h"
    abi_version = runtime_abi_version(abi_header)
    integration_inputs = [
        plugin,
        shim,
        root / "tools/jit/clang_cuda_prelude.h",
        abi_header,
        root / "include/nta/DeviceAPI.cuh",
        root / "include/nta/KernelPolicy.cuh",
        root / "include/nta/FlashInferKernelPolicy.cuh",
        root / "runtime/device/Acquire.cuh",
        root / "runtime/device/JitRuntime.cuh",
    ]
    flashinfer_version = ""
    flashinfer_include = None
    if options.flashinfer_hook:
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or spec.origin is None:
            parser.error("--flashinfer-hook requires flashinfer-python")
        try:
            flashinfer_version = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            parser.error("flashinfer-python distribution metadata is missing")
        flashinfer_include = (
            pathlib.Path(spec.origin).resolve().parent / "data" / "include"
        )
        integration_inputs.extend([
            flashinfer_include / "flashinfer/attention/decode.cuh",
            flashinfer_include / "flashinfer/attention/prefill.cuh",
            root / "tools/flashinfer/prepare_overlay.py",
        ])
    version_tag = f"-fi{flashinfer_version}" if flashinfer_version else ""
    tag = f"nta-abi{abi_version}{version_tag}-{fingerprint(integration_inputs)}"
    cache_root = pathlib.Path(
        options.cache_root
        or os.environ.get("NTA_JIT_CACHE_ROOT", pathlib.Path.home() / ".cache/flashinfer")
    ).expanduser()
    workspace = (cache_root / tag).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    environment = {
        "FLASHINFER_NVCC": str(shim),
        "FLASHINFER_WORKSPACE_BASE": str(workspace),
        "NTA_PROJECT_ROOT": str(root),
        "NTA_PLUGIN": str(plugin),
        "NTA_CLANG": options.clang,
        "NTA_CUDA_PATH": options.cuda_path,
        "NTA_JIT_CACHE_TAG": tag,
        "NTA_ABI_VERSION": str(abi_version),
        "NTA_BUILD_DIR": str(build),
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(root), os.environ.get("PYTHONPATH", "")) if value
        ),
    }
    if options.flashinfer_hook:
        overlay = workspace / "nta-flashinfer-overlay"
        subprocess.run(
            [
                sys.executable,
                str(root / "tools/flashinfer/prepare_overlay.py"),
                "--output",
                str(overlay),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        environment.update({
            "NTA_FLASHINFER_HOOK": "1",
            "NTA_FLASHINFER_OVERLAY": str(overlay),
            "NTA_JIT_ONLY": os.environ.get(
                "NTA_JIT_ONLY",
                "batch_decode_kernel.cu,batch_prefill_paged_kernel_",
            ),
            "NTA_JIT_PHASE_SOURCE": os.environ.get(
                "NTA_JIT_PHASE_SOURCE",
                "batch_decode_kernel.cu,batch_prefill_paged_kernel_mask_0.cu",
            ),
        })
    if options.print_env:
        for name, value in environment.items():
            print(f"export {name}={shlex.quote(value)}")
        return 0
    command = options.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after -- or use --print-env")
    os.execvpe(command[0], command, {**os.environ, **environment})
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
