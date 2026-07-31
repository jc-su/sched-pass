#!/usr/bin/env python3
"""Run a FlashInfer process with NTA's clang JIT and isolated cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shlex
import sys


def fingerprint(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--cache-root")
    parser.add_argument("--clang", default="/usr/bin/clang++-22")
    parser.add_argument("--cuda-path", default="/usr/local/cuda-12.9")
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[2]
    build = (root / options.build_dir).resolve()
    plugin = build / "libNtaPass.so"
    shim = root / "tools" / "jit" / "nvcc_clang.py"
    if not plugin.is_file():
        parser.error(f"pass plugin not found: {plugin}")
    integration_inputs = [
        plugin,
        shim,
        root / "tools/jit/clang_cuda_prelude.h",
        root / "include/nta/RuntimeABI.h",
        root / "include/nta/DeviceAPI.cuh",
        root / "include/nta/KernelPolicy.cuh",
        root / "runtime/device/Acquire.cuh",
    ]
    tag = f"nta-abi9-{fingerprint(integration_inputs)}"
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
    }
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
