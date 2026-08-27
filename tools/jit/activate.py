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

try:
    from cuda_toolkit import nvcc_path, resolve_cuda_home
except ModuleNotFoundError:
    # Installed activation lives in ``bin`` while the JIT helpers live beside
    # the installed NTA data root.
    _installed_jit = (
        pathlib.Path(__file__).resolve().parent.parent
        / "share"
        / "nta"
        / "tools"
        / "jit"
    )
    if not (_installed_jit / "cuda_toolkit.py").is_file():
        raise
    sys.path.insert(0, str(_installed_jit))
    from cuda_toolkit import nvcc_path, resolve_cuda_home


def project_layout(
    script: pathlib.Path, configured_root: str | None
) -> tuple[pathlib.Path, pathlib.Path | None]:
    if configured_root:
        root = pathlib.Path(configured_root).expanduser().resolve()
        prefix = (
            root.parents[1]
            if root.name == "nta" and root.parent.name == "share"
            else None
        )
        return root, prefix
    source_root = script.parents[2]
    if (source_root / "include/nta/RuntimeABI.h").is_file():
        return source_root, None
    if script.parent.name == "bin":
        prefix = script.parent.parent
        root = prefix / "share" / "nta"
        if (root / "include/nta/RuntimeABI.h").is_file():
            return root, prefix
    raise RuntimeError(
        "cannot locate NTA headers; use --project-root or install the CMake "
        "runtime and JIT artifacts together"
    )


def first_file(candidates: list[pathlib.Path], description: str) -> pathlib.Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"{description} not found; checked {rendered}")


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
    parser.add_argument("--project-root")
    parser.add_argument("--plugin")
    parser.add_argument("--runtime-library")
    parser.add_argument("--clang", default="/usr/bin/clang++-22")
    parser.add_argument(
        "--cuda-path",
        help=(
            "CUDA toolkit root; defaults to NTA_CUDA_PATH/CUDA_HOME, then the "
            "installed framework CUDA ABI and finally the system nvcc"
        ),
    )
    parser.add_argument("--flashinfer-hook", action="store_true")
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    script = pathlib.Path(__file__).resolve()
    try:
        root, prefix = project_layout(
            script, options.project_root or os.environ.get("NTA_PROJECT_ROOT")
        )
    except RuntimeError as error:
        parser.error(str(error))
    build_value = pathlib.Path(options.build_dir).expanduser()
    build = (build_value if build_value.is_absolute() else root / build_value).resolve()
    plugin_candidates = []
    configured_plugin = options.plugin or os.environ.get("NTA_PLUGIN")
    if configured_plugin:
        plugin_candidates.append(pathlib.Path(configured_plugin).expanduser())
    plugin_candidates.append(build / "libNtaPass.so")
    if prefix is not None:
        plugin_candidates.append(prefix / "lib" / "nta" / "libNtaPass.so")
    try:
        plugin = first_file(plugin_candidates, "NTA pass plugin")
    except RuntimeError as error:
        parser.error(str(error))
    shim = root / "tools" / "jit" / "nvcc_clang.py"
    runtime_candidates = []
    configured_runtime = options.runtime_library or os.environ.get(
        "NTA_RUNTIME_LIBRARY"
    )
    if configured_runtime:
        runtime_candidates.append(pathlib.Path(configured_runtime).expanduser())
    runtime_candidates.append(build / "libnta-runtime.so")
    if prefix is not None:
        runtime_candidates.append(prefix / "lib" / "libnta-runtime.so")
    try:
        runtime_library = first_file(runtime_candidates, "NTA runtime library")
    except RuntimeError as error:
        parser.error(str(error))
    transport_candidates = []
    configured_transport = os.environ.get("NTA_TRANSPORT_PROGRAM")
    if configured_transport:
        transport_candidates.append(pathlib.Path(configured_transport).expanduser())
    transport_candidates.append(build / "libnta-transport-program.so")
    if prefix is not None:
        transport_candidates.append(prefix / "lib" / "libnta-transport-program.so")
    try:
        transport_program = first_file(
            transport_candidates, "NTA transport phase program"
        )
    except RuntimeError as error:
        parser.error(str(error))
    try:
        cuda_home = resolve_cuda_home(options.cuda_path)
        real_nvcc = first_file([nvcc_path(cuda_home)], "CUDA nvcc")
    except RuntimeError as error:
        parser.error(str(error))
    abi_header = root / "include/nta/RuntimeABI.h"
    abi_version = runtime_abi_version(abi_header)
    integration_inputs = [
        script,
        plugin,
        shim,
        root / "tools/jit/clang_cuda_prelude.h",
        abi_header,
        root / "include/nta/OperatorContract.h",
        root / "include/nta/DeviceAPI.cuh",
        root / "include/nta/KernelPolicy.cuh",
        root / "include/nta/FlashInferKernelPolicy.cuh",
        root / "runtime/device/Acquire.cuh",
        root / "runtime/device/JitRuntime.cuh",
        root / "runtime/device/TransportProgram.cu",
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
        integration_inputs.extend(
            [
                flashinfer_include / "flashinfer/attention/decode.cuh",
                flashinfer_include / "flashinfer/attention/prefill.cuh",
                root / "tools/flashinfer/prepare_overlay.py",
            ]
        )
    version_tag = f"-fi{flashinfer_version}" if flashinfer_version else ""
    tag = f"nta-abi{abi_version}{version_tag}-{fingerprint(integration_inputs)}"
    cache_root_base = pathlib.Path(
        options.cache_root
        or os.environ.get(
            "NTA_JIT_CACHE_ROOT", pathlib.Path.home() / ".cache/flashinfer"
        )
    ).expanduser()
    # Physical NVMe probes may be run through sudo while framework tests run
    # as the experiment user. A shared JIT cache lets the privileged run leave
    # root-owned CUDA objects in the normal user's build tree, turning a later
    # framework test into an unrelated permission failure. Keep the cache
    # content-addressed within each UID namespace.
    cache_root = cache_root_base / f"uid-{os.getuid()}"
    workspace = (cache_root / tag).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    environment = {
        "FLASHINFER_NVCC": str(shim),
        "FLASHINFER_WORKSPACE_BASE": str(workspace),
        "NTA_PROJECT_ROOT": str(root),
        "NTA_PLUGIN": str(plugin),
        "NTA_CLANG": options.clang,
        "NTA_CUDA_PATH": str(cuda_home),
        "NTA_REAL_NVCC": str(real_nvcc),
        "CUDA_HOME": str(cuda_home),
        "CUDA_PATH": str(cuda_home),
        "NTA_JIT_CACHE_TAG": tag,
        "NTA_ABI_VERSION": str(abi_version),
        "NTA_BUILD_DIR": str(build),
        "NTA_RUNTIME_LIBRARY": str(runtime_library),
        "NTA_TRANSPORT_PROGRAM": str(transport_program),
        "PYTHONPATH": os.pathsep.join(
            value
            for value in (
                str(root / "python"),
                str(root),
                os.environ.get("PYTHONPATH", ""),
            )
            if value
        ),
        "LD_LIBRARY_PATH": os.pathsep.join(
            value
            for value in (
                str(runtime_library.parent),
                os.environ.get("LD_LIBRARY_PATH", ""),
            )
            if value
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
                "--fast-reuse",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        environment.update(
            {
                "NTA_FLASHINFER_HOOK": "1",
                "NTA_FLASHINFER_OVERLAY": str(overlay),
                "NTA_JIT_ONLY": os.environ.get(
                    "NTA_JIT_ONLY",
                    "generated/",
                ),
                "NTA_JIT_PHASE_SOURCE": os.environ.get(
                    "NTA_JIT_PHASE_SOURCE",
                    "batch_decode_kernel.cu,batch_prefill_paged_kernel_mask_0.cu",
                ),
                "NTA_JIT_REQUEST_BOUND_SOURCE": os.environ.get(
                    "NTA_JIT_REQUEST_BOUND_SOURCE",
                    "nta_sglang_decode_request_bound,"
                    "nta_sglang_prefill_request_bound,"
                    "nta_batch_prefill_vllm_request_bound",
                ),
            }
        )
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
