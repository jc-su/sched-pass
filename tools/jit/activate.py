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
import shutil
import subprocess
import sys

try:
    from cuda_toolkit import nvcc_path, resolve_cuda_home
except ModuleNotFoundError:
    # A source-tree module import does not automatically add this script's
    # directory to sys.path; installed activation instead lives in ``bin``.
    _script_jit = pathlib.Path(__file__).resolve().parent
    _installed_jit = _script_jit.parent / "share" / "nta" / "tools" / "jit"
    _jit_helpers = (
        _script_jit
        if (_script_jit / "cuda_toolkit.py").is_file()
        else _installed_jit
    )
    if not (_jit_helpers / "cuda_toolkit.py").is_file():
        raise
    sys.path.insert(0, str(_jit_helpers))
    from cuda_toolkit import nvcc_path, resolve_cuda_home


NUMERICAL_CACHE_SCHEMA = "nta-numerical-v1"
DEFAULT_JIT_ONLY = "generated/"
DEFAULT_METADATA_SOURCE = (
    "batch_decode_kernel.cu,batch_prefill_paged_kernel_mask_0.cu"
)
DEFAULT_REQUEST_BOUND_SOURCE = (
    "nta_sglang_decode_request_bound,"
    "nta_sglang_prefill_request_bound,"
    "nta_batch_prefill_vllm_request_bound"
)
DEFAULT_STREAM_ORDERED_SOURCE = (
    "nta_sglang_decode_stream_ordered,"
    "nta_sglang_prefill_stream_ordered,"
    "nta_sglang_prefill_demand_acquire_tier_v4_"
)


def numerical_fingerprint_inputs(
    root: pathlib.Path,
    plugin: pathlib.Path,
    shim: pathlib.Path,
    abi_header: pathlib.Path,
) -> list[pathlib.Path]:
    """Return only files that can change a typed numerical artifact."""

    return [
        plugin,
        shim,
        root / "tools/jit/clang_cuda_prelude.h",
        abi_header,
        root / "include/nta/OperatorContract.h",
        root / "include/nta/TicketProtocol.cuh",
        root / "include/nta/DeviceAPI.cuh",
        root / "include/nta/KernelPolicy.cuh",
        root / "include/nta/FlashInferKernelPolicy.cuh",
        root / "runtime/device/TypedInstrumentation.cuh",
        root / "runtime/device/Acquire.cuh",
        root / "runtime/device/OperatorMetadata.cuh",
    ]


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


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_path(value: str, description: str) -> pathlib.Path:
    candidate = pathlib.Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(value)
    if resolved is None:
        raise RuntimeError(f"{description} not found: {value}")
    return pathlib.Path(resolved).resolve()


def tool_identity(executable: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot query toolchain identity: {executable}") from error
    return f"{executable}\n{result.stdout.strip()}"


def fingerprint(paths: list[pathlib.Path], identities: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    for index, path in enumerate(paths):
        if not path.is_file():
            raise RuntimeError(f"JIT fingerprint input is missing: {path}")
        digest.update(index.to_bytes(8, "little"))
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    for identity in identities or ():
        digest.update(b"\0tool\0")
        digest.update(identity.encode("utf-8"))
    return digest.hexdigest()[:24]


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
        clang = executable_path(options.clang, "Clang CUDA compiler")
        toolchain_identities = [
            tool_identity(clang),
            tool_identity(real_nvcc),
            f"cuda_home={cuda_home.resolve()}",
        ]
    except RuntimeError as error:
        parser.error(str(error))
    abi_header = root / "include/nta/RuntimeABI.h"
    abi_version = runtime_abi_version(abi_header)
    # Only inputs capable of changing the numerical operator or its verified
    # contract belong to the FlashInfer workspace identity. Runtime transport
    # and host-library artifacts carry independent ABI/content checks below;
    # hashing them here would force a 95+ second attention rebuild after an
    # unrelated copy/progress change.
    numerical_inputs = numerical_fingerprint_inputs(root, plugin, shim, abi_header)
    jit_only = os.environ.get("NTA_JIT_ONLY", DEFAULT_JIT_ONLY)
    metadata_source = os.environ.get(
        "NTA_JIT_METADATA_SOURCE", DEFAULT_METADATA_SOURCE
    )
    request_bound_source = os.environ.get(
        "NTA_JIT_REQUEST_BOUND_SOURCE", DEFAULT_REQUEST_BOUND_SOURCE
    )
    stream_ordered_source = os.environ.get(
        "NTA_JIT_STREAM_ORDERED_SOURCE", DEFAULT_STREAM_ORDERED_SOURCE
    )
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
        numerical_inputs.append(root / "tools/flashinfer/prepare_overlay.py")
        numerical_inputs.extend(
            sorted(
                path
                for path in (flashinfer_include / "flashinfer").rglob("*")
                if path.is_file() and path.suffix in {".cuh", ".h", ".hpp"}
            )
        )
    version_tag = f"-fi{flashinfer_version}" if flashinfer_version else ""
    numerical_identities = [
        *toolchain_identities,
        f"cache_schema={NUMERICAL_CACHE_SCHEMA}",
        f"jit_only={jit_only}",
        f"metadata_source={metadata_source}",
        f"request_bound_source={request_bound_source}",
        f"stream_ordered_source={stream_ordered_source}",
        f"strip_arch={os.environ.get('NTA_STRIP_ARCH', '')}",
        f"staging_streaming={os.environ.get('NTA_STAGING_STREAMING', '')}",
    ]
    source_fingerprint = fingerprint(numerical_inputs, numerical_identities)
    policy_tag = "-stream" if os.environ.get("NTA_STAGING_STREAMING") == "1" else ""
    tag = f"nta-abi{abi_version}{version_tag}{policy_tag}-{source_fingerprint}"
    transport_digest = file_sha256(transport_program)
    toolchain_digest = hashlib.sha256(
        "\n\0\n".join(toolchain_identities).encode("utf-8")
    ).hexdigest()
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
        "NTA_CLANG": str(clang),
        "NTA_CUDA_PATH": str(cuda_home),
        "NTA_REAL_NVCC": str(real_nvcc),
        "CUDA_HOME": str(cuda_home),
        "CUDA_PATH": str(cuda_home),
        "NTA_JIT_CACHE_TAG": tag,
        "NTA_ABI_VERSION": str(abi_version),
        "NTA_BUILD_DIR": str(build),
        "NTA_RUNTIME_LIBRARY": str(runtime_library),
        "NTA_TRANSPORT_PROGRAM": str(transport_program),
        "NTA_TRANSPORT_PROGRAM_SHA256": transport_digest,
        "NTA_JIT_TOOLCHAIN_SHA256": toolchain_digest,
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
                "NTA_JIT_ONLY": jit_only,
                "NTA_JIT_METADATA_SOURCE": metadata_source,
                "NTA_JIT_REQUEST_BOUND_SOURCE": request_bound_source,
                "NTA_JIT_STREAM_ORDERED_SOURCE": stream_ordered_source,
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
