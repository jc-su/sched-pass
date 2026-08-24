#!/usr/bin/env python3
"""Translate an nvcc-style JIT command to clang CUDA and load NtaPass."""

from __future__ import annotations

import hashlib
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


def operator_family(arguments: list[str]) -> int:
    """Return the semantic family of the generated operator module.

    FlashInfer's tensor-core decode module is implemented with paged-prefill
    translation units, so a source filename cannot identify the public
    operator. The generated module directory is present in every compile
    command and remains stable across those internal implementation choices.
    """
    command = " ".join(arguments)
    # The baseline is deliberately compiled through the same launcher for a
    # fair differential measurement, but it has no NTA typed contract.  Its
    # module name still contains the public FlashInfer family token, so this
    # guard must precede family recognition.
    if "_baseline" in command:
        return 0
    module_tokens = (
        (("nta_sglang_decode_", "nta_batch_decode_"), 1),
        (("nta_sglang_prefill_", "nta_batch_prefill_"), 2),
    )
    families = {
        family
        for tokens, family in module_tokens
        if any(token in command for token in tokens)
    }
    if len(families) > 1:
        raise RuntimeError("JIT command names multiple NTA operator families")
    if families:
        return families.pop()

    # Standalone generic integrations do not necessarily use an NTA module
    # name. Preserve source-based identification only for those modules.
    if "batch_decode_kernel.cu" in command:
        return 1
    if "batch_prefill_paged_kernel" in command:
        return 2
    return 0


def has_typed_operator_kernel_source(arguments: list[str]) -> bool:
    """Identify translation units patched with the FlashInfer CTA hooks.

    A generated FlashInfer module contains binding/helper translation units in
    addition to the actual paged attention kernels.  The module-level name is
    present in all of them, but only the kernel sources contain typed NTA
    marker calls.  Contract constants must therefore not be injected into
    helpers; doing so makes the LLVM pass reject an otherwise ordinary helper
    TU for having no acquisition marker.
    """
    command = " ".join(arguments)
    return any(
        token in command
        for token in (
            "batch_decode_kernel.cu",
            "batch_prefill_paged_kernel",
            "batch_prefill_ragged_kernel",
        )
    )


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
            request_bound = matches_filter(
                arguments, "NTA_JIT_REQUEST_BOUND_SOURCE"
            )
            # JitRuntime is emitted by the one source selected by
            # NTA_JIT_PHASE_SOURCE.  Request-bound is a module-wide compile
            # variant, but adding JitRuntime to every request-bound helper TU
            # would create duplicate exported phase symbols at link time.
            # The activate launcher supplies a stable phase-source selector;
            # standalone workers must use that launcher as well.
            phase_runtime = phase_source
            stream_ordered_direct = (
                "nta_sglang_prefill_demand_acquire_tier_v4_"
                in " ".join(arguments)
            )
            family = operator_family(arguments)
            typed_module = (
                family in (1, 2)
                and has_typed_operator_kernel_source(arguments)
                and any(
                    token in " ".join(arguments)
                    for token in ("nta_batch_", "nta_sglang_")
                )
            )
            operator_form = 1 if request_bound else 2
            direct_capabilities = (1 << 0) | (1 << 6) | (1 << 7)
            incremental_capabilities = sum(1 << bit for bit in range(8))
            capabilities = (
                direct_capabilities if request_bound else incremental_capabilities
            )
            fingerprint = hashlib.sha256(
                os.environ.get("NTA_JIT_CACHE_TAG", "").encode("utf-8")
            ).digest()
            hash_low = int.from_bytes(fingerprint[:8], "little")
            hash_high = int.from_bytes(fingerprint[8:16], "little")
            coordinate_map = 1 if family in (1, 2) else 0
            partial_state = 1 if family in (1, 2) else 0
            reduction = 1 if family in (1, 2) else 0
            plan_flags = 0x1F if family in (1, 2) else 0x9
            instrumentation_flags = 0xF if typed_module else 0
            identity_binding = 1 if typed_module else 0
            demand_binding = 1 if typed_module else 0
            access_proof = 3 if typed_module else 0
            tier_mask = 0x3F if typed_module else 0
            plan_fingerprint = hashlib.sha256(
                (
                    f"{os.environ.get('NTA_JIT_CACHE_TAG', '')}|{family}|6|"
                    f"{coordinate_map}|{partial_state}|{reduction}|{plan_flags}|"
                    f"{instrumentation_flags}|{identity_binding}|{demand_binding}|"
                    f"{access_proof}|{tier_mask}"
                ).encode("utf-8")
            ).digest()
            plan_hash_low = int.from_bytes(plan_fingerprint[:8], "little")
            plan_hash_high = int.from_bytes(plan_fingerprint[8:16], "little")
            command.extend([
                f"-DNTA_DEVICE_PHASE_KERNELS={1 if phase_runtime else 0}",
                f"-DNTA_TYPED_OPERATOR_CONTRACT={1 if typed_module else 0}",
                f"-DNTA_FLASHINFER_REQUEST_BOUND={1 if request_bound else 0}",
                "-DNTA_FLASHINFER_PREACQUIRED_ONLY=0",
                "-DNTA_FLASHINFER_STREAM_ORDERED_DIRECT="
                f"{1 if stream_ordered_direct else 0}",
                f"-DNTA_OPERATOR_FAMILY={family}",
                f"-DNTA_OPERATOR_FORM={operator_form}",
                f"-DNTA_OPERATOR_CAPABILITIES={capabilities}ULL",
                f"-DNTA_OPERATOR_SOURCE_HASH_LOW={hash_low}ULL",
                f"-DNTA_OPERATOR_SOURCE_HASH_HIGH={hash_high}ULL",
                "-DNTA_OPERATOR_SUPPORTED_FORMS=6U",
                f"-DNTA_OPERATOR_COORDINATE_MAP={coordinate_map}U",
                f"-DNTA_OPERATOR_PARTIAL_STATE={partial_state}U",
                f"-DNTA_OPERATOR_REDUCTION={reduction}U",
                f"-DNTA_OPERATOR_PLAN_FLAGS={plan_flags}U",
                f"-DNTA_OPERATOR_PLAN_HASH_LOW={plan_hash_low}ULL",
                f"-DNTA_OPERATOR_PLAN_HASH_HIGH={plan_hash_high}ULL",
                f"-DNTA_OPERATOR_INSTRUMENTATION_FLAGS={instrumentation_flags}ULL",
                f"-DNTA_OPERATOR_IDENTITY_BINDING={identity_binding}U",
                f"-DNTA_OPERATOR_DEMAND_BINDING={demand_binding}U",
                f"-DNTA_OPERATOR_ACCESS_PROOF={access_proof}U",
                "-DNTA_OPERATOR_GRANULARITY_BYTES=0U",
                f"-DNTA_OPERATOR_TIER_MASK={tier_mask}ULL",
                "-include",
                str(ROOT / "runtime/device/TypedInstrumentation.cuh"),
                "-include",
                str(
                    ROOT / (
                        "runtime/device/JitRuntime.cuh"
                        if phase_runtime else "runtime/device/Acquire.cuh"
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
            compatibility_launcher = ROOT / "tools" / "flashinfer" / "nvcc_compat.py"
            if compatibility_launcher.is_file():
                os.execv(
                    compatibility_launcher,
                    [str(compatibility_launcher), *arguments],
                )
            os.execv(real_nvcc, [real_nvcc, *arguments])

    cache_tag = os.environ.get("NTA_JIT_CACHE_TAG", "")
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if instrument and cache_tag and cache_tag not in workspace:
        raise RuntimeError(
            "instrumented FlashInfer JIT requires an NTA-tagged workspace")
    if instrument and os.environ.get("NTA_STAGING_STREAMING") == "1":
        # Compile-time cache policy forks the kernel bytes; refuse to bake
        # it into a workspace that does not carry the marker, or a toggled
        # env would silently reuse the other policy's cached kernels.
        if "stream" not in workspace:
            raise RuntimeError(
                "NTA_STAGING_STREAMING requires a workspace tagged with "
                "'stream' (fresh cache), not a policy-unmarked cache")

    command = translate(arguments, instrument)
    if instrument and os.environ.get("NTA_STAGING_STREAMING") == "1":
        command.append("-DNTA_STAGING_STREAMING=1")
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
