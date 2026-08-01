#!/usr/bin/env python3
"""Prepare a source-checked FlashInfer include overlay with NTA CTA hooks."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import shutil


SUPPORTED_VERSION = "0.6.12"
EXPECTED_TREE_HASH = (
    "71b994241bb85a3e3d5d0e40ac02d3c5652a07bbc4afaa8627adcb1914bb8b1c"
)
EXPECTED_HASHES = {
    "attention/decode.cuh": (
        "019d673aa848a938798a2c58b34b9b5813a3f137962cbbd90ef7bd71f636f373"
    ),
    "attention/prefill.cuh": (
        "0620738f4c1f3e64fd1713cec863f3ae67ab2bdbf2f4002d9e1a1bf6d69ea737"
    ),
}

POLICY_INCLUDE_ANCHOR = "#include <cooperative_groups.h>"
POLICY_INCLUDE = '#include "nta/FlashInferKernelPolicy.cuh"\n'

DECODE_ANCHOR = """__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
"""
DECODE_REPLACEMENT = """__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    const uint32_t nta_work_index = blockIdx.x;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (!nta::flashinfer::validWork(params, nta_work_index, nta_request_index)) return;
    auto* nta_runtime = nta::flashinfer::runtime(params);
    nta::kernel::WorkContext nta_work{};
    if (!nta::kernel::acquireWork(nta_runtime, nta::flashinfer::workItems(params),
                                  nta::flashinfer::dependencies(params), nta_work_index,
                                  nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(nta_runtime, nta_work)) return;
  }
  extern __shared__ uint8_t smem[];
"""

MLA_DECODE_ANCHOR = """__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
"""
MLA_DECODE_REPLACEMENT = """__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    const uint32_t nta_work_index = blockIdx.x;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (!nta::flashinfer::validWork(params, nta_work_index, nta_request_index)) return;
    auto* nta_runtime = nta::flashinfer::runtime(params);
    nta::kernel::WorkContext nta_work{};
    if (!nta::kernel::acquireWork(nta_runtime, nta::flashinfer::workItems(params),
                                  nta::flashinfer::dependencies(params), nta_work_index,
                                  nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(nta_runtime, nta_work)) return;
  }
  auto block = cg::this_thread_block();
"""

PAGED_PREFILL_ANCHOR = """__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
"""
PAGED_PREFILL_REPLACEMENT = """__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  if constexpr (nta::flashinfer::HasWorkPlanV<Params>) {
    const uint32_t nta_work_index = blockIdx.x;
    if (params.block_valid_mask && !params.block_valid_mask[nta_work_index]) return;
    const uint32_t nta_request_index = params.request_indices[nta_work_index];
    if (!nta::flashinfer::validWork(params, nta_work_index, nta_request_index)) return;
    auto* nta_runtime = nta::flashinfer::runtime(params);
    nta::kernel::WorkContext nta_work{};
    if (!nta::kernel::acquireWork(nta_runtime, nta::flashinfer::workItems(params),
                                  nta::flashinfer::dependencies(params), nta_work_index,
                                  nta_work)) {
      nta::kernel::defer(nta_runtime, nta_work);
      return;
    }
    if (!nta::flashinfer::shouldRun(nta_runtime, nta_work)) return;
  }
  extern __shared__ uint8_t smem[];
"""


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def flashinfer_include() -> tuple[str, pathlib.Path]:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or spec.origin is None:
        raise RuntimeError("flashinfer-python is not installed")
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("flashinfer-python distribution metadata is missing") from error
    include = pathlib.Path(spec.origin).resolve().parent / "data" / "include"
    if not (include / "flashinfer").is_dir():
        raise RuntimeError(f"FlashInfer include tree is missing: {include}")
    return version, include


def checked_replace(source: str, anchor: str, replacement: str,
                    description: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {description} anchor, found {count}"
        )
    return source.replace(anchor, replacement)


def patch_header(path: pathlib.Path, replacements: list[tuple[str, str, str]]) -> None:
    source = path.read_text(encoding="utf-8")
    source = checked_replace(
        source,
        POLICY_INCLUDE_ANCHOR,
        POLICY_INCLUDE + POLICY_INCLUDE_ANCHOR,
        "policy include",
    )
    for anchor, replacement, description in replacements:
        source = checked_replace(source, anchor, replacement, description)
    path.write_text(source, encoding="utf-8")


def validate_existing(
    output: pathlib.Path, expected: dict[str, object]
) -> dict[str, object] | None:
    manifest_path = output / "manifest.json"
    if not output.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid existing FlashInfer overlay: {output}") from error
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise RuntimeError(f"stale existing FlashInfer overlay: {output}")
    overlay_hashes = manifest.get("overlay_hashes")
    overlay_tree_hash = manifest.get("overlay_tree_hash")
    if (
        not isinstance(overlay_hashes, dict)
        or set(overlay_hashes) != set(EXPECTED_HASHES)
        or not isinstance(overlay_tree_hash, str)
        or tree_hash(output / "flashinfer") != overlay_tree_hash
        or any(
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or not (output / "flashinfer" / relative).is_file()
            or sha256(output / "flashinfer" / relative) != expected_hash
            for relative, expected_hash in overlay_hashes.items()
        )
    ):
        raise RuntimeError(f"corrupt existing FlashInfer overlay: {output}")
    return manifest


def prepare_locked(output: pathlib.Path) -> dict[str, object]:
    version, include = flashinfer_include()
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            f"unsupported FlashInfer {version}; expected {SUPPORTED_VERSION}"
        )
    observed = {
        relative: sha256(include / "flashinfer" / relative)
        for relative in EXPECTED_HASHES
    }
    observed_tree_hash = tree_hash(include / "flashinfer")
    if observed != EXPECTED_HASHES or observed_tree_hash != EXPECTED_TREE_HASH:
        mismatches = [
            relative for relative, expected in EXPECTED_HASHES.items()
            if observed[relative] != expected
        ]
        if observed_tree_hash != EXPECTED_TREE_HASH:
            mismatches.append("complete include tree")
        raise RuntimeError(
            "FlashInfer headers differ from the validated 0.6.12 sources: "
            + ", ".join(mismatches)
        )

    manifest: dict[str, object] = {
        "flashinfer_version": version,
        "source_include": str(include),
        "source_hashes": observed,
        "source_tree_hash": observed_tree_hash,
        "hooks": ["batch-decode", "mla-decode", "paged-prefill-fa2"],
    }
    existing = validate_existing(output, manifest)
    if existing is not None:
        return existing

    destination = output / "flashinfer"
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    shutil.copytree(include / "flashinfer", temporary / "flashinfer")
    patch_header(
        temporary / "flashinfer/attention/decode.cuh",
        [
            (DECODE_ANCHOR, DECODE_REPLACEMENT, "batch decode"),
            (MLA_DECODE_ANCHOR, MLA_DECODE_REPLACEMENT, "MLA decode"),
        ],
    )
    patch_header(
        temporary / "flashinfer/attention/prefill.cuh",
        [(PAGED_PREFILL_ANCHOR, PAGED_PREFILL_REPLACEMENT, "paged prefill")],
    )
    manifest["overlay_hashes"] = {
        relative: sha256(temporary / "flashinfer" / relative)
        for relative in EXPECTED_HASHES
    }
    manifest["overlay_tree_hash"] = tree_hash(temporary / "flashinfer")
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.rename(output)
    if not destination.is_dir():
        raise RuntimeError(f"failed to prepare FlashInfer overlay: {destination}")
    return manifest


def prepare(output: pathlib.Path) -> dict[str, object]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return prepare_locked(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    options = parser.parse_args()
    print(json.dumps(prepare(options.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
