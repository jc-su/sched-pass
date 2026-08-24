#!/usr/bin/env python3
"""Isolated reproducer for the per-step bounded indexed copy seam.

Registers an engine-shaped K/V indexed object pair over fully controlled
buffers, rewrites the shared index arrays' prefix, bounds the copy with
nta_jit_set_indexed_row_counts, runs the validated indexed progress, and
asserts the staged rows equal their host sources byte for byte while
untouched rows stay untouched. Discriminates a transport seam defect from
an engine-tensor-semantics defect.

Requires a compiled phase module; pass it explicitly or via
NTA_PHASE_MODULE.
"""

from __future__ import annotations

import os
import pathlib
import re

import torch

from nta_runtime.runtime import (
    IndexedHostObject,
    JitPhaseProgram,
    Runtime,
    RuntimeConfig,
)

ROWS = 64
HEADS = 2
DIM = 128


def _runtime_abi_version() -> int:
    configured = os.environ.get("NTA_ABI_VERSION")
    if configured:
        return int(configured)
    header = pathlib.Path(__file__).resolve().parents[2] / "include/nta/RuntimeABI.h"
    match = re.search(
        r"inline constexpr std::uint32_t Version = (\d+);",
        header.read_text(encoding="utf-8"),
    )
    if match is None:
        raise RuntimeError(f"cannot read the NTA ABI version from {header}")
    return int(match.group(1))


def locate_module() -> pathlib.Path | None:
    configured = os.environ.get("NTA_PHASE_MODULE")
    if configured:
        path = pathlib.Path(configured)
        if not path.exists():
            raise RuntimeError(f"NTA_PHASE_MODULE does not exist: {path}")
        return path
    roots = []
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if workspace:
        roots.append(pathlib.Path(workspace))
    roots.extend(
        (
            pathlib.Path.cwd() / "flashinfer-jit-cache",
            pathlib.Path.home() / ".cache/flashinfer",
        )
    )
    abi_prefix = f"nta-abi{_runtime_abi_version()}-"
    for root in roots:
        compatible_roots = (
            [root]
            if root.name.startswith(abi_prefix)
            else sorted(
                (candidate for candidate in root.glob(f"{abi_prefix}*") if candidate.is_dir()),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            )
        )
        for compatible_root in compatible_roots:
            for pattern in (
                "nta_batch_decode_default_v2_hooked.so",
                "nta_sglang_decode_demand*.so",
            ):
                candidates = sorted(
                    compatible_root.rglob(pattern),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    return candidates[0]
    return None


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; seam reproducer skipped")
        return 0
    module = locate_module()
    if module is None:
        print("no compiled phase module in the cache; seam reproducer skipped")
        return 0
    # Phase modules link against TVM-FFI, whose symbols the serving process
    # provides by importing flashinfer before loading any module.
    import flashinfer  # noqa: F401
    phases = JitPhaseProgram(module)
    runtime = Runtime(
        RuntimeConfig(
            request_capacity=4,
            object_capacity=8,
            intent_capacity=8,
            work_ticket_capacity=8,
            max_dependencies_per_work_ticket=1,
        )
    )

    def patterned(offset: float) -> torch.Tensor:
        base = torch.arange(
            ROWS * HEADS * DIM, dtype=torch.float32
        ).view(ROWS, HEADS, DIM)
        return (base * 1e-3 + offset).to(torch.float16)

    host_k = patterned(1.0).pin_memory()
    host_v = patterned(2.0).pin_memory()
    device_k = torch.zeros(
        (ROWS, HEADS, DIM), dtype=torch.float16, device="cuda"
    )
    device_v = torch.zeros_like(device_k)

    capacity = 16
    source_index = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    staging_index = torch.zeros(capacity, dtype=torch.int32, device="cuda")

    element = HEADS * DIM * 2
    objects = []
    for source, staging in ((host_k, device_k), (host_v, device_v)):
        objects.append(
            IndexedHostObject(
                0x5345414D_00000000 + len(objects),
                1,
                source.data_ptr(),
                staging.data_ptr(),
                source_index.data_ptr(),
                staging_index.data_ptr(),
                capacity,
                element,
                source.stride(0) * source.element_size(),
                staging.stride(0) * staging.element_size(),
                ROWS,
                ROWS,
            )
        )
    stream = torch.cuda.current_stream()
    runtime.register_indexed_host_objects(0, objects, stream=stream)

    # Step one: a scattered subset, deliberately unordered.
    source_rows = [3, 41, 7, 22, 58]
    staging_rows = [10, 4, 33, 60, 21]
    count = len(source_rows)
    source_index[:count] = torch.tensor(
        source_rows, dtype=torch.int32, device="cuda"
    )
    staging_index[:count] = torch.tensor(
        staging_rows, dtype=torch.int32, device="cuda"
    )
    phases.set_indexed_row_counts(runtime, 0, 2, count, stream=stream)
    phases.progress_validated_indexed_host_range(runtime, 0, 2, stream=stream)
    torch.cuda.synchronize()

    failures = 0
    for name, host, device_buffer in (
        ("K", host_k, device_k),
        ("V", host_v, device_v),
    ):
        staged = device_buffer[
            torch.tensor(staging_rows, dtype=torch.long, device="cuda")
        ].cpu()
        expected = host[torch.tensor(source_rows, dtype=torch.long)]
        if not torch.equal(staged, expected):
            mismatch = (
                staged.view(count, -1) != expected.view(count, -1)
            ).any(dim=1)
            first = int(mismatch.nonzero()[0])
            print(
                f"{name}: DIVERGED at pair {first} "
                f"(src {source_rows[first]} -> dst {staging_rows[first]}): "
                f"staged head {staged.view(count, -1)[first, :4].tolist()} "
                f"expected {expected.view(count, -1)[first, :4].tolist()}"
            )
            failures += 1
        untouched = [r for r in range(ROWS) if r not in staging_rows]
        touched_extra = device_buffer[
            torch.tensor(untouched, dtype=torch.long, device="cuda")
        ]
        if not bool((touched_extra == 0).all()):
            print(f"{name}: untouched destination rows were modified")
            failures += 1

    # Step two: shrink the count and rewrite the prefix; only the new prefix
    # may copy.
    device_k.zero_()
    device_v.zero_()
    source_index[:2] = torch.tensor([11, 12], dtype=torch.int32, device="cuda")
    staging_index[:2] = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    phases.set_indexed_row_counts(runtime, 0, 2, 2, stream=stream)
    phases.progress_validated_indexed_host_range(runtime, 0, 2, stream=stream)
    torch.cuda.synchronize()
    if not torch.equal(device_k[1:3].cpu(), host_k[11:13]):
        print("K: shrunken-count prefix copy diverged")
        failures += 1
    if not bool((device_k[3:] == 0).all()) or not bool(
        (device_k[0] == 0).all()
    ):
        print("K: shrunken count copied beyond its prefix")
        failures += 1

    # Fused mapped-host summary construction reads the pinned source directly
    # and emits only fp32 envelopes in HBM.
    summary_first = 5
    summary_tokens = 19
    summary_page_tokens = 4
    summary_pages = (
        summary_tokens + summary_page_tokens - 1
    ) // summary_page_tokens
    for dtype in (torch.float16, torch.bfloat16):
        summary_source = patterned(3.0).to(dtype).pin_memory()
        summary_min = torch.empty(
            (summary_pages, HEADS, DIM), dtype=torch.float32, device="cuda"
        )
        summary_max = torch.empty_like(summary_min)
        phases.reduce_mapped_key_pages(
            summary_source,
            summary_first,
            summary_tokens,
            summary_page_tokens,
            summary_min,
            summary_max,
            stream=stream,
        )
        torch.cuda.synchronize()
        expected_min = []
        expected_max = []
        selected = summary_source[
            summary_first : summary_first + summary_tokens
        ].float()
        for begin in range(0, summary_tokens, summary_page_tokens):
            page = selected[begin : begin + summary_page_tokens]
            expected_min.append(page.amin(dim=0))
            expected_max.append(page.amax(dim=0))
        if not torch.equal(summary_min.cpu(), torch.stack(expected_min)) or not (
            torch.equal(summary_max.cpu(), torch.stack(expected_max))
        ):
            print(f"mapped summary: {dtype} envelope reduction diverged")
            failures += 1

    # Exact sparse path: page identities, hit filtering, and transfer-list
    # construction, and row-count publication remain on the CUDA stream.
    device_k.zero_()
    device_v.zero_()
    token_count = 32
    page_tokens = 4
    logical_host_rows = torch.arange(
        token_count, dtype=torch.int32, device="cuda"
    )
    logical_device_rows = torch.arange(
        token_count, dtype=torch.int32, device="cuda"
    )
    staged_pages = torch.zeros(
        token_count // page_tokens, dtype=torch.int32, device="cuda"
    )
    copied_rows = torch.zeros(1, dtype=torch.int64, device="cuda")
    selected_pages = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    phases.prepare_selected_indexed_rows(
        runtime,
        0,
        2,
        selected_pages,
        page_tokens,
        token_count,
        logical_host_rows,
        logical_device_rows,
        staged_pages,
        source_index,
        staging_index,
        copied_rows,
        stream=stream,
    )
    phases.progress_validated_indexed_host_range(runtime, 0, 2, stream=stream)
    torch.cuda.synchronize()
    selected_rows = torch.tensor(
        [4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.long, device="cuda"
    )
    if not torch.equal(device_k[selected_rows].cpu(), host_k[selected_rows.cpu()]):
        print("device selection: compacted page copy diverged")
        failures += 1
    if int(copied_rows) != 8:
        print("device selection: copied-row accounting is not exact")
        failures += 1

    # Re-selecting a resident page is a no-copy hit. Adding one page copies
    # only that page, while preserving deterministic selected order.
    phases.prepare_selected_indexed_rows(
        runtime,
        0,
        2,
        torch.tensor([3, 5], dtype=torch.int64, device="cuda"),
        page_tokens,
        token_count,
        logical_host_rows,
        logical_device_rows,
        staged_pages,
        source_index,
        staging_index,
        copied_rows,
        stream=stream,
    )
    phases.progress_validated_indexed_host_range(runtime, 0, 2, stream=stream)
    torch.cuda.synchronize()
    if int(copied_rows) != 12 or not torch.equal(
        device_k[20:24].cpu(), host_k[20:24]
    ):
        print("device selection: hit filtering or incremental copy diverged")
        failures += 1

    # External-prefix path: logical pages occupy a smaller physical cache.
    # The phase emits the exact FlashInfer table and rewrites the indexed
    # transfer prefix with misses only. Re-selection must preserve hits while
    # replacing a page that is outside the new selected set.
    device_k.zero_()
    device_v.zero_()
    bounded_tokens = 30
    bounded_capacity = 16
    bounded_host_rows = torch.arange(
        bounded_tokens, dtype=torch.int32, device="cuda"
    )
    bounded_device_rows = torch.arange(
        40,
        40 + bounded_capacity,
        dtype=torch.int32,
        device="cuda",
    )
    cached_pages = torch.full(
        (bounded_capacity // page_tokens,),
        -1,
        dtype=torch.int64,
        device="cuda",
    )
    selected_rows = torch.empty(
        bounded_capacity, dtype=torch.int32, device="cuda"
    )
    copied_rows.zero_()
    first_selection = torch.tensor(
        [1, 3, 7], dtype=torch.int64, device="cuda"
    )
    phases.prepare_bounded_selected_indexed_rows(
        runtime,
        0,
        2,
        first_selection,
        page_tokens,
        bounded_tokens,
        bounded_host_rows,
        bounded_device_rows,
        cached_pages,
        selected_rows,
        source_index,
        staging_index,
        copied_rows,
        stream=stream,
    )
    phases.progress_validated_indexed_host_range(
        runtime, 0, 2, stream=stream
    )
    torch.cuda.synchronize()
    first_physical = torch.tensor(
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        dtype=torch.long,
        device="cuda",
    )
    first_source = torch.tensor(
        [4, 5, 6, 7, 12, 13, 14, 15, 28, 29], dtype=torch.long
    )
    if int(copied_rows) != 10 or not torch.equal(
        selected_rows[:10].to(torch.long), first_physical
    ) or not torch.equal(device_k[first_physical].cpu(), host_k[first_source]):
        print("bounded selection: initial placement or copy diverged")
        failures += 1

    second_selection = torch.tensor(
        [1, 5, 7], dtype=torch.int64, device="cuda"
    )
    phases.prepare_bounded_selected_indexed_rows(
        runtime,
        0,
        2,
        second_selection,
        page_tokens,
        bounded_tokens,
        bounded_host_rows,
        bounded_device_rows,
        cached_pages,
        selected_rows,
        source_index,
        staging_index,
        copied_rows,
        stream=stream,
    )
    phases.progress_validated_indexed_host_range(
        runtime, 0, 2, stream=stream
    )
    torch.cuda.synchronize()
    second_source = torch.tensor(
        [4, 5, 6, 7, 20, 21, 22, 23, 28, 29], dtype=torch.long
    )
    if int(copied_rows) != 14 or not torch.equal(
        selected_rows[:10].to(torch.long), first_physical
    ) or not torch.equal(device_k[first_physical].cpu(), host_k[second_source]):
        print("bounded selection: replacement copied hits or wrong rows")
        failures += 1

    phases.prepare_bounded_selected_indexed_rows(
        runtime,
        0,
        2,
        second_selection,
        page_tokens,
        bounded_tokens,
        bounded_host_rows,
        bounded_device_rows,
        cached_pages,
        selected_rows,
        source_index,
        staging_index,
        copied_rows,
        stream=stream,
    )
    phases.progress_validated_indexed_host_range(
        runtime, 0, 2, stream=stream
    )
    torch.cuda.synchronize()
    if int(copied_rows) != 14:
        print("bounded selection: all-hit replay performed a copy")
        failures += 1

    for name, pages in (
        ("bounded duplicate page", [2, 2]),
        ("bounded out-of-range page", [bounded_tokens // page_tokens + 1]),
    ):
        sticky_before = runtime.sticky_failed_count
        phases.prepare_bounded_selected_indexed_rows(
            runtime,
            0,
            2,
            torch.tensor(pages, dtype=torch.int64, device="cuda"),
            page_tokens,
            bounded_tokens,
            bounded_host_rows,
            bounded_device_rows,
            cached_pages,
            selected_rows,
            source_index,
            staging_index,
            copied_rows,
            stream=stream,
        )
        torch.cuda.synchronize()
        if runtime.sticky_failed_count <= sticky_before:
            print(f"bounded selection: {name} did not fail closed")
            failures += 1

    for name, pages in (
        ("duplicate page", [2, 2]),
        ("out-of-range page", [token_count // page_tokens]),
    ):
        sticky_before = runtime.sticky_failed_count
        phases.prepare_selected_indexed_rows(
            runtime,
            0,
            2,
            torch.tensor(pages, dtype=torch.int64, device="cuda"),
            page_tokens,
            token_count,
            logical_host_rows,
            logical_device_rows,
            staged_pages,
            source_index,
            staging_index,
            copied_rows,
            stream=stream,
        )
        torch.cuda.synchronize()
        if runtime.sticky_failed_count <= sticky_before:
            print(f"device selection: {name} did not fail closed")
            failures += 1

    # Rewritten prefixes are untrusted device input. Both source and
    # destination bounds must fail before Issued publication and leave the
    # destination untouched.
    for name, invalid_source, invalid_destination in (
        ("source", ROWS, 0),
        ("destination", 0, ROWS),
    ):
        device_k.zero_()
        device_v.zero_()
        source_index[0] = invalid_source
        staging_index[0] = invalid_destination
        sticky_before = runtime.sticky_failed_count
        phases.set_indexed_row_counts(runtime, 0, 2, 1, stream=stream)
        phases.progress_validated_indexed_host_range(
            runtime, 0, 2, stream=stream
        )
        torch.cuda.synchronize()
        if torch.count_nonzero(device_k).item() != 0 or torch.count_nonzero(
            device_v
        ).item() != 0:
            print(f"{name}: invalid indexed prefix modified staging")
            failures += 1
        if runtime.sticky_failed_count <= sticky_before:
            print(f"{name}: invalid indexed prefix did not poison the runtime")
            failures += 1

    if failures:
        print(f"seam reproducer FAILED with {failures} defect(s); module {module}")
        return 1
    print(f"indexed row-count seam holds; module {module.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
