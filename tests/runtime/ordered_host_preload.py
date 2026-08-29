#!/usr/bin/env python3
"""Validate ordered indexed-host production and device-published readiness."""

from __future__ import annotations

import pathlib
import sys

import torch

from nta_runtime import (
    IndexedHostIndexBinding,
    IndexedHostObject,
    JitPhaseProgram,
    Runtime,
    RuntimeConfig,
)


ELEMENT_BYTES = 128


def _indexed_object(
    *,
    slot: int,
    source: torch.Tensor,
    destination: torch.Tensor,
    source_indices: torch.Tensor,
    destination_indices: torch.Tensor,
    index_count: int,
    source_rows: int,
    destination_rows: int,
    version: int,
) -> IndexedHostObject:
    return IndexedHostObject(
        object_id=0x4E54414F52440000 + slot,
        version=version,
        source_device_address=source.data_ptr(),
        staging_device_address=destination.data_ptr(),
        source_indices_device_address=source_indices.data_ptr(),
        staging_indices_device_address=destination_indices.data_ptr(),
        index_count=index_count,
        element_bytes=ELEMENT_BYTES,
        source_stride_bytes=source.stride(0) * source.element_size(),
        staging_stride_bytes=destination.stride(0) * destination.element_size(),
        source_index_limit=source_rows,
        staging_index_limit=destination_rows,
    )


def _check_successful_ordered_preload(
    runtime: Runtime, phases: JitPhaseProgram
) -> None:
    pair_counts = (513, 257, 385, 129)
    pair_count = len(pair_counts)
    row_count = max(pair_counts)
    source = torch.arange(
        pair_count * 2 * row_count * ELEMENT_BYTES,
        dtype=torch.int64,
    ).to(torch.uint8)
    source = source.reshape(pair_count, 2, row_count, ELEMENT_BYTES).pin_memory()
    destination = torch.full(
        (pair_count, 2, row_count, ELEMENT_BYTES),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    source_indices = torch.arange(
        row_count - 1, -1, -1, dtype=torch.int32, device="cuda"
    )
    destination_indices = torch.arange(row_count, dtype=torch.int32, device="cuda")
    objects: list[IndexedHostObject] = []
    for pair, index_count in enumerate(pair_counts):
        for lane in range(2):
            slot = 2 * pair + lane
            objects.append(
                _indexed_object(
                    slot=slot,
                    source=source[pair, lane],
                    destination=destination[pair, lane],
                    source_indices=source_indices,
                    destination_indices=destination_indices,
                    index_count=index_count,
                    source_rows=row_count,
                    destination_rows=row_count,
                    version=17,
                )
            )

    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    registration = torch.cuda.Event()
    producer_done = torch.cuda.Event()
    task_head = torch.empty(1, dtype=torch.int32, device="cuda")
    runtime.register_indexed_host_objects(
        0,
        objects,
        stream=producer,
        index_binding=IndexedHostIndexBinding(
            source_indices.data_ptr(),
            destination_indices.data_ptr(),
            row_count,
        ),
    )
    registration.record(producer)
    phases.preload_host_pairs_ordered(
        runtime,
        first_object=0,
        pair_count=pair_count,
        worker_blocks=8,
        task_head=task_head,
        stream=producer,
    )
    producer_done.record(producer)

    consumer.wait_event(registration)
    for pair in range(pair_count):
        runtime.wait_object_range_terminal(2 * pair, 1, consumer)
    consumer.synchronize()
    if not producer_done.query():
        raise AssertionError("terminal object waits did not cover ordered production")
    if runtime.sticky_failed_count != 0:
        raise AssertionError("valid ordered host production reported a device failure")

    actual = destination.cpu()
    for pair, index_count in enumerate(pair_counts):
        for lane in range(2):
            expected = torch.full(
                (row_count, ELEMENT_BYTES), 0xA5, dtype=torch.uint8
            )
            expected[:index_count].copy_(
                source[pair, lane, source_indices[:index_count].cpu()]
            )
            if not torch.equal(actual[pair, lane], expected):
                raise AssertionError(
                    f"ordered indexed-host contents differ for pair={pair}, lane={lane}"
                )


def _check_failed_object_is_terminal(
    runtime: Runtime, phases: JitPhaseProgram
) -> None:
    source = torch.arange(4 * ELEMENT_BYTES, dtype=torch.uint8).reshape(
        4, ELEMENT_BYTES
    ).pin_memory()
    destination = torch.full(
        (2, 4, ELEMENT_BYTES), 0xA5, dtype=torch.uint8, device="cuda"
    )
    source_indices = torch.tensor((0, 4), dtype=torch.int32, device="cuda")
    destination_indices = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    objects = tuple(
        _indexed_object(
            slot=8 + lane,
            source=source,
            destination=destination[lane],
            source_indices=source_indices,
            destination_indices=destination_indices,
            index_count=2,
            source_rows=4,
            destination_rows=4,
            version=18,
        )
        for lane in range(2)
    )
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    registration = torch.cuda.Event()
    task_head = torch.empty(1, dtype=torch.int32, device="cuda")
    failures_before = runtime.sticky_failed_count
    runtime.register_indexed_host_objects(
        8,
        objects,
        stream=producer,
        index_binding=IndexedHostIndexBinding(
            source_indices.data_ptr(), destination_indices.data_ptr(), 2
        ),
    )
    registration.record(producer)
    phases.preload_host_pairs_ordered(
        runtime,
        first_object=8,
        pair_count=1,
        worker_blocks=2,
        task_head=task_head,
        stream=producer,
    )
    consumer.wait_event(registration)
    runtime.wait_object_range_terminal(8, 1, consumer)
    consumer.synchronize()
    if runtime.sticky_failed_count <= failures_before:
        raise AssertionError("invalid ordered object did not fail closed")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: ordered_host_preload.py <transport-program.so>")
    transport = pathlib.Path(sys.argv[1]).resolve()
    with Runtime(
        RuntimeConfig(
            request_capacity=1,
            object_capacity=10,
            intent_capacity=2,
            work_ticket_capacity=1,
            max_dependencies_per_work_ticket=1,
        )
    ) as runtime, JitPhaseProgram(transport) as phases:
        _check_successful_ordered_preload(runtime, phases)
        _check_failed_object_is_terminal(runtime, phases)
    print("ordered_host_preload=pass")


if __name__ == "__main__":
    main()
