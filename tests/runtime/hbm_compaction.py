#!/usr/bin/env python3
"""Numerical regression for exact transport-scratch HBM compaction."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

from nta_runtime.runtime import (
    JitPhaseProgram,
    Placement,
    Replica,
    Runtime,
    RuntimeConfig,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: hbm_compaction.py <transport-program.so>")
    program_path = Path(sys.argv[1]).resolve(strict=True)
    source = torch.arange(4 * 257, dtype=torch.int32, device="cuda").reshape(4, 257)
    destination = torch.full_like(source, -1)
    order = (2, 0, 3, 1)
    source_addresses = torch.tensor(
        [source[index].data_ptr() for index in order],
        dtype=torch.int64,
        device="cuda",
    )
    destination_addresses = torch.tensor(
        [destination[index].data_ptr() for index in range(4)],
        dtype=torch.int64,
        device="cuda",
    )
    row_table = torch.stack(
        (
            torch.tensor(order, dtype=torch.int64, device="cuda"),
            source_addresses,
            destination_addresses,
        ),
        dim=1,
    )
    with Runtime(
        RuntimeConfig(
            request_capacity=1,
            object_capacity=4,
            intent_capacity=1,
            work_ticket_capacity=1,
            max_dependencies_per_work_ticket=1,
        )
    ) as runtime, JitPhaseProgram(program_path) as phases:
        for slot in range(4):
            runtime.register_object(
                slot,
                object_id=0x4E5441434F4D5000 + slot,
                version=1,
                bytes=source.shape[1] * source.element_size(),
                replicas=(Replica(source[slot].data_ptr(), Placement.HBM),),
            )
        phases.compact_hbm_rows(
            source_addresses,
            destination_addresses,
            source.shape[1] * source.element_size(),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(destination, source[list(order)], rtol=0, atol=0)
        destination.fill_(-1)
        phases.compact_ready_hbm_rows(
            runtime,
            row_table,
            source.shape[1] * source.element_size(),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(destination, source[list(order)], rtol=0, atol=0)
    print("hbm_compaction=pass")


if __name__ == "__main__":
    main()
