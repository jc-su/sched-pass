#!/usr/bin/env python3
"""Validate that stock consumers cannot observe failed acquisition objects."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import torch

from nta_runtime import JitPhaseProgram, Placement, Replica, Runtime, RuntimeConfig


_EXPECTED_TRAP_EXIT = 86


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            request_capacity=1,
            object_capacity=2,
            intent_capacity=1,
            work_ticket_capacity=1,
            max_dependencies_per_work_ticket=1,
        )
    )


def _check_ready(program_path: Path) -> None:
    backing = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    with _runtime() as runtime, JitPhaseProgram(program_path) as phases:
        runtime.register_object(
            0,
            object_id=0x4E54415245414459,
            version=1,
            bytes=backing.numel(),
            replicas=(Replica(backing.data_ptr(), Placement.HBM),),
        )
        phases.require_ready_objects(runtime, 0, 1)
        torch.cuda.synchronize()


def _trap_child(program_path: Path) -> None:
    with _runtime() as runtime, JitPhaseProgram(program_path) as phases:
        # Aliasing an empty source deterministically publishes Failed in slot 1.
        phases.alias_preloaded_objects(
            runtime,
            source_first=0,
            destination_first=1,
            object_count=1,
            object_id_base=0x4E54414641494C00,
            version=1,
        )
        source = torch.ones(4096, dtype=torch.uint8, device="cuda")
        destination = torch.zeros_like(source)
        source_addresses = torch.tensor(
            (source.data_ptr(),), dtype=torch.int64, device="cuda"
        )
        destination_addresses = torch.tensor(
            (destination.data_ptr(),), dtype=torch.int64, device="cuda"
        )
        row_table = torch.stack(
            (
                torch.ones(1, dtype=torch.int64, device="cuda"),
                source_addresses,
                destination_addresses,
            ),
            dim=1,
        )
        phases.compact_ready_hbm_rows(
            runtime,
            row_table,
            source.numel(),
        )
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            # The trap poisons this process's CUDA context by design. Avoid
            # invoking CUDA-owning destructors while unwinding it.
            os._exit(_EXPECTED_TRAP_EXIT)
    raise AssertionError("a Failed object passed the numerical-consumer gate")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: ready_object_gate.py <transport-program.so> [--trap]")
    program_path = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) == 3:
        if sys.argv[2] != "--trap":
            raise SystemExit("unknown ready-object gate mode")
        _trap_child(program_path)
        return

    _check_ready(program_path)
    completed = subprocess.run(
        (sys.executable, str(Path(__file__).resolve()), str(program_path), "--trap"),
        check=False,
        env=os.environ.copy(),
    )
    if completed.returncode != _EXPECTED_TRAP_EXIT:
        raise AssertionError(
            "Failed-object child did not stop at the numerical-consumer gate: "
            f"returncode={completed.returncode}"
        )
    print("ready_object_gate=pass")


if __name__ == "__main__":
    main()
