#!/usr/bin/env python3
"""Validate zero-copy reuse of a stock FlashInfer launch plan."""

from __future__ import annotations

from nta_runtime.flashinfer import adopt_planned_flashinfer_state


class Module:
    def plan(self) -> None:
        return None

    def paged_run(self) -> None:
        return None


class Wrapper:
    def __init__(self, *, typed: bool, plan: object | None) -> None:
        self._kv_layout = "NHD"
        self._backend = "fa2"
        self.device = "cuda:0"
        self._jit_module = Module() if typed else None
        self._cached_module = self._jit_module or Module()
        self._jit_additional_tensor_names = (
            ["nta_runtime", "nta_work_items"] if typed else []
        )
        self._plan_info = plan
        self._int_workspace_buffer = object()
        self._pin_memory_int_workspace_buffer = object()
        self._kv_lens_buffer = object()
        self._paged_kv_indices_buf = object()
        self._batch_size = 7


def main() -> None:
    plan = object()
    source = Wrapper(typed=False, plan=plan)
    target = Wrapper(typed=True, plan=None)
    typed_module = target._jit_module
    owned_resources = (
        target._int_workspace_buffer,
        target._pin_memory_int_workspace_buffer,
        target._kv_lens_buffer,
    )
    adopt_planned_flashinfer_state(target, source)
    assert target._plan_info is plan
    assert target._int_workspace_buffer is source._int_workspace_buffer
    assert (
        target._pin_memory_int_workspace_buffer
        is source._pin_memory_int_workspace_buffer
    )
    assert target._kv_lens_buffer is source._kv_lens_buffer
    assert target._paged_kv_indices_buf is source._paged_kv_indices_buf
    assert target._cached_module is typed_module
    assert target._jit_module is typed_module
    assert target._jit_additional_tensor_names == [
        "nta_runtime",
        "nta_work_items",
    ]
    assert target._nta_owned_plan_resources == owned_resources

    # Re-adoption must retain the target's original allocations while moving
    # to the source's newest launch geometry.
    newer_plan = object()
    source._plan_info = newer_plan
    adopt_planned_flashinfer_state(target, source)
    assert target._plan_info is newer_plan
    assert target._nta_owned_plan_resources == owned_resources

    malformed = Wrapper(typed=False, plan=None)
    try:
        adopt_planned_flashinfer_state(target, malformed)
    except RuntimeError as error:
        assert "no completed plan" in str(error)
    else:
        raise AssertionError("FlashInfer accepted an unplanned source wrapper")


if __name__ == "__main__":
    main()
