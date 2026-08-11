#!/usr/bin/env python3
"""Pure planning checks for compiler-generated request-group overlap."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nta_runtime.engines.selected_form import SelectedAttentionExecutor


class FakeWrapper:
    def __init__(self) -> None:
        self._paged_kv_indices_buf = None
        self.plan_calls: list[tuple[object, ...]] = []
        self.run_calls: list[tuple[object, ...]] = []
        self.use_tensor_cores = False

    def plan(self, *args, **kwargs) -> None:
        del kwargs
        self.plan_calls.append(args)
        self._paged_kv_indices_buf = args[1]

    def run(self, *args, **kwargs):
        self.run_calls.append(args)
        return kwargs.get("out")


def main() -> int:
    executor = SelectedAttentionExecutor(
        4,
        1,
        decode_jit_args=["decode"],
        prefill_jit_args=["prefill"],
        register_wrapper=lambda *_: None,
    )
    external_wrapper = FakeWrapper()
    peer_wrapper = FakeWrapper()
    executor._overlap_decode_wrapper = lambda _: peer_wrapper

    claim = SimpleNamespace(claim_id=7, kept_prefix_rows=4)
    segments = [
        (claim, torch.tensor([50, 51], dtype=torch.int32), 0),
        (None, torch.tensor([1, 2, 3], dtype=torch.int32), 1),
        (None, torch.tensor([4, 5], dtype=torch.int32), 2),
    ]
    q = torch.zeros((3, 4, 8), dtype=torch.float16)
    key_cache = torch.zeros((64, 2, 8), dtype=torch.float16)
    layer = SimpleNamespace(scaling=0.125)
    context = executor._build_request_overlap_ctx(
        (claim,),
        external_wrapper,
        q,
        key_cache,
        layer,
        segments,
        (0,),
        (1, 2),
        7,
    )
    assert context["request_overlap"] is True
    assert context["claim_positions"] == (0,)
    assert context["peer_positions"] == (1, 2)
    assert context["claim_entries"][0]["begin"] == 0
    assert context["claim_entries"][0]["end"] == 4
    assert context["plan_indices"][4:].tolist() == [50, 51]
    assert context["peer_indices"].tolist() == [1, 2, 3, 4, 5]
    assert external_wrapper.plan_calls[0][0].tolist() == [0, 6]
    assert peer_wrapper.plan_calls[0][0].tolist() == [0, 3, 5]

    bindings = tuple(
        SimpleNamespace(request_slot=slot) for slot in (10, 11, 12)
    )
    engine = SimpleNamespace(
        _active_batch=SimpleNamespace(bindings=bindings),
        _runtime=SimpleNamespace(device_view_tensor="runtime"),
        _stats={},
        _phase_program=lambda _: None,
    )
    executor._run_paged(
        engine,
        peer_wrapper,
        q[1:3],
        (key_cache, key_cache),
        layer,
        request_positions=(1, 2),
    )
    assert peer_wrapper.run_calls[-1][4] == 11
    assert engine._stats["selected_compiler_launches"] == 1

    try:
        executor._run_paged(
            engine,
            peer_wrapper,
            q[[0, 2]],
            (key_cache, key_cache),
            layer,
            request_positions=(0, 2),
        )
    except RuntimeError as error:
        assert "contiguous request bindings" in str(error)
    else:
        raise AssertionError("noncontiguous request groups were accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
