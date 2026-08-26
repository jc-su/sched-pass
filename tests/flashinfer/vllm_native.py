#!/usr/bin/env python3
"""Run the pinned vLLM V1 attention consumer against a real GPU batch."""

from __future__ import annotations

from types import SimpleNamespace
import math
import os

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.flashinfer import FIDecode, FIPrefill

from nta_runtime.adapters.vllm_v1 import (
    VllmV1Hook,
    current_vllm_v1_forward_state,
    vllm_v1_forward_state,
)
from nta_runtime.engines.vllm import NtaVllmFlashInferImpl
from nta_runtime.runtime import Runtime, RuntimeConfig


class _BlockTable:
    def __init__(self, group: object) -> None:
        self.block_tables = (group,)
        self._group = group

    def __getitem__(self, index: int) -> object:
        del index
        return self._group


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the vLLM native consumer test")
    os.environ["NTA_VLLM_NATIVE"] = "1"

    device = torch.device("cuda")
    page_size = 16
    num_pages = 4
    num_kv_heads = 2
    num_heads = 4
    head_size = 128
    scale = 1.0 / math.sqrt(head_size)

    key = torch.randn(
        (num_pages, page_size, num_kv_heads, head_size),
        device=device,
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    # vLLM 0.26's packed NHD cache is (blocks, kv_heads, page, 2*head_size).
    kv_cache = torch.cat((key, value), dim=-1).permute(0, 2, 1, 3).contiguous()
    query = torch.randn((2, num_heads, head_size), device=device, dtype=torch.float16)

    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    stock_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size, page_size], dtype=torch.int32)
    stock_wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        q_data_type=query.dtype,
        kv_data_type=kv_cache.dtype,
        sm_scale=scale,
        disable_split_kv=True,
    )
    expected = stock_wrapper.run(query, (key, value))
    prefill_query = torch.randn(
        (4, num_heads, head_size), device=device, dtype=torch.float16
    )
    stock_prefill = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    qo_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    stock_prefill.plan(
        qo_indptr,
        indptr,
        indices,
        last_page_len,
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        q_data_type=prefill_query.dtype,
        kv_data_type=kv_cache.dtype,
        sm_scale=scale,
        causal=False,
        disable_split_kv=True,
    )
    prefill_expected = stock_prefill.run(prefill_query, (key, value))
    mixed_decode = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    mixed_decode.plan(
        torch.tensor([0, 2], dtype=torch.int32),
        indices[:2],
        torch.tensor([page_size], dtype=torch.int32),
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        q_data_type=query.dtype,
        kv_data_type=kv_cache.dtype,
        sm_scale=scale,
        disable_split_kv=True,
    )
    mixed_decode_expected = mixed_decode.run(query[:1], (key, value))
    mixed_prefill = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    mixed_prefill.plan(
        torch.tensor([0, 2], dtype=torch.int32, device=device),
        torch.tensor([0, 2], dtype=torch.int32),
        indices[2:4],
        torch.tensor([page_size], dtype=torch.int32),
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        q_data_type=prefill_query.dtype,
        kv_data_type=kv_cache.dtype,
        sm_scale=scale,
        causal=False,
        disable_split_kv=True,
    )
    mixed_prefill_expected = mixed_prefill.run(prefill_query[:2], (key, value))

    runtime = Runtime(
        RuntimeConfig(
            request_capacity=2,
            object_capacity=4,
            intent_capacity=4,
            work_ticket_capacity=8,
            max_dependencies_per_work_ticket=2,
            device_ordinal=0,
            tenant_capacity=2,
        )
    )
    try:
        hook = VllmV1Hook(
            runtime,
            2,
            page_bytes=page_size * num_kv_heads * 2 * head_size * 2,
            expected_vllm_version="0.26.0",
            version_provider=lambda: "0.26.0",
        )
        group = SimpleNamespace(
            get_numpy_array=lambda: np.asarray([[0, 1], [2, 3]], dtype=np.int32),
            num_blocks_per_row=np.asarray([2, 2], dtype=np.int32),
        )
        input_batch = SimpleNamespace(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            block_table=_BlockTable(group),
        )
        scheduler_output = SimpleNamespace(
            num_scheduled_tokens={"a": 1, "b": 1},
            finished_req_ids=set(),
        )
        metadata = SimpleNamespace(
            num_decodes=2,
            num_decode_tokens=2,
            num_prefills=0,
            num_prefill_tokens=0,
            num_actual_tokens=2,
            causal=False,
            use_cascade=False,
            decode=FIDecode(wrapper=stock_wrapper),
            decode_use_trtllm=False,
        )
        prefill_metadata = SimpleNamespace(
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=2,
            num_prefill_tokens=4,
            num_actual_tokens=4,
            causal=False,
            use_cascade=False,
            prefill=FIPrefill(wrapper=stock_prefill),
            decode=None,
        )
        mixed_metadata = SimpleNamespace(
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=1,
            num_prefill_tokens=2,
            num_actual_tokens=3,
            causal=False,
            use_cascade=False,
            prefill=FIPrefill(wrapper=mixed_prefill),
            decode=FIDecode(wrapper=mixed_decode),
            decode_use_trtllm=False,
        )

        with set_current_vllm_config(VllmConfig()):
            implementation = NtaVllmFlashInferImpl(
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                None,
                None,
                "auto",
            )

            def run_batch(
                current_scheduler: object,
                current_input: object,
                current_query: torch.Tensor,
                current_expected: torch.Tensor,
                epoch: int,
            ) -> float:
                batch = hook.bind_forward(
                    current_scheduler,
                    current_input,
                    epoch=epoch,
                    stream=torch.cuda.current_stream(),
                )
                output = torch.empty_like(current_expected)
                with vllm_v1_forward_state(current_scheduler):
                    state = current_vllm_v1_forward_state()
                    assert state is not None
                    state.batch = batch
                    state.hook = hook
                    state.page_size = page_size
                    implementation._native_forward(
                        None,
                        current_query,
                        torch.empty(
                            (2, num_kv_heads, head_size),
                            device=device,
                            dtype=current_query.dtype,
                        ),
                        torch.empty(
                            (2, num_kv_heads, head_size),
                            device=device,
                            dtype=current_query.dtype,
                        ),
                        kv_cache,
                        metadata,
                        output,
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, current_expected, rtol=2e-3, atol=2e-3
                )
                return (output - current_expected).abs().max().item()

            def run_prefill(epoch: int) -> float:
                batch = hook.bind_forward(
                    SimpleNamespace(
                        num_scheduled_tokens={"a": 2, "b": 2},
                        finished_req_ids=set(),
                    ),
                    input_batch,
                    epoch=epoch,
                    stream=torch.cuda.current_stream(),
                )
                output = torch.empty_like(prefill_expected)
                with vllm_v1_forward_state(
                    SimpleNamespace(
                        num_scheduled_tokens={"a": 2, "b": 2},
                        finished_req_ids=set(),
                    )
                ):
                    state = current_vllm_v1_forward_state()
                    assert state is not None
                    state.batch = batch
                    state.hook = hook
                    state.page_size = page_size
                    implementation._native_prefill_forward(
                        None,
                        prefill_query,
                        kv_cache,
                        prefill_metadata,
                        output,
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, prefill_expected, rtol=2e-3, atol=2e-3
                )
                return (output - prefill_expected).abs().max().item()

            maximum = run_prefill(epoch=0)
            mixed_batch = hook.bind_forward(
                SimpleNamespace(
                    num_scheduled_tokens={"a": 1, "b": 2},
                    finished_req_ids=set(),
                ),
                input_batch,
                epoch=1,
                stream=torch.cuda.current_stream(),
            )
            mixed_query = torch.cat((query[:1], prefill_query[:2]))
            mixed_expected = torch.cat((mixed_decode_expected, mixed_prefill_expected))
            mixed_output = torch.empty_like(mixed_expected)
            with vllm_v1_forward_state(
                SimpleNamespace(
                    num_scheduled_tokens={"a": 1, "b": 2},
                    finished_req_ids=set(),
                )
            ):
                state = current_vllm_v1_forward_state()
                assert state is not None
                state.batch = mixed_batch
                state.hook = hook
                state.page_size = page_size
                implementation.forward(
                    None,
                    mixed_query,
                    torch.empty(
                        (3, num_kv_heads, head_size),
                        device=device,
                        dtype=mixed_query.dtype,
                    ),
                    torch.empty(
                        (3, num_kv_heads, head_size),
                        device=device,
                        dtype=mixed_query.dtype,
                    ),
                    kv_cache,
                    mixed_metadata,
                    mixed_output,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                mixed_output, mixed_expected, rtol=2e-3, atol=2e-3
            )
            maximum = max(
                maximum,
                (mixed_output - mixed_expected).abs().max().item(),
            )
            maximum = max(
                maximum,
                run_batch(scheduler_output, input_batch, query, expected, epoch=2),
            )
            # Rebind both rows to new request generations and run again.  This
            # catches stale runtime tickets/CTA counters that a one-shot
            # numerical smoke cannot observe.
            second_query = torch.randn_like(query)
            second_expected = stock_wrapper.run(second_query, (key, value))
            second_input = SimpleNamespace(
                req_ids=["c", "d"],
                req_id_to_index={"c": 0, "d": 1},
                block_table=_BlockTable(group),
            )
            second_scheduler = SimpleNamespace(
                num_scheduled_tokens={"c": 1, "d": 1},
                finished_req_ids={"a", "b"},
            )
            maximum = max(
                maximum,
                run_batch(
                    second_scheduler,
                    second_input,
                    second_query,
                    second_expected,
                    epoch=3,
                ),
            )

        torch.cuda.synchronize()
        print(
            "vllm_native_attention=pass",
            f"max_abs_error={maximum:.6g}",
            f"contract={hook.consumer_contract().kind.value}",
        )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
