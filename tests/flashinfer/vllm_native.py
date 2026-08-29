#!/usr/bin/env python3
"""Run the pinned vLLM V1 attention consumer against a real GPU batch."""

from __future__ import annotations

from pathlib import Path
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
from nta_runtime.adapters.base import ExactDemandProjection
from nta_runtime.engines.vllm import (
    NtaVllmFlashInferImpl,
    VllmV1WorkerController,
    _new_request_bound_wrapper,
)
from nta_runtime.engines.vllm_config import VllmAttentionConfig
from nta_runtime.engines.vllm_modules import (
    _default_workspace_bytes,
    _prepare_attention_modules,
)
from nta_runtime.connectors.vllm_host import build_indexed_host_resources
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
    workspace_base = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).resolve()

    device = torch.device("cuda")
    page_size = 16
    num_pages = 4
    num_kv_heads = 2
    # Qwen2.5-3B's qualified serving geometry is GQA 16:2.  Group size eight
    # exercises materially different prefill indexing than the former 4:2
    # unit shape and must stay in the numerical gate.
    num_heads = 16
    head_size = 128
    scale = 1.0 / math.sqrt(head_size)
    # Exercise the production lifecycle: typed numerical modules are
    # materialized once during backend setup and execution only loads them.
    _prepare_attention_modules(
        VllmAttentionConfig.from_environment(
            default_workspace_bytes=_default_workspace_bytes()
        ),
        (torch.float16,),
        head_size,
    )

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

    # Match the causal chunked-prefill shape that external-tier vLLM serving
    # actually exercises.  Keep its workspace independent: FlashInfer plans
    # store scheduler arrays in the wrapper workspace, so sharing one with the
    # smaller decode/prefill fixtures would make this differential ambiguous.
    causal_tokens = 212
    causal_pages = 14
    causal_cache_pages = 32
    causal_key = torch.randn(
        (causal_cache_pages, page_size, num_kv_heads, head_size),
        device=device,
        dtype=torch.float16,
    )
    causal_value = torch.randn_like(causal_key)
    causal_kv_cache = (
        torch.cat((causal_key, causal_value), dim=-1).permute(0, 2, 1, 3).contiguous()
    )
    causal_query = torch.randn(
        (causal_tokens, num_heads, head_size),
        device=device,
        dtype=torch.float16,
    )
    causal_workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    causal_stock_prefill = BatchPrefillWithPagedKVCacheWrapper(causal_workspace, "NHD")
    causal_qo_indptr = torch.tensor(
        [0, causal_tokens], dtype=torch.int32, device=device
    )
    causal_kv_indptr = torch.tensor([0, causal_pages], dtype=torch.int32)
    causal_indices = torch.arange(1, causal_pages + 1, dtype=torch.int32, device=device)
    causal_last_page_len = torch.tensor([4], dtype=torch.int32)
    causal_stock_prefill.plan(
        causal_qo_indptr,
        causal_kv_indptr,
        causal_indices,
        causal_last_page_len,
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        q_data_type=causal_query.dtype,
        kv_data_type=causal_kv_cache.dtype,
        sm_scale=scale,
        causal=True,
        # vLLM leaves split-K enabled for this production shape.  The NTA
        # external-tier wrapper currently replans it without split-K, so this
        # reference intentionally preserves the framework plan choice.
        disable_split_kv=False,
    )
    causal_expected = causal_stock_prefill.run(causal_query, (causal_key, causal_value))
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
            request_capacity=4,
            object_capacity=4,
            intent_capacity=4,
            # The causal schedule below contains 28 canonical work items.
            # PREACQUIRED execution must not confuse those structural IDs
            # with this deliberately smaller runtime ticket directory.
            work_ticket_capacity=4,
            max_dependencies_per_work_ticket=2,
            device_ordinal=0,
            tenant_capacity=4,
        )
    )
    try:
        hook = VllmV1Hook(
            runtime,
            4,
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
                    state.connector_validated = True
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
                    state.connector_validated = True
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

            def run_causal_prefill(epoch: int) -> float:
                causal_group = SimpleNamespace(
                    get_numpy_array=lambda: np.arange(
                        1, causal_pages + 1, dtype=np.int32
                    ).reshape(1, causal_pages),
                    num_blocks_per_row=np.asarray([causal_pages], dtype=np.int32),
                )
                causal_input = SimpleNamespace(
                    req_ids=["causal"],
                    req_id_to_index={"causal": 0},
                    block_table=_BlockTable(causal_group),
                )
                causal_scheduler = SimpleNamespace(
                    num_scheduled_tokens={"causal": causal_tokens},
                    finished_req_ids=set(),
                )
                batch = hook.bind_forward(
                    causal_scheduler,
                    causal_input,
                    epoch=epoch,
                    stream=torch.cuda.current_stream(),
                )
                causal_metadata = SimpleNamespace(
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_prefills=1,
                    num_prefill_tokens=causal_tokens,
                    num_actual_tokens=causal_tokens,
                    causal=True,
                    use_cascade=False,
                    prefill=FIPrefill(wrapper=causal_stock_prefill),
                    decode=None,
                )
                output = torch.empty_like(causal_expected)
                with vllm_v1_forward_state(causal_scheduler):
                    state = current_vllm_v1_forward_state()
                    assert state is not None
                    state.batch = batch
                    state.hook = hook
                    state.connector_validated = True
                    state.page_size = page_size
                    implementation._native_prefill_forward(
                        None,
                        causal_query,
                        causal_kv_cache,
                        causal_metadata,
                        output,
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, causal_expected, rtol=2e-3, atol=2e-3
                )
                return (output - causal_expected).abs().max().item()

            maximum = run_prefill(epoch=0)
            maximum = max(maximum, run_causal_prefill(epoch=1))
            mixed_batch = hook.bind_forward(
                SimpleNamespace(
                    num_scheduled_tokens={"a": 1, "b": 2},
                    finished_req_ids={"causal"},
                ),
                input_batch,
                epoch=2,
                stream=torch.cuda.current_stream(),
            )
            mixed_query = torch.cat((query[:1], prefill_query[:2]))
            mixed_expected = torch.cat((mixed_decode_expected, mixed_prefill_expected))
            mixed_output = torch.empty_like(mixed_expected)
            with vllm_v1_forward_state(
                SimpleNamespace(
                    num_scheduled_tokens={"a": 1, "b": 2},
                    finished_req_ids={"causal"},
                )
            ):
                state = current_vllm_v1_forward_state()
                assert state is not None
                state.batch = mixed_batch
                state.hook = hook
                state.connector_validated = True
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

            # Exercise the request-bound direct ABI with the same mixed
            # decode+prefill phase split. FlashInfer restarts requestIndex at
            # zero for the prefill wrapper, so its launch must receive row 1 of
            # the full-forward binding table rather than decode row 0.
            mixed_direct_decode = _new_request_bound_wrapper(
                "decode",
                torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device),
                query_dtype=query.dtype,
                kv_dtype=kv_cache.dtype,
                head_size=head_size,
                workspace_base=workspace_base,
            )
            mixed_direct_decode.plan(
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
            mixed_direct_prefill = _new_request_bound_wrapper(
                "prefill",
                torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device),
                query_dtype=query.dtype,
                kv_dtype=kv_cache.dtype,
                head_size=head_size,
                workspace_base=workspace_base,
            )
            mixed_direct_prefill.plan(
                torch.tensor([0, 2], dtype=torch.int32, device=device),
                torch.tensor([0, 2], dtype=torch.int32),
                indices[2:4],
                torch.tensor([page_size], dtype=torch.int32),
                num_heads,
                num_kv_heads,
                head_size,
                page_size,
                q_data_type=query.dtype,
                kv_data_type=kv_cache.dtype,
                sm_scale=scale,
                causal=False,
                disable_split_kv=True,
            )
            mixed_direct_output = torch.empty_like(mixed_expected)
            mixed_binding_values = tuple(
                value
                for binding in mixed_batch.bindings
                for value in (binding.request_slot, binding.generation)
            )
            with vllm_v1_forward_state(SimpleNamespace()):
                state = current_vllm_v1_forward_state()
                assert state is not None
                state.batch = mixed_batch
                state.hook = hook
                state.connector_validated = True
                state.page_size = page_size
                state.request_bindings_tensor = torch.tensor(
                    mixed_binding_values, dtype=torch.int64, device=device
                )
                state.execution_owner = SimpleNamespace(
                    record_request_binding_consumer=lambda *_args: None
                )
                implementation._run_request_bound(
                    state,
                    state.phase_batch(0, 1),
                    mixed_direct_decode,
                    mixed_decode,
                    query[:1],
                    kv_cache,
                    mixed_direct_output[:1],
                    kind="decode",
                    framework_owned=True,
                    phase_start=0,
                )
                implementation._run_request_bound(
                    state,
                    state.phase_batch(1, 1),
                    mixed_direct_prefill,
                    mixed_prefill,
                    prefill_query[:2],
                    kv_cache,
                    mixed_direct_output[1:],
                    kind="prefill",
                    framework_owned=True,
                    phase_start=1,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                mixed_direct_output, mixed_expected, rtol=2e-3, atol=2e-3
            )
            maximum = max(
                maximum,
                (mixed_direct_output - mixed_expected).abs().max().item(),
            )

            # A stale expected generation must suppress the direct kernel; the
            # old self-validating guard would read the same slot's current
            # generation and incorrectly execute it.
            stale_output = torch.full_like(mixed_prefill_expected, float("nan"))
            stale_values = list(mixed_binding_values)
            stale_values[3] += 1
            with vllm_v1_forward_state(SimpleNamespace()):
                state = current_vllm_v1_forward_state()
                assert state is not None
                state.batch = mixed_batch
                state.hook = hook
                state.connector_validated = True
                state.page_size = page_size
                state.request_bindings_tensor = torch.tensor(
                    stale_values, dtype=torch.int64, device=device
                )
                state.execution_owner = SimpleNamespace(
                    record_request_binding_consumer=lambda *_args: None
                )
                implementation._run_request_bound(
                    state,
                    state.phase_batch(1, 1),
                    mixed_direct_prefill,
                    mixed_prefill,
                    prefill_query[:2],
                    kv_cache,
                    stale_output,
                    kind="prefill",
                    framework_owned=True,
                    phase_start=1,
                )
            torch.cuda.synchronize()
            assert torch.isnan(stale_output).all()
            maximum = max(
                maximum,
                run_batch(scheduler_output, input_batch, query, expected, epoch=3),
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
                    epoch=4,
                ),
            )

            # The direct vLLM ABI must bind the request index through an
            # explicit device map, not by assuming framework rows occupy
            # consecutive runtime slots.  Reverse and gap the slots so an
            # offset-based implementation either skips or binds the wrong
            # generation.
            mapped_batch = hook.adapter.bind_batch(
                ("mapped-x", "mapped-y"),
                (3, 1),
                epoch=5,
                stream=torch.cuda.current_stream(),
                exact_demand=ExactDemandProjection(
                    ((0, 1), (2, 3)),
                    page_size * num_kv_heads * 2 * head_size * 2,
                ),
            )
            mapped_wrapper = _new_request_bound_wrapper(
                "decode",
                workspace,
                query_dtype=query.dtype,
                kv_dtype=kv_cache.dtype,
                head_size=head_size,
                workspace_base=workspace_base,
            )
            mapped_wrapper.plan(
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
                disable_split_kv=False,
            )
            mapped_query = torch.randn_like(query)
            mapped_expected = stock_wrapper.run(mapped_query, (key, value))
            mapped_output = torch.empty_like(mapped_expected)
            with vllm_v1_forward_state(
                SimpleNamespace(
                    num_scheduled_tokens={"mapped-x": 1, "mapped-y": 1},
                    finished_req_ids={"c", "d"},
                )
            ):
                state = current_vllm_v1_forward_state()
                assert state is not None
                state.batch = mapped_batch
                state.hook = hook
                state.connector_validated = True
                state.page_size = page_size
                state.request_bindings_tensor = torch.tensor(
                    [
                        3,
                        mapped_batch.bindings[0].generation,
                        1,
                        mapped_batch.bindings[1].generation,
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                state.execution_owner = SimpleNamespace(
                    record_request_binding_consumer=lambda *_args: None
                )
                implementation._run_request_bound(
                    state,
                    mapped_batch,
                    mapped_wrapper,
                    stock_wrapper,
                    mapped_query,
                    kv_cache,
                    mapped_output,
                    kind="decode",
                    framework_owned=True,
                    phase_start=0,
                )
            torch.cuda.synchronize()
            torch.testing.assert_close(
                mapped_output, mapped_expected, rtol=2e-3, atol=2e-3
            )
            maximum = max(
                maximum,
                (mapped_output - mapped_expected).abs().max().item(),
            )

            # Exercise the real host-staged path with vLLM's packed backing
            # geometry.  Prefill consumes destination rows 2/3 before decode
            # consumes rows 0/1, so this also catches phase-local index arrays
            # that are incorrectly treated as offsets into the full load map.
            logical_row_elements = int(kv_cache[0].numel())
            packed_offset_elements = 64
            packed_stride_elements = logical_row_elements + 128
            packed_destination = torch.full(
                (num_pages, packed_stride_elements),
                7.0,
                dtype=kv_cache.dtype,
                device=device,
            )
            host_kv_cache = packed_destination[
                :,
                packed_offset_elements : packed_offset_elements + logical_row_elements,
            ].view((num_pages, *kv_cache.shape[1:]))
            host_kv_cache.zero_()
            packed_source = torch.zeros(
                (num_pages, packed_stride_elements),
                dtype=kv_cache.dtype,
                device="cpu",
                pin_memory=True,
            )
            packed_source[
                :,
                packed_offset_elements : packed_offset_elements + logical_row_elements,
            ].copy_(kv_cache.detach().cpu().view(num_pages, logical_row_elements))
            host_resources = build_indexed_host_resources(
                {"model.layers.0.self_attn": host_kv_cache},
                {"packed": packed_destination},
                {"packed": packed_source},
            )
            os.environ["NTA_SERVING_TIER"] = "host_staged"
            host_implementation = NtaVllmFlashInferImpl(
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                None,
                None,
                "auto",
            )
            host_input = SimpleNamespace(
                req_ids=["host-decode", "host-prefill"],
                req_id_to_index={"host-decode": 0, "host-prefill": 1},
                block_table=_BlockTable(group),
            )
            host_scheduler = SimpleNamespace(
                num_scheduled_tokens={"host-decode": 1, "host-prefill": 2},
                finished_req_ids={"c", "d"},
            )
            host_batch = hook.bind_forward(
                host_scheduler,
                host_input,
                epoch=6,
                stream=torch.cuda.current_stream(),
            )
            host_output = torch.empty_like(mixed_expected)
            layer = SimpleNamespace(layer_name="model.layers.0.self_attn")
            runner = type("HostRunner", (), {})()
            runner.kv_cache_config = SimpleNamespace(
                kv_cache_groups=(SimpleNamespace(layer_names=(layer.layer_name,)),)
            )
            host_owner = VllmV1WorkerController(runner)
            with vllm_v1_forward_state(host_scheduler):
                state = current_vllm_v1_forward_state()
                assert state is not None
                state.batch = host_batch
                state.hook = hook
                state.connector_validated = True
                state.page_size = page_size
                state.host_transfer_pairs = ((0, 0), (1, 1), (2, 2), (3, 3))
                state.host_resources = host_resources
                state.execution_owner = host_owner
                host_owner.begin_forward(state)
                try:
                    host_implementation.forward(
                        layer,
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
                        host_kv_cache,
                        mixed_metadata,
                        host_output,
                    )
                    host_owner.commit_forward(state)
                    state.commit_evidence()
                except BaseException:
                    host_owner.abort_forward(state)
                    state.abort_evidence()
                    raise
            torch.cuda.synchronize()
            torch.testing.assert_close(host_kv_cache, kv_cache, rtol=0, atol=0)
            torch.testing.assert_close(
                host_output, mixed_expected, rtol=2e-3, atol=2e-3
            )
            assert torch.all(packed_destination[:, :packed_offset_elements] == 7).item()
            assert torch.all(
                packed_destination[
                    :,
                    packed_offset_elements + logical_row_elements :,
                ]
                == 7
            ).item()
            maximum = max(
                maximum,
                (host_output - mixed_expected).abs().max().item(),
            )
            os.environ["NTA_SERVING_TIER"] = "hbm"

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
