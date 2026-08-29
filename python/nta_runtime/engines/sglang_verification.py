"""Opt-in numerical and transfer verification for SGLang attention."""

from __future__ import annotations

from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from nta_runtime.engines.sglang_state import _ActiveBatch


class SglangAttentionVerifier:
    """Run expensive correctness specifications outside production policy.

    The verifier owns no request, CUDA stream, or tier resource. It receives
    the immutable forward state being checked and records only diagnostic
    results in the shared engine statistics dictionary.
    """

    def __init__(
        self,
        *,
        decode_use_tensor_cores: bool,
        stats: dict[str, Any],
    ) -> None:
        self._decode_use_tensor_cores = bool(decode_use_tensor_cores)
        self._stats = stats

    def verify_attention_output(
        self,
        wrapper: Any,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        actual: torch.Tensor,
        layer: Any,
        *,
        causal: bool,
        window_left: int,
    ) -> None:
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        batch_size = int(wrapper._batch_size)
        kv_indptr = wrapper._paged_kv_indptr_buf[: batch_size + 1]
        page_count = int(kv_indptr[-1].item())
        kv_indices = wrapper._paged_kv_indices_buf[:page_count]
        last_page_len = wrapper._paged_kv_last_page_len_buf[:batch_size]
        num_kv_heads = int(kv_cache[0].shape[-2])
        if isinstance(wrapper, BatchDecodeWithPagedKVCacheWrapper):
            reference_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend="fa2",
                use_tensor_cores=self._decode_use_tensor_cores,
            )
            reference_wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page_len,
                int(q.shape[1]),
                num_kv_heads,
                int(q.shape[2]),
                1,
                window_left=window_left,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=kv_cache[0].dtype,
            )
        else:
            qo_indptr = wrapper._qo_indptr_buf[: batch_size + 1]
            reference_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            reference_wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                int(q.shape[1]),
                num_kv_heads,
                int(q.shape[2]),
                1,
                causal=causal,
                window_left=window_left,
                sm_scale=layer.scaling,
                q_data_type=q.dtype,
                kv_data_type=kv_cache[0].dtype,
            )
        expected = reference_wrapper.run(
            q,
            kv_cache,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        torch.cuda.current_stream().synchronize()
        difference = (actual.float() - expected.float()).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
        finite_fraction = float(torch.isfinite(actual).float().mean().item())
        actual_absmax = float(torch.nan_to_num(actual.float()).abs().max().item())
        expected_absmax = float(expected.float().abs().max().item())
        self._stats["last_attention_max_abs_error"] = maximum
        self._stats["last_attention_mean_abs_error"] = mean
        if not torch.allclose(actual, expected, rtol=2e-3, atol=2e-3):
            raise RuntimeError(
                "instrumented FlashInfer output differs from stock "
                f"(layer={layer.layer_id}, max={maximum:.6g}, mean={mean:.6g}, "
                f"finite={finite_fraction:.6g}, actual_absmax={actual_absmax:.6g}, "
                f"expected_absmax={expected_absmax:.6g})"
            )

    @staticmethod
    def verify_layer_transfer(
        batch: _ActiveBatch,
        layer_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        pending = batch.pending_host_load
        if pending is None:
            raise RuntimeError("layer transfer verification has no HiCache transfer")
        controller = pending.controller
        local_layer = layer_id - int(
            getattr(controller.mem_pool_device, "start_layer", 0)
        )
        mapping = pending.materialize_mapping()
        device_pages = torch.tensor(
            tuple(mapping), dtype=torch.long, device=kv_cache[0].device
        )
        host_pages = torch.tensor(tuple(mapping.values()), dtype=torch.long)
        torch.cuda.current_stream().synchronize()
        expected_key = controller.mem_pool_host.k_data_refs[local_layer].index_select(
            0, host_pages
        )
        expected_value = controller.mem_pool_host.v_data_refs[local_layer].index_select(
            0, host_pages
        )
        actual_key = kv_cache[0].index_select(0, device_pages).cpu()
        actual_value = kv_cache[1].index_select(0, device_pages).cpu()
        for name, actual, expected in (
            ("key", actual_key, expected_key),
            ("value", actual_value, expected_value),
        ):
            unequal = actual != expected
            if unequal.any():
                bad_pages = unequal.flatten(1).any(1).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"indexed {name} transfer mismatch on logical pages "
                    f"{bad_pages[:16]} ({len(bad_pages)}/{len(mapping)})"
                )
