"""Opt-in numerical and transfer verification for SGLang attention."""

from __future__ import annotations

from typing import Any

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from nta_runtime.engines.sglang_state import SglangForwardEpoch


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
        # FlashInfer sizes split-K scratch against the complete request
        # geometry.  A fixed verifier workspace made the oracle unusable for
        # exactly the long-context mixed batches it is meant to check (for
        # example, a 32K prefill needs substantially more than 64 MiB).  The
        # production wrapper has already planned and executed this geometry,
        # so its workspace capacity is a fail-closed sufficient bound.  Keep
        # the reference allocation distinct: replanning in the production
        # buffer would overwrite scheduler state needed by following layers.
        source_workspace = getattr(wrapper, "_float_workspace_buffer", None)
        if (
            not isinstance(source_workspace, torch.Tensor)
            or source_workspace.dtype != torch.uint8
            or source_workspace.device != q.device
            or source_workspace.numel() <= 0
            or not source_workspace.is_contiguous()
        ):
            raise RuntimeError(
                "attention verification requires the production FlashInfer "
                "workspace capacity"
            )

        workspace = torch.empty_like(source_workspace)
        self._stats["attention_verification_attempts"] = (
            self._stats.get("attention_verification_attempts", 0) + 1
        )
        self._stats["attention_verification_workspace_bytes"] = max(
            self._stats.get("attention_verification_workspace_bytes", 0),
            int(workspace.numel() * workspace.element_size()),
        )
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
        self._stats["attention_verification_max_abs_error"] = max(
            self._stats.get("attention_verification_max_abs_error", 0.0), maximum
        )
        if not torch.allclose(actual, expected, rtol=2e-3, atol=2e-3):
            row_difference = difference.reshape(difference.shape[0], -1)
            row_maximum = row_difference.max(dim=1).values
            row_tolerance = (
                2e-3
                + 2e-3
                * expected.float().abs().reshape(expected.shape[0], -1)
            )
            bad_rows = (
                (row_difference > row_tolerance).any(dim=1).nonzero().flatten()
            )
            request_errors: list[tuple[int, int, int, float]] = []
            if isinstance(wrapper, BatchPrefillWithPagedKVCacheWrapper):
                query_indptr = tuple(
                    int(value)
                    for value in wrapper._qo_indptr_buf[: batch_size + 1]
                    .detach()
                    .to(device="cpu")
                    .tolist()
                )
            else:
                query_indptr = tuple(range(batch_size + 1))
            bad_row_mask = torch.zeros(
                row_maximum.shape[0], dtype=torch.bool, device=row_maximum.device
            )
            bad_row_mask[bad_rows] = True
            for request in range(batch_size):
                begin = query_indptr[request]
                end = query_indptr[request + 1]
                request_errors.append(
                    (
                        request,
                        begin,
                        int(bad_row_mask[begin:end].sum().item()),
                        float(row_maximum[begin:end].max().item()),
                    )
                )
            self._stats["verification_failures"] = (
                self._stats.get("verification_failures", 0) + 1
            )
            raise RuntimeError(
                "instrumented FlashInfer output differs from stock "
                f"(layer={layer.layer_id}, max={maximum:.6g}, mean={mean:.6g}, "
                f"finite={finite_fraction:.6g}, actual_absmax={actual_absmax:.6g}, "
                f"expected_absmax={expected_absmax:.6g}, "
                f"bad_rows={bad_rows[:16].tolist()}/{difference.shape[0]}, "
                f"request_errors={request_errors})"
            )
        self._stats["attention_verification_passes"] = (
            self._stats.get("attention_verification_passes", 0) + 1
        )

    def poison_prefill_split_scratch(
        self,
        wrapper: Any,
        q: torch.Tensor,
    ) -> None:
        """Make missing partial contributors fail visibly in the final merge."""

        if not isinstance(wrapper, BatchPrefillWithPagedKVCacheWrapper):
            raise RuntimeError("partial-consumer verification requires paged prefill")
        plan_info = getattr(wrapper, "_plan_info", None)
        workspace = getattr(wrapper, "_float_workspace_buffer", None)
        try:
            plan_values = tuple(int(plan_info[index]) for index in range(15))
        except (IndexError, TypeError, ValueError):
            plan_values = ()
        if (
            len(plan_values) != 15
            or not bool(plan_values[14])
            or not isinstance(workspace, torch.Tensor)
            or workspace.dtype != torch.uint8
            or workspace.device != q.device
            or not workspace.is_contiguous()
        ):
            plan_length = len(plan_info) if hasattr(plan_info, "__len__") else None
            raise RuntimeError(
                "partial-consumer verification requires split-K workspace metadata "
                f"(plan_type={type(plan_info).__name__}, plan_length={plan_length}, "
                f"plan={plan_info!r}, workspace_type={type(workspace).__name__})"
            )
        padded_work = plan_values[0]
        cta_tile_q = plan_values[3]
        v_offset = plan_values[10]
        s_offset = plan_values[11]
        if (
            padded_work <= 0
            or cta_tile_q <= 0
            or v_offset < 0
            or s_offset < 0
            or v_offset % 4 != 0
            or s_offset % 4 != 0
        ):
            raise RuntimeError("split-K workspace metadata is malformed")
        partial_rows = padded_work * cta_tile_q * int(q.shape[1])
        v_elements = partial_rows * int(q.shape[2])
        s_elements = partial_rows
        v_workspace = workspace.view(q.dtype)
        float_workspace = workspace.view(torch.float32)
        v_begin = v_offset // q.element_size()
        s_begin = s_offset // 4
        if (
            v_offset % q.element_size() != 0
            or v_begin + v_elements > v_workspace.numel()
            or s_begin + s_elements > float_workspace.numel()
            or not (
                v_offset + v_elements * q.element_size() <= s_offset
                or s_offset + s_elements * 4 <= v_offset
            )
        ):
            raise RuntimeError("split-K scratch exceeds or overlaps its workspace")
        v_workspace.narrow(0, v_begin, v_elements).fill_(float("nan"))
        float_workspace.narrow(0, s_begin, s_elements).fill_(float("nan"))
        self._stats["attention_split_scratch_poison_attempts"] = (
            self._stats.get("attention_split_scratch_poison_attempts", 0) + 1
        )

    def verify_prefill_split_scratch_coverage(
        self,
        wrapper: Any,
        q: torch.Tensor,
    ) -> None:
        """Check every partial state referenced by FlashInfer's final merge."""

        plan_info = getattr(wrapper, "_plan_info", None)
        float_workspace = getattr(wrapper, "_float_workspace_buffer", None)
        int_workspace = getattr(wrapper, "_int_workspace_buffer", None)
        try:
            plan = tuple(int(plan_info[index]) for index in range(15))
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError("split-K coverage has no readable plan") from error
        total_rows = plan[1]
        merge_offset = plan[7]
        v_offset = plan[10]
        s_offset = plan[11]
        if (
            len(plan) != 15
            or not bool(plan[14])
            or total_rows != int(q.shape[0])
            or min(merge_offset, v_offset, s_offset) < 0
            or any(offset % 4 != 0 for offset in (merge_offset, v_offset, s_offset))
            or not isinstance(float_workspace, torch.Tensor)
            or not isinstance(int_workspace, torch.Tensor)
            or float_workspace.dtype != torch.uint8
            or int_workspace.dtype != torch.uint8
        ):
            raise RuntimeError("split-K coverage metadata is malformed")
        merge_begin = merge_offset // 4
        merge_words = total_rows + 1
        int_values = int_workspace.view(torch.int32)
        if merge_begin + merge_words > int_values.numel():
            raise RuntimeError("split-K merge indptr exceeds integer workspace")
        merge_indptr = int_values.narrow(0, merge_begin, merge_words)
        active_partials = int(merge_indptr[-1].item())
        if active_partials <= 0:
            raise RuntimeError("split-K merge has no active partial states")
        num_heads = int(q.shape[1])
        head_dim = int(q.shape[2])
        v_values = float_workspace.view(q.dtype)
        float_values = float_workspace.view(torch.float32)
        v_begin = v_offset // q.element_size()
        s_begin = s_offset // 4
        v_elements = active_partials * num_heads * head_dim
        s_elements = active_partials * num_heads
        if (
            v_offset % q.element_size() != 0
            or v_begin + v_elements > v_values.numel()
            or s_begin + s_elements > float_values.numel()
        ):
            raise RuntimeError("active split-K states exceed floating workspace")
        v_states = v_values.narrow(0, v_begin, v_elements).view(
            active_partials, num_heads, head_dim
        )
        s_states = float_values.narrow(0, s_begin, s_elements).view(
            active_partials, num_heads
        )
        missing_v = ~torch.isfinite(v_states).all(dim=(1, 2))
        missing_s = ~torch.isfinite(s_states).all(dim=1)
        missing = missing_v | missing_s
        missing_indices = missing.nonzero().flatten()
        self._stats["attention_split_active_partials"] = active_partials
        self._stats["attention_split_missing_partials"] = int(
            missing_indices.numel()
        )
        self._stats["attention_split_missing_v_partials"] = int(
            missing_v.sum().item()
        )
        self._stats["attention_split_missing_s_partials"] = int(
            missing_s.sum().item()
        )
        if missing_indices.numel() == 0:
            return
        merge_cpu = merge_indptr.detach().to(device="cpu")
        missing_cpu = missing_indices.detach().to(device="cpu")
        output_rows = torch.bucketize(missing_cpu, merge_cpu[1:], right=False)
        unique_rows, row_counts = output_rows.unique(return_counts=True)
        row_summary = tuple(
            (int(row), int(count))
            for row, count in zip(unique_rows[:32], row_counts[:32], strict=True)
        )
        raise RuntimeError(
            "partial consumer left merge-visible split-K states unwritten "
            f"(active={active_partials}, missing={missing_indices.numel()}, "
            f"missing_v={missing_v.sum().item()}, "
            f"missing_s={missing_s.sum().item()}, "
            f"first_missing={missing_cpu[:64].tolist()}, "
            f"row_missing_counts={row_summary})"
        )

    def verify_typed_projection(
        self,
        wrapper: Any,
        semantic: Any,
        pending: Any,
        *,
        default_page_size: int,
    ) -> None:
        """Cross-check the fast logical projection against FlashInfer pages.

        Production constructs typed lease slices from request-local logical
        spans and therefore avoids downloading a context-sized page table.
        Verification must independently prove that optimization: for every
        work unit, the projected contiguous lease interval must name exactly
        the host/device pages that FlashInfer will dereference.  This catches a
        soundness gap that output-only checks can observe but cannot localize.
        """

        from nta_runtime.engines.sglang_semantics import (
            work_page_pairs,
            wrapper_page_layout,
        )

        if semantic.dependency_kind != "typed_lease":
            raise RuntimeError("typed projection verification requires a lease plan")
        page_pairs = work_page_pairs(
            semantic.schedule,
            pending,
            layout=wrapper_page_layout(
                wrapper, default_page_size=default_page_size
            ),
            host_staged=True,
            physical_catalog=None,
        )
        host_rows = tuple(
            int(value)
            for value in pending.host_indices.detach().to(device="cpu").tolist()
        )
        device_rows = tuple(
            int(value)
            for value in pending.device_indices.detach().to(device="cpu").tolist()
        )
        if len(host_rows) != len(device_rows):
            raise RuntimeError("typed projection lease maps have different lengths")
        operations = {
            operation.operation_id: operation
            for operation in pending.operation_ranges()
        }
        if len(operations) != len(pending.operation_ranges()):
            raise RuntimeError("typed projection lease repeats an operation")
        dependencies = semantic.acquisition_slices
        if len(page_pairs) != len(dependencies):
            raise RuntimeError("typed projection and FlashInfer work counts disagree")
        self._stats["typed_projection_verification_attempts"] = (
            self._stats.get("typed_projection_verification_attempts", 0) + 1
        )
        for work_id, (dependency, actual_pair) in enumerate(
            zip(dependencies, page_pairs, strict=True)
        ):
            if dependency is None:
                expected_pair = ((), ())
            else:
                operation = operations.get(dependency.operation_id)
                if operation is None or dependency.row_end > operation.row_count:
                    raise RuntimeError(
                        "typed projection work exceeds its lease operation"
                    )
                begin = operation.row_begin + dependency.row_begin
                end = begin + dependency.row_count
                expected_pair = (host_rows[begin:end], device_rows[begin:end])
            if actual_pair != expected_pair:
                expected_host, expected_device = expected_pair
                actual_host, actual_device = actual_pair
                raise RuntimeError(
                    "typed lease projection differs from FlashInfer page demand "
                    f"(work={work_id}, request="
                    f"{semantic.schedule.request_indices[work_id]}, tile="
                    f"{semantic.schedule.kv_tile_indices[work_id]}, "
                    f"expected_rows={len(expected_device)}, "
                    f"actual_rows={len(actual_device)}, "
                    f"expected_device={expected_device[:8]}, "
                    f"actual_device={actual_device[:8]}, "
                    f"expected_host={expected_host[:8]}, "
                    f"actual_host={actual_host[:8]})"
                )
        self._stats["typed_projection_verification_passes"] = (
            self._stats.get("typed_projection_verification_passes", 0) + 1
        )

    @staticmethod
    def verify_layer_transfer(
        batch: SglangForwardEpoch,
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
