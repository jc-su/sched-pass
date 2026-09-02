#!/usr/bin/env python3
"""Exercise SGLang's proactive NVMe acquisition owner on real hardware.

The fixture is intentionally numerical-kernel independent: it verifies the
new cross-layer producer contract itself. Two framework batches remain live
at once, each with four model layers and two request generations. Their exact
groups occupy disjoint ranges in one service epoch without forcing the first
numerical consumer to retire. One consumer is cancelled; every live generation
must still materialize the exact namespace bytes into its framework-owned HBM
cache. This lifetime test deliberately does not claim dynamic issue-order
overtaking.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch

from nta_runtime.engines.sglang_nvme import (
    NvmeSchedulingMode,
    SglangNvmeAcquisitionPipeline,
)
from nta_runtime.requests import RequestBinding
from nta_runtime.runtime import (
    JitPhaseProgram,
    NvmeDmaTarget,
    NvmeHbmMappingPolicy,
    NvmeOptions,
    NvmeTransport,
    Runtime,
    RuntimeConfig,
    WorkTicketState,
)
from nta_runtime.tier import PageExtent
from nta_runtime.transport_program import load_activated_transport_program


LAYER_COUNT = 4
ROW_COUNT = 2
ROW_BYTES = 4096
SOURCE_OFFSET = 512


def _mapping_policy() -> NvmeHbmMappingPolicy:
    selected = os.environ.get("NTA_NVME_HBM_BACKEND", "auto")
    try:
        return {
            "auto": NvmeHbmMappingPolicy.AUTO,
            "nvidia-peer-pages": NvmeHbmMappingPolicy.NVIDIA_PEER_PAGES,
            "cuda-dmabuf-ioas": NvmeHbmMappingPolicy.CUDA_DMA_BUF_IOAS,
        }[selected]
    except KeyError as error:
        raise RuntimeError(f"unknown NVMe HBM backend: {selected}") from error


class _PhysicalTier:
    def __init__(self, transport: NvmeTransport) -> None:
        capabilities = transport.capabilities
        self.nvme_lba_size = int(capabilities.lba_size)
        self.nvme_max_transfer_bytes = int(capabilities.max_transfer_bytes)
        self.config = SimpleNamespace(
            queue_depth=int(capabilities.queue_depth),
            issue_budget=int(capabilities.queue_depth),
            completion_budget=int(capabilities.queue_depth),
            progress_timeout_ns=1_000_000_000,
        )

    def extent(
        self, layer: int, ordinals: tuple[int, ...], component: str, row_bytes: int
    ) -> PageExtent:
        if (
            layer < 0
            or layer >= LAYER_COUNT
            or not ordinals
            or row_bytes != ROW_BYTES
            or component not in {"key", "value"}
        ):
            raise RuntimeError("physical NVMe fixture received invalid geometry")
        if ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals))):
            raise RuntimeError("physical NVMe fixture requires contiguous ordinals")
        lane_bytes = ROW_COUNT * ROW_BYTES
        layer_bytes = 2 * lane_bytes
        component_offset = 0 if component == "key" else lane_bytes
        return PageExtent(
            SOURCE_OFFSET
            + layer * layer_bytes
            + component_offset
            + ordinals[0] * row_bytes,
            len(ordinals) * row_bytes,
        )


def _reference_lane(reference: Path, layer: int, component: str) -> torch.Tensor:
    lane_bytes = ROW_COUNT * ROW_BYTES
    component_offset = 0 if component == "key" else lane_bytes
    with reference.open("rb") as handle:
        handle.seek(SOURCE_OFFSET + layer * 2 * lane_bytes + component_offset)
        payload = handle.read(lane_bytes)
    if len(payload) != lane_bytes:
        raise RuntimeError("NVMe reference does not cover the pipeline fixture")
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
        ROW_COUNT, ROW_BYTES
    )


def main() -> None:
    endpoint = os.environ.get("NTA_NVME_ENDPOINT", "")
    reference = Path(os.environ.get("NTA_NVME_REFERENCE", ""))
    media_policy = os.environ.get("NTA_NVME_MEDIA_POLICY", "")
    if not endpoint or not reference.is_file():
        raise RuntimeError("physical SGLang NVMe test requires endpoint and reference")
    if os.environ.get("NTA_NVME_DMA_TARGET", "") != "hbm-peer":
        raise RuntimeError("physical SGLang NVMe test requires direct HBM DMA")

    transport = NvmeTransport(
        NvmeOptions(
            endpoint=endpoint,
            namespace_id=int(os.environ.get("NTA_NVME_NSID", "1")),
            queue_depth=int(os.environ.get("NTA_NVME_QUEUE_DEPTH", "64")),
            trust_read_only_device_code=media_policy == "trusted-read-only-code",
            dma_target=NvmeDmaTarget.HBM_PEER,
            hbm_mapping_policy=_mapping_policy(),
        )
    )
    runtime: Runtime | None = None
    program: JitPhaseProgram | None = None
    pipeline: SglangNvmeAcquisitionPipeline | None = None
    region = None
    try:
        capabilities = transport.capabilities
        fixture_bytes = LAYER_COUNT * 2 * ROW_COUNT * ROW_BYTES
        if (
            SOURCE_OFFSET % capabilities.lba_size
            or ROW_BYTES % capabilities.lba_size
            or SOURCE_OFFSET + fixture_bytes > capabilities.namespace_bytes
            or ROW_COUNT * ROW_BYTES > capabilities.max_transfer_bytes
        ):
            raise RuntimeError("physical pipeline fixture is not NVMe materializable")

        storage = torch.full(
            (2, 2, LAYER_COUNT, ROW_COUNT, ROW_BYTES),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        region = transport.register_hbm_region(storage.data_ptr(), storage.numel())
        runtime = Runtime(
            RuntimeConfig(
                request_capacity=4,
                object_capacity=4 * LAYER_COUNT,
                intent_capacity=4 * LAYER_COUNT,
                work_ticket_capacity=4 * LAYER_COUNT,
                max_dependencies_per_work_ticket=2,
            ),
            nvme=transport,
        )
        runtime.set_tenant_budget(0, 1 << 30)
        runtime.set_request(0, 101, 1, tenant_id=0)
        runtime.set_request(1, 202, 1, tenant_id=0)
        runtime.set_request(2, 303, 1, tenant_id=0)
        runtime.set_request(3, 404, 1, tenant_id=0)
        # Both requests consume the same physical group.  Cancelling one is the
        # regression case: the live generation must still drive the shared DMA.
        runtime.cancel_request(0, 1)

        program, _path, _digest = load_activated_transport_program()
        progress_stream = torch.cuda.Stream()
        stats: dict[str, int] = {}
        regions = {
            (layer, component): region
            for layer in range(LAYER_COUNT)
            for component in ("key", "value")
        }
        pipeline = SglangNvmeAcquisitionPipeline(
            runtime=runtime,
            tier_service=_PhysicalTier(transport),
            transport_program=lambda: program,
            progress_stream=progress_stream,
            layer_start=0,
            layer_count=LAYER_COUNT,
            object_capacity=4 * LAYER_COUNT,
            work_ticket_capacity=4 * LAYER_COUNT,
            tenant_isolation=False,
            regions=regions,
            stats=stats,
            scheduling_mode=NvmeSchedulingMode.SHARED_APPEND,
        )
        pair = (tuple(range(ROW_COUNT)), tuple(range(ROW_COUNT)))
        semantic = SimpleNamespace(
            dependency_kind="physical_pages",
            schedule=SimpleNamespace(request_indices=(0, 1)),
            page_pairs=(pair, pair),
        )
        bindings = (
            RequestBinding(0, 0, 1, 101),
            RequestBinding(1, 1, 1, 202),
        )

        def kv_cache(batch: int, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
            return storage[batch, 0, layer], storage[batch, 1, layer]

        current = torch.cuda.current_stream()
        acquisition = pipeline.prepare(
            semantic_plans={0: semantic},
            bindings=bindings,
            ordering_stream=current,
            prepare_consumers=lambda _stream, _geometry, _frontier: None,
            select_progressive_layers=(
                lambda _geometry, _frontier, _windows: frozenset(
                    range(LAYER_COUNT)
                )
            ),
            kv_cache_for_layer=lambda layer: kv_cache(0, layer),
            inter_layer_compute_ns=100_000,
        )
        second_bindings = (
            RequestBinding(0, 2, 1, 303),
            RequestBinding(1, 3, 1, 404),
        )
        second = pipeline.prepare(
            semantic_plans={0: semantic},
            bindings=second_bindings,
            ordering_stream=current,
            prepare_consumers=lambda _stream, _geometry, _frontier: None,
            select_progressive_layers=(
                lambda _geometry, _frontier, _windows: frozenset(
                    range(LAYER_COUNT)
                )
            ),
            kv_cache_for_layer=lambda layer: kv_cache(1, layer),
            inter_layer_compute_ns=100_000,
        )
        if second.layers[0].first_object_slot <= acquisition.layers[-1].first_object_slot:
            raise RuntimeError("concurrent NVMe batches aliased directory ranges")
        if any(
            len(layer.wave_events) != 1 or layer.group_wave_indices != (0, 0)
            for layer in acquisition.layers
        ):
            raise RuntimeError("physical NVMe exact groups lost their ready wave")
        for acquired_layer in acquisition.layers:
            pipeline.consume_layer(
                acquisition, acquired_layer, current, wait_for_ready=True
            )
        pipeline.record_consumer(acquisition, current)
        for acquired_layer in second.layers:
            pipeline.consume_layer(second, acquired_layer, current, wait_for_ready=True)
        pipeline.record_consumer(second, current)
        torch.cuda.synchronize()

        for batch in range(2):
            for layer in range(LAYER_COUNT):
                torch.testing.assert_close(
                    storage[batch, 0, layer].cpu(),
                    _reference_lane(reference, layer, "key"),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    storage[batch, 1, layer].cpu(),
                    _reference_lane(reference, layer, "value"),
                    rtol=0,
                    atol=0,
                )
        states = tuple(
            runtime.work_ticket_state(index) for index in range(4 * LAYER_COUNT)
        )
        if states.count(WorkTicketState.CANCELLED) != LAYER_COUNT or states.count(
            WorkTicketState.DONE
        ) != 3 * LAYER_COUNT:
            raise RuntimeError(f"generation fan-out retired incorrectly: {states}")
        queue = transport.stats
        sticky_failed = runtime.sticky_failed_count
        if (
            sticky_failed != 0
            or queue.submitted == 0
            or queue.completed != queue.submitted
            or queue.failed != 0
            or queue.outstanding != 0
            or queue.error != 0
        ):
            raise RuntimeError(
                "proactive NVMe pipeline did not quiesce: "
                f"sticky_failed={sticky_failed} queue={queue} states={states}"
            )
        if (
            stats.get("nvme_pipeline_batches") != 2
            or stats.get("nvme_pipeline_layers") != 2 * LAYER_COUNT
            or stats.get("nvme_pipeline_groups") != 4 * LAYER_COUNT
            or stats.get("nvme_pipeline_packets") != 2 * LAYER_COUNT
            or stats.get("nvme_pipeline_work_items") != 4 * LAYER_COUNT
            or stats.get("nvme_pipeline_physical_bytes") != 2 * fixture_bytes
            or stats.get("nvme_epochs") != 1
        ):
            raise RuntimeError(f"proactive NVMe accounting is incomplete: {stats}")
        print(
            "sglang_nvme_physical=pass "
            f"batches=2 layers={2 * LAYER_COUNT} submitted={queue.submitted} "
            f"work_items={stats['nvme_pipeline_work_items']}"
        )
    finally:
        try:
            torch.cuda.synchronize()
        except BaseException:
            pass
        if pipeline is not None:
            pipeline.close()
        if runtime is not None:
            runtime.close()
        if region is not None:
            region.close()
        if program is not None:
            program.close()
        transport.close()


if __name__ == "__main__":
    main()
