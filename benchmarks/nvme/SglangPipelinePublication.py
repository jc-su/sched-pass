#!/usr/bin/env python3
"""Causal physical benchmark for the SGLang NVMe acquisition pipeline.

Four read-only arms hold exact demand and HBM destinations constant while
changing one mechanism boundary at a time:

* ``ordered_batch`` is the production queue-fill window, bulk publication,
  and device-validated ordered EDF cursor.
* ``ordered_scalar`` changes only directory publication to the former
  one-object Python/C path.
* ``heap_batch`` changes only dispatch to the generic dynamic EDF heap.
* ``layered_batch`` changes only global windowing to one epoch per layer.

Arm order rotates within one process so controller, GPU, JIT, and thermal
state are shared. This is a regression/causal gate, not a serving result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.atomic_io import atomic_write_text  # noqa: E402

from nta_runtime.engines.sglang_nvme import (  # noqa: E402
    SglangNvmeAcquisitionPipeline,
)
from nta_runtime.nvme_granularity import (  # noqa: E402
    NvmeTransferServiceModel,
    plan_nvme_scratch_capacity,
)
from nta_runtime.nvme_materialization import NvmeScratchArena  # noqa: E402
from nta_runtime.requests import RequestBinding  # noqa: E402
from nta_runtime.runtime import (  # noqa: E402
    JitPhaseProgram,
    NvmeDmaTarget,
    NvmeHbmMappingPolicy,
    NvmeOptions,
    NvmeTransport,
    RegisteredNvmeObjectInstall,
    Runtime,
    RuntimeConfig,
)
from nta_runtime.tier import PageExtent  # noqa: E402
from nta_runtime.transport_program import (  # noqa: E402
    load_activated_transport_program,
)


ROW_BYTES = 4096
SOURCE_OFFSET = 4096


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument(
        "--source-stride",
        type=int,
        default=2,
        help="source-row spacing for the controlled granularity envelope",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument(
        "--issue-budget",
        type=int,
        default=0,
        help="commands issued per progress round (0 uses controller depth)",
    )
    parser.add_argument(
        "--production-window-layers",
        type=int,
        default=0,
        help="ordered/heap runtime capacity in layers (0 uses all layers)",
    )
    parser.add_argument(
        "--pipeline-window-layers",
        type=int,
        default=0,
        help="benchmark-only layer window cap (0 uses the production planner)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--minimum-publication-prepare-speedup", type=float, default=0.0
    )
    parser.add_argument("--minimum-publication-wall-speedup", type=float, default=0.0)
    parser.add_argument("--minimum-ordered-wall-speedup", type=float, default=0.0)
    parser.add_argument("--minimum-window-wall-speedup", type=float, default=0.0)
    parser.add_argument("--command-service-ns", type=int, default=0)
    parser.add_argument("--read-bandwidth-bps", type=int, default=0)
    parser.add_argument("--compaction-bandwidth-bps", type=int, default=0)
    parser.add_argument("--compaction-launch-ns", type=int, default=0)
    parser.add_argument("--minimum-granularity-gain", type=float, default=1.03)
    parser.add_argument("--minimum-granularity-wall-speedup", type=float, default=0.0)
    arguments = parser.parse_args()
    if (
        min(
            arguments.layers,
            arguments.runs,
            arguments.source_stride,
            arguments.trials,
        )
        <= 0
    ):
        parser.error("layers, runs, source-stride, and trials must be positive")
    if (
        arguments.warmup < 0
        or arguments.issue_budget < 0
        or arguments.production_window_layers < 0
        or arguments.production_window_layers > arguments.layers
        or arguments.pipeline_window_layers < 0
        or arguments.pipeline_window_layers > arguments.layers
        or arguments.compaction_launch_ns < 0
    ):
        parser.error("warmup must be non-negative")
    service_values = (
        arguments.command_service_ns,
        arguments.read_bandwidth_bps,
        arguments.compaction_bandwidth_bps,
    )
    if any(service_values) and (not all(value > 0 for value in service_values)):
        parser.error("all three NVMe service measurements must be positive")
    if arguments.minimum_granularity_gain < 1.0:
        parser.error("minimum granularity gain must be at least one")
    if arguments.minimum_granularity_wall_speedup > 0 and not all(service_values):
        parser.error("the granularity gate requires a calibrated service model")
    if (
        min(
            arguments.minimum_publication_prepare_speedup,
            arguments.minimum_publication_wall_speedup,
            arguments.minimum_ordered_wall_speedup,
            arguments.minimum_window_wall_speedup,
            arguments.minimum_granularity_wall_speedup,
        )
        < 0
    ):
        parser.error("minimum speedups must be non-negative")
    return arguments


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
    def __init__(
        self,
        transport: NvmeTransport,
        *,
        layer_count: int,
        source_rows: int,
        issue_budget: int,
        service_model: NvmeTransferServiceModel | None = None,
    ) -> None:
        capabilities = transport.capabilities
        self.nvme_lba_size = int(capabilities.lba_size)
        self.nvme_max_transfer_bytes = int(capabilities.max_transfer_bytes)
        self.config = SimpleNamespace(
            queue_depth=int(capabilities.queue_depth),
            issue_budget=(
                int(capabilities.queue_depth)
                if issue_budget == 0
                else min(issue_budget, int(capabilities.queue_depth))
            ),
            completion_budget=int(capabilities.queue_depth),
            progress_timeout_ns=1_000_000_000,
            nvme_service_model=service_model or NvmeTransferServiceModel(),
        )
        self.nvme_controller_page_size = int(capabilities.controller_page_size)
        self._layer_count = layer_count
        self._source_rows = source_rows

    @property
    def source_bytes(self) -> int:
        return self._layer_count * 2 * self._source_rows * ROW_BYTES

    def extent(
        self, layer: int, ordinals: tuple[int, ...], component: str, row_bytes: int
    ) -> PageExtent:
        if (
            layer < 0
            or layer >= self._layer_count
            or not ordinals
            or row_bytes != ROW_BYTES
            or component not in {"key", "value"}
            or ordinals != tuple(range(ordinals[0], ordinals[0] + len(ordinals)))
            or ordinals[-1] >= self._source_rows
        ):
            raise RuntimeError("publication benchmark received invalid geometry")
        lane_bytes = self._source_rows * ROW_BYTES
        layer_bytes = 2 * lane_bytes
        component_offset = 0 if component == "key" else lane_bytes
        return PageExtent(
            SOURCE_OFFSET
            + layer * layer_bytes
            + component_offset
            + ordinals[0] * ROW_BYTES,
            len(ordinals) * ROW_BYTES,
        )


class _PublicationRuntime:
    """Delegate everything except the publication arm under measurement."""

    def __init__(self, runtime: Runtime, *, scalar: bool) -> None:
        self._runtime = runtime
        self._scalar = scalar

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def install_registered_nvme_objects_async(
        self,
        objects: tuple[RegisteredNvmeObjectInstall, ...],
        stream: torch.cuda.Stream,
    ) -> tuple[int, ...]:
        if not self._scalar:
            return self._runtime.install_registered_nvme_objects_async(objects, stream)
        return tuple(
            self._runtime.install_registered_nvme_object_async(
                object_.slot,
                object_.object_id,
                object_.version,
                object_.source_byte_offset,
                object_.bytes,
                object_.region,
                object_.destination_device_address,
                stream,
                object_.prior_consumer_event,
            )
            for object_ in objects
        )


class _SchedulerProgram:
    """Select one transport scheduler arm without changing production code."""

    def __init__(self, program: JitPhaseProgram, *, generic_heap: bool) -> None:
        self._program = program
        self._generic_heap = generic_heap

    def __getattr__(self, name: str) -> Any:
        return getattr(self._program, name)

    def discover_ordered_nvme(
        self,
        runtime: Runtime,
        plan: Any,
        first_intent: int,
        intent_count: int,
        stream: torch.cuda.Stream,
    ) -> None:
        if self._generic_heap:
            self._program.discover(runtime, plan, stream)
            return
        self._program.discover_ordered_nvme(
            runtime, plan, first_intent, intent_count, stream
        )


class _Arm:
    def __init__(
        self,
        *,
        name: str,
        transport: NvmeTransport,
        tier: _PhysicalTier,
        program: JitPhaseProgram,
        layer_count: int,
        run_count: int,
        source_stride: int,
        scalar_publication: bool,
        generic_heap: bool,
        window_layers: int,
        pipeline_window_layers: int | None,
    ) -> None:
        self.name = name
        self.storage = torch.full(
            (2, layer_count, run_count, ROW_BYTES),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        self.region = transport.register_hbm_region(
            self.storage.data_ptr(), self.storage.numel()
        )
        self.scratch_storage: torch.Tensor | None = None
        self.scratch_region: Any | None = None
        scratch: NvmeScratchArena | None = None
        if tier.config.nvme_service_model.calibrated:
            scratch_bytes = plan_nvme_scratch_capacity(
                queue_depth=int(tier.config.queue_depth),
                max_transfer_bytes=tier.nvme_max_transfer_bytes,
            )
            self.scratch_storage = torch.empty(
                scratch_bytes,
                dtype=torch.uint8,
                device="cuda",
            )
            self.scratch_region = transport.register_hbm_region(
                self.scratch_storage.data_ptr(), self.scratch_storage.numel()
            )
            scratch = NvmeScratchArena(self.scratch_storage, self.scratch_region)
        objects_per_layer = 2 * run_count
        work_items_per_layer = 2 * run_count
        if window_layers <= 0 or window_layers > layer_count:
            raise ValueError("benchmark window must fit the model layer count")
        object_count = window_layers * objects_per_layer
        work_item_count = window_layers * work_items_per_layer
        self.runtime = Runtime(
            RuntimeConfig(
                request_capacity=2,
                object_capacity=object_count,
                intent_capacity=object_count,
                work_ticket_capacity=work_item_count,
                max_dependencies_per_work_ticket=2,
            ),
            nvme=transport,
        )
        self.runtime.set_tenant_budget(0, 1 << 30)
        self.runtime.set_request(0, 101, 1, tenant_id=0, deadline_clock=900)
        self.runtime.set_request(1, 202, 1, tenant_id=0, deadline_clock=500)
        # Preserve the shared-resource cancellation case while one live
        # generation remains responsible for every physical transfer.
        self.runtime.cancel_request(0, 1)
        self.progress_stream = torch.cuda.Stream()
        self.stats: dict[str, int] = {}
        regions = {
            (layer, component): self.region
            for layer in range(layer_count)
            for component in ("key", "value")
        }
        runtime_view = _PublicationRuntime(self.runtime, scalar=scalar_publication)
        scheduler = _SchedulerProgram(program, generic_heap=generic_heap)
        self.pipeline = SglangNvmeAcquisitionPipeline(
            runtime=runtime_view,
            tier_service=tier,
            transport_program=lambda: scheduler,
            progress_stream=self.progress_stream,
            layer_start=0,
            layer_count=layer_count,
            object_capacity=object_count,
            work_ticket_capacity=work_item_count,
            tenant_isolation=False,
            regions=regions,
            scratch=scratch,
            stats=self.stats,
            window_layer_limit=pipeline_window_layers,
        )
        source = tuple(source_stride * index for index in range(run_count))
        destination = tuple(range(run_count))
        pair = (source, destination)
        self.semantic = SimpleNamespace(
            dependency_kind="physical_pages",
            schedule=SimpleNamespace(request_indices=(0, 1)),
            page_pairs=(pair, pair),
        )
        self.bindings = (
            RequestBinding(0, 0, 1, 101, deadline_clock=900),
            RequestBinding(1, 1, 1, 202, deadline_clock=500),
        )
        self.window_count = 0

    def _kv_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.storage[0, layer], self.storage[1, layer]

    def run(self) -> dict[str, float]:
        current = torch.cuda.current_stream()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_stop = torch.cuda.Event(enable_timing=True)
        stage_counters = (
            "nvme_geometry_cpu_ns",
            "nvme_descriptor_cpu_ns",
            "nvme_publication_cpu_ns",
            "nvme_finalization_cpu_ns",
            "nvme_topology_cpu_ns",
        )
        stage_before = {name: int(self.stats.get(name, 0)) for name in stage_counters}
        torch.cuda.nvtx.range_push(self.name)
        try:
            gpu_start.record(self.progress_stream)
            wall_start = time.perf_counter_ns()
            acquisition = self.pipeline.prepare(
                semantic_plans={0: self.semantic},
                bindings=self.bindings,
                ordering_stream=current,
                prepare_consumers=lambda _stream: None,
                kv_cache_for_layer=self._kv_cache,
                inter_layer_compute_ns=100_000,
            )
            self.window_count = acquisition.window_count
            prepare_stop = time.perf_counter_ns()
            gpu_stop.record(self.progress_stream)
            for acquired_layer in acquisition.layers:
                self.pipeline.wait_layer(acquisition, acquired_layer, current)
            self.pipeline.record_consumer(acquisition, current)
            torch.cuda.synchronize()
            wall_stop = time.perf_counter_ns()
        finally:
            torch.cuda.nvtx.range_pop()
        result = {
            "prepare_ms": (prepare_stop - wall_start) / 1e6,
            "gpu_pipeline_ms": gpu_start.elapsed_time(gpu_stop),
            "wall_ms": (wall_stop - wall_start) / 1e6,
        }
        result.update(
            {
                name.removeprefix("nvme_").removesuffix("_cpu_ns") + "_cpu_ms": (
                    int(self.stats[name]) - stage_before[name]
                )
                / 1e6
                for name in stage_counters
            }
        )
        return result

    def close(self) -> None:
        self.pipeline.close()
        self.runtime.close()
        if self.scratch_region is not None:
            self.scratch_region.close()
        self.region.close()


def _verify(
    arm: _Arm,
    *,
    reference: Path,
    layer_count: int,
    source_rows: int,
    run_count: int,
    source_stride: int,
) -> None:
    source_ordinals = tuple(source_stride * index for index in range(run_count))
    lane_bytes = source_rows * ROW_BYTES
    with reference.open("rb") as handle:
        for layer in range(layer_count):
            for lane, component_offset in enumerate((0, lane_bytes)):
                rows = bytearray()
                for ordinal in source_ordinals:
                    handle.seek(
                        SOURCE_OFFSET
                        + layer * 2 * lane_bytes
                        + component_offset
                        + ordinal * ROW_BYTES
                    )
                    payload = handle.read(ROW_BYTES)
                    if len(payload) != ROW_BYTES:
                        raise RuntimeError("NVMe reference is shorter than the fixture")
                    rows.extend(payload)
                expected = torch.frombuffer(rows, dtype=torch.uint8).reshape(
                    run_count, ROW_BYTES
                )
                actual = arm.storage[lane, layer].cpu()
                mismatch = actual != expected
                if bool(mismatch.any()):
                    coordinates = mismatch.nonzero()[:8].tolist()
                    examples = tuple(
                        (
                            int(row),
                            int(byte),
                            int(actual[row, byte]),
                            int(expected[row, byte]),
                        )
                        for row, byte in coordinates
                    )
                    raise RuntimeError(
                        f"{arm.name} corrupted NVMe publication at layer={layer} "
                        f"lane={lane}: mismatched_bytes={int(mismatch.sum())} "
                        f"first(row,byte,actual,expected)={examples}"
                    )
    if arm.runtime.sticky_failed_count != 0:
        raise RuntimeError(f"{arm.name} publication poisoned the runtime")


def _median(samples: list[dict[str, float]], field: str) -> float:
    return statistics.median(sample[field] for sample in samples)


def _write_output(path: Path, encoded: str) -> None:
    parent_existed = path.parent.exists()
    atomic_write_text(path, encoded + "\n")
    if os.geteuid() != 0 or "SUDO_UID" not in os.environ:
        return
    owner = int(os.environ["SUDO_UID"])
    group = int(os.environ.get("SUDO_GID", owner))
    os.chown(path, owner, group)
    if not parent_existed:
        os.chown(path.parent, owner, group)


def main() -> None:
    arguments = _arguments()
    endpoint = os.environ.get("NTA_NVME_ENDPOINT", "")
    reference = Path(os.environ.get("NTA_NVME_REFERENCE", ""))
    media_policy = os.environ.get("NTA_NVME_MEDIA_POLICY", "")
    if not endpoint or not reference.is_file():
        raise RuntimeError("physical publication benchmark requires endpoint/reference")
    if os.environ.get("NTA_NVME_DMA_TARGET", "") != "hbm-peer":
        raise RuntimeError("publication benchmark requires direct HBM DMA")

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
    source_rows = 1 + arguments.source_stride * (arguments.runs - 1)
    direct_tier = _PhysicalTier(
        transport,
        layer_count=arguments.layers,
        source_rows=source_rows,
        issue_budget=arguments.issue_budget,
    )
    calibrated_model = (
        None
        if arguments.command_service_ns == 0
        else NvmeTransferServiceModel(
            command_service_ns=arguments.command_service_ns,
            read_bandwidth_bytes_per_second=arguments.read_bandwidth_bps,
            compaction_bandwidth_bytes_per_second=(arguments.compaction_bandwidth_bps),
            compaction_launch_ns=arguments.compaction_launch_ns,
            minimum_gain=arguments.minimum_granularity_gain,
        )
    )
    program: JitPhaseProgram | None = None
    arms: dict[str, _Arm] = {}
    try:
        capabilities = transport.capabilities
        if (
            SOURCE_OFFSET % capabilities.lba_size
            or ROW_BYTES % capabilities.lba_size
            or ROW_BYTES > capabilities.max_transfer_bytes
            or SOURCE_OFFSET + direct_tier.source_bytes > capabilities.namespace_bytes
            or SOURCE_OFFSET + direct_tier.source_bytes > reference.stat().st_size
        ):
            raise RuntimeError("publication benchmark is not NVMe materializable")
        program, _path, _digest = load_activated_transport_program()
        production_window_layers = (
            arguments.layers
            if arguments.production_window_layers == 0
            else arguments.production_window_layers
        )
        arm_configurations = {
            "ordered_batch": (
                False,
                False,
                production_window_layers,
                arguments.pipeline_window_layers or None,
            ),
            "ordered_scalar": (
                True,
                False,
                production_window_layers,
                arguments.pipeline_window_layers or None,
            ),
            "heap_batch": (
                False,
                True,
                production_window_layers,
                arguments.pipeline_window_layers or None,
            ),
            "layered_batch": (False, False, 1, 1),
        }
        arm_tiers = {name: direct_tier for name in arm_configurations}
        if calibrated_model is not None:
            arm_configurations["granularity_auto"] = (
                False,
                False,
                production_window_layers,
                arguments.pipeline_window_layers or None,
            )
            arm_tiers["granularity_auto"] = _PhysicalTier(
                transport,
                layer_count=arguments.layers,
                source_rows=source_rows,
                issue_budget=arguments.issue_budget,
                service_model=calibrated_model,
            )
        arms = {
            name: _Arm(
                name=name,
                transport=transport,
                tier=arm_tiers[name],
                program=program,
                layer_count=arguments.layers,
                run_count=arguments.runs,
                source_stride=arguments.source_stride,
                scalar_publication=configuration[0],
                generic_heap=configuration[1],
                window_layers=configuration[2],
                pipeline_window_layers=configuration[3],
            )
            for name, configuration in arm_configurations.items()
        }
        arm_names = tuple(arm_configurations)
        for _ in range(arguments.warmup):
            for name in arm_names:
                arms[name].run()

        samples: dict[str, list[dict[str, float]]] = {name: [] for name in arm_names}
        for trial in range(arguments.trials):
            pivot = trial % len(arm_names)
            order = arm_names[pivot:] + arm_names[:pivot]
            for name in order:
                samples[name].append(arms[name].run())

        for arm in arms.values():
            _verify(
                arm,
                reference=reference,
                layer_count=arguments.layers,
                source_rows=source_rows,
                run_count=arguments.runs,
                source_stride=arguments.source_stride,
            )
        queue = transport.stats
        if queue.failed or queue.outstanding or queue.error:
            raise RuntimeError(f"publication benchmark left NVMe queue failed: {queue}")

        medians = {
            name: {field: _median(values, field) for field in values[0]}
            for name, values in samples.items()
        }
        comparisons = {
            "bulk_publication": {
                field: medians["ordered_scalar"][field]
                / medians["ordered_batch"][field]
                for field in ("prepare_ms", "gpu_pipeline_ms", "wall_ms")
            },
            "ordered_dispatch": {
                field: medians["heap_batch"][field] / medians["ordered_batch"][field]
                for field in ("prepare_ms", "gpu_pipeline_ms", "wall_ms")
            },
            "queue_fill_window": {
                field: medians["layered_batch"][field] / medians["ordered_batch"][field]
                for field in ("prepare_ms", "gpu_pipeline_ms", "wall_ms")
            },
        }
        if "granularity_auto" in medians:
            comparisons["exact_granularity"] = {
                field: medians["ordered_batch"][field]
                / medians["granularity_auto"][field]
                for field in ("prepare_ms", "gpu_pipeline_ms", "wall_ms")
            }
        gate = {
            "minimum_publication_prepare_speedup": (
                arguments.minimum_publication_prepare_speedup
            ),
            "minimum_publication_wall_speedup": (
                arguments.minimum_publication_wall_speedup
            ),
            "minimum_ordered_wall_speedup": (arguments.minimum_ordered_wall_speedup),
            "minimum_window_wall_speedup": arguments.minimum_window_wall_speedup,
            "minimum_granularity_wall_speedup": (
                arguments.minimum_granularity_wall_speedup
            ),
            "passed": (
                comparisons["bulk_publication"]["prepare_ms"]
                >= arguments.minimum_publication_prepare_speedup
                and comparisons["bulk_publication"]["wall_ms"]
                >= arguments.minimum_publication_wall_speedup
                and comparisons["ordered_dispatch"]["wall_ms"]
                >= arguments.minimum_ordered_wall_speedup
                and comparisons["queue_fill_window"]["wall_ms"]
                >= arguments.minimum_window_wall_speedup
                and (
                    arguments.minimum_granularity_wall_speedup == 0
                    or comparisons["exact_granularity"]["wall_ms"]
                    >= arguments.minimum_granularity_wall_speedup
                )
            ),
        }
        transfer_runs_per_lane = (
            (arguments.runs + capabilities.max_transfer_bytes // ROW_BYTES - 1)
            // (capabilities.max_transfer_bytes // ROW_BYTES)
            if arguments.source_stride == 1
            else arguments.runs
        )
        objects_per_layer = 2 * transfer_runs_per_lane
        report = {
            "schema": 3,
            "benchmark": "sglang_nvme_pipeline_causal",
            "read_only": True,
            "layers": arguments.layers,
            "runs_per_layer": arguments.runs,
            "source_stride_rows": arguments.source_stride,
            "objects_per_layer": objects_per_layer,
            "commands_per_trial": arguments.layers * objects_per_layer,
            "bytes_per_trial": 2 * arguments.layers * arguments.runs * ROW_BYTES,
            "warmup": arguments.warmup,
            "trials": arguments.trials,
            "issue_budget": direct_tier.config.issue_budget,
            "runtime_window_capacity_layers": production_window_layers,
            "pipeline_window_layer_limit": arguments.pipeline_window_layers,
            "granularity_service_model": {
                "calibrated": calibrated_model is not None,
                "command_service_ns": arguments.command_service_ns or None,
                "read_bandwidth_bytes_per_second": (
                    arguments.read_bandwidth_bps or None
                ),
                "compaction_bandwidth_bytes_per_second": (
                    arguments.compaction_bandwidth_bps or None
                ),
                "compaction_launch_ns": (
                    arguments.compaction_launch_ns
                    if calibrated_model is not None
                    else None
                ),
                "minimum_gain": (
                    arguments.minimum_granularity_gain
                    if calibrated_model is not None
                    else None
                ),
            },
            "window_count": {name: arm.window_count for name, arm in arms.items()},
            "pipeline_stats": {name: arm.stats for name, arm in arms.items()},
            "samples": samples,
            "median": medians,
            "comparison_speedup": comparisons,
            "gate": gate,
            "queue": {
                "submitted": queue.submitted,
                "completed": queue.completed,
                "failed": queue.failed,
                "outstanding": queue.outstanding,
                "error": queue.error,
            },
        }
        encoded = json.dumps(report, indent=2, sort_keys=True)
        if arguments.output is not None:
            _write_output(arguments.output, encoded)
        print(encoded)
        if not gate["passed"]:
            raise RuntimeError(
                "NVMe pipeline missed its causal performance gate: "
                f"publication_prepare="
                f"{comparisons['bulk_publication']['prepare_ms']:.3f}x "
                f"publication_wall="
                f"{comparisons['bulk_publication']['wall_ms']:.3f}x "
                f"ordered_wall="
                f"{comparisons['ordered_dispatch']['wall_ms']:.3f}x "
                f"window_wall="
                f"{comparisons['queue_fill_window']['wall_ms']:.3f}x "
                f"granularity_wall="
                f"{comparisons.get('exact_granularity', {}).get('wall_ms', 1.0):.3f}x"
            )
    finally:
        try:
            torch.cuda.synchronize()
        except BaseException:
            pass
        for arm in arms.values():
            arm.close()
        if program is not None:
            program.close()
        transport.close()


if __name__ == "__main__":
    main()
