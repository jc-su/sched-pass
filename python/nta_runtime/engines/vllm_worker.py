"""Worker-local vLLM runtime, identity, and publication ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any, Literal
import weakref

import torch

from nta_runtime.adapters.base import EngineBatch
from nta_runtime.adapters.vllm_v1 import VllmV1Hook
from nta_runtime.engines.vllm_config import (
    SUPPORTED_VLLM_VERSION,
    VllmWorkerConfig,
)
from nta_runtime.engines.vllm_modules import VLLM_STATS
from nta_runtime.hbm_registration import HbmDestinationSlice
from nta_runtime.runtime import Runtime
from nta_runtime.runtime_resources import (
    RuntimeResourceConfig,
    ServingRuntimeResources,
)
from nta_runtime.tier import PageTransferRun, TierPageCatalog
from nta_runtime.work_unit import Granularity


def _build_resources(
    runner: Any, request_capacity: int, config: VllmWorkerConfig
) -> ServingRuntimeResources:
    device = getattr(runner, "device", None)
    device_ordinal = (
        int(device.index)
        if isinstance(device, torch.device) and device.index is not None
        else int(torch.cuda.current_device())
    )
    return ServingRuntimeResources.open(
        tier_config=config.tier,
        runtime_config=RuntimeResourceConfig(
            request_capacity=request_capacity,
            # Physical objects are maximal source/destination-contiguous runs.
            # Keep the bound explicit: normal sequential prefixes collapse to
            # one run, while fragmented catalogs fail closed instead of
            # silently dropping dependencies.
            object_capacity=config.object_capacity,
            intent_capacity=config.object_capacity,
            work_ticket_capacity=config.work_ticket_capacity,
            max_dependencies_per_work_ticket=(config.max_dependencies_per_work_ticket),
            device_ordinal=device_ordinal,
            tenant_capacity=config.tenant_capacity,
            staging_byte_capacity=config.staging_byte_capacity,
        ),
    )


@dataclass
class _WorkerAttentionPhase:
    """One worker-scoped FlashInfer resource and its epoch-scoped plan result."""

    resource: Any
    workspace: "AttentionWorkspaceContract"
    planned_epoch: int | None = None
    plan_result: Any = None


@dataclass(frozen=True, slots=True)
class AttentionWorkspaceContract:
    """Physical ownership of one FlashInfer float-workspace binding.

    Framework-owned buffers may be shared by multiple wrappers because vLLM
    already serializes their plans and launches on one worker stream. Worker-
    owned buffers are reserved for differential execution, where the stock and
    custom wrappers must remain independent.
    """

    capacity_bytes: int
    ownership: Literal["worker", "framework"]
    identity: int | None = None

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError("vLLM attention workspace must be positive")
        if self.ownership == "worker":
            if self.identity is not None:
                raise ValueError("worker-owned workspace cannot borrow an identity")
        elif self.ownership == "framework":
            if self.identity is None or self.identity <= 0:
                raise ValueError(
                    "framework-owned workspace requires a physical identity"
                )
        else:
            raise ValueError(f"unknown vLLM workspace owner {self.ownership!r}")

    @classmethod
    def worker_owned(cls, capacity_bytes: int) -> "AttentionWorkspaceContract":
        return cls(capacity_bytes, "worker")

    @classmethod
    def framework_owned(
        cls, capacity_bytes: int, identity: int
    ) -> "AttentionWorkspaceContract":
        return cls(capacity_bytes, "framework", identity)


@dataclass
class _RequestBindingBuffer:
    """One reusable, stream-fenced request binding upload.

    A tiny ring avoids synchronizing every forward while protecting both the
    pinned host source from H2D write-after-read and the device table from
    reuse before the last request-bound attention launch has consumed it.
    """

    host: torch.Tensor
    host_numpy: Any
    device: torch.Tensor
    in_use_until: torch.cuda.Event | None = None


@dataclass(frozen=True)
class _PhysicalLayerDestination:
    """One framework-owned, setup-time registered packed KV tensor."""

    layer_name: str
    catalog_layer: int
    tensor_address: int
    block_count: int
    block_bytes: int
    region: Any

    def address(self, first_block: int, block_count: int) -> int:
        if first_block < 0 or block_count <= 0:
            raise ValueError("vLLM physical destination range must be positive")
        if first_block > self.block_count - block_count:
            raise RuntimeError("vLLM physical destination exceeds its KV tensor")
        return self.tensor_address + first_block * self.block_bytes


@dataclass(frozen=True)
class _PhysicalTransferLayout:
    """Pure directory result consumed by the runtime publication step."""

    runs: tuple[PageTransferRun, ...]
    run_indices_by_work: tuple[tuple[int, ...], ...]
    unique_block_count: int


def _physical_transfer_layout(
    catalog: TierPageCatalog,
    *,
    catalog_layer: int,
    work_bindings: tuple[tuple[tuple[str, int], ...], ...],
    row_bytes: int,
    max_transfer_bytes: int,
) -> _PhysicalTransferLayout:
    """Coalesce one layer while preserving every work item's ready frontier."""

    key_by_destination: dict[int, str] = {}
    for bindings in work_bindings:
        for storage_key, block in bindings:
            previous = key_by_destination.setdefault(block, storage_key)
            if previous != storage_key:
                raise RuntimeError(
                    "vLLM destination block is bound to conflicting storage keys"
                )
    ordered = tuple(sorted(key_by_destination.items()))
    runs = (
        catalog.transfer_runs(
            layer=catalog_layer,
            storage_keys=tuple(storage_key for _, storage_key in ordered),
            destination_indices=tuple(block for block, _ in ordered),
            component="packed_kv",
            row_bytes=row_bytes,
            max_transfer_bytes=max_transfer_bytes,
        )
        if ordered
        else ()
    )
    run_by_destination: dict[int, int] = {}
    for run_index, run in enumerate(runs):
        for block in range(
            run.destination_first, run.destination_first + run.row_count
        ):
            if block in run_by_destination:
                raise RuntimeError("vLLM physical transfer runs overlap")
            run_by_destination[block] = run_index
    run_indices_by_work = tuple(
        tuple(dict.fromkeys(run_by_destination[block] for _, block in bindings))
        for bindings in work_bindings
    )
    return _PhysicalTransferLayout(runs, run_indices_by_work, len(ordered))


class VllmV1WorkerController:
    """Own one worker-local runtime and identity hook.

    The class name retains the vLLM ``v1`` API namespace distinction; the
    current 0.26 profile uses the ``v1.worker.gpu.model_runner`` V2 runner.
    The controller owns the native runtime for exactly that runner lifetime.
    """

    def __init__(self, runner: Any) -> None:
        self._runner_ref = weakref.ref(runner)
        self._worker_config: VllmWorkerConfig | None = None
        self._runtime: Runtime | None = None
        self._resources: ServingRuntimeResources | None = None
        self._hook: VllmV1Hook | None = None
        self._page_size = 0
        self._page_bytes = 0
        self._request_capacity = 0
        self._epoch = 0
        # FlashInfer planner state and its large workspace are worker resources,
        # not transformer-layer resources.  vLLM invokes one AttentionImpl per
        # layer with the same batch metadata, so a per-impl owner multiplies
        # workspace residency and planner/readback cost by the layer count.
        self._attention_phases: dict[tuple[Any, ...], _WorkerAttentionPhase] = {}
        # Only worker-owned bytes are additional HBM. Framework-owned
        # workspaces are reference-counted by allocation identity so prefill
        # and decode wrappers sharing vLLM's one buffer are not double-counted.
        self._attention_workspace_bytes = 0
        self._attention_borrowed_workspace_bytes = 0
        self._attention_borrowed_workspace_refs: dict[int, tuple[int, int]] = {}
        self._request_binding_buffers: list[_RequestBindingBuffer] = []
        self._request_binding_cursor = 0
        self._request_bindings_device: torch.Tensor | None = None
        self._layer_ordinals: dict[str, int] = {}
        self._physical_destinations: dict[str, _PhysicalLayerDestination] = {}
        self._physical_destinations_prepared = False
        # The runtime object directory is worker-global, while vLLM constructs
        # one AttentionImpl per transformer layer.  Directory generations and
        # consumer quiescence must therefore live here, not in an impl.
        self._external_object_version = 0
        self._external_consumer_event: torch.cuda.Event | None = None
        self._active_forward_state: Any | None = None
        self._forward_quiescence_event: torch.cuda.Event | None = None
        self._external_publication_open = False
        self._external_publication_stream: torch.cuda.Stream | None = None
        self._tenant_isolation_enabled = False
        self._closed = False

    @staticmethod
    def _cache_geometry(runner: Any) -> tuple[int, int]:
        groups = getattr(
            getattr(runner, "kv_cache_config", None), "kv_cache_groups", ()
        )
        if len(groups) != 1:
            raise RuntimeError("NTA vLLM currently requires exactly one KV cache group")
        spec = groups[0].kv_cache_spec
        page_size = int(getattr(spec, "block_size", 0))
        if page_size <= 0:
            raise RuntimeError("vLLM KV cache spec has no positive block_size")
        page_bytes = int(getattr(spec, "page_size_bytes", 0))
        if page_bytes <= 0:
            raise RuntimeError("vLLM KV cache spec has no page_size_bytes")
        return page_size, page_bytes

    def _ensure_hook(
        self,
        runner: Any,
        request_capacity: int,
        page_size: int,
        page_bytes: int,
    ) -> VllmV1Hook:
        if self._closed:
            raise RuntimeError("vLLM worker controller is closed")
        if self._runtime is None:
            scheduler_config = getattr(
                getattr(runner, "vllm_config", None), "scheduler_config", None
            )
            max_batched_tokens = int(
                getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
            )
            config = VllmWorkerConfig.from_environment(
                request_capacity=request_capacity,
                max_batched_tokens=max_batched_tokens,
            )
            resources = _build_resources(runner, request_capacity, config)
            runtime = resources.runtime
            try:
                tenant_capacity = int(runtime.config.tenant_capacity)
                for tenant_id, max_bytes in config.tenant_specs:
                    if tenant_id >= tenant_capacity:
                        raise RuntimeError(
                            f"tenant {tenant_id} exceeds NTA_TENANT_CAPACITY="
                            f"{tenant_capacity}"
                        )
                    runtime.set_tenant_budget(tenant_id, max_bytes)
                hook = VllmV1Hook(
                    runtime,
                    request_capacity,
                    page_bytes=page_bytes,
                    expected_vllm_version=SUPPORTED_VLLM_VERSION,
                    tenant_for_request=config.tenant_for_request,
                    profile_cpu=config.profile_cpu,
                )
            except BaseException:
                try:
                    resources.close()
                except BaseException:
                    pass
                raise
            self._resources = resources
            self._runtime = runtime
            self._hook = hook
            self._worker_config = config
            self._tenant_isolation_enabled = config.tenant_isolation_enabled
            self._request_capacity = request_capacity
            self._page_bytes = page_bytes
        elif (
            self._request_capacity != request_capacity or self._page_bytes != page_bytes
        ):
            raise RuntimeError(
                "vLLM KV cache geometry changed while the worker runtime was live"
            )
        self._page_size = page_size
        assert self._hook is not None
        return self._hook

    def _publish_request_bindings(self, batch: EngineBatch) -> None:
        """Publish phase-sliceable ``(slot, generation)`` rows per forward."""
        if not self._request_binding_buffers:
            runner = self._runner_ref()
            if runner is None:
                raise RuntimeError("vLLM model runner was destroyed")
            device = getattr(runner, "device", torch.device("cuda"))
            for _ in range(2):
                host = torch.empty(
                    2 * self._request_capacity,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                self._request_binding_buffers.append(
                    _RequestBindingBuffer(
                        host=host,
                        host_numpy=host.numpy().reshape(self._request_capacity, 2),
                        device=torch.empty(
                            2 * self._request_capacity,
                            dtype=torch.int64,
                            device=device,
                        ),
                    )
                )
        buffer = self._request_binding_buffers[self._request_binding_cursor]
        self._request_binding_cursor = (self._request_binding_cursor + 1) % len(
            self._request_binding_buffers
        )
        if buffer.in_use_until is not None:
            buffer.in_use_until.synchronize()
        rows = buffer.host_numpy
        rows.fill(-1)
        rows[: len(batch.bindings), 0] = batch.request_slots
        rows[: len(batch.bindings), 1] = batch.generations
        stream = torch.cuda.current_stream(buffer.device.device)
        buffer.device.copy_(buffer.host, non_blocking=True)
        uploaded = torch.cuda.Event()
        uploaded.record(stream)
        buffer.in_use_until = uploaded
        self._request_bindings_device = buffer.device

    def record_request_binding_consumer(
        self, bindings: torch.Tensor, stream: torch.cuda.Stream
    ) -> None:
        """Fence the ring entry used by one request-bound launch."""

        address = int(bindings.data_ptr())
        for buffer in self._request_binding_buffers:
            begin = int(buffer.device.data_ptr())
            end = begin + int(buffer.device.numel() * buffer.device.element_size())
            if begin <= address < end:
                event = torch.cuda.Event()
                event.record(stream)
                buffer.in_use_until = event
                return
        raise RuntimeError("vLLM request-bound launch used an unowned binding table")

    def bind(self, scheduler_output: Any) -> EngineBatch:
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V1 model runner was destroyed")
        input_batch = getattr(runner, "input_batch", None)
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if input_batch is None or request_capacity <= 0:
            raise RuntimeError("vLLM V1 runner is not initialized with InputBatch")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(runner, request_capacity, page_size, page_bytes)
        started = time.perf_counter_ns()
        batch = hook.bind_forward(
            scheduler_output,
            input_batch,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._publish_request_bindings(batch)
        assert self._worker_config is not None
        if self._worker_config.profile_cpu:
            VLLM_STATS["bridge_bind_cpu_ns"] += time.perf_counter_ns() - started
            VLLM_STATS["bridge_bind_calls"] += 1
        self._epoch += 1
        return batch

    def bind_v2(
        self,
        scheduler_output: Any,
        input_batch: Any,
        *,
        block_tables: Any,
        num_blocks: Any,
    ) -> EngineBatch:
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V2 model runner was destroyed")
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if request_capacity <= 0:
            raise RuntimeError("vLLM V2 runner has no positive request capacity")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(runner, request_capacity, page_size, page_bytes)
        started = time.perf_counter_ns()
        batch = hook.bind_v2_forward(
            scheduler_output,
            input_batch,
            block_tables=block_tables,
            num_blocks=num_blocks,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._publish_request_bindings(batch)
        assert self._worker_config is not None
        if self._worker_config.profile_cpu:
            VLLM_STATS["bridge_bind_cpu_ns"] += time.perf_counter_ns() - started
            VLLM_STATS["bridge_bind_calls"] += 1
            VLLM_STATS.update(hook.last_bind_profile)
        self._epoch += 1
        return batch

    def bind_connector(self, metadata: Any, input_batch: Any) -> EngineBatch:
        """Bind one official KVConnector metadata object before FI planning."""
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM V2 model runner was destroyed")
        request_capacity = int(getattr(runner, "max_num_reqs", 0))
        if request_capacity <= 0:
            raise RuntimeError("vLLM V2 runner has no positive request capacity")
        page_size, page_bytes = self._cache_geometry(runner)
        hook = self._ensure_hook(runner, request_capacity, page_size, page_bytes)
        batch = hook.bind_connector_forward(
            metadata.request_ids,
            metadata.block_tables,
            metadata.finished_request_ids,
            input_batch=input_batch,
            epoch=self._epoch,
            stream=torch.cuda.current_stream(),
            granularity=Granularity.PAGE_GROUP,
        )
        self._publish_request_bindings(batch)
        self._epoch += 1
        return batch

    def prepare_physical_destinations(self) -> None:
        """Register every packed vLLM layer destination exactly once.

        vLLM owns allocation and numerical lifetime.  NTA owns only peer
        mapping views over those tensors, and installs per-transfer object
        views later without allocation, registration, or ioctl work in the
        forward path.
        """

        if self._resources is None:
            raise RuntimeError("vLLM physical setup ran before runtime binding")
        tier = self._resources.tier
        if tier.is_hbm or tier.is_host_staged:
            return
        if not tier.is_nvme:
            raise RuntimeError(
                f"vLLM {tier.tier.value} numerical consumption is deferred; "
                "the profile fails closed instead of preparing the wrong address space"
            )
        if self._physical_destinations_prepared:
            return
        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("vLLM model runner was destroyed")
        catalog = tier.catalog
        if catalog is None or catalog.components != ("packed_kv",):
            raise RuntimeError("vLLM NVMe setup requires a packed_kv catalog")
        groups = tuple(getattr(runner.kv_cache_config, "kv_cache_groups", ()))
        if len(groups) != 1:
            raise RuntimeError("vLLM NVMe setup requires exactly one KV group")
        layer_names = tuple(str(value) for value in groups[0].layer_names)
        if len(layer_names) != catalog.layer_count:
            raise RuntimeError("vLLM KV layers do not match the physical catalog")
        context = getattr(
            getattr(runner, "compilation_config", None),
            "static_forward_context",
            None,
        )
        if not isinstance(context, dict):
            raise RuntimeError("vLLM runner has no static attention-layer directory")

        destinations: list[HbmDestinationSlice] = []
        geometry: dict[str, tuple[int, int]] = {}
        addresses: set[int] = set()
        for layer_name in layer_names:
            layer = context.get(layer_name)
            tensor = getattr(layer, "kv_cache", None)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise RuntimeError(
                    f"vLLM layer {layer_name!r} has no CUDA KV destination"
                )
            if tensor.ndim < 1 or tensor.shape[0] <= 0:
                raise RuntimeError(f"vLLM layer {layer_name!r} has invalid KV shape")
            block_count = int(tensor.shape[0])
            block_stride = int(tensor.stride(0)) * int(tensor.element_size())
            if block_stride != self._page_bytes:
                raise RuntimeError(
                    f"vLLM layer {layer_name!r} block stride {block_stride} "
                    f"does not match catalog payload bytes {self._page_bytes}"
                )
            address = int(tensor.data_ptr())
            if address <= 0 or address in addresses:
                raise RuntimeError(
                    "vLLM physical profile does not support shared/aliased layer "
                    "KV destinations"
                )
            addresses.add(address)
            total_bytes = block_count * self._page_bytes
            destinations.append(HbmDestinationSlice(layer_name, address, total_bytes))
            geometry[layer_name] = (address, block_count)

        preparation = tier.prepare_nvme_hbm_destinations(tuple(destinations))
        prepared: dict[str, _PhysicalLayerDestination] = {}
        for catalog_layer, layer_name in enumerate(layer_names):
            address, block_count = geometry[layer_name]
            region = preparation.regions.get(layer_name)
            if region is None:
                raise RuntimeError("vLLM NVMe setup lost a registered layer mapping")
            prepared[layer_name] = _PhysicalLayerDestination(
                layer_name,
                catalog_layer,
                address,
                block_count,
                self._page_bytes,
                region,
            )
        self._physical_destinations = prepared
        self._physical_destinations_prepared = True
        VLLM_STATS["physical_destination_layers"] = len(prepared)
        VLLM_STATS["physical_destination_registrations"] = (
            preparation.registration_count
        )
        VLLM_STATS["physical_destination_bytes"] = preparation.destination_bytes
        VLLM_STATS["physical_registration_bytes"] = preparation.registration_bytes

    def physical_destination(self, layer: Any) -> _PhysicalLayerDestination:
        """Resolve an Attention layer through vLLM's stable layer-name seam."""

        if not self._physical_destinations_prepared:
            raise RuntimeError("vLLM physical destinations were not prepared")
        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM attention layer has no stable layer_name")
        try:
            return self._physical_destinations[layer_name]
        except KeyError:
            raise RuntimeError(
                f"vLLM layer {layer_name!r} is absent from the physical directory"
            ) from None

    def semantic_layer(self, layer: Any) -> int:
        """Resolve one framework layer to a stable model-local ordinal."""

        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM attention layer has no stable layer_name")
        if not self._layer_ordinals:
            runner = self._runner_ref()
            groups = tuple(
                getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", ())
            )
            if len(groups) != 1:
                raise RuntimeError(
                    "vLLM semantic layer directory requires one KV group"
                )
            names = tuple(str(value) for value in groups[0].layer_names)
            if not names or len(set(names)) != len(names):
                raise RuntimeError("vLLM KV layer directory is empty or ambiguous")
            self._layer_ordinals = {name: ordinal for ordinal, name in enumerate(names)}
        try:
            return self._layer_ordinals[layer_name]
        except KeyError:
            raise RuntimeError(
                f"vLLM layer {layer_name!r} is absent from the semantic directory"
            ) from None

    def attention_phase(
        self,
        form: str,
        key: tuple[Any, ...],
        epoch: int,
        build: Callable[[], Any],
        plan: Callable[[Any], Any],
        *,
        workspace: AttentionWorkspaceContract,
    ) -> tuple[Any, Any]:
        """Acquire one worker-shared wrapper and plan it once per batch epoch.

        vLLM shares one FlashInfer metadata plan across all transformer layers.
        NTA must preserve the same ownership boundary: planning a custom wrapper
        in every ``AttentionImpl`` adds repeated planner work, device-to-host
        schedule readback, and a full workspace per layer.  The worker executes
        layers serially, so one resource per form/phase/signature is the
        narrowest safe lifetime.  ``plan_result`` lets incremental consumers
        cache the extracted structural schedule alongside the wrapper.
        """
        if form not in {"request_bound", "incremental"}:
            raise ValueError(f"unknown vLLM attention form {form!r}")
        if epoch < 0:
            raise ValueError("vLLM attention phase epoch cannot be negative")
        phase_key = (form, *key)
        phase = self._attention_phases.get(phase_key)
        if phase is not None and phase.workspace != workspace:
            # A framework metadata builder may replace its workspace between
            # forwards. Rebuild this logical phase instead of retaining a
            # wrapper that pins stale framework storage indefinitely.
            self._release_attention_workspace(phase.workspace)
            del self._attention_phases[phase_key]
            phase = None
        if phase is None:
            phase = _WorkerAttentionPhase(build(), workspace)
            self._attention_phases[phase_key] = phase
            self._retain_attention_workspace(workspace)
            prefix = f"worker_{form}"
            VLLM_STATS[f"{prefix}_wrapper_builds"] += 1
            if workspace.ownership == "worker":
                VLLM_STATS[f"{prefix}_workspace_allocated_bytes"] += (
                    workspace.capacity_bytes
                )
            else:
                VLLM_STATS[f"{prefix}_workspace_borrowed_bindings"] += 1
        if phase.planned_epoch == epoch:
            VLLM_STATS[f"worker_{form}_plan_reuses"] += 1
            return phase.resource, phase.plan_result
        if phase.planned_epoch is not None and epoch < phase.planned_epoch:
            raise RuntimeError("vLLM attention phase epoch moved backwards")
        phase.plan_result = plan(phase.resource)
        phase.planned_epoch = epoch
        VLLM_STATS[f"worker_{form}_plan_builds"] += 1
        return phase.resource, phase.plan_result

    def _retain_attention_workspace(
        self, workspace: AttentionWorkspaceContract
    ) -> None:
        if workspace.ownership == "worker":
            self._attention_workspace_bytes += workspace.capacity_bytes
            VLLM_STATS["worker_attention_workspace_peak_bytes"] = max(
                VLLM_STATS["worker_attention_workspace_peak_bytes"],
                self._attention_workspace_bytes,
            )
            return
        assert workspace.identity is not None
        prior = self._attention_borrowed_workspace_refs.get(workspace.identity)
        if prior is None:
            self._attention_borrowed_workspace_refs[workspace.identity] = (
                workspace.capacity_bytes,
                1,
            )
            self._attention_borrowed_workspace_bytes += workspace.capacity_bytes
            VLLM_STATS["worker_attention_borrowed_workspace_peak_bytes"] = max(
                VLLM_STATS["worker_attention_borrowed_workspace_peak_bytes"],
                self._attention_borrowed_workspace_bytes,
            )
            return
        capacity, references = prior
        if capacity != workspace.capacity_bytes:
            raise RuntimeError("vLLM framework workspace identity changed capacity")
        self._attention_borrowed_workspace_refs[workspace.identity] = (
            capacity,
            references + 1,
        )

    def _release_attention_workspace(
        self, workspace: AttentionWorkspaceContract
    ) -> None:
        if workspace.ownership == "worker":
            self._attention_workspace_bytes -= workspace.capacity_bytes
            if self._attention_workspace_bytes < 0:
                raise RuntimeError("vLLM worker workspace accounting underflow")
            return
        assert workspace.identity is not None
        prior = self._attention_borrowed_workspace_refs.get(workspace.identity)
        if prior is None:
            raise RuntimeError("vLLM borrowed workspace accounting underflow")
        capacity, references = prior
        if capacity != workspace.capacity_bytes:
            raise RuntimeError("vLLM borrowed workspace capacity changed")
        if references == 1:
            del self._attention_borrowed_workspace_refs[workspace.identity]
            self._attention_borrowed_workspace_bytes -= capacity
        else:
            self._attention_borrowed_workspace_refs[workspace.identity] = (
                capacity,
                references - 1,
            )

    def begin_forward(self, state: Any) -> None:
        """Open one worker transaction before identity or acquisition binding."""
        if self._active_forward_state is not None:
            raise RuntimeError("vLLM worker forward lifetimes overlap")
        self._active_forward_state = state
        self._forward_quiescence_event = self._external_consumer_event
        self._external_publication_open = False
        self._external_publication_stream = None

    def validate_forward_commit(self, state: Any) -> None:
        if self._active_forward_state is not state:
            raise RuntimeError("vLLM worker committed the wrong forward")
        if self._external_publication_open:
            raise RuntimeError("vLLM external publication has no consumer fence")

    def commit_forward(self, state: Any) -> None:
        """Atomically retain only quiescence produced by a successful forward."""
        self.validate_forward_commit(state)
        self._external_consumer_event = self._forward_quiescence_event
        self._active_forward_state = None
        self._forward_quiescence_event = None
        self._external_publication_stream = None

    def abort_forward(self, state: Any) -> None:
        """Close every publication lease while preserving GPU quiescence."""
        if self._active_forward_state is None:
            return
        if self._active_forward_state is not state:
            raise RuntimeError("vLLM worker aborted the wrong forward")
        if self._external_publication_open:
            stream = self._external_publication_stream
            if stream is None:
                raise RuntimeError("vLLM external publication lost its CUDA stream")
            event = torch.cuda.Event()
            event.record(stream)
            self._forward_quiescence_event = event
        self._external_consumer_event = self._forward_quiescence_event
        self._active_forward_state = None
        self._forward_quiescence_event = None
        self._external_publication_open = False
        self._external_publication_stream = None

    def begin_external_publication(
        self, stream: torch.cuda.Stream
    ) -> tuple[int, torch.cuda.Event | None]:
        """Allocate a directory generation inside the active forward lease."""
        if self._active_forward_state is None:
            raise RuntimeError("vLLM external publication ran outside a forward")
        if self._external_publication_open:
            raise RuntimeError("vLLM external publications overlap")
        self._external_object_version = (self._external_object_version + 1) & 0xFFFFFFFF
        self._external_object_version = self._external_object_version or 1
        self._external_publication_open = True
        self._external_publication_stream = stream
        return self._external_object_version, self._forward_quiescence_event

    def record_external_consumer(self, stream: torch.cuda.Stream) -> None:
        """Publish completion ordering for the next directory replacement."""
        if self._active_forward_state is None or not self._external_publication_open:
            raise RuntimeError("vLLM external consumer has no publication lease")
        expected_stream = self._external_publication_stream
        expected_handle = getattr(expected_stream, "cuda_stream", expected_stream)
        actual_handle = getattr(stream, "cuda_stream", stream)
        if expected_stream is None or actual_handle != expected_handle:
            raise RuntimeError(
                "vLLM external publication and consumer use different CUDA streams"
            )
        event = torch.cuda.Event()
        event.record(stream)
        self._forward_quiescence_event = event
        self._external_publication_open = False
        self._external_publication_stream = None

    def close(self) -> None:
        """Close the runtime after the framework has stopped using the runner."""
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        active = self._active_forward_state
        if active is not None:
            try:
                self.abort_forward(active)
            except BaseException as error:
                failure = error
        runtime, self._runtime = self._runtime, None
        resources, self._resources = self._resources, None
        self._hook = None
        self._page_size = 0
        self._page_bytes = 0
        self._request_capacity = 0
        for phase in self._attention_phases.values():
            self._release_attention_workspace(phase.workspace)
        self._attention_phases.clear()
        self._attention_workspace_bytes = 0
        self._attention_borrowed_workspace_bytes = 0
        self._attention_borrowed_workspace_refs.clear()
        self._request_binding_buffers.clear()
        self._request_binding_cursor = 0
        self._request_bindings_device = None
        self._layer_ordinals.clear()
        self._physical_destinations.clear()
        self._physical_destinations_prepared = False
        self._external_consumer_event = None
        self._external_object_version = 0
        self._active_forward_state = None
        self._forward_quiescence_event = None
        self._external_publication_open = False
        self._external_publication_stream = None
        self._tenant_isolation_enabled = False
        try:
            if resources is not None:
                resources.close()
            elif runtime is not None:
                runtime.close()
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"vLLM runtime close also failed: {error!r}")
        if failure is not None:
            raise failure

    def __del__(self) -> None:
        # vLLM may abandon a worker during initialization (for example after
        # a tenant policy or KV geometry rejection).  The normal shutdown hook
        # is not guaranteed to run for that partial worker, so retain the same
        # best-effort runtime ownership fallback as the serving adapters.
        try:
            self.close()
        except BaseException:
            pass

    @property
    def hook(self) -> VllmV1Hook:
        if self._hook is None:
            raise RuntimeError("vLLM NTA worker controller has not bound a forward")
        return self._hook

    @property
    def page_size(self) -> int:
        if self._page_size <= 0:
            raise RuntimeError("vLLM V1 worker controller has no page size")
        return self._page_size

    @property
    def tier_service(self) -> Any:
        if self._resources is None:
            raise RuntimeError("vLLM worker controller has no serving tier")
        return self._resources.tier

    @property
    def tenant_isolation_enabled(self) -> bool:
        return self._tenant_isolation_enabled

    @property
    def request_bindings_tensor(self) -> torch.Tensor:
        if self._request_bindings_device is None:
            raise RuntimeError("vLLM worker has no published request-binding map")
        return self._request_bindings_device


def _controller(runner: Any) -> VllmV1WorkerController:
    controller = getattr(runner, "_nta_vllm_controller", None)
    if controller is None:
        controller = VllmV1WorkerController(runner)
        setattr(runner, "_nta_vllm_controller", controller)
    return controller


def _commit_forward(state: Any) -> None:
    """Commit lifecycle ownership and evidence after numerical success."""
    owner = getattr(state, "execution_owner", None)
    connector = getattr(state, "connector_owner", None)
    counters = state.validate_evidence_commit()
    native_launches = counters.get("native_decode_launches", 0) + counters.get(
        "native_prefill_launches", 0
    )
    if native_launches and state.hook is None:
        raise RuntimeError("vLLM direct forward completed without an identity hook")
    if state.batch is not None:
        if not isinstance(owner, VllmV1WorkerController):
            raise RuntimeError("vLLM forward has no worker transaction owner")
        if connector is None:
            raise RuntimeError("vLLM forward has no connector transaction owner")
        owner.validate_forward_commit(state)
        connector.validate_forward_commit()
        # Everything that can reject the transaction has now been checked.
        # The following commits are field publications only, so an exception
        # cannot leave one owner committed while another remains active.
        owner.commit_forward(state)
        connector.commit_forward()
    committed = state.commit_evidence()
    if committed != counters:  # pragma: no cover - same-thread context contract
        raise RuntimeError("vLLM forward evidence changed during commit")
    if native_launches:
        state.hook.record_native_launch(native_launches)
    VLLM_STATS.update(counters)


def _abort_forward(state: Any) -> None:
    """Abort all forward-local owners and discard uncommitted evidence."""
    failures: list[BaseException] = []
    owner = getattr(state, "execution_owner", None)
    if isinstance(owner, VllmV1WorkerController):
        try:
            owner.abort_forward(state)
        except BaseException as error:
            failures.append(error)
    connector = getattr(state, "connector_owner", None)
    if connector is not None:
        try:
            connector.abort_forward()
        except BaseException as error:
            failures.append(error)
    try:
        state.abort_evidence()
    except BaseException as error:
        failures.append(error)
    if failures:
        primary = failures[0]
        for failure in failures[1:]:
            primary.add_note(f"additional vLLM abort failure: {failure!r}")
        raise primary


__all__ = [
    "AttentionWorkspaceContract",
    "VllmV1WorkerController",
    "_PhysicalLayerDestination",
    "_controller",
    "_commit_forward",
    "_abort_forward",
    "_physical_transfer_layout",
]
