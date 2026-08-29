"""Pinned vLLM V1 worker boundary.

This module is the only place where the vLLM 0.26 V1 worker object layout is
known.  It intentionally keeps the vLLM import optional: contract tests can
use small structural doubles, while a real worker hook verifies the installed
distribution before accepting a forward.

The vLLM input-batch row is a mutable implementation detail and can move when
the persistent batch is compacted.  It is therefore not used as NTA's request
slot.  ``VllmV1Hook`` owns a bounded stable slot table and treats vLLM's CPU
block-table mirror as an exact-demand source, never as a policy decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field
import heapq
import importlib.metadata
from operator import index as integer_index
import os
import time
from typing import Any, Protocol

from .base import ConsumerContract, ExactDemandProjection, EngineBatch
from .vllm import VllmAdapter
from ..resource_contract import (
    ResourceAddressSpace,
    ResourceOwner,
    ResourcePath,
    require_numerical_binding,
    resource_contract,
)
from ..work_unit import Granularity


SUPPORTED_VLLM_V1_VERSION = "0.26.0"


def validate_vllm_attention_tier(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Validate the tier visible to the vLLM numerical consumer.

    A physical tier is accepted only as an explicit NTA native profile.  The
    worker resource owner opens and validates the transport, while the
    attention consumer still fails closed if it is not enabled.  This avoids
    silently treating a physical-tier configuration as resident vLLM data.
    """
    values = os.environ if environ is None else environ
    selected = values.get("NTA_SERVING_TIER", "hbm").strip().lower()
    if selected not in {"hbm", "host_mapped", "host_staged", "nvme", "cxl_dax"}:
        raise RuntimeError(
            "NTA_SERVING_TIER must be hbm, host_mapped, host_staged, nvme, "
            "or cxl_dax for vLLM"
        )
    require_numerical_binding(
        resource_contract(selected),
        ResourceOwner.ENGINE,
        ResourceAddressSpace.HBM,
        frozenset((ResourcePath.RESIDENT, ResourcePath.MATERIALIZED)),
        consumer="vLLM FlashInfer paged-KV attention",
    )
    if selected == "host_staged" and values.get("NTA_VLLM_NATIVE", "0") != "1":
        raise RuntimeError("vLLM host_staged requires NTA_VLLM_NATIVE=1")
    if selected == "nvme":
        if values.get("NTA_VLLM_NATIVE", "0") != "1":
            raise RuntimeError(
                "vLLM physical tiers require NTA_VLLM_NATIVE=1; stock resident "
                "attention cannot consume NVMe or CXL-DAX data"
            )
        if values.get("NTA_VLLM_PHYSICAL_CATALOG", "0") != "1":
            raise RuntimeError(
                "vLLM physical tiers require the explicit "
                "NTA_VLLM_PHYSICAL_CATALOG=1 replay profile; the upstream "
                "KVConnector ownership/write-back lifecycle is not implicit"
            )
    return selected


@dataclass
class VllmV1ForwardState:
    """Per-forward sidecar shared by the worker hook and attention layers.

    vLLM deliberately keeps ``SchedulerOutput`` out of ``AttentionImpl``.
    The sidecar is therefore the narrow control-plane bridge: the worker
    publishes one exact :class:`EngineBatch` after ``_update_states`` and
    every attention layer consumes the same immutable batch.  It is mutable
    only while the worker is establishing the sidecar; attention code must
    never replace ``batch`` after the first layer has entered.
    """

    scheduler_output: Any | None = None
    input_batch: Any | None = None
    batch: EngineBatch | None = None
    connector_metadata: Any | None = None
    hook: "VllmV1Hook | None" = None
    tier_service: Any | None = None
    execution_owner: Any | None = None
    connector_owner: Any | None = None
    request_bindings_tensor: Any | None = None
    connector_validated: bool = False
    page_size: int = 0
    storage_key_tables: tuple[tuple[str | None, ...], ...] = ()
    host_transfer_pairs: tuple[tuple[int, int], ...] = ()
    host_resources: dict[str, Any] = field(default_factory=dict)
    host_acquisition: Any | None = None
    tenant_isolation_enabled: bool = False
    host_index_tensors: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]], tuple[Any, Any]
    ] = field(default_factory=dict)
    reference_warmup: bool = False
    _phase_batches: dict[tuple[int, int], EngineBatch] = field(
        default_factory=dict, init=False, repr=False
    )
    _forward_evidence: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _evidence_committed: bool = field(default=False, init=False, repr=False)
    _evidence_aborted: bool = field(default=False, init=False, repr=False)
    _active_host_layer: str | None = field(default=None, init=False, repr=False)
    _host_consumed_destinations: set[int] = field(
        default_factory=set, init=False, repr=False
    )
    _host_schedule_observations: list[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]
    ] = field(default_factory=list, init=False, repr=False)
    _host_waited_fences: set[int] = field(default_factory=set, init=False, repr=False)

    def begin_host_layer(self, layer_name: str) -> None:
        """Open exact host-transfer accounting for one numerical layer."""
        if not layer_name or self._active_host_layer is not None:
            raise RuntimeError("vLLM host layer lifetime is overlapping or unnamed")
        self._active_host_layer = layer_name
        self._host_consumed_destinations.clear()
        self._host_schedule_observations.clear()

    def consumed_host_destinations(self, layer_name: str) -> frozenset[int]:
        if self._active_host_layer != layer_name:
            raise RuntimeError("vLLM host transfer ran outside its active layer")
        return frozenset(self._host_consumed_destinations)

    def record_host_destinations(
        self, layer_name: str, destinations: tuple[int, ...]
    ) -> None:
        if self._active_host_layer != layer_name:
            raise RuntimeError("vLLM host transfer published into the wrong layer")
        self._host_consumed_destinations.update(destinations)

    def record_host_schedule(
        self,
        layer_name: str,
        chunk_tokens: int,
        request_indices: tuple[int, ...],
        kv_tile_indices: tuple[int, ...],
        selected_pages: tuple[tuple[int, ...], ...],
    ) -> None:
        if self._active_host_layer != layer_name:
            raise RuntimeError("vLLM host schedule belongs to the wrong layer")
        if not (len(request_indices) == len(kv_tile_indices) == len(selected_pages)):
            raise RuntimeError("vLLM host schedule diagnostics are misaligned")
        self._host_schedule_observations.append(
            (chunk_tokens, request_indices, kv_tile_indices, selected_pages)
        )

    def finish_host_layer(self, layer_name: str) -> None:
        """Prove that every admitted host block became numerical HBM state."""
        if self._active_host_layer != layer_name:
            raise RuntimeError("vLLM host layer finalization is unbalanced")
        expected = {destination for _, destination in self.host_transfer_pairs}
        acquisition = self.host_acquisition
        if acquisition is not None:
            fence, _ = acquisition.fence_for(layer_name)
            if expected and fence not in self._host_waited_fences:
                raise RuntimeError(
                    "vLLM Host attention did not wait for its layer acquisition"
                )
            self._host_consumed_destinations.update(expected)
            acquisition.retire(layer_name)
            self.record_evidence("host_acquisition_retirements")
        consumed = set(self._host_consumed_destinations)
        observations = tuple(self._host_schedule_observations)
        self._active_host_layer = None
        self._host_consumed_destinations.clear()
        self._host_schedule_observations.clear()
        if consumed != expected:
            missing = sorted(expected - consumed)
            extra = sorted(consumed - expected)
            raise RuntimeError(
                "vLLM host layer did not materialize its exact admitted set: "
                f"expected_count={len(expected)}, consumed_count={len(consumed)}, "
                f"missing={missing[:8]}, extra={extra[:8]}, "
                f"consumed={sorted(consumed)[:16]}, schedules={observations[:2]}"
            )

    def abort_host_layer(self, layer_name: str) -> None:
        if self._active_host_layer == layer_name:
            self._active_host_layer = None
            self._host_consumed_destinations.clear()
            self._host_schedule_observations.clear()

    def wait_for_host_layer(self, stream: Any) -> bool:
        """Order one numerical layer behind its exact readiness fence."""

        layer_name = self._active_host_layer
        if layer_name is None:
            raise RuntimeError("vLLM Host wait ran outside an active layer")
        acquisition = self.host_acquisition
        if acquisition is None:
            if self.host_transfer_pairs:
                raise RuntimeError(
                    "vLLM direct Host attention has no layer acquisition"
                )
            return False
        if self.tenant_isolation_enabled and not acquisition.tenant_accounted:
            raise RuntimeError(
                "vLLM Host acquisition bypassed finite tenant byte credits"
            )
        fence, event = acquisition.fence_for(layer_name)
        if fence in self._host_waited_fences:
            return False
        stream.wait_event(event)
        self._host_waited_fences.add(fence)
        self.record_evidence("host_acquisition_waits")
        return True

    def phase_batch(self, start: int, count: int) -> EngineBatch:
        """Validate and cache one framework phase for all model layers."""
        if self.batch is None:
            raise RuntimeError("vLLM forward sidecar has no engine batch")
        key = (start, count)
        phase = self._phase_batches.get(key)
        if phase is None:
            phase = self.batch.phase(start, count)
            self._phase_batches[key] = phase
        return phase

    def phase_request_bindings(self, start: int, count: int) -> Any:
        """Return the phase-local ``(slot, generation)`` device table.

        FlashInfer resets its request index within each decode/prefill phase.
        Narrowing here keeps that local index aligned with the same immutable
        :class:`EngineBatch.phase` range used by the numerical wrapper.
        """

        self.phase_batch(start, count)
        bindings = self.request_bindings_tensor
        if bindings is None:
            raise RuntimeError("vLLM forward sidecar has no request bindings")
        if getattr(bindings, "ndim", None) != 1 or getattr(
            bindings, "numel", lambda: 0
        )() < 2 * (start + count):
            raise RuntimeError("vLLM request-binding table is incomplete")
        return bindings.narrow(0, 2 * start, 2 * count)

    def storage_keys_for(
        self, phase: EngineBatch
    ) -> tuple[tuple[str | None, ...], ...]:
        """Return connector storage identities aligned to one phase batch."""

        if self.batch is None:
            raise RuntimeError("vLLM forward sidecar has no engine batch")
        if len(self.storage_key_tables) != len(self.batch.bindings):
            raise RuntimeError(
                "vLLM physical storage identities are not aligned to the forward"
            )
        by_request = {
            binding.request_id: table
            for binding, table in zip(
                self.batch.bindings, self.storage_key_tables, strict=True
            )
        }
        if len(by_request) != len(self.batch.bindings):
            raise RuntimeError("vLLM forward contains duplicate stable request IDs")
        try:
            return tuple(by_request[binding.request_id] for binding in phase.bindings)
        except KeyError as error:
            raise RuntimeError(
                "vLLM phase contains a request absent from connector metadata"
            ) from error

    def record_native_launch(
        self,
        kind: str,
        work_items: int,
        *,
        form: str,
        framework_owned: bool,
        serving_tier: str = "hbm",
    ) -> None:
        """Stage numerical evidence until the complete worker forward commits."""
        if kind not in {"decode", "prefill"}:
            raise ValueError(f"unknown vLLM attention phase {kind!r}")
        if form not in {"request_bound", "incremental"}:
            raise ValueError(f"unknown vLLM numerical form {form!r}")
        if work_items <= 0:
            raise ValueError("vLLM native launch must contain work")
        if self._evidence_committed or self._evidence_aborted:
            raise RuntimeError("vLLM forward evidence is already finalized")
        values = self._forward_evidence
        increments = [
            (f"native_{kind}_launches", 1),
            (f"native_{kind}_work_items", work_items),
            ("semantic_dense_tiles", work_items),
            (f"{form}_{kind}_launches", 1),
        ]
        if serving_tier in {"nvme", "cxl_dax"}:
            increments.append((f"physical_{kind}_launches", 1))
        elif serving_tier == "host_staged":
            increments.append((f"host_{kind}_launches", 1))
        for key, increment in increments:
            values[key] = values.get(key, 0) + increment
        if framework_owned:
            key = f"framework_owned_{kind}_launches"
            values[key] = values.get(key, 0) + 1

    def record_evidence(self, name: str, increment: int = 1) -> None:
        """Stage a nonnegative per-forward counter for atomic publication."""
        if not name or increment < 0:
            raise ValueError("vLLM evidence requires a name and nonnegative value")
        if self._evidence_committed or self._evidence_aborted:
            raise RuntimeError("vLLM forward evidence is already finalized")
        self._forward_evidence[name] = self._forward_evidence.get(name, 0) + increment

    def record_profile_ns(self, name: str, elapsed_ns: int) -> None:
        """Accumulate opt-in CPU timing at the same forward boundary."""
        if not name.endswith("_cpu_ns") or elapsed_ns < 0:
            raise ValueError("vLLM CPU profile counters must be nonnegative *_cpu_ns")
        self.record_evidence(name, elapsed_ns)

    def validate_evidence_commit(self) -> dict[str, int]:
        """Validate and snapshot evidence without changing transaction state."""
        if self._evidence_committed or self._evidence_aborted:
            raise RuntimeError("vLLM forward evidence was finalized twice")
        if self._active_host_layer is not None:
            raise RuntimeError("vLLM forward ended with an active host layer")
        if self.host_acquisition is not None and not self.host_acquisition.terminal:
            raise RuntimeError("vLLM forward ended with unconsumed Host layers")
        return dict(self._forward_evidence)

    def commit_evidence(self) -> dict[str, int]:
        """Return this forward's counters exactly once at its owner boundary."""
        values = self.validate_evidence_commit()
        self._evidence_committed = True
        return values

    def abort_evidence(self) -> None:
        """Discard every staged counter when numerical execution fails."""
        if self._evidence_committed:
            raise RuntimeError("committed vLLM evidence cannot be aborted")
        if self._evidence_aborted:
            return
        if self._active_host_layer is not None:
            self.abort_host_layer(self._active_host_layer)
        self._forward_evidence.clear()
        self._evidence_aborted = True


_FORWARD_STATE: contextvars.ContextVar[VllmV1ForwardState | None] = (
    contextvars.ContextVar("nta_vllm_v1_forward_state", default=None)
)


@contextmanager
def vllm_v1_forward_state(
    scheduler_output: Any,
) -> Any:
    """Install one worker-forward sidecar for opaque vLLM attention ops."""
    state = VllmV1ForwardState(scheduler_output)
    token = _FORWARD_STATE.set(state)
    try:
        yield state
    finally:
        _FORWARD_STATE.reset(token)


@contextmanager
def vllm_v1_reference_warmup_state() -> Any:
    """Mark a framework-owned dummy forward as reference-only.

    vLLM builds profiling and attention-workspace warmup batches directly in
    ``GPUModelRunner._dummy_run``.  Those batches have no scheduler request
    identity and must never be passed to the NTA work-unit path.  Keeping this
    marker separate from :func:`vllm_v1_forward_state` prevents a missing real
    worker sidecar from being mistaken for a harmless warmup.
    """
    state = VllmV1ForwardState(reference_warmup=True)
    token = _FORWARD_STATE.set(state)
    try:
        yield state
    finally:
        _FORWARD_STATE.reset(token)


def current_vllm_v1_forward_state() -> VllmV1ForwardState | None:
    """Return the active sidecar, if execution is inside a vLLM forward."""
    return _FORWARD_STATE.get()


class VllmV1NumericalConsumer(Protocol):
    """The explicit handoff from vLLM V1 metadata to numerical execution.

    A scheduler projection is not an attention implementation.  The concrete
    vLLM ``AttentionImpl`` adapter supplies this delegate after it has been
    constructed in the worker, so the identity/demand boundary remains
    testable without importing vLLM in the core runtime.
    """

    def consumer_contract(self) -> ConsumerContract: ...

    def consume(self, batch: EngineBatch, **attention_args: Any) -> Any: ...


def _installed_vllm_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def _exact_integer(value: Any, description: str) -> int:
    """Accept Python/NumPy integer scalars, but never truncate floats."""

    if isinstance(value, bool):
        raise RuntimeError(f"vLLM {description} must be an integer")
    try:
        normalized = integer_index(value)
    except TypeError:
        raise RuntimeError(f"vLLM {description} must be an integer") from None
    return int(normalized)


def _retired_request_ids(scheduler_output: Any) -> tuple[str, ...]:
    """Return every request whose runtime generation must end this step."""

    retired = {
        str(request_id)
        for field in ("finished_req_ids", "preempted_req_ids")
        for request_id in (getattr(scheduler_output, field, ()) or ())
    }
    if any(not request_id for request_id in retired):
        raise RuntimeError("vLLM retired request IDs must be non-empty")
    return tuple(sorted(retired))


@dataclass(frozen=True)
class VllmV1SchedulerProjection:
    """Exact NTA projection extracted from one vLLM V1 worker step."""

    request_ids: tuple[str, ...]
    block_tables: tuple[tuple[int, ...], ...]
    page_bytes: int
    request_rows: tuple[int, ...] = ()

    @classmethod
    def from_forward(
        cls,
        scheduler_output: Any,
        input_batch: Any,
        *,
        page_bytes: int,
    ) -> "VllmV1SchedulerProjection":
        if page_bytes <= 0:
            raise ValueError("vLLM V1 page_bytes must be positive")
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, Mapping) or not scheduled:
            raise RuntimeError(
                "vLLM V1 scheduler output must expose non-empty num_scheduled_tokens"
            )
        scheduled_ids = {str(request_id) for request_id in scheduled}
        input_request_ids = getattr(input_batch, "req_ids", None)
        if not isinstance(input_request_ids, Sequence) or isinstance(
            input_request_ids, (str, bytes)
        ):
            raise RuntimeError(
                "vLLM V1 hook requires InputBatch.req_ids to preserve the "
                "attention row order"
            )
        request_ids = tuple(
            str(request_id)
            for request_id in input_request_ids
            if str(request_id) in scheduled_ids
        )
        if len(request_ids) != len(scheduled_ids) or set(request_ids) != scheduled_ids:
            raise RuntimeError(
                "vLLM V1 input batch and scheduler output disagree on scheduled "
                "request IDs or contain duplicate attention rows"
            )
        if any(not request_id for request_id in request_ids):
            raise RuntimeError("vLLM V1 scheduled request IDs must be non-empty")
        retired_ids = set(_retired_request_ids(scheduler_output))
        if retired_ids.intersection(request_ids):
            raise RuntimeError(
                "vLLM V1 output retires and schedules the same request ID; "
                "the hook cannot disambiguate generations"
            )

        request_indices = getattr(input_batch, "req_id_to_index", None)
        block_tables = getattr(input_batch, "block_table", None)
        if not isinstance(request_indices, Mapping) or block_tables is None:
            raise RuntimeError(
                "vLLM V1 hook requires InputBatch.req_id_to_index and block_table"
            )
        try:
            group_count = len(block_tables.block_tables)
        except (AttributeError, TypeError):
            raise RuntimeError(
                "vLLM V1 hook requires the MultiGroupBlockTable contract"
            ) from None
        if group_count != 1:
            raise RuntimeError(
                "vLLM V1 exact-demand hook currently requires exactly one KV "
                f"cache group, got {group_count}"
            )
        group = block_tables[0]
        try:
            table = group.get_numpy_array()
            row_counts = group.num_blocks_per_row
        except AttributeError:
            raise RuntimeError(
                "vLLM V1 hook requires the CPU block-table mirror"
            ) from None

        rows: list[tuple[int, ...]] = []
        used_rows: set[int] = set()
        for request_id in request_ids:
            if request_id not in request_indices:
                raise RuntimeError(
                    f"vLLM V1 input batch has no row for scheduled request {request_id!r}"
                )
            row_index = _exact_integer(request_indices[request_id], "request row index")
            if row_index < 0 or row_index >= len(row_counts):
                raise RuntimeError(
                    f"vLLM V1 request row {row_index} is outside the input batch"
                )
            if row_index in used_rows:
                raise RuntimeError("vLLM V1 scheduled requests reuse an allocation row")
            used_rows.add(row_index)
            row_count = _exact_integer(row_counts[row_index], "block count")
            if row_count <= 0:
                raise RuntimeError(
                    f"vLLM V1 request {request_id!r} has no exact KV blocks"
                )
            try:
                values = table[row_index, :row_count].tolist()
            except (AttributeError, IndexError, TypeError, ValueError):
                raise RuntimeError(
                    f"vLLM V1 block-table row is not readable for {request_id!r}"
                ) from None
            if not isinstance(values, Sequence) or len(values) != row_count:
                raise RuntimeError(
                    f"vLLM V1 block-table row is shorter than its block count "
                    f"for {request_id!r}"
                )
            row = tuple(_exact_integer(page, "block-table page ID") for page in values)
            if any(page < 0 for page in row):
                raise RuntimeError(
                    f"vLLM V1 block-table row has a negative page ID for {request_id!r}"
                )
            rows.append(row)
        return cls(request_ids, tuple(rows), page_bytes)

    def exact_demand(self) -> ExactDemandProjection:
        return ExactDemandProjection(self.block_tables, self.page_bytes)

    @classmethod
    def from_v2_forward(
        cls,
        scheduler_output: Any,
        input_batch: Any,
        *,
        block_tables: Sequence[Any],
        num_blocks: Sequence[Any],
        page_bytes: int,
    ) -> "VllmV1SchedulerProjection":
        """Project vLLM's V2 batch after its device table gather.

        V2 intentionally removed the V1 CPU ``MultiGroupBlockTable`` object.
        The worker bridge supplies an adapter-owned CPU mirror maintained from
        vLLM's allocation writes, so this projection does not copy a GPU block
        table back to the host on every forward.
        """
        if page_bytes <= 0:
            raise ValueError("vLLM V2 page_bytes must be positive")
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, Mapping) or not scheduled:
            raise RuntimeError(
                "vLLM V2 scheduler output must expose non-empty num_scheduled_tokens"
            )
        request_ids_in_batch = getattr(input_batch, "req_ids", None)
        row_indices = getattr(input_batch, "idx_mapping_np", None)
        if not isinstance(request_ids_in_batch, Sequence) or isinstance(
            request_ids_in_batch, (str, bytes)
        ):
            raise RuntimeError("vLLM V2 input batch has no request-id sequence")
        if row_indices is None or isinstance(row_indices, (str, bytes)):
            raise RuntimeError("vLLM V2 input batch has no CPU request index map")
        try:
            row_indices = tuple(
                _exact_integer(value, "CPU request row index") for value in row_indices
            )
        except TypeError:
            raise RuntimeError(
                "vLLM V2 input batch has no CPU request index map"
            ) from None
        if len(request_ids_in_batch) != len(row_indices):
            raise RuntimeError("vLLM V2 request IDs and row indices are misaligned")
        scheduled_ids = {str(request_id) for request_id in scheduled}
        request_ids = tuple(
            str(request_id)
            for request_id in request_ids_in_batch
            if str(request_id) in scheduled_ids
        )
        if len(request_ids) != len(scheduled_ids) or set(request_ids) != scheduled_ids:
            raise RuntimeError(
                "vLLM V2 input batch and scheduler output disagree on scheduled "
                "request IDs"
            )
        if set(_retired_request_ids(scheduler_output)).intersection(request_ids):
            raise RuntimeError(
                "vLLM V2 output retires and schedules the same request ID"
            )
        if len(block_tables) != 1 or len(num_blocks) != 1:
            raise RuntimeError(
                "vLLM V2 exact-demand hook currently requires exactly one KV group"
            )
        table = block_tables[0]
        counts = num_blocks[0]
        rows: list[tuple[int, ...]] = []
        request_indices: dict[str, int] = {}
        used_rows: set[int] = set()
        for request_id, row_index in zip(
            request_ids_in_batch, row_indices, strict=True
        ):
            request_id = str(request_id)
            if request_id not in scheduled_ids:
                continue
            if row_index < 0:
                raise RuntimeError(
                    f"vLLM V2 request {request_id!r} has a negative row index"
                )
            if row_index in used_rows:
                raise RuntimeError("vLLM V2 scheduled requests reuse an allocation row")
            used_rows.add(row_index)
            request_indices[request_id] = row_index
            try:
                row_count = _exact_integer(counts[row_index], "block count")
                if row_count <= 0:
                    raise RuntimeError(
                        f"vLLM V2 request {request_id!r} has no exact KV blocks"
                    )
                values = table[row_index, :row_count].tolist()
            except (AttributeError, IndexError, TypeError, ValueError):
                raise RuntimeError(
                    f"vLLM V2 block-table row is not readable for {request_id!r}"
                ) from None
            if not isinstance(values, Sequence) or len(values) != row_count:
                raise RuntimeError(
                    f"vLLM V2 block-table row is shorter than its block count "
                    f"for {request_id!r}"
                )
            row = tuple(_exact_integer(page, "block-table page ID") for page in values)
            if any(page < 0 for page in row):
                raise RuntimeError(
                    f"vLLM V2 block-table row has a negative page ID for {request_id!r}"
                )
            rows.append(row)
        if set(request_indices) != set(request_ids):
            raise RuntimeError("vLLM V2 scheduled request rows are incomplete")
        return cls(
            request_ids,
            tuple(rows),
            page_bytes,
            tuple(request_indices[request_id] for request_id in request_ids),
        )


class _StableVllmSlots:
    """Bounded stable request-ID to NTA-slot mapping."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("vLLM V1 request capacity must be positive")
        self._request_to_slot: dict[str, int] = {}
        self._request_to_row: dict[str, int] = {}
        self._row_to_request: dict[int, str] = {}
        self._free = list(range(capacity))
        heapq.heapify(self._free)

    def replacements(
        self, request_ids: Sequence[str], request_rows: Sequence[int]
    ) -> tuple[str, ...]:
        if len(request_ids) != len(request_rows):
            raise RuntimeError("vLLM request IDs and allocation rows are misaligned")
        normalized_rows = tuple(
            _nonnegative_integer(row, "allocation row") for row in request_rows
        )
        current = set(request_ids)
        replaced = {
            previous
            for request_id, row in zip(request_ids, normalized_rows, strict=True)
            if (previous := self._row_to_request.get(row)) is not None
            and previous != request_id
            and previous not in current
        }
        return tuple(sorted(replaced))

    def assign(
        self,
        request_ids: Sequence[str],
        finished_request_ids: Sequence[str],
        request_rows: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("vLLM V1 scheduled request IDs are not unique")
        current = set(request_ids)
        finished = {str(request_id) for request_id in finished_request_ids}
        if current.intersection(finished):
            raise RuntimeError(
                "vLLM V1 cannot reuse a request ID in the same finish/schedule step"
            )
        if request_rows is not None and len(request_rows) != len(request_ids):
            raise RuntimeError("vLLM request IDs and allocation rows are misaligned")
        if request_rows is not None:
            request_rows = tuple(
                _nonnegative_integer(row, "allocation row") for row in request_rows
            )
            if len(set(request_rows)) != len(request_rows):
                raise RuntimeError("vLLM allocation rows are not unique")

        def release(request_id: str) -> None:
            slot = self._request_to_slot.pop(request_id, None)
            row = self._request_to_row.pop(request_id, None)
            if row is not None and self._row_to_request.get(row) == request_id:
                self._row_to_request.pop(row, None)
            if slot is not None:
                heapq.heappush(self._free, slot)

        for request_id in finished:
            if request_id not in current:
                release(request_id)
        if request_rows is not None:
            # vLLM may compact or swap persistent input-batch rows.  Detach
            # every moving live request before resolving target-row owners;
            # otherwise a two-way swap leaves a stale reverse entry and the
            # next row replacement can leak a request slot.
            targets = dict(zip(request_ids, request_rows, strict=True))
            for request_id, target_row in targets.items():
                previous_row = self._request_to_row.get(request_id)
                if previous_row is not None and previous_row != target_row:
                    if self._row_to_request.get(previous_row) == request_id:
                        self._row_to_request.pop(previous_row, None)
                    self._request_to_row.pop(request_id, None)
            for request_id, row in targets.items():
                previous = self._row_to_request.get(row)
                if previous is not None and previous != request_id:
                    if previous in current:
                        self._request_to_row.pop(previous, None)
                    else:
                        release(previous)
                    self._row_to_request.pop(row, None)
        result: list[int] = []
        for offset, request_id in enumerate(request_ids):
            slot = self._request_to_slot.get(request_id)
            if slot is None:
                if not self._free:
                    raise RuntimeError("vLLM V1 request slot capacity is exhausted")
                slot = heapq.heappop(self._free)
                self._request_to_slot[request_id] = slot
            if request_rows is not None:
                row = request_rows[offset]
                self._request_to_row[request_id] = row
                self._row_to_request[row] = request_id
            result.append(slot)
        return tuple(result)


def _nonnegative_integer(value: Any, description: str) -> int:
    normalized = _exact_integer(value, description)
    if normalized < 0:
        raise RuntimeError(f"vLLM {description} must be nonnegative")
    return normalized


class VllmV1Hook:
    """Adapter entry point for a pinned vLLM V1 model runner.

    A custom model runner calls ``bind_forward`` after vLLM's
    ``_update_states`` and before the attention launch.  The call is control
    plane work: it reads vLLM's CPU mirror, publishes NTA identity, and
    returns an engine-neutral batch.  It does not perform a per-request I/O
    ioctl or copy a block table from GPU to host.
    """

    def __init__(
        self,
        runtime: Any,
        request_capacity: int,
        *,
        page_bytes: int,
        expected_vllm_version: str = SUPPORTED_VLLM_V1_VERSION,
        version_provider: Callable[[], str | None] = _installed_vllm_version,
        tenant_for_request: Callable[[str], int] | None = None,
        priority_for_request: Callable[[str], int] | None = None,
        deadline_for_request: Callable[[str], int] | None = None,
        consumer: VllmV1NumericalConsumer | None = None,
        profile_cpu: bool = False,
    ) -> None:
        if page_bytes <= 0:
            raise ValueError("vLLM V1 page_bytes must be positive")
        if not expected_vllm_version:
            raise ValueError("expected vLLM version must be non-empty")
        self._runtime = runtime
        self._adapter = VllmAdapter(runtime, request_capacity)
        self._slots = _StableVllmSlots(request_capacity)
        self._page_bytes = page_bytes
        self._expected_version = expected_vllm_version
        installed = version_provider()
        if installed != expected_vllm_version:
            raise RuntimeError(
                "unsupported vLLM version for NTA V1 hook: "
                f"expected {expected_vllm_version}, found {installed or 'missing'}"
            )
        self._tenant_for_request = tenant_for_request
        self._priority_for_request = priority_for_request
        self._deadline_for_request = deadline_for_request
        self._consumer = consumer
        self._profile_cpu = bool(profile_cpu)
        self._native_launches = 0
        self._last_bind_profile: dict[str, int] = {}

    @property
    def adapter(self) -> VllmAdapter:
        return self._adapter

    @property
    def runtime(self) -> Any:
        """Return the runtime shared by the worker and attention consumer."""
        return self._runtime

    @property
    def last_bind_profile(self) -> dict[str, int]:
        """Return opt-in timing for the most recently completed bind."""
        return dict(self._last_bind_profile)

    def record_native_launch(self, count: int = 1) -> None:
        """Promote evidence only after an NTA attention launch completed."""
        if count <= 0:
            raise ValueError("native launch count must be positive")
        self._native_launches += count

    def projection_contract(self) -> ConsumerContract:
        """Describe this hook without overstating what it executes.

        The V1 hook publishes identity and exact block demand.  A model
        runner or attention backend must replace this projection-only
        contract with a native or framework-reference consumer contract
        after it has actually consumed the returned batch.
        """
        return ConsumerContract.projection_only(
            engine="vllm",
            backend="vllm_v1_worker_projection",
            engine_version=self._expected_version,
        )

    def consumer_contract(self) -> ConsumerContract:
        """Return the contract of the numerical delegate, if one is wired."""
        if self._native_launches:
            return ConsumerContract.native_work_unit(
                engine="vllm",
                backend="nta_flashinfer",
                engine_version=self._expected_version,
            )
        if self._consumer is None:
            return self.projection_contract()
        contract = self._consumer.consumer_contract()
        if not isinstance(contract, ConsumerContract):
            raise TypeError("vLLM V1 numerical consumer returned no typed contract")
        if contract.engine != "vllm" or not contract.formal_execution:
            raise RuntimeError(
                "vLLM V1 numerical consumer must provide a formal vLLM "
                "consumer contract"
            )
        return contract

    def bind_forward(
        self,
        scheduler_output: Any,
        input_batch: Any,
        *,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        projection = VllmV1SchedulerProjection.from_forward(
            scheduler_output,
            input_batch,
            page_bytes=self._page_bytes,
        )
        retired = _retired_request_ids(scheduler_output)
        for request_id in retired:
            self._adapter.retire_request(request_id)
        slots = self._slots.assign(projection.request_ids, retired)
        priorities = (
            None
            if self._priority_for_request is None
            else tuple(
                self._priority_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        deadlines = (
            None
            if self._deadline_for_request is None
            else tuple(
                self._deadline_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        tenants = (
            None
            if self._tenant_for_request is None
            else tuple(
                self._tenant_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        return self._adapter.bind_batch(
            projection.request_ids,
            slots,
            epoch=epoch,
            stream=stream,
            priorities=priorities,
            deadline_clocks=deadlines,
            tenant_ids=tenants,
            exact_demand=projection.exact_demand(),
            granularity=granularity,
        )

    def bind_v2_forward(
        self,
        scheduler_output: Any,
        input_batch: Any,
        *,
        block_tables: Sequence[Any],
        num_blocks: Sequence[Any],
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        """Bind vLLM V2's request batch through its typed CPU table mirror."""
        profiling = self._profile_cpu
        phase_started = time.perf_counter_ns()
        profile: dict[str, int] = {}

        def finish_phase(name: str) -> None:
            nonlocal phase_started
            if profiling:
                now = time.perf_counter_ns()
                profile[name] = now - phase_started
                phase_started = now

        projection = VllmV1SchedulerProjection.from_v2_forward(
            scheduler_output,
            input_batch,
            block_tables=block_tables,
            num_blocks=num_blocks,
            page_bytes=self._page_bytes,
        )
        finish_phase("bridge_projection_cpu_ns")
        retired = _retired_request_ids(scheduler_output)
        for request_id in retired:
            self._adapter.retire_request(request_id)
        finish_phase("bridge_finished_retire_cpu_ns")
        replacements = self._slots.replacements(
            projection.request_ids, projection.request_rows
        )
        for request_id in replacements:
            self._adapter.retire_request(request_id)
        finish_phase("bridge_replacement_retire_cpu_ns")
        slots = self._slots.assign(
            projection.request_ids,
            retired,
            projection.request_rows,
        )
        finish_phase("bridge_slot_assign_cpu_ns")
        priorities = (
            None
            if self._priority_for_request is None
            else tuple(
                self._priority_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        deadlines = (
            None
            if self._deadline_for_request is None
            else tuple(
                self._deadline_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        tenants = (
            None
            if self._tenant_for_request is None
            else tuple(
                self._tenant_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        finish_phase("bridge_policy_cpu_ns")
        exact_demand = projection.exact_demand()
        finish_phase("bridge_exact_demand_cpu_ns")
        result = self._adapter.bind_batch(
            projection.request_ids,
            slots,
            epoch=epoch,
            stream=stream,
            priorities=priorities,
            deadline_clocks=deadlines,
            tenant_ids=tenants,
            exact_demand=exact_demand,
            granularity=granularity,
        )
        finish_phase("bridge_identity_publish_cpu_ns")
        self._last_bind_profile = profile
        return result

    def bind_connector_forward(
        self,
        request_ids: Sequence[str],
        block_tables: Sequence[Sequence[int]],
        finished_request_ids: Sequence[str],
        *,
        input_batch: Any,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        """Bind connector demand only after proving its numerical row order."""
        framework_ids = getattr(input_batch, "req_ids", None)
        framework_rows = getattr(input_batch, "idx_mapping_np", None)
        if not isinstance(framework_ids, Sequence) or isinstance(
            framework_ids, (str, bytes)
        ):
            raise RuntimeError("vLLM connector input batch has no request row order")
        if framework_rows is None or isinstance(framework_rows, (str, bytes)):
            raise RuntimeError("vLLM connector input batch has no allocation rows")
        try:
            actual_ids = tuple(str(request_id) for request_id in framework_ids)
            actual_rows = tuple(
                _nonnegative_integer(row, "allocation row") for row in framework_rows
            )
        except TypeError:
            raise RuntimeError(
                "vLLM connector input batch has no allocation rows"
            ) from None
        metadata_ids = tuple(str(request_id) for request_id in request_ids)
        if any(not request_id for request_id in actual_ids + metadata_ids):
            raise RuntimeError("vLLM connector request IDs must be non-empty")
        if len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError("vLLM input batch contains duplicate request rows")
        if len(actual_ids) != len(actual_rows):
            raise RuntimeError("vLLM request IDs and allocation rows are misaligned")
        if metadata_ids != actual_ids:
            mismatch = next(
                (
                    index
                    for index, (metadata_id, actual_id) in enumerate(
                        zip(metadata_ids, actual_ids, strict=False)
                    )
                    if metadata_id != actual_id
                ),
                min(len(metadata_ids), len(actual_ids)),
            )
            raise RuntimeError(
                "vLLM connector request order does not match InputBatch.req_ids "
                f"at attention row {mismatch}: metadata={metadata_ids!r}, "
                f"input_batch={actual_ids!r}"
            )
        projection = VllmV1SchedulerProjection(
            actual_ids,
            tuple(tuple(row) for row in block_tables),
            self._page_bytes,
            actual_rows,
        )
        finished = tuple(str(request_id) for request_id in finished_request_ids)
        for request_id in finished:
            self._adapter.retire_request(request_id)
        replacements = self._slots.replacements(
            projection.request_ids, projection.request_rows
        )
        for request_id in replacements:
            self._adapter.retire_request(request_id)
        slots = self._slots.assign(
            projection.request_ids, finished, projection.request_rows
        )
        priorities = (
            None
            if self._priority_for_request is None
            else tuple(
                self._priority_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        deadlines = (
            None
            if self._deadline_for_request is None
            else tuple(
                self._deadline_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        tenants = (
            None
            if self._tenant_for_request is None
            else tuple(
                self._tenant_for_request(request_id)
                for request_id in projection.request_ids
            )
        )
        return self._adapter.bind_batch(
            projection.request_ids,
            slots,
            epoch=epoch,
            stream=stream,
            priorities=priorities,
            deadline_clocks=deadlines,
            tenant_ids=tenants,
            exact_demand=projection.exact_demand(),
            granularity=granularity,
        )

    def consume_forward(
        self,
        scheduler_output: Any,
        input_batch: Any,
        *,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
        **attention_args: Any,
    ) -> Any:
        """Bind one V1 step and hand it to the real numerical consumer.

        This is the only vLLM-to-NTA execution seam.  It performs no transport
        operation itself; a concrete ``AttentionImpl`` owns the numerical
        launch and may consume the typed batch plus framework attention
        arguments.  Calling it without that implementation is an explicit
        error rather than a stock-attention disguise.
        """
        if self._consumer is None:
            raise RuntimeError("vLLM V1 projection has no numerical attention consumer")
        batch = self.bind_forward(
            scheduler_output,
            input_batch,
            epoch=epoch,
            stream=stream,
            granularity=granularity,
        )
        self.consumer_contract()
        return self._consumer.consume(batch, **attention_args)
