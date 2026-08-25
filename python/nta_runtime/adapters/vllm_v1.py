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
from dataclasses import dataclass
import heapq
import importlib.metadata
import os
from typing import Any, Protocol

from .base import ConsumerContract, ExactDemandProjection, EngineBatch
from .vllm import VllmAdapter
from ..work_unit import Granularity


SUPPORTED_VLLM_V1_VERSION = "0.26.0"


def validate_vllm_attention_tier(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Validate the tier visible to the vLLM numerical consumer.

    The current vLLM ``AttentionImpl`` consumes resident KV pages.  A
    physical tier must therefore be rejected before construction; otherwise
    an operator could set ``NTA_SERVING_TIER=nvme`` while the framework still
    served the resident cache and produce a misleading artifact.  The
    framework-neutral host-staged default remains allowed for the reference
    and resident qualification profiles; it is not treated as proof of an
    external vLLM transfer.
    """
    values = os.environ if environ is None else environ
    selected = values.get("NTA_SERVING_TIER", "host_staged").strip().lower()
    if selected == "host":
        selected = "host_staged"
    if selected == "cxl":
        selected = "cxl_dax"
    if selected not in {"host_staged", "nvme", "cxl_dax"}:
        raise RuntimeError(
            "NTA_SERVING_TIER must be host_staged, nvme, or cxl_dax for vLLM"
        )
    if selected in {"nvme", "cxl_dax"}:
        raise RuntimeError(
            "vLLM resident attention cannot consume a physical tier; configure "
            "a tested V1 KVConnector/data-lifetime adapter first"
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
    hook: "VllmV1Hook | None" = None
    page_size: int = 0
    reference_warmup: bool = False


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
                "vLLM V1 scheduler output must expose non-empty "
                "num_scheduled_tokens"
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
        finished = getattr(scheduler_output, "finished_req_ids", frozenset())
        finished_ids = {str(request_id) for request_id in finished}
        if finished_ids.intersection(request_ids):
            raise RuntimeError(
                "vLLM V1 output finishes and schedules the same request ID; "
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
        for request_id in request_ids:
            if request_id not in request_indices:
                raise RuntimeError(
                    f"vLLM V1 input batch has no row for scheduled request {request_id!r}"
                )
            row_index = int(request_indices[request_id])
            if row_index < 0 or row_index >= len(row_counts):
                raise RuntimeError(
                    f"vLLM V1 request row {row_index} is outside the input batch"
                )
            row_count = int(row_counts[row_index])
            if row_count <= 0:
                raise RuntimeError(
                    f"vLLM V1 request {request_id!r} has no exact KV blocks"
                )
            try:
                row = tuple(int(page) for page in table[row_index, :row_count].tolist())
            except (AttributeError, IndexError, TypeError, ValueError):
                raise RuntimeError(
                    f"vLLM V1 block-table row is not readable for {request_id!r}"
                ) from None
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
                "vLLM V2 scheduler output must expose non-empty "
                "num_scheduled_tokens"
            )
        request_ids_in_batch = getattr(input_batch, "req_ids", None)
        row_indices = getattr(input_batch, "idx_mapping_np", None)
        if not isinstance(request_ids_in_batch, Sequence) or isinstance(
            request_ids_in_batch, (str, bytes)
        ):
            raise RuntimeError("vLLM V2 input batch has no request-id sequence")
        if isinstance(row_indices, (str, bytes)):
            raise RuntimeError("vLLM V2 input batch has no CPU request index map")
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
        if len(block_tables) != 1 or len(num_blocks) != 1:
            raise RuntimeError(
                "vLLM V2 exact-demand hook currently requires exactly one KV group"
            )
        table = block_tables[0]
        counts = num_blocks[0]
        rows: list[tuple[int, ...]] = []
        request_indices: dict[str, int] = {}
        for request_id, row_index in zip(
            request_ids_in_batch, row_indices, strict=True
        ):
            request_id = str(request_id)
            if request_id not in scheduled_ids:
                continue
            row_index = int(row_index)
            request_indices[request_id] = row_index
            try:
                row_count = int(counts[row_index])
                if row_count <= 0:
                    raise RuntimeError(
                        f"vLLM V2 request {request_id!r} has no exact KV blocks"
                    )
                row = tuple(int(page) for page in table[row_index, :row_count].tolist())
            except (AttributeError, IndexError, TypeError, ValueError):
                raise RuntimeError(
                    f"vLLM V2 block-table row is not readable for {request_id!r}"
                ) from None
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
        current = set(request_ids)
        replaced = {
            previous
            for request_id, row in zip(request_ids, request_rows, strict=True)
            if (previous := self._row_to_request.get(int(row))) is not None
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
            request_rows = tuple(int(row) for row in request_rows)
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
                row = int(request_rows[offset])
                self._request_to_row[request_id] = row
                self._row_to_request[row] = request_id
            result.append(slot)
        return tuple(result)


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
        self._version_provider = version_provider
        self._tenant_for_request = tenant_for_request
        self._priority_for_request = priority_for_request
        self._deadline_for_request = deadline_for_request
        self._consumer = consumer
        self._native_launches = 0

    @property
    def adapter(self) -> VllmAdapter:
        return self._adapter

    @property
    def runtime(self) -> Any:
        """Return the runtime shared by the worker and attention consumer."""
        return self._runtime

    def record_native_launch(self) -> None:
        """Promote evidence only after an NTA attention launch completed."""
        self._native_launches += 1

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

    def _check_version(self) -> None:
        installed = self._version_provider()
        if installed != self._expected_version:
            raise RuntimeError(
                "unsupported vLLM version for NTA V1 hook: "
                f"expected {self._expected_version}, found {installed or 'missing'}"
            )

    def bind_forward(
        self,
        scheduler_output: Any,
        input_batch: Any,
        *,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        self._check_version()
        projection = VllmV1SchedulerProjection.from_forward(
            scheduler_output,
            input_batch,
            page_bytes=self._page_bytes,
        )
        finished = tuple(
            str(request_id)
            for request_id in getattr(scheduler_output, "finished_req_ids", ())
        )
        for request_id in finished:
            self._adapter.retire_request(request_id)
        slots = self._slots.assign(projection.request_ids, finished)
        priorities = (
            None
            if self._priority_for_request is None
            else tuple(self._priority_for_request(request_id) for request_id in projection.request_ids)
        )
        deadlines = (
            None
            if self._deadline_for_request is None
            else tuple(self._deadline_for_request(request_id) for request_id in projection.request_ids)
        )
        tenants = (
            None
            if self._tenant_for_request is None
            else tuple(self._tenant_for_request(request_id) for request_id in projection.request_ids)
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
        self._check_version()
        projection = VllmV1SchedulerProjection.from_v2_forward(
            scheduler_output,
            input_batch,
            block_tables=block_tables,
            num_blocks=num_blocks,
            page_bytes=self._page_bytes,
        )
        finished = tuple(
            str(request_id)
            for request_id in getattr(scheduler_output, "finished_req_ids", ())
        )
        for request_id in finished:
            self._adapter.retire_request(request_id)
        replacements = self._slots.replacements(
            projection.request_ids, projection.request_rows
        )
        for request_id in replacements:
            self._adapter.retire_request(request_id)
        slots = self._slots.assign(
            projection.request_ids,
            finished,
            projection.request_rows,
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
            raise RuntimeError(
                "vLLM V1 projection has no numerical attention consumer"
            )
        batch = self.bind_forward(
            scheduler_output,
            input_batch,
            epoch=epoch,
            stream=stream,
            granularity=granularity,
        )
        self.consumer_contract()
        return self._consumer.consume(batch, **attention_args)
