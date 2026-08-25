"""Pinned vLLM V1 worker boundary.

This module is the only place where the vLLM 0.13 V1 worker object layout is
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
from dataclasses import dataclass
import heapq
import importlib.metadata
from typing import Any, Protocol

from .base import ConsumerContract, ExactDemandProjection, EngineBatch
from .vllm import VllmAdapter
from ..work_unit import Granularity


SUPPORTED_VLLM_V1_VERSION = "0.13.0"


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
        request_ids = tuple(str(request_id) for request_id in scheduled)
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


class _StableVllmSlots:
    """Bounded stable request-ID to NTA-slot mapping."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("vLLM V1 request capacity must be positive")
        self._request_to_slot: dict[str, int] = {}
        self._free = list(range(capacity))
        heapq.heapify(self._free)

    def assign(
        self, request_ids: Sequence[str], finished_request_ids: Sequence[str]
    ) -> tuple[int, ...]:
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("vLLM V1 scheduled request IDs are not unique")
        current = set(request_ids)
        finished = {str(request_id) for request_id in finished_request_ids}
        if current.intersection(finished):
            raise RuntimeError(
                "vLLM V1 cannot reuse a request ID in the same finish/schedule step"
            )
        for request_id in finished:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is not None and request_id not in current:
                heapq.heappush(self._free, slot)
        result: list[int] = []
        for request_id in request_ids:
            slot = self._request_to_slot.get(request_id)
            if slot is None:
                if not self._free:
                    raise RuntimeError("vLLM V1 request slot capacity is exhausted")
                slot = heapq.heappop(self._free)
                self._request_to_slot[request_id] = slot
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
        self._adapter = VllmAdapter(runtime, request_capacity)
        self._slots = _StableVllmSlots(request_capacity)
        self._page_bytes = page_bytes
        self._expected_version = expected_vllm_version
        self._version_provider = version_provider
        self._tenant_for_request = tenant_for_request
        self._priority_for_request = priority_for_request
        self._deadline_for_request = deadline_for_request
        self._consumer = consumer

    @property
    def adapter(self) -> VllmAdapter:
        return self._adapter

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
