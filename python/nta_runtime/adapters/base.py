"""Common request-identity adapter with no engine-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from enum import Enum
from operator import index as integer_index
from typing import Any, Protocol, runtime_checkable

from ..requests import RequestBinding, RequestIdentityRegistry
from ..work_unit import Granularity


class ConsumerKind(str, Enum):
    """Numerical consumer reached after an engine projection.

    A projection is deliberately not an execution result.  Keeping these
    states explicit prevents a scheduler hook from being reported as a
    native work-unit consumer in an artifact.
    """

    NATIVE_WORK_UNIT = "native_work_unit"
    FRAMEWORK_REFERENCE = "framework_reference"
    PROJECTION_ONLY = "projection_only"


@dataclass(frozen=True)
class ConsumerContract:
    """Typed evidence for the boundary that consumed an :class:`EngineBatch`.

    ``exact_demand`` describes the numerical operation, while the remaining
    flags describe how that operation was reached.  They are separate on
    purpose: an exact vLLM scheduler projection has exact demand but has not
    yet submitted or consumed an NTA work plan.
    """

    engine: str
    backend: str
    kind: ConsumerKind
    exact_demand: bool
    typed_work_plan: bool
    native_submission: bool
    numerical_consumer: bool
    engine_version: str = "unknown"

    def __post_init__(self) -> None:
        if not self.engine or not self.backend or not self.engine_version:
            raise ValueError("consumer contract identity must be non-empty")
        if not isinstance(self.kind, ConsumerKind):
            raise TypeError("consumer contract kind must be a ConsumerKind")
        for field in (
            "exact_demand",
            "typed_work_plan",
            "native_submission",
            "numerical_consumer",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"consumer contract {field} must be bool")
        if self.kind is ConsumerKind.NATIVE_WORK_UNIT and not all(
            (self.exact_demand, self.typed_work_plan, self.native_submission, self.numerical_consumer)
        ):
            raise ValueError(
                "native work-unit consumer requires exact demand, typed plan, "
                "native submission, and numerical consumption"
            )
        if self.kind is ConsumerKind.FRAMEWORK_REFERENCE and not (
            self.exact_demand
            and not self.typed_work_plan
            and not self.native_submission
            and self.numerical_consumer
        ):
            raise ValueError(
                "framework reference consumer requires exact demand, no native "
                "submission, and numerical consumption"
            )
        if self.kind is ConsumerKind.PROJECTION_ONLY and (
            self.native_submission or self.numerical_consumer
        ):
            raise ValueError(
                "projection-only contract cannot claim native submission or "
                "numerical consumption"
            )

    @classmethod
    def native_work_unit(
        cls, *, engine: str, backend: str, engine_version: str = "unknown"
    ) -> "ConsumerContract":
        return cls(
            engine=engine,
            backend=backend,
            kind=ConsumerKind.NATIVE_WORK_UNIT,
            exact_demand=True,
            typed_work_plan=True,
            native_submission=True,
            numerical_consumer=True,
            engine_version=engine_version,
        )

    @classmethod
    def framework_reference(
        cls, *, engine: str, backend: str, engine_version: str = "unknown"
    ) -> "ConsumerContract":
        return cls(
            engine=engine,
            backend=backend,
            kind=ConsumerKind.FRAMEWORK_REFERENCE,
            exact_demand=True,
            typed_work_plan=False,
            native_submission=False,
            numerical_consumer=True,
            engine_version=engine_version,
        )

    @classmethod
    def projection_only(
        cls, *, engine: str, backend: str, engine_version: str = "unknown"
    ) -> "ConsumerContract":
        return cls(
            engine=engine,
            backend=backend,
            kind=ConsumerKind.PROJECTION_ONLY,
            exact_demand=True,
            typed_work_plan=False,
            native_submission=False,
            numerical_consumer=False,
            engine_version=engine_version,
        )

    @property
    def formal_execution(self) -> bool:
        """Whether this contract is eligible for formal serving evidence."""
        return self.kind in {
            ConsumerKind.NATIVE_WORK_UNIT,
            ConsumerKind.FRAMEWORK_REFERENCE,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": 1,
            "engine": self.engine,
            "backend": self.backend,
            "kind": self.kind.value,
            "exact_demand": self.exact_demand,
            "typed_work_plan": self.typed_work_plan,
            "native_submission": self.native_submission,
            "numerical_consumer": self.numerical_consumer,
            "engine_version": self.engine_version,
        }


@dataclass(frozen=True)
class ExactDemandProjection:
    """Engine-neutral exact page/unit demand supplied by an adapter."""

    request_unit_ids: tuple[tuple[int, ...], ...]
    unit_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.unit_bytes, bool):
            raise ValueError("exact demand unit_bytes must be an integer")
        try:
            unit_bytes = integer_index(self.unit_bytes)
        except TypeError:
            raise ValueError("exact demand unit_bytes must be an integer") from None
        if unit_bytes <= 0:
            raise ValueError("exact demand unit_bytes must be positive")
        try:
            normalized_rows = tuple(
                tuple(integer_index(unit) for unit in units)
                for units in self.request_unit_ids
            )
        except (TypeError, ValueError):
            raise ValueError(
                "exact demand unit IDs must be integer sequences"
            ) from None
        if not normalized_rows or any(not units for units in normalized_rows):
            raise ValueError("exact demand must contain units for every request")
        if any(
            any(unit < 0 for unit in units) or len(set(units)) != len(units)
            for units in normalized_rows
        ):
            raise ValueError("exact demand unit IDs must be unique and nonnegative")
        object.__setattr__(self, "request_unit_ids", normalized_rows)
        object.__setattr__(self, "unit_bytes", int(unit_bytes))


@dataclass(frozen=True)
class EngineBatch:
    """The minimum engine-to-runtime handoff for one forward."""

    engine: str
    epoch: int
    bindings: tuple[RequestBinding, ...]
    granularity: Granularity
    exact_demand: ExactDemandProjection | None = None

    def __post_init__(self) -> None:
        if not self.engine:
            raise ValueError("engine name must be non-empty")
        if isinstance(self.epoch, bool):
            raise ValueError("engine batch epoch must be an integer")
        try:
            epoch = integer_index(self.epoch)
        except TypeError:
            raise ValueError("engine batch epoch must be an integer") from None
        if epoch < 0:
            raise ValueError("engine batch epoch cannot be negative")
        object.__setattr__(self, "epoch", int(epoch))
        if not self.bindings:
            raise ValueError("engine batch must contain at least one request")
        if len({binding.request_slot for binding in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("engine batch cannot reuse a request slot")
        if len({binding.request_index for binding in self.bindings}) != len(
            self.bindings
        ):
            raise ValueError("engine batch cannot reuse a request index")
        if self.exact_demand is not None and len(
            self.exact_demand.request_unit_ids
        ) != len(self.bindings):
            raise ValueError("exact demand rows must match the engine batch")

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(binding.request_id for binding in self.bindings)

    @property
    def request_slots(self) -> tuple[int, ...]:
        return tuple(binding.request_slot for binding in self.bindings)

    @property
    def generations(self) -> tuple[int, ...]:
        return tuple(binding.generation for binding in self.bindings)

    @property
    def tenant_ids(self) -> tuple[int, ...]:
        """Logical tenant for each request in the engine batch."""
        return tuple(binding.tenant_id for binding in self.bindings)

    def phase(self, start: int, count: int) -> "EngineBatch":
        """Return a validated contiguous phase view of this forward.

        Frameworks may reorder one forward into contiguous phases (vLLM, for
        example, places decode rows before prefill rows).  A phase view keeps
        the original epoch and immutable request bindings while narrowing the
        exact-demand rows to the wrapper that consumes them.  It is metadata
        only: no request is rebound and no runtime state is duplicated.
        """
        if isinstance(start, bool) or isinstance(count, bool):
            raise TypeError("engine batch phase bounds must be integers")
        try:
            start = integer_index(start)
            count = integer_index(count)
        except TypeError:
            raise TypeError("engine batch phase bounds must be integers") from None
        if start < 0 or count <= 0 or start + count > len(self.bindings):
            raise ValueError(
                "engine batch phase must be a non-empty contiguous binding range"
            )
        demand = self.exact_demand
        if demand is not None:
            demand = ExactDemandProjection(
                demand.request_unit_ids[start : start + count], demand.unit_bytes
            )
        bindings = tuple(
            RequestBinding(
                request_index=local_index,
                request_slot=binding.request_slot,
                generation=binding.generation,
                request_id=binding.request_id,
                priority=binding.priority,
                deadline_clock=binding.deadline_clock,
                tenant_id=binding.tenant_id,
            )
            for local_index, binding in enumerate(
                self.bindings[start : start + count]
            )
        )
        return EngineBatch(
            self.engine,
            self.epoch,
            bindings,
            self.granularity,
            demand,
        )


@runtime_checkable
class EngineBoundary(Protocol):
    """The one lifecycle boundary shared by SGLang and vLLM adapters."""

    engine: str

    def bind_forward(
        self,
        forward_batch: Any,
        *,
        epoch: int,
        stream: Any = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
        **kwargs: Any,
    ) -> EngineBatch: ...

    def cancel_matching(
        self, request_id_prefix: str = "", *, all: bool = False
    ) -> int: ...


class RequestIdentityAdapter:
    """Shared slot/generation implementation used by frontend adapters."""

    def __init__(self, runtime: Any, request_capacity: int, *, engine: str) -> None:
        self.engine = engine
        self._identity = RequestIdentityRegistry(runtime, request_capacity)

    def bind(
        self,
        request_ids: Sequence[str],
        request_slots: Sequence[int],
        *,
        priorities: Sequence[int] | None = None,
        deadline_clocks: Sequence[int] | None = None,
        tenant_ids: Sequence[int] | None = None,
        stream: Any = None,
    ) -> tuple[RequestBinding, ...]:
        return self._identity.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            deadline_clocks=deadline_clocks,
            tenant_ids=tenant_ids,
            stream=stream,
        )

    def cancel_matching(self, request_id_prefix: str = "", *, all: bool = False) -> int:
        return self._identity.cancel_matching(request_id_prefix, all=all)

    def retire_request(self, request_id: str) -> bool:
        """Retire one engine-confirmed request at the lifecycle boundary."""
        return self._identity.cancel(request_id)

    @property
    def last_publish_count(self) -> int:
        return self._identity.last_publish_count

    @property
    def last_metadata_publish_count(self) -> int:
        return self._identity.last_metadata_publish_count
