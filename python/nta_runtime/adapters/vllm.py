"""vLLM boundary adapter seam.

vLLM owns request IDs, block tables, cancellation, and scheduler slots.  The
adapter accepts those already-normalized values and deliberately does not
import vLLM, so contract tests run without a vLLM installation.
"""

from __future__ import annotations

from typing import Any

from .base import EngineBatch, RequestIdentityAdapter
from ..work_unit import Granularity


class VllmAdapter(RequestIdentityAdapter):
    def __init__(self, runtime: Any, request_capacity: int) -> None:
        super().__init__(runtime, request_capacity, engine="vllm")

    def bind_batch(
        self,
        request_ids: tuple[str, ...],
        request_slots: tuple[int, ...],
        *,
        epoch: int,
        stream: Any = None,
        priorities: tuple[int, ...] | None = None,
        deadline_clocks: tuple[int, ...] | None = None,
        granularity: Granularity = Granularity.PAGE_GROUP,
    ) -> EngineBatch:
        bindings = self.bind(
            request_ids,
            request_slots,
            priorities=priorities,
            deadline_clocks=deadline_clocks,
            stream=stream,
        )
        return EngineBatch(self.engine, epoch, bindings, granularity)
