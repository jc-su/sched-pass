"""Machine-checkable numerical-consumer contract for experiment evidence.

This module intentionally has no CUDA, engine, or runtime imports. Serving and
paired-evaluation validators use it as the one schema implementation so a
scheduler projection cannot be promoted to numerical evidence by a validator
that drifted from the runtime contract.
"""

from __future__ import annotations

from typing import Any


CONSUMER_CONTRACT_SCHEMA = 1
CONSUMER_KINDS = frozenset(
    {"native_work_unit", "framework_reference", "projection_only"}
)
_BOOLEAN_FIELDS = (
    "exact_demand",
    "typed_work_plan",
    "native_submission",
    "numerical_consumer",
)


def validate_consumer_contract(
    value: Any,
    *,
    expected_engine: str | None = None,
    expected_backend: str | None = None,
    require_formal_execution: bool = False,
) -> dict[str, Any]:
    """Validate and return a normalized consumer contract."""

    if not isinstance(value, dict):
        raise ValueError("consumer contract must be an object")
    if value.get("schema") != CONSUMER_CONTRACT_SCHEMA:
        raise ValueError("unsupported consumer contract schema")
    for field in ("engine", "backend", "kind", "engine_version"):
        field_value = value.get(field)
        if not isinstance(field_value, str) or not field_value:
            raise ValueError(f"consumer contract lacks {field}")
    if expected_engine is not None and value["engine"] != expected_engine:
        raise ValueError(
            "consumer contract engine diverges from the expected engine"
        )
    if expected_backend is not None and value["backend"] != expected_backend:
        raise ValueError(
            "consumer contract backend diverges from the expected backend"
        )
    kind = value["kind"]
    if kind not in CONSUMER_KINDS:
        raise ValueError(f"unknown consumer contract kind: {kind!r}")
    booleans: dict[str, bool] = {}
    for field in _BOOLEAN_FIELDS:
        field_value = value.get(field)
        # bool is intentionally exact: Python's bool is an int subclass.
        if type(field_value) is not bool:
            raise ValueError(f"consumer contract {field} is not boolean")
        booleans[field] = field_value

    if kind == "native_work_unit":
        if not all(booleans.values()):
            raise ValueError("native consumer contract is incomplete")
    elif kind == "framework_reference":
        if not (
            booleans["exact_demand"]
            and not booleans["typed_work_plan"]
            and not booleans["native_submission"]
            and booleans["numerical_consumer"]
        ):
            raise ValueError("framework consumer contract is invalid")
    elif booleans["native_submission"] or booleans["numerical_consumer"]:
        raise ValueError("projection-only contract claims numerical execution")

    if require_formal_execution and kind == "projection_only":
        raise ValueError(
            "projection-only engine hook cannot be serving evidence"
        )
    return dict(value)
