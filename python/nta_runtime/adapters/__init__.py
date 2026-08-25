"""Engine adapters for the engine-neutral NTA execution contract."""

from .base import (
    ConsumerContract,
    ConsumerKind,
    ExactDemandProjection,
    EngineBatch,
    EngineBoundary,
    RequestIdentityAdapter,
)
from .sglang import SglangAdapter, SglangExecutionConfig
from .vllm import VllmAdapter, VllmSchedulerProjection
from .vllm_v1 import (
    SUPPORTED_VLLM_V1_VERSION,
    VllmV1Hook,
    VllmV1NumericalConsumer,
    VllmV1SchedulerProjection,
)

__all__ = [
    "ConsumerContract",
    "ConsumerKind",
    "EngineBatch",
    "ExactDemandProjection",
    "EngineBoundary",
    "RequestIdentityAdapter",
    "SglangAdapter",
    "SglangExecutionConfig",
    "VllmAdapter",
    "VllmSchedulerProjection",
    "SUPPORTED_VLLM_V1_VERSION",
    "VllmV1Hook",
    "VllmV1NumericalConsumer",
    "VllmV1SchedulerProjection",
]
