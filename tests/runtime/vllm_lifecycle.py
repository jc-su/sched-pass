#!/usr/bin/env python3
"""Test vLLM worker-runtime rollback and idempotent shutdown ownership."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines import vllm as vllm_engine  # noqa: E402
from nta_runtime.adapters.base import EngineBatch, ExactDemandProjection  # noqa: E402
from nta_runtime.requests import RequestBinding  # noqa: E402
from nta_runtime.work_unit import Granularity  # noqa: E402


class Runner:
    pass


class FakeRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(tenant_capacity=1)
        self.closed = 0

    def set_tenant_budget(self, *_args) -> None:
        raise AssertionError("invalid tenant should be rejected before budget upload")

    def close(self) -> None:
        self.closed += 1


class FakeResources:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.tier = SimpleNamespace()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        self.runtime.close()


def main() -> None:
    batch = EngineBatch(
        "vllm",
        4,
        (
            RequestBinding(0, 0, 1, 11),
            RequestBinding(1, 1, 1, 12),
        ),
        Granularity.PAGE_GROUP,
        ExactDemandProjection(((10, 11, 12, 13), (20, 21)), 4096),
    )
    split_schedule = SimpleNamespace(kv_chunk_tokens=32)
    assert vllm_engine.NtaVllmFlashInferImpl._physical_pages(
        batch, split_schedule, 0, 1, 16
    ) == (12, 13)
    assert vllm_engine.NtaVllmFlashInferImpl._physical_pages(
        batch, SimpleNamespace(kv_chunk_tokens=0), 1, 0, 16
    ) == (20, 21)
    for schedule, tile in (
        (SimpleNamespace(kv_chunk_tokens=16), 4),
        (SimpleNamespace(kv_chunk_tokens=15), 0),
    ):
        try:
            vllm_engine.NtaVllmFlashInferImpl._physical_pages(
                batch, schedule, 0, tile, 16
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid vLLM physical page selection was accepted")

    runtime = FakeRuntime()
    resources = FakeResources(runtime)
    controller = vllm_engine.VllmV1WorkerController(Runner())
    with patch.dict("os.environ", {"NTA_TENANT_BUDGETS": "7:4096"}, clear=False):
        with patch.object(vllm_engine, "_build_resources", return_value=resources):
            try:
                controller._ensure_hook(
                    controller._runner_ref(),
                    request_capacity=1,
                    page_size=16,
                    page_bytes=4096,
                )
            except RuntimeError as error:
                assert "tenant 7" in str(error)
            else:
                raise AssertionError("invalid vLLM tenant policy was accepted")
    assert resources.closed == 1
    assert runtime.closed == 1
    assert controller._runtime is None
    assert controller._hook is None
    controller.close()
    assert runtime.closed == 1
    print("vllm_lifecycle=pass")


if __name__ == "__main__":
    main()
