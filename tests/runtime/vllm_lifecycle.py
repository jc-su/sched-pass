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


def main() -> None:
    runtime = FakeRuntime()
    controller = vllm_engine.VllmV1WorkerController(Runner())
    with patch.dict("os.environ", {"NTA_TENANT_BUDGETS": "7:4096"}, clear=False):
        with patch.object(vllm_engine, "_build_runtime", return_value=runtime):
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
    assert runtime.closed == 1
    assert controller._runtime is None
    assert controller._hook is None
    controller.close()
    assert runtime.closed == 1
    print("vllm_lifecycle=pass")


if __name__ == "__main__":
    main()
