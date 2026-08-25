#!/usr/bin/env python3
"""Verify the pinned SGLang/vLLM dual-engine runtime profile in one process."""

from __future__ import annotations

import importlib.metadata


EXPECTED = {
    "torch": "2.11.0",
    "flashinfer-python": "0.6.14",
    "sglang": "0.5.16",
    "sglang-kernel": "0.4.5",
    "vllm": "0.26.0",
}


def main() -> None:
    for name, expected in EXPECTED.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(
                f"dual-engine profile requires {name}=={expected}, found {actual}"
            )

    import torch
    import flashinfer  # noqa: F401
    import sglang  # noqa: F401
    import vllm  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("dual-engine profile requires CUDA visibility")

    from nta_runtime.plugins.sglang import register as register_sglang
    from nta_runtime.plugins.vllm import register as register_vllm

    register_sglang()
    register_vllm()
    print(
        "engine_environment=pass",
        f"torch={EXPECTED['torch']}",
        f"flashinfer={EXPECTED['flashinfer-python']}",
        f"sglang={EXPECTED['sglang']}",
        f"sglang-kernel={EXPECTED['sglang-kernel']}",
        f"vllm={EXPECTED['vllm']}",
    )


if __name__ == "__main__":
    main()
