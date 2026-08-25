#!/usr/bin/env python3
"""Compare NTA's externally acquired paged decode with real FlashInfer."""

from __future__ import annotations

import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch


HEADER = struct.Struct("<8I")
MAGIC = 0x4E544146


def take(data: memoryview, offset: int, dtype: np.dtype, count: int):
    size = np.dtype(dtype).itemsize * count
    end = offset + size
    if end > len(data):
        raise RuntimeError("truncated NTA FlashInfer fixture")
    return np.frombuffer(data[offset:end], dtype=dtype, count=count).copy(), end


def read_fixture(path: pathlib.Path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise RuntimeError("missing NTA FlashInfer fixture header")
    fields = HEADER.unpack_from(raw)
    magic, version, requests, pages, head_dim, page_tokens, references, _ = fields
    if magic != MAGIC or version != 1 or head_dim != 128 or page_tokens != 16:
        raise RuntimeError(f"unsupported NTA FlashInfer fixture: {fields}")

    data = memoryview(raw)
    offset = HEADER.size
    indptr, offset = take(data, offset, np.int32, requests + 1)
    indices, offset = take(data, offset, np.int32, references)
    last_len, offset = take(data, offset, np.int32, requests)
    query, offset = take(data, offset, np.float16, requests * head_dim)
    kv, offset = take(
        data, offset, np.float16, pages * 2 * page_tokens * head_dim
    )
    nta_output, offset = take(data, offset, np.float32, requests * head_dim)
    if offset != len(data):
        raise RuntimeError("unexpected trailing data in NTA FlashInfer fixture")
    return (
        requests,
        pages,
        head_dim,
        page_tokens,
        indptr,
        indices,
        last_len,
        query,
        kv,
        nta_output,
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: differential_decode.py <nta-paged-attention>")
    executable = pathlib.Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="nta-flashinfer-") as directory:
        fixture = pathlib.Path(directory) / "decode.bin"
        command = [
            str(executable),
            "--mode=host-staged",
            "--copy=global",
            "--requests=7",
            "--min-pages=2",
            "--max-pages=5",
            "--iterations=1",
            "--progress-rounds=5",
            f"--dump-output={fixture}",
        ]
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
        parsed = read_fixture(fixture)

    (
        requests,
        pages,
        head_dim,
        page_tokens,
        indptr,
        indices,
        last_len,
        query,
        kv,
        nta_output,
    ) = parsed

    import flashinfer

    device = torch.device("cuda")
    q_tensor = torch.from_numpy(query).reshape(requests, 1, head_dim).to(device)
    kv_tensor = (
        torch.from_numpy(kv)
        .reshape(pages, 2, page_tokens, 1, head_dim)
        .to(device)
    )
    indptr_tensor = torch.from_numpy(indptr).to(device)
    indices_tensor = torch.from_numpy(indices).to(device)
    last_len_tensor = torch.from_numpy(last_len).to(device)
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        indptr_tensor,
        indices_tensor,
        last_len_tensor,
        1,
        1,
        head_dim,
        page_tokens,
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    flashinfer_output = wrapper.run(q_tensor, kv_tensor).float().cpu().numpy()
    nta_output = nta_output.reshape(requests, 1, head_dim)
    difference = np.abs(flashinfer_output - nta_output)
    maximum = float(difference.max())
    mean = float(difference.mean())
    finite = bool(np.isfinite(flashinfer_output).all())
    matched = finite and bool(np.allclose(flashinfer_output, nta_output, 2e-3, 2e-3))
    print(
        f"flashinfer_version={flashinfer.__version__} requests={requests} "
        f"physical_pages={pages} max_abs_error={maximum:.6g} "
        f"mean_abs_error={mean:.6g} matched={int(matched)}"
    )
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
