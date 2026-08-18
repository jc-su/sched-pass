#!/usr/bin/env python3
"""Prove the extend staging chain captures and replays under CUDA graphs.

The extend-capture design bakes workspace pointers into one graph per
token bucket and lets every per-claim difference live in buffer contents:
envelopes, host rows, table indices, and the pinned source bytes. This
probe captures the real fused chain (scoring, top-k, prep, validated
indexed transfer) for a multi-layer workspace, then replays it against
changed envelope and source contents and byte-verifies that each replay
stages exactly the rows its own selection chose. It also times eager
versus replayed execution of the same chain.
"""

from __future__ import annotations

import argparse
import time

import torch
import flashinfer  # noqa: F401  (provides the TVM FFI symbols the JIT module links)

from nta_runtime.runtime import (
    IndexedHostObject,
    JitPhaseProgram,
    Runtime,
    RuntimeConfig,
)

PAGE = 16
LAYERS = 8
PAGES = 64
TOKENS = PAGES * PAGE
BUDGET = 8
HEADS = 2
DIM = 128
FREE_BUDGET = BUDGET - 3
KEPT_ROWS = (FREE_BUDGET + 2) * PAGE + PAGE  # full pages + tail page
CAPACITY = BUDGET * PAGE
OBJECT_BASE = 2


def reference_free_pages(kmin, kmax, queries):
    scores = torch.zeros(PAGES, dtype=torch.float32, device="cuda")
    q = queries.to(torch.float32)
    for head in range(q.shape[1]):
        kv_head = head // (q.shape[1] // HEADS)
        contribution = torch.maximum(
            kmin[:, kv_head] * q[0, head], kmax[:, kv_head] * q[0, head]
        ).sum(dim=-1)
        scores += contribution
    scores[torch.tensor([0, PAGES - 2, PAGES - 1], device="cuda")] = float(
        "-inf"
    )
    order = torch.argsort(scores, descending=True, stable=True)
    return order[:FREE_BUDGET]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    args = parser.parse_args()
    torch.cuda.init()
    device = torch.device("cuda")
    runtime = Runtime(
        RuntimeConfig(
            request_capacity=8,
            object_capacity=256,
            intent_capacity=64,
            work_ticket_capacity=64,
        )
    )
    phases = JitPhaseProgram(args.module)

    host_sources = []
    stagings = []
    for _ in range(LAYERS * 2):
        host_sources.append(
            torch.randn(
                TOKENS, HEADS, DIM, dtype=torch.float16
            ).pin_memory()
        )
        stagings.append(
            torch.zeros(CAPACITY, HEADS, DIM, dtype=torch.float16, device=device)
        )
    source_index = torch.zeros(
        LAYERS, CAPACITY, dtype=torch.int32, device=device
    )
    staging_index = torch.zeros(
        LAYERS, CAPACITY, dtype=torch.int32, device=device
    )
    element = HEADS * DIM * 2
    objects = []
    for layer in range(LAYERS):
        for half in range(2):
            source = host_sources[layer * 2 + half]
            staging = stagings[layer * 2 + half]
            objects.append(
                IndexedHostObject(
                    0x54455354 + len(objects),
                    1,
                    source.data_ptr(),
                    staging.data_ptr(),
                    source_index[layer].data_ptr(),
                    staging_index[layer].data_ptr(),
                    CAPACITY,
                    element,
                    source.stride(0) * source.element_size(),
                    staging.stride(0) * staging.element_size(),
                    TOKENS,
                    CAPACITY,
                )
            )
    runtime.register_indexed_host_objects(OBJECT_BASE, objects)
    torch.cuda.synchronize()

    kmin = torch.randn(LAYERS, PAGES, HEADS, DIM, dtype=torch.float32, device=device)
    kmax = kmin + torch.rand_like(kmin)
    queries = torch.randn(1, HEADS, DIM, dtype=torch.float16, device=device)
    page_scores = torch.empty(PAGES, dtype=torch.float32, device=device)
    full_forced = torch.tensor([0, PAGES - 2], dtype=torch.int64, device=device)
    ordered_pages = torch.empty(
        2 + FREE_BUDGET + 1, dtype=torch.int64, device=device
    )
    host_rows = torch.arange(TOKENS, dtype=torch.int32, device=device)
    device_rows = torch.arange(CAPACITY, dtype=torch.int32, device=device)
    cached_pages = torch.full(
        (LAYERS, BUDGET), -1, dtype=torch.int64, device=device
    )
    selected_rows = torch.empty(CAPACITY, dtype=torch.int32, device=device)
    copied_rows = torch.zeros(1, dtype=torch.int64, device=device)

    def chain(stream):
        for layer in range(LAYERS):
            base = OBJECT_BASE + 2 * layer
            phases.select_prepare_claim_rows(
                runtime,
                base,
                queries.contiguous(),
                kmin[layer],
                kmax[layer],
                page_scores,
                full_forced,
                PAGES - 1,
                FREE_BUDGET,
                ordered_pages,
                PAGE,
                TOKENS,
                host_rows,
                device_rows,
                cached_pages[layer],
                selected_rows,
                source_index[layer],
                staging_index[layer],
                copied_rows,
                stream=stream,
            )
            phases.progress_validated_indexed_host_range(
                runtime, base, 2, stream=stream
            )

    stream = torch.cuda.current_stream()
    # Eager timing + warmup.
    chain(stream)
    torch.cuda.synchronize()
    eager_start = time.monotonic()
    for _ in range(10):
        chain(stream)
    torch.cuda.synchronize()
    eager_ms = (time.monotonic() - eager_start) * 100.0

    # Capture.
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        chain(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain(torch.cuda.current_stream())
    torch.cuda.synchronize()

    def verify(tag):
        cached_pages.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        free = reference_free_pages(kmin[LAYERS - 1], kmax[LAYERS - 1], queries)
        ordered = torch.cat(
            [
                full_forced,
                free,
                torch.tensor([PAGES - 1], dtype=torch.int64, device=device),
            ]
        )
        kernel_free = ordered_pages[2 : 2 + FREE_BUDGET]
        if set(kernel_free.tolist()) != set(free.tolist()):
            raise SystemExit(
                f"{tag}: selection diverged {sorted(kernel_free.tolist())} "
                f"vs {sorted(free.tolist())}"
            )
        source = host_sources[(LAYERS - 1) * 2].to(device)
        for slot, page in enumerate(ordered.tolist()):
            rows = min(PAGE, TOKENS - page * PAGE)
            staged = stagings[(LAYERS - 1) * 2][
                slot * PAGE : slot * PAGE + rows
            ]
            expected = source[page * PAGE : page * PAGE + rows]
            if not torch.equal(staged, expected):
                raise SystemExit(f"{tag}: staged bytes wrong for page {page}")
        print(f"{tag}: selection + staged bytes exact")

    verify("replay-1 (capture-time contents)")

    # Change every content input and replay again: the graph must stage the
    # NEW selection from the NEW source bytes.
    for source in host_sources:
        source.copy_(torch.randn_like(source))
    kmin.copy_(torch.randn_like(kmin))
    kmax.copy_(kmin + torch.rand_like(kmax))
    queries.copy_(torch.randn_like(queries))
    verify("replay-2 (changed envelopes, queries, and source bytes)")

    replay_start = time.monotonic()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    replay_ms = (time.monotonic() - replay_start) * 100.0
    print(
        f"eager chain: {eager_ms:.2f}ms/iter | replayed: {replay_ms:.2f}ms/iter "
        f"({LAYERS} layers; speedup {eager_ms / max(replay_ms, 1e-6):.1f}x)"
    )
    print("extend staging chain is capture-sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
