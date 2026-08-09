#!/usr/bin/env python3
"""Property tests for incremental page-envelope maintenance.

The load-bearing invariant: after any interleaving of appends, whole-page
writes, and slot-reusing resets, every live page's incremental envelope must
equal the envelope recomputed from scratch over the tokens the page actually
holds. A seeded random mutation storm checks exactly that against a reference
pool, on CPU and CUDA.
"""

from __future__ import annotations

import torch

from nta_runtime.page_summaries import PageSummaryTable
from nta_runtime.quest_selector import page_key_envelopes, quest_page_scores

PAGES = 24
PAGE_TOKENS = 8
KV_HEADS = 2
HEAD_DIM = 16


def verify_against_reference(
    table: PageSummaryTable, reference: dict[int, list[torch.Tensor]]
) -> None:
    for page in range(PAGES):
        tokens = reference.get(page, [])
        if not tokens:
            if int(table.filled[page]) != 0:
                raise AssertionError(f"page {page} should be empty")
            continue
        stacked = torch.stack(tokens).unsqueeze(0)
        expect_min, expect_max = page_key_envelopes(
            stacked.permute(0, 1, 2, 3).reshape(
                1, len(tokens), KV_HEADS, HEAD_DIM
            )
        )
        kmin, kmax = table.envelopes(torch.tensor([page]))
        if not torch.allclose(kmin.cpu(), expect_min.cpu(), atol=1e-6):
            raise AssertionError(f"page {page} min envelope diverged")
        if not torch.allclose(kmax.cpu(), expect_max.cpu(), atol=1e-6):
            raise AssertionError(f"page {page} max envelope diverged")
        if int(table.filled[page]) != len(tokens):
            raise AssertionError(f"page {page} fill count diverged")


def mutation_storm(device: str) -> None:
    generator = torch.Generator().manual_seed(20260809)
    table = PageSummaryTable(
        PAGES, PAGE_TOKENS, KV_HEADS, HEAD_DIM, device=device
    )
    reference: dict[int, list[torch.Tensor]] = {}
    generations = [0] * PAGES
    for step in range(400):
        op = int(torch.randint(0, 3, (1,), generator=generator))
        if op == 0:
            # Batched decode append: one token to each of several distinct,
            # non-full pages.
            candidates = [
                p
                for p in range(PAGES)
                if len(reference.get(p, [])) < PAGE_TOKENS
            ]
            if not candidates:
                continue
            width = 1 + int(
                torch.randint(0, min(4, len(candidates)), (1,), generator=generator)
            )
            chosen = [
                candidates[int(i)]
                for i in torch.randperm(len(candidates), generator=generator)[
                    :width
                ]
            ]
            tokens = torch.randn(
                (len(chosen), KV_HEADS, HEAD_DIM), generator=generator
            )
            table.append_tokens(
                torch.tensor(chosen, device=device), tokens.to(device)
            )
            for slot, index in enumerate(chosen):
                reference.setdefault(index, []).append(tokens[slot])
        elif op == 1:
            # Promotion: whole or partial page materialization.
            page = int(torch.randint(0, PAGES, (1,), generator=generator))
            tokens = 1 + int(
                torch.randint(0, PAGE_TOKENS, (1,), generator=generator)
            )
            payload = torch.randn(
                (1, tokens, KV_HEADS, HEAD_DIM), generator=generator
            )
            table.write_pages(
                torch.tensor([page], device=device), payload.to(device)
            )
            reference[page] = [payload[0, t] for t in range(tokens)]
        else:
            # Eviction and slot reuse.
            page = int(torch.randint(0, PAGES, (1,), generator=generator))
            table.reset_pages(torch.tensor([page], device=device))
            reference.pop(page, None)
            generations[page] += 1
            if int(table.generation[page]) != generations[page]:
                raise AssertionError("generation did not advance on reset")
        if step % 40 == 0:
            verify_against_reference(table, reference)
    verify_against_reference(table, reference)

    # Scoring consistency and empty-page masking on the final state.
    live = [p for p in range(PAGES) if reference.get(p)]
    empty = [p for p in range(PAGES) if not reference.get(p)]
    if live and empty:
        query = torch.randn((2, 4, HEAD_DIM), generator=generator).to(device)
        pages = torch.tensor(live + empty[:1], device=device)
        scores = table.scores(query, pages, group_size=2)
        kmin, kmax = table.envelopes(torch.tensor(live, device=device))
        direct = quest_page_scores(query, kmin, kmax, group_size=2)
        if not torch.allclose(scores[:, : len(live)].cpu(), direct.cpu(),
                              atol=1e-5):
            raise AssertionError("table scoring diverged from direct scoring")
        if not bool(torch.isinf(scores[:, -1]).all()) or not bool(
            (scores[:, -1] < 0).all()
        ):
            raise AssertionError("empty page did not score -inf")


def rejections() -> None:
    table = PageSummaryTable(4, 2, KV_HEADS, HEAD_DIM)
    token = torch.randn((1, KV_HEADS, HEAD_DIM))
    cases = [
        lambda: PageSummaryTable(0, 2, 2, 4),
        lambda: table.append_tokens(torch.tensor([9]), token),
        lambda: table.append_tokens(torch.tensor([0, 0]), token.repeat(2, 1, 1)),
        lambda: table.append_tokens(
            torch.tensor([1]), torch.randn((1, KV_HEADS + 1, HEAD_DIM))
        ),
        lambda: table.write_pages(
            torch.tensor([1]), torch.randn((1, 3, KV_HEADS, HEAD_DIM))
        ),
    ]
    for index, case in enumerate(cases):
        try:
            case()
        except ValueError:
            continue
        raise AssertionError(f"invalid-input case {index} was not rejected")
    # Capacity: the third append to a two-token page must fail.
    table.append_tokens(torch.tensor([2]), token)
    table.append_tokens(torch.tensor([2]), token)
    try:
        table.append_tokens(torch.tensor([2]), token)
    except ValueError:
        pass
    else:
        raise AssertionError("append past page capacity was not rejected")


def main() -> int:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        mutation_storm(device)
    rejections()
    print(f"page summary maintenance properties hold on {devices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
