#!/usr/bin/env python3
"""Property tests for Quest-style page selection.

The scores must genuinely upper-bound attainable attention logits and route
through the correct GQA head groups; both properties are what make selection
quality-safe, so they are asserted mathematically rather than eyeballed.
"""

from __future__ import annotations

import torch

from nta_runtime.quest_selector import (
    page_key_envelopes,
    quest_candidate_scores,
    quest_page_scores,
)


def reference_per_head_max_logits(
    query: torch.Tensor, key_pages: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Sum over query heads of the exact per-head max token logit per page."""
    batch, query_heads, head_dim = query.shape
    pages, tokens, kv_heads, _ = key_pages.shape
    result = torch.zeros((batch, pages), dtype=torch.float64)
    for b in range(batch):
        for h in range(query_heads):
            kv_head = h // group_size
            # (pages, tokens)
            logits = torch.einsum(
                "d,ptd->pt",
                query[b, h].to(torch.float64),
                key_pages[:, :, kv_head, :].to(torch.float64),
            )
            result[b] += logits.amax(dim=1)
    return result


def check_upper_bound(device: str) -> None:
    generator = torch.Generator().manual_seed(20260809)
    key_pages = torch.randn((24, 16, 2, 32), generator=generator).to(device)
    query = torch.randn((3, 4, 32), generator=generator).to(device)
    kmin, kmax = page_key_envelopes(key_pages)
    scores = quest_page_scores(query, kmin, kmax, group_size=2).cpu()
    reference = reference_per_head_max_logits(
        query.cpu(), key_pages.cpu(), group_size=2
    )
    slack = scores.to(torch.float64) - reference
    if not bool((slack >= -1e-3).all()):
        raise AssertionError(
            f"quest score fails to bound attainable logits: min slack "
            f"{slack.min().item()}"
        )


def check_planted_signal(device: str) -> None:
    generator = torch.Generator().manual_seed(7)
    key_pages = 0.01 * torch.randn((16, 8, 2, 16), generator=generator)
    query = torch.randn((1, 4, 16), generator=generator)
    planted = 9
    for kv_head in range(2):
        for query_head in range(2 * kv_head, 2 * kv_head + 2):
            key_pages[planted, 3, kv_head] += 10.0 * torch.sign(
                query[0, query_head]
            )
    kmin, kmax = page_key_envelopes(key_pages.to(device))
    scores = quest_page_scores(query.to(device), kmin, kmax, group_size=2)
    if int(scores[0].argmax()) != planted:
        raise AssertionError(
            f"planted page {planted} did not rank first: "
            f"{int(scores[0].argmax())}"
        )


def check_gqa_routing(device: str) -> None:
    key_pages = torch.zeros((4, 4, 2, 8))
    query = torch.zeros((1, 4, 8))
    # Query lives only in heads 2-3, which belong to KV head 1.
    query[0, 2:] = 1.0
    # Page 1 carries signal only where the query looks; page 2 carries a
    # larger signal in the head group the query ignores.
    key_pages[1, 0, 1] = 5.0
    key_pages[2, 0, 0] = 50.0
    kmin, kmax = page_key_envelopes(key_pages.to(device))
    scores = quest_page_scores(query.to(device), kmin, kmax, group_size=2)[0]
    if not scores[1] > scores[2]:
        raise AssertionError(
            f"GQA routing selected the wrong head group: {scores.tolist()}"
        )


def check_candidate_view(device: str) -> None:
    generator = torch.Generator().manual_seed(11)
    candidate_pages = torch.randn((3, 6, 4, 2, 16), generator=generator).to(
        device
    )
    query = torch.randn((3, 4, 16), generator=generator).to(device)
    combined = quest_candidate_scores(query, candidate_pages, group_size=2)
    for request in range(3):
        kmin, kmax = page_key_envelopes(candidate_pages[request])
        alone = quest_page_scores(
            query[request : request + 1], kmin, kmax, group_size=2
        )[0]
        if not torch.allclose(combined[request], alone, atol=1e-4):
            raise AssertionError("per-request candidate scoring disagrees")


def check_rejections() -> None:
    bad = [
        lambda: page_key_envelopes(torch.zeros((2, 3, 4))),
        lambda: quest_page_scores(
            torch.zeros((1, 4, 8)),
            torch.zeros((2, 2, 8)),
            torch.zeros((2, 2, 4)),
            group_size=2,
        ),
        lambda: quest_page_scores(
            torch.zeros((1, 3, 8)),
            torch.zeros((2, 2, 8)),
            torch.zeros((2, 2, 8)),
            group_size=2,
        ),
        lambda: quest_candidate_scores(
            torch.zeros((1, 2, 3, 4)), torch.zeros((1, 2, 3, 4)), group_size=1
        ),
    ]
    for index, case in enumerate(bad):
        try:
            case()
        except ValueError:
            continue
        raise AssertionError(f"invalid-input case {index} was not rejected")


def main() -> int:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        check_upper_bound(device)
        check_planted_signal(device)
        check_gqa_routing(device)
        check_candidate_view(device)
    check_rejections()
    print(f"quest selector properties hold on {devices}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
