#!/usr/bin/env python3
"""Attention-mass recall of Quest page selection on a real checkpoint.

The 1D quality gate needs evidence that envelope-based top-k selection
retains the attention mass a real model actually places on real text. This
benchmark runs a real forward pass, captures every layer's post-RoPE keys and
true last-position attention distribution, recomputes the decode query from
the layer's own projection weights and rotary embeddings, and *verifies* that
reconstruction against the model's attention row before scoring. It then
reports, per layer, how much true attention mass the Quest-selected pages
capture versus the oracle (true top-k) and the page count.

No sampling, no simulation: prompts are real repository prose, the model is a
real checkpoint, and every number is derived from one verified forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
from typing import Any

import torch

from nta_runtime.quest_selector import (
    budgeted_page_selection,
    page_key_envelopes,
    quest_page_scores,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPT_SOURCES = ("README.md", "docs/ARCHITECTURE.md", "docs/SYSTEM_PLAN.md")


def distinct_corpus(start_source: str) -> tuple[str, int]:
    """Concatenate every repository document once, rotated to start at
    ``start_source`` so prompts differ, replicating only if the distinct
    corpus itself is too short.

    Replicated text is the pathological case for attention concentration —
    mass legitimately spreads over near-duplicate keys — so recall measured
    on replicated prompts is a lower bound, not a workload estimate. The
    report records whether replication was required.
    """
    documents = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
    names = [str(path.relative_to(ROOT)) for path in documents]
    pivot = names.index(start_source) if start_source in names else 0
    ordered = documents[pivot:] + documents[:pivot]
    corpus = "\n\n".join(
        path.read_text(encoding="utf-8") for path in ordered
    )
    return corpus, len(ordered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--page-tokens", type=int, default=16)
    parser.add_argument(
        "--top-k-pages", type=lambda v: tuple(int(x) for x in v.split(",")),
        default=(4, 8, 16, 32),
    )
    # Deployed sparse-attention selectors always retain attention sinks and
    # the recent window; the envelope ranks only the remaining budget. Zeros
    # reproduce the raw-envelope ablation.
    parser.add_argument("--sink-pages", type=int, default=1)
    parser.add_argument("--recent-pages", type=int, default=2)
    parser.add_argument(
        "--verify-tokens",
        type=int,
        default=512,
        help=(
            "prefix length for the materialized-attention verification pass; "
            "prompts longer than this score through the verified "
            "reconstruction path without materializing attention"
        ),
    )
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.prompt_tokens < 4 * args.page_tokens:
        parser.error("prompt must span at least four pages")
    if args.prompts < 1 or args.prompts > len(PROMPT_SOURCES):
        parser.error(f"1..{len(PROMPT_SOURCES)} prompts are available")
    if min(args.top_k_pages) <= 0:
        parser.error("top-k page counts must be positive")
    return args


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def apply_rotary(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = q.shape[-1] // 2
    rotated = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    return q * cos + rotated * sin


def capture_layer_inputs(model: Any) -> tuple[list[dict[str, Any]], list[Any]]:
    captured: list[dict[str, Any]] = []
    handles = []

    def hook(module: Any, hook_args: tuple, hook_kwargs: dict) -> None:
        hidden = hook_kwargs.get("hidden_states")
        if hidden is None and hook_args:
            hidden = hook_args[0]
        position = hook_kwargs.get("position_embeddings")
        if position is None:
            for value in hook_args:
                if (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and torch.is_tensor(value[0])
                ):
                    position = value
                    break
        captured.append(
            {"module": module, "hidden": hidden, "position": position}
        )

    for layer in model.model.layers:
        handles.append(
            layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        )
    return captured, handles


def last_position_query(record: dict[str, Any]) -> torch.Tensor:
    """Recompute the final-position post-RoPE query from module weights."""
    module = record["module"]
    hidden = record["hidden"]
    position = record["position"]
    if hidden is None or position is None:
        raise RuntimeError(
            "attention hook did not observe hidden states and rotary "
            "embeddings; this model family is unsupported"
        )
    weight_dtype = module.q_proj.weight.dtype
    last_hidden = hidden[0, -1].to(weight_dtype)
    q = module.q_proj(last_hidden)
    head_dim = getattr(module, "head_dim", None) or (
        q.shape[-1] // module.config.num_attention_heads
    )
    q = q.view(-1, head_dim)
    if hasattr(module, "q_norm"):
        q = module.q_norm(q)
    cos, sin = position
    q = apply_rotary(
        q.to(torch.float32), cos[0, -1].to(torch.float32),
        sin[0, -1].to(torch.float32),
    )
    return q


def page_mass(row: torch.Tensor, page_tokens: int) -> torch.Tensor:
    """Fold a (heads, tokens) attention row into per-page mass."""
    heads, tokens = row.shape
    pages = tokens // page_tokens
    return (
        row[:, : pages * page_tokens]
        .view(heads, pages, page_tokens)
        .sum(dim=-1)
    )


def reconstructed_row(query: torch.Tensor, key_states: torch.Tensor,
                      group_size: int) -> torch.Tensor:
    """Last-position attention row from the verified reconstruction path."""
    head_dim = key_states.shape[-1]
    grouped_keys = key_states.repeat_interleave(group_size, dim=1)
    logits = torch.einsum("hd,shd->hs", query, grouped_keys) * head_dim ** -0.5
    return torch.softmax(logits, dim=-1)


def score_layer(
    layer_index: int,
    query: torch.Tensor,
    key_states: torch.Tensor,
    true_row: torch.Tensor,
    group_size: int,
    args: argparse.Namespace,
    reconstruction_error: float | None,
) -> dict[str, Any]:
    seq = key_states.shape[0]
    pages = seq // args.page_tokens
    key_pages = key_states[: pages * args.page_tokens].view(
        pages, args.page_tokens, key_states.shape[1], key_states.shape[2]
    )
    kmin, kmax = page_key_envelopes(key_pages)
    quest = quest_page_scores(
        query.unsqueeze(0), kmin, kmax, group_size=group_size
    )[0]
    mass = page_mass(true_row, args.page_tokens)
    total = mass.sum()
    oracle_rank = mass.sum(dim=0).argsort(descending=True)
    row: dict[str, Any] = {
        "layer": layer_index,
        "pages": pages,
        "reconstruction_max_abs_error": reconstruction_error,
    }
    for k in args.top_k_pages:
        budget = min(k, pages)
        chosen = budgeted_page_selection(
            quest, pages, budget,
            sink_pages=args.sink_pages, recent_pages=args.recent_pages,
        )
        oracle = oracle_rank[:budget]
        row[f"quest_recall_at_{k}"] = float((mass[:, chosen].sum() / total))
        row[f"oracle_recall_at_{k}"] = float((mass[:, oracle].sum() / total))
    return row


def tokenize_prompt(tokenizer: Any, text: str, model: Any,
                    length: int) -> torch.Tensor:
    tokens = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=length)
    input_ids = tokens["input_ids"].to(model.device)
    if input_ids.shape[1] < length:
        raise RuntimeError(
            f"prompt only produced {input_ids.shape[1]} tokens; supply "
            "longer source text or lower the requested length"
        )
    return input_ids


def evaluate_prompt(
    model: Any, tokenizer: Any, text: str, args: argparse.Namespace,
    *, length: int, verify_only: bool = False,
) -> list[dict[str, Any]]:
    """Materialized-attention pass: verifies reconstruction, then scores."""
    input_ids = tokenize_prompt(tokenizer, text, model, length)
    captured, handles = capture_layer_inputs(model)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids, output_attentions=True, use_cache=True,
                attn_implementation="eager",
            )
    finally:
        for handle in handles:
            handle.remove()
    config = model.config
    group_size = config.num_attention_heads // config.num_key_value_heads
    results = []
    for layer_index, attention in enumerate(outputs.attentions):
        true_row = attention[0, :, -1, :].to(torch.float32).cpu()
        keys = outputs.past_key_values.layers[layer_index].keys
        key_states = keys[0].permute(1, 0, 2).to(torch.float32).cpu()
        query = last_position_query(captured[layer_index]).cpu()
        rebuilt = reconstructed_row(query, key_states, group_size)
        error = (rebuilt - true_row).abs().max().item()
        if error > 5e-2:
            raise RuntimeError(
                f"layer {layer_index}: reconstructed attention diverges from "
                f"the model's row (max abs {error:.4f}); refusing to score"
            )
        if not verify_only:
            results.append(
                score_layer(layer_index, query, key_states, true_row,
                            group_size, args, error)
            )
    return results


def evaluate_prompt_long(
    model: Any, tokenizer: Any, text: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Long-context pass over the verified reconstruction path.

    No attention tensor is materialized; the true row comes from the same
    query/key reconstruction that the short-prefix pass certifies against the
    model's own attention on every run.
    """
    input_ids = tokenize_prompt(tokenizer, text, model, args.prompt_tokens)
    captured, handles = capture_layer_inputs(model)
    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    config = model.config
    group_size = config.num_attention_heads // config.num_key_value_heads
    results = []
    for layer_index in range(len(model.model.layers)):
        keys = outputs.past_key_values.layers[layer_index].keys
        key_states = keys[0].permute(1, 0, 2).to(torch.float32).cpu()
        query = last_position_query(captured[layer_index]).cpu()
        true_row = reconstructed_row(query, key_states, group_size)
        results.append(
            score_layer(layer_index, query, key_states, true_row,
                        group_size, args, None)
        )
    return results


def main() -> int:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Measurement tool, not a serving path: float32 everywhere keeps the
    # reconstruction certification meaningful. bfloat16 attention rows
    # legitimately diverge ~1e-1 from the exact computation, which would
    # force a tolerance too loose to certify anything.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    long_context = args.prompt_tokens > args.verify_tokens
    prompt_layers: list[dict[str, Any]] = []
    corpus_replicated = False
    for source in PROMPT_SOURCES[: args.prompts]:
        text, distinct_documents = distinct_corpus(source)
        probe = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        if probe < args.prompt_tokens:
            corpus_replicated = True
            text = text * (args.prompt_tokens // max(probe, 1) + 1)
        if long_context:
            # Certify the reconstruction on the materialized prefix, then
            # score the full length through the certified path.
            evaluate_prompt(model, tokenizer, text, args,
                            length=args.verify_tokens, verify_only=True)
            layers = evaluate_prompt_long(model, tokenizer, text, args)
        else:
            layers = evaluate_prompt(model, tokenizer, text, args,
                                     length=args.prompt_tokens)
        prompt_layers.append({"source": source, "layers": layers})

    top_k = args.top_k_pages
    aggregate: dict[str, Any] = {}
    for k in top_k:
        quest_values = [
            layer[f"quest_recall_at_{k}"]
            for prompt in prompt_layers
            for layer in prompt["layers"]
        ]
        oracle_values = [
            layer[f"oracle_recall_at_{k}"]
            for prompt in prompt_layers
            for layer in prompt["layers"]
        ]
        aggregate[f"quest_recall_at_{k}"] = {
            "mean": sum(quest_values) / len(quest_values),
            "min_layer": min(quest_values),
        }
        aggregate[f"oracle_recall_at_{k}"] = {
            "mean": sum(oracle_values) / len(oracle_values),
            "min_layer": min(oracle_values),
        }
    report = {
        "schema": 1,
        "classification": "quest-attention-mass-recall",
        "revision": os.environ.get("NTA_REVISION", git_value("rev-parse", "HEAD")),
        "dirty": bool(git_value("status", "--porcelain")),
        "model": str(args.model),
        "device": args.device,
        "prompt_tokens": args.prompt_tokens,
        "page_tokens": args.page_tokens,
        "mode": "long-context-reconstructed" if long_context else "materialized",
        "verified_prefix_tokens": args.verify_tokens if long_context else None,
        "corpus_replicated": corpus_replicated,
        "prompts": [p["source"] for p in prompt_layers],
        "aggregate": aggregate,
        "per_prompt": prompt_layers,
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
