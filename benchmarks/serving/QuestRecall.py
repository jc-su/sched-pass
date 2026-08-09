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

from nta_runtime.quest_selector import page_key_envelopes, quest_page_scores

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPT_SOURCES = ("README.md", "docs/ARCHITECTURE.md", "docs/SYSTEM_PLAN.md")


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


def evaluate_prompt(
    model: Any, tokenizer: Any, text: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    tokens = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=args.prompt_tokens)
    input_ids = tokens["input_ids"].to(model.device)
    if input_ids.shape[1] < args.prompt_tokens:
        raise RuntimeError(
            f"prompt only produced {input_ids.shape[1]} tokens; supply "
            "longer source text or lower --prompt-tokens"
        )
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
        # (heads, source) true attention of the final position.
        true_row = attention[0, :, -1, :].to(torch.float32).cpu()
        keys = outputs.past_key_values.layers[layer_index].keys
        key_states = keys[0].permute(1, 0, 2).to(torch.float32).cpu()
        seq, kv_heads, head_dim = key_states.shape
        pages = seq // args.page_tokens
        key_pages = key_states[: pages * args.page_tokens].view(
            pages, args.page_tokens, kv_heads, head_dim
        )
        query = last_position_query(captured[layer_index]).cpu()

        # Fail-closed verification: the reconstructed query must reproduce
        # the model's own attention row.
        scale = head_dim ** -0.5
        grouped_keys = key_states.repeat_interleave(group_size, dim=1)
        logits = torch.einsum("hd,shd->hs", query, grouped_keys) * scale
        rebuilt = torch.softmax(logits, dim=-1)
        error = (rebuilt - true_row).abs().max().item()
        if error > 5e-2:
            raise RuntimeError(
                f"layer {layer_index}: reconstructed attention diverges from "
                f"the model's row (max abs {error:.4f}); refusing to score"
            )

        kmin, kmax = page_key_envelopes(key_pages)
        quest = quest_page_scores(
            query.unsqueeze(0), kmin, kmax, group_size=group_size
        )[0]
        mass = page_mass(true_row, args.page_tokens)
        total = mass.sum(dim=-1)
        aggregate_mass = mass.sum(dim=0)
        quest_rank = quest.argsort(descending=True)
        oracle_rank = aggregate_mass.argsort(descending=True)
        row: dict[str, Any] = {
            "layer": layer_index,
            "pages": pages,
            "reconstruction_max_abs_error": error,
        }
        for k in args.top_k_pages:
            chosen = quest_rank[: min(k, pages)]
            oracle = oracle_rank[: min(k, pages)]
            row[f"quest_recall_at_{k}"] = float(
                (mass[:, chosen].sum() / total.sum()).item()
            )
            row[f"oracle_recall_at_{k}"] = float(
                (mass[:, oracle].sum() / total.sum()).item()
            )
        results.append(row)
    return results


def main() -> int:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    prompt_layers: list[dict[str, Any]] = []
    for source in PROMPT_SOURCES[: args.prompts]:
        text = (ROOT / source).read_text(encoding="utf-8")
        layers = evaluate_prompt(model, tokenizer, text * 8, args)
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
