#!/usr/bin/env python3
"""Analytic regime map: where can selected/tiered serving win at all?

Decode attention reads every resident KV byte of a request each step, so
the latency headroom available to *any* KV-selection mechanism is bounded
by attention's share of the step, and the capacity headroom is bounded by
KV pressure against the pool. Both are arithmetic over model geometry,
batch, context, and budget — no simulation. This map exists because the
project spent weeks optimizing at an operating point (2-KV-head 3B model,
batch <= 2, uncontended 96GB pool) whose available decode win is ~4%: the
mechanism could not have won there no matter how good the implementation
became. Every future experiment runs only at points this map says can pay,
and the map's negative region is the paper's measured applicability
boundary, not an embarrassment.

Constants carry provenance. Anchored step times come from measured
artifacts in this repository; estimated ones are marked and must be
replaced by the P1 anchor runs before any derived number is published.
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Measured on this host (four-tier ladder, docs/VALIDATION.md): HBM ~1.6TB/s
# on the RTX PRO 6000 Blackwell; matches the part's GDDR7 specification.
HBM_BYTES_PER_SECOND = 1.6e12

# results/serving/selected-load-16k-b128-{optimized,scan,misspath}.json,
# arms.stock.resident_p95_tpot_seconds: 0.0100-0.0104 for Qwen2.5-3B
# resident decode (2K context, effective batch ~1 per forward in separate
# mode). Attention at that point reads ~74MB (~46us), so the anchor is
# almost pure base cost.
ANCHOR_TPOT_SECONDS = {"qwen2.5-3b": 0.0101}

# Kernel-launch/setup floor added by the selected path per layer, beyond
# dense: selection + staging-check + table-rewrite launches. Assumption
# (~3 launches x ~8us) pending the per-launch anomaly profile; the map's
# conclusions are insensitive to 2x error here.
MECHANISM_LAUNCHES_PER_LAYER = 3
LAUNCH_SECONDS = 8e-6

MODELS = {
    # name: (layers, kv_heads, head_dim, params_billions)  [config.json]
    "qwen2.5-3b": (36, 2, 128, 3.09),
    "qwen3-4b": (36, 8, 128, 4.02),
    "qwen3-8b": (36, 8, 128, 8.19),
    "qwen3-30b-a3b": (48, 4, 128, 30.5),
}

GPU_MEMORY_BYTES = 96.0e9 * 0.85  # RTX PRO 6000 Blackwell, SGLang default
STAGING_POOL_BYTES = 2.0e9        # NTA bounded staging reservation
RECENT_SINK_TOKENS = 64           # retention rows kept beyond the budget
PAGE_TOKENS = 16
OUTPUT_TOKENS = 256               # decode-phase KV growth allowance


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int) -> int:
    return layers * kv_heads * head_dim * 2 * 2  # K+V, fp16


def base_step_seconds(name: str) -> tuple[float, bool]:
    """Anchored where measured; weight-read floor plus the anchor's fixed
    overhead where not. Estimated entries must be re-anchored by P1."""
    layers, kv_heads, head_dim, params = MODELS[name]
    weight_read = params * 1e9 * 2 / HBM_BYTES_PER_SECOND
    if name in ANCHOR_TPOT_SECONDS:
        return ANCHOR_TPOT_SECONDS[name], True
    anchor_layers, _, _, anchor_params = MODELS["qwen2.5-3b"]
    anchor_overhead = ANCHOR_TPOT_SECONDS["qwen2.5-3b"] - (
        anchor_params * 1e9 * 2 / HBM_BYTES_PER_SECOND
    )
    return weight_read + anchor_overhead, False


def evaluate_point(
    name: str, context: int, batch: int, budget_pages: int
) -> dict:
    layers, kv_heads, head_dim, params = MODELS[name]
    per_token = kv_bytes_per_token(layers, kv_heads, head_dim)
    base, anchored = base_step_seconds(name)

    dense_kv_per_request = (context + OUTPUT_TOKENS) * per_token
    pool = GPU_MEMORY_BYTES - params * 1e9 * 2
    dense_max_batch = int(pool // dense_kv_per_request)

    kept_tokens = budget_pages * PAGE_TOKENS + RECENT_SINK_TOKENS
    selected_kv_per_request = (kept_tokens + OUTPUT_TOKENS) * per_token
    selected_pool = pool - STAGING_POOL_BYTES
    selected_max_batch = int(selected_pool // selected_kv_per_request)

    dense_attention = batch * context * per_token / HBM_BYTES_PER_SECOND
    selected_attention = (
        batch * kept_tokens * per_token / HBM_BYTES_PER_SECOND
    )
    mechanism_floor = (
        layers * MECHANISM_LAUNCHES_PER_LAYER * LAUNCH_SECONDS
    )
    dense_step = base + dense_attention
    selected_step = base + selected_attention + mechanism_floor

    return {
        "model": name,
        "anchored": anchored,
        "context": context,
        "batch": batch,
        "budget_pages": budget_pages,
        "kv_bytes_per_token": per_token,
        "dense_kv_gb": batch * dense_kv_per_request / 1e9,
        "dense_max_batch": dense_max_batch,
        "selected_max_batch": selected_max_batch,
        "admission_ratio": (
            selected_max_batch / dense_max_batch
            if dense_max_batch > 0
            else float("inf")
        ),
        "dense_feasible": batch <= dense_max_batch,
        "attention_share_pct": 100 * dense_attention / dense_step,
        "dense_step_ms": dense_step * 1e3,
        "selected_step_ms": selected_step * 1e3,
        "latency_win_pct": 100 * (1 - selected_step / dense_step),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--min-win-pct", type=float, default=30.0)
    args = parser.parse_args()

    grid = []
    for name in MODELS:
        for context in (16384, 32768, 65536, 131072):
            for batch in (1, 4, 8, 16, 32, 64):
                for budget in (64, 128, 256):
                    grid.append(evaluate_point(name, context, batch, budget))

    # Self-check against history: the operating point the project measured
    # for weeks must show near-zero available win, or the map is wrong.
    historical = evaluate_point("qwen2.5-3b", 16384, 1, 128)
    if historical["attention_share_pct"] > 6.0:
        raise AssertionError(
            "regime map contradicts the measured no-win history at the "
            f"3B/16K/batch-1 point: {historical['attention_share_pct']:.1f}%"
        )

    frontier = [
        p
        for p in grid
        if p["dense_feasible"]
        and p["latency_win_pct"] >= args.min_win_pct
    ]
    capacity_wins = [
        p
        for p in grid
        if not p["dense_feasible"] and p["batch"] <= p["selected_max_batch"]
    ]

    print(
        f"historical point (3B/16K/b1/128): attention share "
        f"{historical['attention_share_pct']:.1f}%, available win "
        f"{historical['latency_win_pct']:.1f}%  <- weeks were spent here"
    )
    print(f"\nlatency frontier (win >= {args.min_win_pct:.0f}%):")
    for p in sorted(
        frontier, key=lambda p: -p["latency_win_pct"]
    )[:12]:
        print(
            f"  {p['model']:14s} ctx={p['context']:6d} bs={p['batch']:3d} "
            f"budget={p['budget_pages']:3d}  share={p['attention_share_pct']:5.1f}%  "
            f"win={p['latency_win_pct']:5.1f}%  dense_step={p['dense_step_ms']:6.1f}ms"
            f"{'' if p['anchored'] else '  [T_base estimated]'}"
        )
    print("\ncapacity-only wins (dense cannot even run the point):")
    for p in sorted(
        capacity_wins, key=lambda p: (p["model"], p["context"], p["batch"])
    )[:8]:
        print(
            f"  {p['model']:14s} ctx={p['context']:6d} bs={p['batch']:3d} "
            f"dense_max={p['dense_max_batch']:3d} selected_max={p['selected_max_batch']:4d} "
            f"admission x{p['admission_ratio']:.1f}"
        )

    report = {
        "classification": "selected-serving-regime-map",
        "schema": 1,
        "constants": {
            "hbm_bytes_per_second": HBM_BYTES_PER_SECOND,
            "gpu_pool_bytes": GPU_MEMORY_BYTES,
            "anchors": ANCHOR_TPOT_SECONDS,
            "mechanism_floor_launches": MECHANISM_LAUNCHES_PER_LAYER,
        },
        "historical_point": historical,
        "grid": grid,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
