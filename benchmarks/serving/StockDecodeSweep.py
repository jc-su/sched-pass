#!/usr/bin/env python3
"""Stock decode TPOT versus (batch, context): the regime map's ground truth.

Measures pure decode step time on stock SGLang at controlled batch and
per-request context, by differencing two generations that share the same
prefill (prefill and setup cancel; the quotient is decode). Compares each
point against the analytic prediction from RegimeMap.py so the map's
central claim — attention-byte share grows to dominance with batch and
context — is validated or falsified by measurement before any operating
point derived from it is used.

Dense attention reads the same bytes regardless of token content, so
prompts are synthetic repeated tokens; this harness measures physics, not
quality.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        metavar="BATCH,CONTEXT",
        help="sweep point; repeatable",
    )
    parser.add_argument("--short-output-tokens", type=int, default=32)
    parser.add_argument("--long-output-tokens", type=int, default=160)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--cuda-host-cxx", type=pathlib.Path)
    parser.add_argument(
        "--flashinfer-workspace-base",
        type=pathlib.Path,
        default=ROOT / "results" / "analysis" / "stock-sweep-flashinfer",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    args.points = []
    for raw in args.point:
        batch, context = (int(v) for v in raw.split(","))
        if batch <= 0 or context <= 0:
            parser.error(f"invalid point {raw!r}")
        args.points.append((batch, context))
    model_config = json.loads((args.model / "config.json").read_text())
    position_limit = int(model_config.get("max_position_embeddings", 1 << 30))
    for batch, context in args.points:
        needed = context + args.long_output_tokens + 64
        if needed > position_limit:
            parser.error(
                f"point ({batch},{context}) needs {needed} positions but the "
                f"model supports {position_limit}; reduce the context"
            )
    return args


def measure_point(args: argparse.Namespace, batch: int, context: int) -> dict[str, Any]:
    """One engine per point: context length and pool sizing differ."""
    import sglang as sgl

    input_ids = [[1000] * context for _ in range(batch)]
    engine = sgl.Engine(
        model_path=str(args.model.resolve()),
        context_length=context + args.long_output_tokens + 64,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=batch,
        disable_radix_cache=True,
    )
    try:
        timings = {}
        for label, new_tokens in (
            ("warmup", args.short_output_tokens),
            ("short", args.short_output_tokens),
            ("long", args.long_output_tokens),
        ):
            sampling = {
                "temperature": 0,
                "max_new_tokens": new_tokens,
                "ignore_eos": True,
            }
            started = time.perf_counter()
            results = engine.generate(input_ids=input_ids, sampling_params=sampling)
            elapsed = time.perf_counter() - started
            produced = 0
            for result in results:
                info = result["meta_info"]
                produced += int(
                    info.get(
                        "completion_tokens",
                        info.get("completion_tokens_without_jump_forward", 0),
                    )
                )
            if produced != batch * new_tokens:
                raise RuntimeError(
                    f"{label} generation produced {produced} tokens, "
                    f"expected {batch * new_tokens}; decode differencing "
                    "would be invalid"
                )
            timings[label] = elapsed
    finally:
        engine.shutdown()

    extra_tokens = args.long_output_tokens - args.short_output_tokens
    tpot = (timings["long"] - timings["short"]) / extra_tokens
    return {
        "batch": batch,
        "context": context,
        "short_seconds": timings["short"],
        "long_seconds": timings["long"],
        "decode_tpot_seconds": tpot,
    }


def regime_model_for(model_dir: pathlib.Path) -> str:
    """Resolve the regime-map entry by measured geometry, never by name.

    A silent wrong-model comparison would invalidate every prediction
    column, so an unmapped geometry is an error, not a fallback.
    """
    from RegimeMap import MODELS

    config = json.loads((model_dir / "config.json").read_text())
    geometry = (
        int(config["num_hidden_layers"]),
        int(config["num_key_value_heads"]),
        int(
            config.get(
                "head_dim",
                config["hidden_size"] // config["num_attention_heads"],
            )
        ),
    )
    for name, (layers, kv_heads, head_dim, *_) in MODELS.items():
        if geometry == (layers, kv_heads, head_dim):
            return name
    raise RuntimeError(
        f"model geometry {geometry} has no regime-map entry; add it to "
        "RegimeMap.MODELS before sweeping"
    )


def predicted(model_name: str, batch: int, context: int) -> dict[str, float]:
    from RegimeMap import evaluate_point

    point = evaluate_point(model_name, context, batch, 128)
    return {
        "dense_step_ms": point["dense_step_ms"],
        "attention_share_pct": point["attention_share_pct"],
    }


def main() -> int:
    args = parse_args()
    # FlashInfer JIT requires the CUDA-matched host toolchain; reuse the
    # smoke harness's dual-consumer configuration (CC drives both Triton's
    # C launchers and nvcc's -ccbin).
    from SglangSmoke import configure_jit_environment

    configure_jit_environment(args)
    regime_model = regime_model_for(args.model)
    measured = []
    for batch, context in args.points:
        result = measure_point(args, batch, context)
        model = predicted(regime_model, batch, context)
        result["predicted_dense_step_ms"] = model["dense_step_ms"]
        result["predicted_attention_share_pct"] = model["attention_share_pct"]
        result["measured_over_predicted"] = (
            result["decode_tpot_seconds"] * 1e3 / model["dense_step_ms"]
        )
        measured.append(result)
        if args.output:
            # Persist after every point: a later point's failure must not
            # discard completed measurements.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"partial": True, "points": measured}) + "\n"
            )
        print(
            f"bs={batch} ctx={context}: measured TPOT "
            f"{result['decode_tpot_seconds'] * 1e3:.2f}ms, predicted "
            f"{model['dense_step_ms']:.2f}ms "
            f"(x{result['measured_over_predicted']:.2f}), predicted "
            f"attention share {model['attention_share_pct']:.1f}%",
            flush=True,
        )

    report = {
        "classification": "stock-decode-sweep",
        "schema": 1,
        "model": str(args.model),
        "revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip(),
        "points": measured,
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
