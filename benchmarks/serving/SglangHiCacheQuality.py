#!/usr/bin/env python3
"""Score external-prefix retrieval tasks through one SGLang backend."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Any

from SglangHiCache import (
    configure_environment,
    device_cached_tokens,
    generated_text,
    git_value,
    host_cached_tokens,
    make_prompt,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QualityTask:
    name: str
    prefix: str
    question: str
    answer: str
    needle_token_offset: int
    kind: str = "needle"

    @property
    def prompt(self) -> str:
        return self.prefix + self.question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        required=True,
    )
    parser.add_argument("--external-tokens", type=int, default=16384)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument(
        "--task-kinds",
        default="needle",
        help=(
            "comma list from {needle, multikey, count}; needle is single-"
            "position retrieval, multikey requires six dispersed parts in "
            "order, count requires tallying eight dispersed markers — the "
            "aggregation kinds exist because low budgets can pass retrieval "
            "while failing global-attention tasks"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--resident-tokens", type=int, default=2048)
    parser.add_argument("--resident-output-tokens", type=int, default=64)
    parser.add_argument("--churn-tokens", type=int, default=17000)
    parser.add_argument("--max-total-tokens", type=int, default=19000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--flashinfer-workspace-base", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if min(
        args.external_tokens,
        args.task_count,
        args.max_new_tokens,
        args.resident_tokens,
        args.resident_output_tokens,
        args.churn_tokens,
        args.max_total_tokens,
        args.context_length,
    ) <= 0:
        parser.error("token counts must be positive")
    if args.external_tokens + args.churn_tokens <= args.max_total_tokens:
        parser.error("external prefix and churn must exceed the device token pool")
    args.task_kinds = tuple(
        kind.strip() for kind in args.task_kinds.split(",") if kind.strip()
    )
    for kind in args.task_kinds:
        if kind not in ("needle", "multikey", "count"):
            parser.error(f"unknown task kind {kind!r}")
    if not args.task_kinds:
        parser.error("at least one task kind is required")
    return args


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def repeated_ids(tokenizer: Any, label: str, minimum: int) -> list[int]:
    seed = (
        f"{label}: request scoped GPU data dependencies need bounded external "
        "cache acquisition and compiler checked consumers. "
    )
    ids: list[int] = []
    seed_ids = token_ids(tokenizer, seed)
    while len(ids) < minimum:
        ids.extend(seed_ids)
    return ids[:minimum]


def _assemble_prefix(
    tokenizer: Any,
    args: argparse.Namespace,
    label: str,
    segments: list[tuple[float, str]],
    question: str,
) -> tuple[str, int]:
    """Build an exact-length prefix with texts at fractional offsets.

    Returns the decoded prefix and the first segment's token offset.
    """
    question_ids = token_ids(tokenizer, question)
    pieces = sorted(segments)
    ids: list[int] = []
    first_offset = -1
    for position, (fraction, text) in enumerate(pieces):
        target = int(args.external_tokens * fraction)
        if target > len(ids):
            ids += repeated_ids(
                tokenizer, f"{label}-fill-{position}", target - len(ids)
            )
        segment_ids = token_ids(tokenizer, text)
        if first_offset < 0:
            first_offset = len(ids)
        ids += segment_ids
    tail = args.external_tokens - len(ids) - len(question_ids)
    if tail < 64:
        raise RuntimeError("quality prefix has no room for its question")
    ids += repeated_ids(tokenizer, f"{label}-fill-tail", tail)
    ids += question_ids
    if len(ids) != args.external_tokens:
        raise RuntimeError("quality prefix token construction drifted")
    return tokenizer.decode(ids, skip_special_tokens=True), first_offset


def build_tasks(tokenizer: Any, args: argparse.Namespace) -> list[QualityTask]:
    rng = random.Random(args.seed)
    # Keep planted content away from the sink and recent pages the
    # selector always retains, or the task measures retention, not
    # selection.
    fractions = [0.18, 0.43, 0.68, 0.82, 0.31, 0.57]
    tasks: list[QualityTask] = []
    for kind in args.task_kinds:
        for index in range(args.task_count):
            label = f"{kind}-{index}"
            if kind == "needle":
                answer = f"NTAKEY{index:02d}{rng.randrange(1000, 9999)}"
                needle = (
                    f"\nThe exact retrieval key for quality task {index} is "
                    f"{answer}. When asked about quality task {index}, answer "
                    f"only {answer}.\n"
                )
                question = (
                    f"\nQuestion: What is the exact retrieval key for quality "
                    f"task {index}? Answer with only the key.\nAnswer:"
                )
                prefix, offset = _assemble_prefix(
                    tokenizer, args, label,
                    [(fractions[index % len(fractions)], needle)], question,
                )
            elif kind == "multikey":
                parts = [
                    f"P{part}X{rng.randrange(100, 999)}" for part in range(6)
                ]
                segments = [
                    (
                        [0.15, 0.28, 0.41, 0.54, 0.67, 0.80][part],
                        f"\nPart {part + 1} of the master passcode for vault "
                        f"{index} is {parts[part]}. Remember part {part + 1}: "
                        f"{parts[part]}.\n",
                    )
                    for part in range(6)
                ]
                question = (
                    f"\nQuestion: State every part of the master passcode for "
                    f"vault {index} in order, separated by spaces.\nAnswer:"
                )
                answer = " ".join(parts)
                prefix, offset = _assemble_prefix(
                    tokenizer, args, label, segments, question
                )
            elif kind == "count":
                marker = f"AUDITCODE{index:02d}"
                segments = [
                    (
                        [0.12, 0.22, 0.34, 0.46, 0.58, 0.66, 0.74, 0.84][hit],
                        f"\nThe audit marker {marker} appears at this "
                        f"checkpoint.\n",
                    )
                    for hit in range(8)
                ]
                question = (
                    f"\nQuestion: How many times does the audit marker "
                    f"{marker} appear in this document? Answer with the "
                    f"number only.\nAnswer:"
                )
                answer = "8"
                prefix, offset = _assemble_prefix(
                    tokenizer, args, label, segments, question
                )
            else:
                raise RuntimeError(f"unknown quality task kind {kind!r}")
            tasks.append(
                QualityTask(
                    name=label,
                    prefix=prefix,
                    question="",
                    answer=answer,
                    needle_token_offset=offset,
                    kind=kind,
                )
            )
    return tasks


def score_output(text: str, task: QualityTask) -> bool:
    compact = "".join(text.upper().split())
    if task.kind == "multikey":
        cursor = 0
        for part in task.answer.upper().split():
            found = compact.find(part, cursor)
            if found < 0:
                return False
            cursor = found + len(part)
        return True
    if task.kind == "count":
        head = compact.lstrip(".,:;")[:8]
        return head.startswith(task.answer) or head.startswith("EIGHT")
    return task.answer.upper() in compact


async def stream_request(
    engine: Any,
    prompt: str,
    sampling: dict[str, Any],
    *,
    rid: str,
    gate: asyncio.Event | None = None,
    first_token_event: asyncio.Event | None = None,
) -> tuple[dict[str, Any], float]:
    if gate is not None:
        await gate.wait()
    started = time.perf_counter()
    stream = await engine.async_generate(prompt, sampling, stream=True, rid=rid)
    final: dict[str, Any] | None = None
    async for result in stream:
        if final is None and first_token_event is not None:
            first_token_event.set()
        final = result
    if final is None:
        raise RuntimeError(f"SGLang returned no streamed output for {rid}")
    return final, time.perf_counter() - started


async def run_quality_pair(
    engine: Any,
    *,
    resident_prompt: str,
    resident_sampling: dict[str, Any],
    external_prompt: str,
    external_sampling: dict[str, Any],
    index: int,
) -> tuple[tuple[dict[str, Any], float], tuple[dict[str, Any], float]]:
    resident_started = asyncio.Event()
    resident = asyncio.create_task(
        stream_request(
            engine,
            resident_prompt,
            resident_sampling,
            rid=f"nta-quality-resident-{index}",
            first_token_event=resident_started,
        )
    )
    external = asyncio.create_task(
        stream_request(
            engine,
            external_prompt,
            external_sampling,
            rid=f"nta-quality-external-{index}",
            gate=resident_started,
        )
    )
    resident_result, external_result = await asyncio.gather(resident, external)
    return resident_result, external_result


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    tasks = build_tasks(tokenizer, args)
    setup_sampling = {"temperature": 0, "max_new_tokens": 1}
    task_sampling = {
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": False,
    }
    resident_sampling = {
        "temperature": 0,
        "max_new_tokens": args.resident_output_tokens,
        "ignore_eos": True,
    }
    churn_prompts = [
        make_prompt(tokenizer, f"quality-churn-{index}", args.churn_tokens)
        for index in range(args.task_count)
    ]
    shape_prompt = make_prompt(tokenizer, "quality-shape", args.external_tokens)
    resident_prompts = [
        make_prompt(tokenizer, f"quality-resident-{index}", args.resident_tokens)
        for index in range(args.task_count)
    ]
    load_started = time.perf_counter()
    records = []
    digest = hashlib.sha256()
    with sgl.Engine(
        model_path=str(args.model.resolve()),
        attention_backend=args.attention_backend,
        dtype="float16",
        mem_fraction_static=0.35,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=8,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=args.context_length,
        enable_mixed_chunk=True,
        enable_hierarchical_cache=True,
        hicache_ratio=8.0,
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
    ) as engine:
        load_seconds = time.perf_counter() - load_started
        generated_text(engine.generate(shape_prompt, setup_sampling))
        generated_text(engine.generate(shape_prompt, setup_sampling))
        for task, churn in zip(tasks, churn_prompts, strict=True):
            generated_text(engine.generate(task.prefix, setup_sampling))
            generated_text(engine.generate(churn, setup_sampling))
            resident_prompt = resident_prompts[len(records)]
            generated_text(engine.generate(resident_prompt, setup_sampling))
            resident_probe = engine.generate(resident_prompt, setup_sampling)
            if device_cached_tokens(resident_probe) <= 0:
                raise RuntimeError("quality resident request did not stay on device")
            (resident_result, _), (result, elapsed) = engine.loop.run_until_complete(
                run_quality_pair(
                    engine,
                    resident_prompt=resident_prompt,
                    resident_sampling=resident_sampling,
                    external_prompt=task.prompt,
                    external_sampling=task_sampling,
                    index=len(records),
                )
            )
            text = generated_text(result)
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
            host_tokens = host_cached_tokens(result)
            device_tokens = device_cached_tokens(result)
            passed = score_output(text, task)
            records.append(
                {
                    "name": task.name,
                    "kind": task.kind,
                    "answer": task.answer,
                    "needle_token_offset": task.needle_token_offset,
                    "host_cached_tokens": host_tokens,
                    "device_cached_tokens": device_tokens,
                    "served_from_host": host_tokens > 0,
                    "resident_device_cached_tokens": device_cached_tokens(
                        resident_result
                    ),
                    "resident_host_cached_tokens": host_cached_tokens(
                        resident_result
                    ),
                    "passed": passed,
                    "elapsed_seconds": elapsed,
                    "generated_text": text,
                }
            )

    stats = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(workspace.glob("nta-engine.*.json"))
    ]
    passed = sum(int(record["passed"]) for record in records)
    report = {
        "schema": 1,
        "classification": "sglang-hicache-quality",
        "revision": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "engine": "sglang",
        "engine_version": importlib.metadata.version("sglang"),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "attention_backend": args.attention_backend,
        "model": str(args.model.resolve()),
        "external_tokens": args.external_tokens,
        "task_count": len(records),
        "passed": passed,
        "pass_rate": passed / max(1, len(records)),
        "pass_rate_by_kind": {
            kind: (
                sum(r["passed"] for r in records if r["kind"] == kind)
                / max(1, sum(1 for r in records if r["kind"] == kind))
            )
            for kind in sorted({r["kind"] for r in records})
        },
        "all_tasks_host_served": all(record["served_from_host"] for record in records),
        "load_seconds": load_seconds,
        "generated_text_sha256": digest.hexdigest(),
        "records": records,
        "engine_stats": stats,
    }
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
