#!/usr/bin/env python3
"""Score real LongBench tasks through one SGLang backend.

Same serving architecture as SglangHiCacheQuality (exact-prefix host-cache
externals with an async resident peer, churn-forced eviction, fail-closed
host-cache attestation), but the cached prefixes are real LongBench
documents and the score is the benchmark's own metric (qa-F1 or Rouge-L)
instead of a synthetic pass/fail. One invocation serves one backend;
parity is judged across two invocations by the caller, per arm.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import pathlib
import re
import time
import zipfile
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
from SglangHiCacheQuality import run_quality_pair, stream_request  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_ROOT = pathlib.Path(os.environ.get("NTA_RESULTS_DIR", "/tmp/nta-results"))

# English tasks whose answers score with token F1; summarization tasks
# score with Rouge-L. Prompts follow LongBench's own templates.
TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, followed by a question. Answer the question "
        "as concisely as you can, based on the story.\n\nStory: {context}\n\n"
        "Question: {input}\n\nAnswer:",
        "f1",
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the "
        "question as concisely as you can. If the question cannot be "
        "answered, reply 'unanswerable'.\n\nArticle: {context}\n\n"
        "Question: {input}\n\nAnswer:",
        "f1",
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Question: {input}\nAnswer:",
        "f1",
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Answer briefly.\n\n"
        "{context}\n\nQuestion: {input}\nAnswer:",
        "f1",
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Answer briefly.\n\n"
        "{context}\n\nQuestion: {input}\nAnswer:",
        "f1",
    ),
    "gov_report": (
        "Write a one-page summary of the following government report.\n\n"
        "{context}\n\nSummary:",
        "rouge",
    ),
}


@dataclass(frozen=True)
class LongBenchTask:
    name: str
    dataset: str
    prefix: str
    question_suffix: str
    answers: tuple[str, ...]
    metric: str
    context_tokens: int

    @property
    def prompt(self) -> str:
        return self.prefix + self.question_suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "nta_flashinfer"),
        required=True,
    )
    parser.add_argument(
        "--tasks",
        default="narrativeqa,qasper,multifieldqa_en,hotpotqa,gov_report",
    )
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--external-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--summary-new-tokens", type=int, default=384)
    parser.add_argument("--resident-tokens", type=int, default=2048)
    parser.add_argument("--resident-output-tokens", type=int, default=64)
    parser.add_argument("--churn-tokens", type=int, default=17000)
    parser.add_argument("--max-total-tokens", type=int, default=19000)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--flashinfer-workspace-base",
        type=pathlib.Path,
        default=RESULTS_ROOT / "serving" / "longbench-cache",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    args.task_names = tuple(
        name.strip() for name in args.tasks.split(",") if name.strip()
    )
    for name in args.task_names:
        if name not in TASK_PROMPTS:
            parser.error(f"unknown LongBench task {name!r}")
    if args.external_tokens + args.churn_tokens <= args.max_total_tokens:
        parser.error("external prefix and churn must exceed the device pool")
    return args


def _longbench_rows(dataset: str) -> list[dict[str, Any]]:
    pattern = str(
        pathlib.Path.home()
        / ".cache/huggingface/hub/datasets--THUDM--LongBench/snapshots/*/data.zip"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise RuntimeError(
            "LongBench data.zip not found; run: hf download THUDM/LongBench "
            "--repo-type dataset"
        )
    with zipfile.ZipFile(matches[0]) as archive:
        with archive.open(f"data/{dataset}.jsonl") as handle:
            return [json.loads(line) for line in handle if line.strip()]


_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")


def _normalized_tokens(text: str) -> list[str]:
    stop = {"a", "an", "the", "and", "or", "of", "to", "in", "is", "was"}
    return [
        token
        for token in _TOKEN_SPLIT.split(text.lower())
        if token and token not in stop
    ]


def qa_f1(prediction: str, answers: tuple[str, ...]) -> float:
    best = 0.0
    predicted = _normalized_tokens(prediction)
    if not predicted:
        return 0.0
    for answer in answers:
        gold = _normalized_tokens(answer)
        if not gold:
            continue
        common = collections.Counter(predicted) & collections.Counter(gold)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def rouge_l(prediction: str, answers: tuple[str, ...]) -> float:
    def lcs(a: list[str], b: list[str]) -> int:
        table = [0] * (len(b) + 1)
        for x in a:
            previous = 0
            for j, y in enumerate(b, 1):
                previous, table[j] = (
                    table[j],
                    (previous + 1 if x == y else max(table[j], table[j - 1])),
                )
        return table[-1]

    best = 0.0
    predicted = _normalized_tokens(prediction)
    if not predicted:
        return 0.0
    for answer in answers:
        gold = _normalized_tokens(answer)
        if not gold:
            continue
        overlap = lcs(predicted, gold)
        if overlap == 0:
            continue
        precision = overlap / len(predicted)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def build_tasks(tokenizer: Any, args: argparse.Namespace) -> list[LongBenchTask]:
    import random
    import zlib

    tasks: list[LongBenchTask] = []
    for dataset in args.task_names:
        template, metric = TASK_PROMPTS[dataset]
        head, tail = template.split("{context}", 1)
        rows = [
            row
            for row in _longbench_rows(dataset)
            if row.get("context") and row.get("answers")
        ]
        # Per-dataset RNG, independent of the task list: one shared RNG
        # left later datasets' shuffles at a task-list-dependent state,
        # so a single-task run sampled different rows than the same task
        # inside the full battery — which invalidated a whole budget
        # ladder (0/16 row overlap) before it was caught by comparing
        # stored gold answers across runs. crc32, not hash(): Python
        # salts str hashes per process.
        rng = random.Random(args.seed * 1000003 + zlib.crc32(dataset.encode()))
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            if taken >= args.samples_per_task:
                break
            suffix = tail.replace("{input}", str(row.get("input", "")))
            head_ids = tokenizer.encode(head, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            budget = args.external_tokens - len(head_ids) - len(suffix_ids)
            if budget < 2048:
                continue
            context_ids = tokenizer.encode(
                str(row["context"]), add_special_tokens=False
            )
            if len(context_ids) < 4096:
                # Short documents do not exercise external-cache execution; the
                # battery targets the serving regime, not padded fillers.
                continue
            if len(context_ids) > budget:
                # LongBench's own convention: keep head and tail halves.
                keep = budget // 2
                context_ids = context_ids[:keep] + context_ids[-keep:]
            prefix = tokenizer.decode(head_ids + context_ids, skip_special_tokens=True)
            tasks.append(
                LongBenchTask(
                    name=f"{dataset}-{taken}",
                    dataset=dataset,
                    prefix=prefix,
                    question_suffix=tokenizer.decode(
                        suffix_ids, skip_special_tokens=True
                    ),
                    answers=tuple(str(a) for a in row["answers"]),
                    metric=metric,
                    context_tokens=len(context_ids),
                )
            )
            taken += 1
        if taken < args.samples_per_task:
            raise RuntimeError(
                f"LongBench task {dataset} yielded only {taken} usable "
                f"samples of {args.samples_per_task}"
            )
    return tasks


def main() -> int:
    args = parse_args()
    workspace = configure_environment(args)
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()))
    tasks = build_tasks(tokenizer, args)
    setup_sampling = {"temperature": 0, "max_new_tokens": 1}
    resident_sampling = {
        "temperature": 0,
        "max_new_tokens": args.resident_output_tokens,
        "ignore_eos": True,
    }
    churn_prompts = [
        make_prompt(tokenizer, f"longbench-churn-{index}", args.churn_tokens)
        for index in range(len(tasks))
    ]
    resident_prompts = [
        make_prompt(tokenizer, f"longbench-resident-{index}", args.resident_tokens)
        for index in range(len(tasks))
    ]
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    load_started = time.perf_counter()
    with sgl.Engine(
        model_path=str(args.model.resolve()),
        attention_backend=args.attention_backend,
        dtype="float16",
        mem_fraction_static=0.35,
        context_length=args.context_length,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=8,
        cuda_graph_backend_decode=os.environ.get(
            "NTA_QUALITY_CUDA_GRAPH_DECODE", "disabled"
        ),
        cuda_graph_backend_prefill=os.environ.get(
            "NTA_QUALITY_CUDA_GRAPH_PREFILL", "disabled"
        ),
        chunked_prefill_size=args.context_length,
        enable_mixed_chunk=True,
        enable_hierarchical_cache=True,
        hicache_ratio=8.0,
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_mem_layout="page_first",
    ) as engine:
        load_seconds = time.perf_counter() - load_started
        for task, churn in zip(tasks, churn_prompts, strict=True):
            index = len(records)
            task_sampling = {
                "temperature": 0,
                "max_new_tokens": (
                    args.summary_new_tokens
                    if task.metric == "rouge"
                    else args.max_new_tokens
                ),
                "ignore_eos": False,
            }
            generated_text(engine.generate(task.prefix, setup_sampling))
            generated_text(engine.generate(churn, setup_sampling))
            resident_prompt = resident_prompts[index]
            generated_text(engine.generate(resident_prompt, setup_sampling))
            resident_probe = engine.generate(resident_prompt, setup_sampling)
            if device_cached_tokens(resident_probe) <= 0:
                generated_text(engine.generate(resident_prompt, setup_sampling))
                resident_probe = engine.generate(resident_prompt, setup_sampling)
            if device_cached_tokens(resident_probe) <= 0:
                raise RuntimeError(
                    f"resident request lost device residency at task {index}"
                )
            (resident_result, _), (result, elapsed) = engine.loop.run_until_complete(
                run_quality_pair(
                    engine,
                    resident_prompt=resident_prompt,
                    resident_sampling=resident_sampling,
                    external_prompt=task.prompt,
                    external_sampling=task_sampling,
                    index=index,
                )
            )
            text = generated_text(result)
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
            host_tokens = host_cached_tokens(result)
            if host_tokens <= 0:
                raise RuntimeError(
                    f"LongBench external {task.name} was not served from "
                    "host cache; the run does not exercise the mechanism"
                )
            scorer = qa_f1 if task.metric == "f1" else rouge_l
            records.append(
                {
                    "name": task.name,
                    "dataset": task.dataset,
                    "metric": task.metric,
                    "score": scorer(text, task.answers),
                    "context_tokens": task.context_tokens,
                    "host_cached_tokens": host_tokens,
                    "device_cached_tokens": device_cached_tokens(result),
                    "resident_device_cached_tokens": device_cached_tokens(
                        resident_result
                    ),
                    "elapsed_seconds": elapsed,
                    "generated_text": text,
                    "answers": list(task.answers),
                }
            )
    stats = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(workspace.glob("nta-engine.*.json"))
    ]
    by_dataset: dict[str, list[float]] = collections.defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record["score"])
    report = {
        "classification": "sglang-longbench-quality",
        "backend": args.attention_backend,
        "revision": git_value("rev-parse", "--short", "HEAD"),
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "external_tokens": args.external_tokens,
        "samples_per_task": args.samples_per_task,
        "load_seconds": load_seconds,
        "task_scores": {
            dataset: sum(scores) / len(scores)
            for dataset, scores in sorted(by_dataset.items())
        },
        "mean_score": sum(r["score"] for r in records) / len(records),
        "generated_sha256": digest.hexdigest(),
        "records": records,
        "engine_stats": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("backend", "task_scores", "mean_score")},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
