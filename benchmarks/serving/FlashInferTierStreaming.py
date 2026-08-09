#!/usr/bin/env python3
"""Measure exact request-aware KV streaming with canonical FlashInfer math."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
import statistics
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

# CUDA 13 rejects this host's default GCC 15. Select a supported compiler for
# stock FlashInfer JITs without weakening nvcc's compatibility checks.
if "CC" not in os.environ and pathlib.Path("/usr/bin/gcc-14").is_file():
    os.environ["CC"] = "/usr/bin/gcc-14"
if "CXX" not in os.environ and pathlib.Path("/usr/bin/g++-14").is_file():
    os.environ["CXX"] = "/usr/bin/g++-14"

import flashinfer
import torch

from nta_runtime import (
    FlashInferTierStreamingOperator,
    TierStreamingRequest,
    build_tier_streaming_schedule,
)


ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class SampleSet:
    values_us: list[float]

    def report(self) -> dict[str, Any]:
        return {
            "median": statistics.median(self.values_us),
            "p95": percentile(self.values_us, 0.95),
            "minimum": min(self.values_us),
            "maximum": max(self.values_us),
            "samples": self.values_us,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-tokens", default="256")
    parser.add_argument("--context-tokens", default="16384")
    parser.add_argument("--resident-fractions", default="0,0.25,0.5,1")
    parser.add_argument("--group-tokens", type=int, default=2048)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--qo-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--backend", choices=("auto", "fa2", "fa3"), default="fa2")
    parser.add_argument("--workspace-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--verify-graph", action="store_true")
    parser.add_argument("--verify-lifecycle", action="store_true")
    parser.add_argument(
        "--compiler-transform",
        action="store_true",
        help="run streaming partials through a paired NTA FlashInfer JIT plan",
    )
    parser.add_argument(
        "--gpu-initiated-host",
        action="store_true",
        help="move bounded host waves with NTA's finite GPU progress kernels",
    )
    parser.add_argument(
        "--primary-arm",
        choices=("direct", "atomic", "streaming"),
        default="streaming",
        help="publish this arm as the top-level metric for qualified trial runners",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    if (
        min(
            arguments.batch_size,
            arguments.group_tokens,
            arguments.slots,
            arguments.qo_heads,
            arguments.kv_heads,
            arguments.head_dim,
            arguments.workspace_mib,
            arguments.warmup,
            arguments.iterations,
            arguments.trials,
        )
        <= 0
    ):
        parser.error("all sizes, warmups, iterations, and trials must be positive")
    if arguments.slots < 2:
        parser.error("tier streaming requires at least two staging slots")
    if arguments.qo_heads % arguments.kv_heads != 0:
        parser.error("query heads must be divisible by KV heads")
    if arguments.gpu_initiated_host and not arguments.compiler_transform:
        parser.error("--gpu-initiated-host requires --compiler-transform")
    return arguments


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def expand_int_vector(text: str, count: int, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if len(values) == 1:
        values *= count
    if len(values) != count or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain one or {count} positive integers")
    return values


def expand_fraction_vector(text: str, count: int) -> tuple[float, ...]:
    try:
        values = tuple(
            float(value.strip()) for value in text.split(",") if value.strip()
        )
    except ValueError as error:
        raise ValueError("resident fractions must be numeric") from error
    if len(values) == 1:
        values *= count
    if len(values) != count or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            f"resident fractions must contain one or {count} values in [0, 1]"
        )
    return values


def cumulative(values: Sequence[int]) -> tuple[int, ...]:
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return tuple(result)


def device_indptr(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(cumulative(values), dtype=torch.int32, device="cuda")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    """Bootstrap a median ratio while preserving each trial's arm pairing."""

    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap requires matched non-empty samples")
    generator = random.Random(seed)
    count = len(numerator)
    ratios: list[float] = []
    for _ in range(resamples):
        indices = [generator.randrange(count) for _ in range(count)]
        numerator_median = statistics.median(numerator[index] for index in indices)
        denominator_median = statistics.median(denominator[index] for index in indices)
        ratios.append(numerator_median / denominator_median)
    return {
        "confidence": 0.95,
        "lower": percentile(ratios, 0.025),
        "upper": percentile(ratios, 0.975),
        "resamples": resamples,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def event_sample(
    call: Callable[[], None], iterations: int, stream: torch.cuda.Stream
) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record(stream)
    for _ in range(iterations):
        call()
    end.record(stream)
    end.synchronize()
    return begin.elapsed_time(end) * 1_000.0 / iterations


class TierStreamingFixture:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.compute_stream = torch.cuda.current_stream()
        self.copy_stream = torch.cuda.Stream(priority=0)
        self.element_bytes = torch.empty((), dtype=torch.float16).element_size()

        query_tokens = expand_int_vector(
            arguments.query_tokens, arguments.batch_size, "query tokens"
        )
        context_tokens = expand_int_vector(
            arguments.context_tokens, arguments.batch_size, "context tokens"
        )
        resident_fractions = expand_fraction_vector(
            arguments.resident_fractions, arguments.batch_size
        )
        requests = [
            TierStreamingRequest(
                request_id=index,
                query_tokens=query_tokens[index],
                context_tokens=context_tokens[index],
                resident_tokens=round(
                    context_tokens[index] * resident_fractions[index]
                ),
                priority=index % 8,
                deadline_ns=(index + 1) * 1_000_000,
            )
            for index in range(arguments.batch_size)
        ]
        self.schedule = build_tier_streaming_schedule(requests, arguments.group_tokens)
        self.requests = self.schedule.requests
        self.query_lengths = tuple(request.query_tokens for request in self.requests)
        self.context_lengths = tuple(
            request.context_tokens for request in self.requests
        )
        self.external_lengths = tuple(
            request.external_tokens for request in self.requests
        )
        self.resident_lengths = tuple(
            request.resident_tokens for request in self.requests
        )
        self.query_offsets = cumulative(self.query_lengths)
        self.external_offsets = cumulative(self.external_lengths)
        self.total_query_tokens = sum(self.query_lengths)
        self.total_external_tokens = sum(self.external_lengths)
        self.total_resident_tokens = sum(self.resident_lengths)
        self.external_request_count = next(
            (
                index
                for index, length in enumerate(self.external_lengths)
                if length == 0
            ),
            len(self.requests),
        )
        if any(
            length != 0
            for length in self.external_lengths[self.external_request_count :]
        ):
            raise RuntimeError("external requests are not a compact prefix")

        torch.manual_seed(arguments.seed)
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(arguments.seed)
        kv_shape = (arguments.kv_heads, arguments.head_dim)
        self.query = torch.randn(
            (self.total_query_tokens, arguments.qo_heads, arguments.head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        self.external_key = torch.randn(
            (self.total_external_tokens, *kv_shape),
            dtype=torch.float16,
            pin_memory=True,
            generator=cpu_generator,
        )
        self.external_value = torch.randn(
            (self.total_external_tokens, *kv_shape),
            dtype=torch.float16,
            pin_memory=True,
            generator=cpu_generator,
        )
        self.resident_key = torch.randn(
            (self.total_resident_tokens, *kv_shape),
            dtype=torch.float16,
            device="cuda",
        )
        self.resident_value = torch.randn_like(self.resident_key)
        self.local_key = torch.randn(
            (self.total_query_tokens, *kv_shape),
            dtype=torch.float16,
            device="cuda",
        )
        self.local_value = torch.randn_like(self.local_key)

        self.workspace = torch.empty(
            arguments.workspace_mib * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda",
        )
        self._metadata: list[torch.Tensor] = []
        self.direct_wrapper = self._wrapper(
            self.query_lengths,
            tuple(
                context + query
                for context, query in zip(self.context_lengths, self.query_lengths)
            ),
            causal=True,
        )
        self.atomic_external_wrapper = (
            self._wrapper(
                self.query_lengths[: self.external_request_count],
                self.external_lengths[: self.external_request_count],
                causal=False,
            )
            if self.external_request_count != 0
            else None
        )

        self.atomic_key = torch.empty(
            (self.total_external_tokens, *kv_shape),
            dtype=torch.float16,
            device="cuda",
        )
        self.atomic_value = torch.empty_like(self.atomic_key)

        self.direct_key, self.direct_value = self._build_direct_kv()
        output_shape = (
            self.total_query_tokens,
            arguments.qo_heads,
            arguments.head_dim,
        )
        lse_shape = (self.total_query_tokens, arguments.qo_heads)
        self.direct_output = torch.empty(
            output_shape, dtype=torch.float16, device="cuda"
        )
        self.direct_lse = torch.empty(lse_shape, dtype=torch.float32, device="cuda")
        self.output = torch.empty_like(self.direct_output)
        self.lse = torch.empty_like(self.direct_lse)
        self.partial_output = torch.empty_like(self.direct_output)
        self.partial_lse = torch.empty_like(self.direct_lse)
        self.streaming = FlashInferTierStreamingOperator(
            self.schedule,
            self.external_key,
            self.external_value,
            self.workspace,
            qo_heads=arguments.qo_heads,
            backend=arguments.backend,
            slot_count=arguments.slots,
            device=self.query.device,
            copy_stream=self.copy_stream,
            compute_stream=self.compute_stream,
            compiler_module_tag=(
                f"tier_b{arguments.batch_size}_g{arguments.group_tokens}"
                if arguments.compiler_transform
                else None
            ),
            gpu_initiated_host=arguments.gpu_initiated_host,
        )
        self.call_start = torch.cuda.Event()
        torch.cuda.synchronize()

    def _wrapper(
        self,
        query_lengths: Sequence[int],
        kv_lengths: Sequence[int],
        *,
        causal: bool,
    ) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        if not query_lengths or len(query_lengths) != len(kv_lengths):
            raise ValueError("FlashInfer wrapper requires matched non-empty lengths")
        if any(length <= 0 for length in kv_lengths):
            # Zero-resident requests still get their local causal partial. A
            # synthetic resident token would change the exact output.
            positive_start = next(
                (index for index, length in enumerate(kv_lengths) if length > 0),
                len(kv_lengths),
            )
            if any(length <= 0 for length in kv_lengths[positive_start:]):
                raise ValueError("positive KV lengths must be a compact suffix")
            query_lengths = query_lengths[positive_start:]
            kv_lengths = kv_lengths[positive_start:]
            if not kv_lengths:
                return None  # type: ignore[return-value]
        query_indptr = device_indptr(query_lengths)
        kv_indptr = device_indptr(kv_lengths)
        self._metadata.extend((query_indptr, kv_indptr))
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, "NHD", backend=self.arguments.backend
        )
        wrapper.plan(
            query_indptr,
            kv_indptr,
            self.arguments.qo_heads,
            self.arguments.kv_heads,
            self.arguments.head_dim,
            causal=causal,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            o_data_type=torch.float16,
            non_blocking=False,
            disable_split_kv=False,
        )
        return wrapper

    def _build_direct_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        kv_shape = (self.arguments.kv_heads, self.arguments.head_dim)
        total_tokens = sum(self.context_lengths) + self.total_query_tokens
        key = torch.empty((total_tokens, *kv_shape), dtype=torch.float16, device="cuda")
        value = torch.empty_like(key)
        destination = 0
        external = 0
        resident = 0
        local = 0
        for external_count, resident_count, query_count in zip(
            self.external_lengths, self.resident_lengths, self.query_lengths
        ):
            key[destination : destination + external_count].copy_(
                self.external_key[external : external + external_count],
                non_blocking=True,
            )
            value[destination : destination + external_count].copy_(
                self.external_value[external : external + external_count],
                non_blocking=True,
            )
            destination += external_count
            external += external_count
            key[destination : destination + resident_count].copy_(
                self.resident_key[resident : resident + resident_count]
            )
            value[destination : destination + resident_count].copy_(
                self.resident_value[resident : resident + resident_count]
            )
            destination += resident_count
            resident += resident_count
            key[destination : destination + query_count].copy_(
                self.local_key[local : local + query_count]
            )
            value[destination : destination + query_count].copy_(
                self.local_value[local : local + query_count]
            )
            destination += query_count
            local += query_count
        torch.cuda.synchronize()
        return key, value

    def _run_base_partials(self) -> None:
        self.streaming.enqueue_base(
            self.query,
            self.resident_key,
            self.resident_value,
            self.local_key,
            self.local_value,
            self.output,
            self.lse,
            self.partial_output,
            self.partial_lse,
        )

    def direct_call(self) -> None:
        self.direct_wrapper.run(
            self.query,
            self.direct_key,
            self.direct_value,
            out=self.direct_output,
            lse=self.direct_lse,
            return_lse=True,
        )

    def atomic_call(self) -> None:
        self.call_start.record(self.compute_stream)
        self.copy_stream.wait_event(self.call_start)
        with torch.cuda.stream(self.copy_stream):
            self.atomic_key.copy_(self.external_key, non_blocking=True)
            self.atomic_value.copy_(self.external_value, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        self._run_base_partials()
        self.compute_stream.wait_event(ready)
        if self.atomic_external_wrapper is not None:
            rows = self.query_offsets[self.external_request_count]
            self.atomic_external_wrapper.run(
                self.query[:rows],
                self.atomic_key,
                self.atomic_value,
                out=self.partial_output[:rows],
                lse=self.partial_lse[:rows],
                return_lse=True,
            )
            flashinfer.merge_state_in_place(
                self.output[:rows],
                self.lse[:rows],
                self.partial_output[:rows],
                self.partial_lse[:rows],
            )

    def streaming_call(
        self,
        completion_events: dict[tuple[int, int], torch.cuda.Event] | None = None,
    ) -> None:
        self.streaming.run(
            self.query,
            self.resident_key,
            self.resident_value,
            self.local_key,
            self.local_value,
            self.output,
            self.lse,
            self.partial_output,
            self.partial_lse,
            completion_events=completion_events,
        )

    def capture_streaming(self):
        return self.streaming.capture(
            self.query,
            self.resident_key,
            self.resident_value,
            self.local_key,
            self.local_value,
            self.output,
            self.lse,
            self.partial_output,
            self.partial_lse,
        )

    def bulk_copy_call(self) -> None:
        self.atomic_key.copy_(self.external_key, non_blocking=True)
        self.atomic_value.copy_(self.external_value, non_blocking=True)

    def completion_times_us(self) -> dict[int, float]:
        events = {
            request.key: torch.cuda.Event(enable_timing=True)
            for request in self.requests
        }
        begin = torch.cuda.Event(enable_timing=True)
        begin.record(self.compute_stream)
        self.streaming_call(events)
        self.compute_stream.synchronize()
        return {
            request_id: begin.elapsed_time(events[request.key]) * 1_000.0
            for request in self.requests
            for request_id in (request.request_id,)
        }

    def kv_bytes(self, tokens: int) -> int:
        return (
            2
            * tokens
            * self.arguments.kv_heads
            * self.arguments.head_dim
            * self.element_bytes
        )


def main() -> int:
    arguments = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    fixture = TierStreamingFixture(arguments)

    for _ in range(arguments.warmup):
        fixture.direct_call()
        fixture.atomic_call()
        fixture.streaming_call()
    torch.cuda.synchronize()

    samples = {
        "direct": SampleSet([]),
        "atomic": SampleSet([]),
        "streaming": SampleSet([]),
        "bulk_copy": SampleSet([]),
    }
    arms = (
        ("direct", fixture.direct_call),
        ("atomic", fixture.atomic_call),
        ("streaming", fixture.streaming_call),
    )
    arm_order: list[list[str]] = []
    for trial in range(arguments.trials):
        order = []
        for offset in range(len(arms)):
            name, call = arms[(trial + offset) % len(arms)]
            order.append(name)
            samples[name].values_us.append(
                event_sample(call, arguments.iterations, fixture.compute_stream)
            )
        arm_order.append(order)
        samples["bulk_copy"].values_us.append(
            event_sample(
                fixture.bulk_copy_call, arguments.iterations, fixture.compute_stream
            )
        )

    fixture.direct_call()
    torch.cuda.synchronize()
    direct_output = fixture.direct_output.clone()
    fixture.atomic_call()
    torch.cuda.synchronize()
    atomic_output = fixture.output.clone()
    fixture.streaming_call()
    torch.cuda.synchronize()
    streaming_output = fixture.output.clone()
    compiler_runtime_protocol_active = False
    if fixture.streaming.compiler_transformed:
        fixture.streaming.verify_compiler_epoch()
        compiler_runtime_protocol_active = (
            fixture.streaming.compiler_runtime_protocol_active
        )
    torch.testing.assert_close(atomic_output, direct_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(streaming_output, direct_output, rtol=2e-3, atol=2e-3)
    graph_replay_verified = False
    graph_dynamic_source_verified = False
    graph_dynamic_max_abs_error = None
    captured = None
    if arguments.verify_graph:
        captured = fixture.capture_streaming()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(arguments.seed + 1)
        replacement_key = torch.randn(
            fixture.external_key.shape,
            dtype=fixture.external_key.dtype,
            pin_memory=True,
            generator=generator,
        )
        replacement_value = torch.randn(
            fixture.external_value.shape,
            dtype=fixture.external_value.dtype,
            pin_memory=True,
            generator=generator,
        )
        captured.replay(replacement_key, replacement_value)
        torch.cuda.synchronize()
        graph_output = fixture.output.clone()
        fixture.streaming.bind_external(replacement_key, replacement_value)
        fixture.streaming_call()
        torch.cuda.synchronize()
        graph_dynamic_max_abs_error = float(
            (graph_output - fixture.output).abs().max()
        )
        torch.testing.assert_close(
            graph_output, fixture.output, rtol=2e-3, atol=2e-3
        )
        fixture.streaming.bind_external(fixture.external_key, fixture.external_value)
        graph_replay_verified = True
        graph_dynamic_source_verified = True

    generation_reuse_verified = False
    cancellation_isolation_verified = False
    if arguments.verify_lifecycle:
        if not fixture.streaming.compiler_transformed:
            raise RuntimeError("lifecycle verification requires compiler transformation")
        request = fixture.requests[0]
        generation = request.generation + 1
        fixture.streaming.rebind_request(0, request.request_id + 10_000, generation)
        if captured is None:
            fixture.streaming_call()
        else:
            captured.replay()
        torch.cuda.synchronize()
        fixture.streaming.verify_compiler_epoch()
        torch.testing.assert_close(
            fixture.output, direct_output, rtol=2e-3, atol=2e-3
        )
        generation_reuse_verified = True

        fixture.streaming.cancel_request(0, generation)
        if captured is None:
            fixture.streaming_call()
        else:
            captured.replay()
        torch.cuda.synchronize()
        lifecycle_status = fixture.streaming.compiler_epoch_status()
        if lifecycle_status.cancelled == 0 or lifecycle_status.failed != 0:
            raise RuntimeError("cancelled request did not retire as Cancelled")
        unaffected_begin = fixture.query_offsets[1]
        torch.testing.assert_close(
            fixture.output[unaffected_begin:],
            direct_output[unaffected_begin:],
            rtol=2e-3,
            atol=2e-3,
        )
        cancellation_isolation_verified = True

        fixture.streaming.rebind_request(0, request.request_id + 20_000, generation + 1)
        if captured is None:
            fixture.streaming_call()
        else:
            captured.replay()
        torch.cuda.synchronize()
        fixture.streaming.verify_compiler_epoch()
        torch.testing.assert_close(
            fixture.output, direct_output, rtol=2e-3, atol=2e-3
        )

    completion_times = fixture.completion_times_us()
    direct_median = statistics.median(samples["direct"].values_us)
    atomic_median = statistics.median(samples["atomic"].values_us)
    streaming_median = statistics.median(samples["streaming"].values_us)
    bulk_copy_median = statistics.median(samples["bulk_copy"].values_us)
    speedup_ci = paired_bootstrap_ratio(
        samples["atomic"].values_us,
        samples["streaming"].values_us,
        seed=arguments.seed,
    )
    external_bytes = fixture.kv_bytes(fixture.total_external_tokens)
    bulk_staging_bytes = external_bytes
    streaming_staging_bytes = fixture.kv_bytes(
        fixture.streaming.staging_tokens
    )
    primary_median = {
        "direct": direct_median,
        "atomic": atomic_median,
        "streaming": streaming_median,
    }[arguments.primary_arm]
    primary_staging_bytes = {
        "direct": 0,
        "atomic": bulk_staging_bytes,
        "streaming": streaming_staging_bytes,
    }[arguments.primary_arm]
    result: dict[str, Any] = {
        "schema": 1,
        "classification": "flashinfer-request-aware-tier-streaming",
        "revision": git_value("rev-parse", "HEAD"),
        "dirty": bool(git_value("status", "--porcelain")),
        "flashinfer_version": flashinfer.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compiler": {
            "cc": os.environ.get("CC"),
            "cxx": os.environ.get("CXX"),
            "nvcc": shutil.which("nvcc"),
        },
        "real_flashinfer_attention": True,
        "real_flashinfer_online_softmax_merge": True,
        "compiler_transformed_attention": fixture.streaming.compiler_transformed,
        "compiler_runtime_protocol_active": compiler_runtime_protocol_active,
        "host_transfer_initiator": (
            "gpu" if fixture.streaming.gpu_initiated_host else "cpu_copy_engine"
        ),
        "compiler_operator_plan": (
            None
            if fixture.streaming.operator_plan is None
            else {
                "schema_version": fixture.streaming.operator_plan.schema_version,
                "runtime_abi_version": (
                    fixture.streaming.operator_plan.runtime_abi_version
                ),
                "family": fixture.streaming.operator_plan.family.name.lower(),
                "supported_forms": fixture.streaming.operator_plan.supported_forms,
                "coordinate_map": (
                    fixture.streaming.operator_plan.coordinate_map.name.lower()
                ),
                "partial_state": (
                    fixture.streaming.operator_plan.partial_state.name.lower()
                ),
                "reduction": fixture.streaming.operator_plan.reduction.name.lower(),
                "flags": int(fixture.streaming.operator_plan.flags),
                "source_fingerprint": (
                    fixture.streaming.operator_plan.source_fingerprint
                ),
                "plan_fingerprint": (
                    fixture.streaming.operator_plan.plan_fingerprint
                ),
            }
        ),
        "graph_replay_verified": graph_replay_verified,
        "graph_dynamic_source_verified": graph_dynamic_source_verified,
        "graph_dynamic_max_abs_error": graph_dynamic_max_abs_error,
        "generation_reuse_verified": generation_reuse_verified,
        "cancellation_isolation_verified": cancellation_isolation_verified,
        "custom_attention_kernel": False,
        "output_parity": True,
        "output_sha256": tensor_sha256(direct_output),
        "atomic_max_abs_error": float((atomic_output - direct_output).abs().max()),
        "streaming_max_abs_error": float(
            (streaming_output - direct_output).abs().max()
        ),
        "request_semantics_retained": True,
        "requests": [
            asdict(request) | {"external_tokens": request.external_tokens}
            for request in fixture.requests
        ],
        "waves": [
            {
                "index": wave.index,
                "active_request_ids": [segment.request_id for segment in wave.segments],
                "active_request_keys": [
                    [segment.request_id, segment.request_generation]
                    for segment in wave.segments
                ],
                "request_token_counts": [
                    segment.token_count for segment in wave.segments
                ],
                "token_count": wave.token_count,
                "completed_request_ids": list(wave.completed_request_ids),
                "completed_request_keys": [
                    list(request_key) for request_key in wave.completed_request_keys
                ],
            }
            for wave in fixture.schedule.waves
        ],
        "request_completion_us": {
            str(request_id): completion_times[request_id]
            for request_id in sorted(completion_times)
        },
        "group_tokens": arguments.group_tokens,
        "slot_count": arguments.slots,
        "primary_arm": arguments.primary_arm,
        "primary_latency_us": primary_median,
        "primary_staging_bytes": primary_staging_bytes,
        "external_bytes": external_bytes,
        "bulk_staging_bytes": bulk_staging_bytes,
        "streaming_staging_bytes": streaming_staging_bytes,
        "staging_capacity_reduction": (
            bulk_staging_bytes / streaming_staging_bytes
            if streaming_staging_bytes
            else None
        ),
        "arm_order": arm_order,
        "direct_us": samples["direct"].report(),
        "atomic_us": samples["atomic"].report(),
        "streaming_us": samples["streaming"].report(),
        "bulk_copy_us": samples["bulk_copy"].report(),
        "streaming_speedup_over_atomic": atomic_median / streaming_median,
        "streaming_speedup_95ci": speedup_ci,
        "streaming_overhead_over_resident_direct": streaming_median / direct_median,
        "atomic_exposed_over_direct_us": atomic_median - direct_median,
        "streaming_exposed_over_direct_us": streaming_median - direct_median,
        "bulk_copy_bandwidth_gbps": external_bytes / bulk_copy_median / 1_000.0,
        "pass_gate": {
            "minimum_speedup": 1.15,
            "minimum_staging_capacity_reduction": 4.0,
            "speedup_pass": atomic_median / streaming_median >= 1.15,
            "speedup_ci_lower_above_one": speedup_ci["lower"] > 1.0,
            "capacity_pass": (
                streaming_staging_bytes != 0
                and bulk_staging_bytes / streaming_staging_bytes >= 4.0
            ),
        },
    }
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
