#!/usr/bin/env python3
"""CPU-only tests for race-safe typed FlashInfer JIT materialization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.flashinfer_jit import (  # noqa: E402
    FlashInferMaterializationOrigin,
    materialize_typed_flashinfer_module,
)


@dataclass(frozen=True, slots=True)
class _FakeModule:
    name: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class _FakeSpec:
    root: Path
    name: str = "nta_fake_typed_module"
    build_delay_seconds: float = 0.0
    fail_build: bool = False
    fail_load: bool = False
    return_none: bool = False
    lock_variant: str = ""

    @property
    def is_aot(self) -> bool:
        return False

    @property
    def jit_library_path(self) -> Path:
        return self.root / self.name / f"{self.name}.so"

    @property
    def lock_path(self) -> Path:
        suffix = f"-{self.lock_variant}" if self.lock_variant else ""
        return self.root / "locks" / f"{self.name}{suffix}.lock"

    @property
    def build_log(self) -> Path:
        return self.root / "build.log"

    @property
    def load_log(self) -> Path:
        return self.root / "load.log"

    def build(self, verbose: bool, need_lock: bool = True) -> None:
        if need_lock:
            raise AssertionError("materializer asked FlashInfer to reacquire its lock")
        with self.build_log.open("a", encoding="utf-8") as output:
            output.write(f"{self.name} verbose={int(verbose)}\n")
        if self.fail_build:
            raise RuntimeError("injected build failure")
        if self.build_delay_seconds:
            time.sleep(self.build_delay_seconds)
        self.jit_library_path.parent.mkdir(parents=True, exist_ok=True)
        self.jit_library_path.write_bytes(f"module:{self.name}".encode())

    def load(self, so_path: Path) -> _FakeModule | None:
        if so_path.resolve() != self.jit_library_path.resolve():
            raise AssertionError("materializer did not load the exact JitSpec path")
        with self.load_log.open("a", encoding="utf-8") as output:
            output.write(f"{self.name}\n")
        if self.fail_load:
            raise RuntimeError("injected load failure")
        if self.return_none:
            return None
        return _FakeModule(self.name, so_path.read_bytes())


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _materialize_in_process(
    root: str,
    barrier: Any,
    result: Any,
    *,
    build_delay_seconds: float,
) -> None:
    try:
        spec = _FakeSpec(Path(root), build_delay_seconds=build_delay_seconds)
        barrier.wait(timeout=10.0)
        materialized = materialize_typed_flashinfer_module(spec, verbose=False)
        result.put(
            {
                "module": materialized.module.payload.decode(),
                "provenance": materialized.provenance.as_dict(),
            }
        )
    except BaseException as error:
        result.put({"error": f"{type(error).__name__}: {error}"})


def _materialize_after_fork(root: str, result: Any) -> None:
    try:
        materialized = materialize_typed_flashinfer_module(
            _FakeSpec(Path(root)), verbose=False
        )
        result.put(materialized.provenance.as_dict())
    except BaseException as error:
        result.put({"error": f"{type(error).__name__}: {error}"})


def _join_owned_processes(processes: list[Any]) -> None:
    for process in processes:
        process.join(timeout=15.0)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5.0)
    if alive:
        raise AssertionError("owned materializer test process did not terminate")
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise AssertionError(f"materializer child exit failures: {failures}")


def _test_cold_build_and_process_cache() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-cold-") as directory:
        spec = _FakeSpec(Path(directory))
        first = materialize_typed_flashinfer_module(spec, verbose=True)
        assert first.provenance.origin is (
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER
        )
        assert first.provenance.producer_origin is first.provenance.origin
        assert first.provenance.build_ns >= 0
        assert first.provenance.load_ns >= 0
        assert first.provenance.artifact_bytes == spec.jit_library_path.stat().st_size
        assert first.provenance.as_dict()["origin"] == "cold_build_owner"
        assert _lines(spec.build_log) == [f"{spec.name} verbose=1"]
        assert _lines(spec.load_log) == [spec.name]

        second = materialize_typed_flashinfer_module(spec)
        assert second.module is first.module
        assert second.provenance.origin is (
            FlashInferMaterializationOrigin.PROCESS_CACHE_HIT
        )
        assert second.provenance.producer_origin is (
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER
        )
        assert second.provenance.build_ns == 0
        assert second.provenance.load_ns == 0
        assert second.provenance.lock_wait_ns == 0
        assert _lines(spec.build_log) == [f"{spec.name} verbose=1"]
        assert _lines(spec.load_log) == [spec.name]


def _test_disk_cache_load() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-disk-") as directory:
        spec = _FakeSpec(Path(directory))
        spec.jit_library_path.parent.mkdir(parents=True)
        spec.jit_library_path.write_bytes(b"prebuilt")
        result = materialize_typed_flashinfer_module(spec)
        assert result.provenance.origin is (
            FlashInferMaterializationOrigin.DISK_CACHE_LOAD
        )
        assert result.provenance.build_ns == 0
        assert result.module.payload == b"prebuilt"
        assert _lines(spec.build_log) == []
        assert _lines(spec.load_log) == [spec.name]


def _test_failures_do_not_enter_process_cache() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-failure-") as directory:
        root = Path(directory)
        failing_build = _FakeSpec(root, fail_build=True)
        try:
            materialize_typed_flashinfer_module(failing_build)
        except RuntimeError as error:
            assert "injected build failure" in str(error)
        else:
            raise AssertionError("injected FlashInfer build failure was accepted")

        recovered = materialize_typed_flashinfer_module(_FakeSpec(root), verbose=False)
        assert recovered.provenance.origin is (
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER
        )
        assert len(_lines(failing_build.build_log)) == 2
        assert len(_lines(failing_build.load_log)) == 1

    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-load-failure-") as directory:
        root = Path(directory)
        failing_load = _FakeSpec(root, fail_load=True)
        failing_load.jit_library_path.parent.mkdir(parents=True)
        failing_load.jit_library_path.write_bytes(b"prebuilt")
        try:
            materialize_typed_flashinfer_module(failing_load)
        except RuntimeError as error:
            assert "injected load failure" in str(error)
        else:
            raise AssertionError("injected FlashInfer load failure was accepted")

        recovered = materialize_typed_flashinfer_module(_FakeSpec(root), verbose=False)
        assert recovered.provenance.origin is (
            FlashInferMaterializationOrigin.DISK_CACHE_LOAD
        )
        assert len(_lines(failing_load.load_log)) == 2

    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-none-") as directory:
        root = Path(directory)
        no_module = _FakeSpec(root, return_none=True)
        no_module.jit_library_path.parent.mkdir(parents=True)
        no_module.jit_library_path.write_bytes(b"prebuilt")
        try:
            materialize_typed_flashinfer_module(no_module)
        except RuntimeError as error:
            assert "returned no module" in str(error)
        else:
            raise AssertionError("an empty FlashInfer module was cached")
        recovered = materialize_typed_flashinfer_module(_FakeSpec(root), verbose=False)
        assert recovered.provenance.origin is (
            FlashInferMaterializationOrigin.DISK_CACHE_LOAD
        )


def _test_thread_concurrency_loads_once() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-thread-") as directory:
        spec = _FakeSpec(Path(directory), build_delay_seconds=0.2)
        barrier = threading.Barrier(2)

        def run() -> Any:
            barrier.wait(timeout=5.0)
            return materialize_typed_flashinfer_module(spec, verbose=False)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(lambda _index: run(), range(2)))
        origins = {result.provenance.origin for result in results}
        assert origins == {
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER,
            FlashInferMaterializationOrigin.PROCESS_CACHE_HIT,
        }
        assert results[0].module is results[1].module
        process_hit = next(
            result
            for result in results
            if result.provenance.origin
            is FlashInferMaterializationOrigin.PROCESS_CACHE_HIT
        )
        assert process_hit.provenance.process_wait_ns > 0
        assert _lines(spec.build_log) == [f"{spec.name} verbose=0"]
        assert _lines(spec.load_log) == [spec.name]


def _test_multiprocess_lock_has_one_cold_owner() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-process-") as directory:
        context = mp.get_context("spawn")
        barrier = context.Barrier(2)
        result = context.Queue()
        processes = [
            context.Process(
                target=_materialize_in_process,
                args=(directory, barrier, result),
                kwargs={"build_delay_seconds": 0.25},
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        _join_owned_processes(processes)
        reports = [result.get(timeout=5.0) for _ in processes]
        result.close()
        result.join_thread()
        errors = [report["error"] for report in reports if "error" in report]
        assert not errors, errors
        origins = sorted(report["provenance"]["origin"] for report in reports)
        assert origins == ["cold_build_owner", "disk_cache_load"]
        assert {report["module"] for report in reports} == {
            "module:nta_fake_typed_module"
        }
        spec = _FakeSpec(Path(directory))
        assert _lines(spec.build_log) == [f"{spec.name} verbose=0"]
        assert _lines(spec.load_log) == [spec.name, spec.name]
        disk = next(
            report
            for report in reports
            if report["provenance"]["origin"] == "disk_cache_load"
        )
        assert disk["provenance"]["build_ns"] == 0
        assert disk["provenance"]["lock_wait_ns"] > 0


def _test_fork_does_not_inherit_loaded_modules() -> None:
    if "fork" not in mp.get_all_start_methods():
        return
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-fork-") as directory:
        spec = _FakeSpec(Path(directory))
        parent = materialize_typed_flashinfer_module(spec, verbose=False)
        assert parent.provenance.origin is (
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER
        )
        context = mp.get_context("fork")
        result = context.Queue()
        process = context.Process(
            target=_materialize_after_fork,
            args=(directory, result),
        )
        process.start()
        _join_owned_processes([process])
        report = result.get(timeout=5.0)
        result.close()
        result.join_thread()
        assert "error" not in report, report
        assert report["origin"] == "disk_cache_load"
        assert report["process_id"] != parent.provenance.process_id
        assert _lines(spec.build_log) == [f"{spec.name} verbose=0"]
        assert _lines(spec.load_log) == [spec.name, spec.name]


def _test_invalid_contracts_are_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-fi-jit-contract-") as directory:
        root = Path(directory)
        spec = _FakeSpec(root)
        spec.jit_library_path.parent.mkdir(parents=True)
        spec.jit_library_path.write_bytes(b"prebuilt")
        materialize_typed_flashinfer_module(spec)
        try:
            materialize_typed_flashinfer_module(
                _FakeSpec(root, lock_variant="conflict")
            )
        except RuntimeError as error:
            assert "conflicting lock paths" in str(error)
        else:
            raise AssertionError("one artifact accepted two lock identities")

        class _AotSpec(_FakeSpec):
            @property
            def is_aot(self) -> bool:
                return True

        try:
            materialize_typed_flashinfer_module(_AotSpec(root, name="aot"))
        except ValueError as error:
            assert "non-AOT" in str(error)
        else:
            raise AssertionError("AOT module entered the typed JIT contract")


def main() -> None:
    _test_cold_build_and_process_cache()
    _test_disk_cache_load()
    _test_failures_do_not_enter_process_cache()
    _test_thread_concurrency_loads_once()
    _test_multiprocess_lock_has_one_cold_owner()
    _test_fork_does_not_inherit_loaded_modules()
    _test_invalid_contracts_are_rejected()
    print("flashinfer_jit=pass")


if __name__ == "__main__":
    main()
