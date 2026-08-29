"""Race-safe materialization for typed FlashInfer JIT modules.

FlashInfer 0.6.14 serializes ``JitSpec.build_and_load`` with a
``FileLock(spec.lock_path, thread_local=False)`` and invokes
``spec.build(verbose, need_lock=False)`` while holding that lock.  This module
uses the same paths and lock boundary, but checks the exact JIT library path
inside the lock so callers can distinguish the process that owned a cold build
from a process that loaded an existing artifact.

The reported cold-build duration covers ``JitSpec.build`` (normally source
preparation, compilation, and linking as driven by Ninja).  It is deliberately
not called compiler time.  A successful result is cached only after
``JitSpec.load`` returns a non-None module; failed builds and loads therefore
never become process-cache hits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import threading
import time
from typing import Any, Generic, Protocol, TypeVar


ModuleT = TypeVar("ModuleT")


class FlashInferJitSpec(Protocol[ModuleT]):
    """The FlashInfer 0.6.14 ``JitSpec`` surface used by the materializer."""

    name: str

    @property
    def is_aot(self) -> bool: ...

    @property
    def jit_library_path(self) -> Path: ...

    @property
    def lock_path(self) -> Path: ...

    def build(self, verbose: bool, need_lock: bool = True) -> None: ...

    def load(self, so_path: Path) -> ModuleT: ...


class FlashInferMaterializationOrigin(str, Enum):
    """The operation performed by one materialization call."""

    COLD_BUILD_OWNER = "cold_build_owner"
    DISK_CACHE_LOAD = "disk_cache_load"
    PROCESS_CACHE_HIT = "process_cache_hit"


@dataclass(frozen=True, slots=True)
class FlashInferMaterializationProvenance:
    """Immutable module origin and non-overlapping startup timings."""

    module_name: str
    origin: FlashInferMaterializationOrigin
    producer_origin: FlashInferMaterializationOrigin
    library_path: Path
    lock_path: Path
    artifact_bytes: int
    process_id: int
    process_wait_ns: int
    lock_wait_ns: int
    build_ns: int
    load_ns: int
    total_ns: int

    def __post_init__(self) -> None:
        producer_origins = {
            FlashInferMaterializationOrigin.COLD_BUILD_OWNER,
            FlashInferMaterializationOrigin.DISK_CACHE_LOAD,
        }
        if not self.module_name:
            raise ValueError("FlashInfer materialization has an empty module name")
        if self.producer_origin not in producer_origins:
            raise ValueError("FlashInfer producer origin is not a materialization")
        if self.origin is not FlashInferMaterializationOrigin.PROCESS_CACHE_HIT:
            if self.origin is not self.producer_origin:
                raise ValueError("FlashInfer origin disagrees with its producer")
        if self.artifact_bytes <= 0 or self.process_id <= 0:
            raise ValueError("FlashInfer materialization has invalid artifact metadata")
        timings = (
            self.process_wait_ns,
            self.lock_wait_ns,
            self.build_ns,
            self.load_ns,
            self.total_ns,
        )
        if min(timings) < 0:
            raise ValueError("FlashInfer materialization has a negative timing")
        measured_ns = sum(timings[:-1])
        if self.total_ns < measured_ns:
            raise ValueError("FlashInfer materialization timings overlap")
        if self.origin is FlashInferMaterializationOrigin.PROCESS_CACHE_HIT:
            if self.build_ns != 0 or self.load_ns != 0 or self.lock_wait_ns != 0:
                raise ValueError("a process-cache hit performed JIT or disk work")
        elif self.origin is FlashInferMaterializationOrigin.DISK_CACHE_LOAD:
            if self.build_ns != 0:
                raise ValueError("a disk-cache load performed a cold build")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible stats record."""

        return {
            "module_name": self.module_name,
            "origin": self.origin.value,
            "producer_origin": self.producer_origin.value,
            "library_path": str(self.library_path),
            "lock_path": str(self.lock_path),
            "artifact_bytes": self.artifact_bytes,
            "process_id": self.process_id,
            "process_wait_ns": self.process_wait_ns,
            "lock_wait_ns": self.lock_wait_ns,
            "build_ns": self.build_ns,
            "load_ns": self.load_ns,
            "total_ns": self.total_ns,
        }


@dataclass(frozen=True, slots=True)
class MaterializedFlashInferModule(Generic[ModuleT]):
    """A loaded module and the work performed by this call to obtain it."""

    module: ModuleT
    provenance: FlashInferMaterializationProvenance


@dataclass(frozen=True, slots=True)
class _ProcessCacheEntry:
    module: Any
    producer_origin: FlashInferMaterializationOrigin
    artifact_bytes: int


_ModuleKey = tuple[str, Path]
_STATE_LOCK = threading.Lock()
_PROCESS_ID = os.getpid()
_PROCESS_CACHE: dict[_ModuleKey, _ProcessCacheEntry] = {}
_PROCESS_GUARDS: dict[_ModuleKey, threading.Lock] = {}
_ARTIFACT_LOCK_PATHS: dict[_ModuleKey, Path] = {}


def _reset_after_fork() -> None:
    """Discard parent-process modules and locks in a forked worker."""

    global _STATE_LOCK, _PROCESS_ID
    global _PROCESS_CACHE, _PROCESS_GUARDS, _ARTIFACT_LOCK_PATHS
    _STATE_LOCK = threading.Lock()
    _PROCESS_ID = os.getpid()
    _PROCESS_CACHE = {}
    _PROCESS_GUARDS = {}
    _ARTIFACT_LOCK_PATHS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _ensure_current_process_locked() -> None:
    global _PROCESS_ID
    if _PROCESS_ID == os.getpid():
        return
    # ``register_at_fork`` handles supported POSIX runtimes.  This fallback
    # also protects unusual process launchers that replace the PID without
    # running the callback.
    _PROCESS_CACHE.clear()
    _PROCESS_GUARDS.clear()
    _ARTIFACT_LOCK_PATHS.clear()
    _PROCESS_ID = os.getpid()


def _lookup_or_guard(
    key: _ModuleKey, lock_path: Path
) -> tuple[_ProcessCacheEntry | None, threading.Lock]:
    with _STATE_LOCK:
        _ensure_current_process_locked()
        recorded_lock = _ARTIFACT_LOCK_PATHS.setdefault(key, lock_path)
        if recorded_lock != lock_path:
            raise RuntimeError(
                "the same FlashInfer JIT artifact was assigned conflicting lock paths"
            )
        return _PROCESS_CACHE.get(key), _PROCESS_GUARDS.setdefault(
            key, threading.Lock()
        )


def _lookup_process_cache(key: _ModuleKey) -> _ProcessCacheEntry | None:
    with _STATE_LOCK:
        _ensure_current_process_locked()
        return _PROCESS_CACHE.get(key)


def _store_process_cache(key: _ModuleKey, entry: _ProcessCacheEntry) -> None:
    with _STATE_LOCK:
        _ensure_current_process_locked()
        existing = _PROCESS_CACHE.setdefault(key, entry)
        if existing is not entry:
            raise RuntimeError(
                "FlashInfer process cache raced despite its per-artifact guard"
            )


def _process_hit(
    *,
    entry: _ProcessCacheEntry,
    module_name: str,
    library_path: Path,
    lock_path: Path,
    started_ns: int,
    process_wait_ns: int,
) -> MaterializedFlashInferModule[Any]:
    provenance = FlashInferMaterializationProvenance(
        module_name=module_name,
        origin=FlashInferMaterializationOrigin.PROCESS_CACHE_HIT,
        producer_origin=entry.producer_origin,
        library_path=library_path,
        lock_path=lock_path,
        artifact_bytes=entry.artifact_bytes,
        process_id=os.getpid(),
        process_wait_ns=process_wait_ns,
        lock_wait_ns=0,
        build_ns=0,
        load_ns=0,
        total_ns=time.perf_counter_ns() - started_ns,
    )
    return MaterializedFlashInferModule(entry.module, provenance)


def materialize_typed_flashinfer_module(
    spec: FlashInferJitSpec[ModuleT],
    *,
    verbose: bool | None = None,
) -> MaterializedFlashInferModule[ModuleT]:
    """Build or load one exact FlashInfer JIT artifact exactly once per process.

    ``cold_build_owner`` means that the exact ``spec.jit_library_path`` was
    absent *after* this process acquired ``spec.lock_path`` and this call ran
    ``spec.build``.  ``disk_cache_load`` means the artifact already existed at
    that same point.  ``process_cache_hit`` performs neither file locking nor
    loading and names the original producer in ``producer_origin``.

    The function intentionally rejects AOT specs: typed NTA modules are custom
    JIT artifacts, and FlashInfer's AOT path has a different no-lock lifecycle.
    """

    started_ns = time.perf_counter_ns()
    module_name = spec.name
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("FlashInfer JIT spec has an empty name")
    if verbose is not None and not isinstance(verbose, bool):
        raise TypeError("FlashInfer JIT verbosity must be boolean")
    if bool(spec.is_aot):
        raise ValueError("typed FlashInfer materialization requires a non-AOT JitSpec")

    # Use the paths returned by JitSpec verbatim. In particular, resolving a
    # symlinked lock path here would no longer synchronize with FlashInfer's
    # own build_and_load(), which passes spec.lock_path directly to FileLock.
    library_path = Path(spec.jit_library_path)
    lock_path = Path(spec.lock_path)
    if not library_path.is_absolute() or not lock_path.is_absolute():
        raise ValueError("FlashInfer JitSpec paths must be absolute")
    if library_path == lock_path:
        raise ValueError("FlashInfer JIT library and lock paths must differ")
    if library_path.name != f"{module_name}.so":
        raise ValueError("FlashInfer JIT library path disagrees with its spec name")
    key = (module_name, library_path)

    cached, process_guard = _lookup_or_guard(key, lock_path)
    if cached is not None:
        return _process_hit(
            entry=cached,
            module_name=module_name,
            library_path=library_path,
            lock_path=lock_path,
            started_ns=started_ns,
            process_wait_ns=0,
        )

    process_wait_started_ns = time.perf_counter_ns()
    with process_guard:
        process_wait_ns = time.perf_counter_ns() - process_wait_started_ns
        cached = _lookup_process_cache(key)
        if cached is not None:
            return _process_hit(
                entry=cached,
                module_name=module_name,
                library_path=library_path,
                lock_path=lock_path,
                started_ns=started_ns,
                process_wait_ns=process_wait_ns,
            )

        # Import lazily so the engine-neutral base package can be imported
        # without the FlashInfer extra. FlashInfer 0.6.14 itself depends on and
        # uses this exact FileLock implementation for JitSpec.build_and_load.
        from filelock import FileLock

        file_lock = FileLock(lock_path, thread_local=False)
        lock_wait_started_ns = time.perf_counter_ns()
        with file_lock:
            lock_wait_ns = time.perf_counter_ns() - lock_wait_started_ns
            build_ns = 0
            if library_path.is_file():
                origin = FlashInferMaterializationOrigin.DISK_CACHE_LOAD
            else:
                origin = FlashInferMaterializationOrigin.COLD_BUILD_OWNER
                build_started_ns = time.perf_counter_ns()
                resolved_verbose = (
                    os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
                    if verbose is None
                    else verbose
                )
                spec.build(bool(resolved_verbose), need_lock=False)
                build_ns = time.perf_counter_ns() - build_started_ns
                if not library_path.is_file():
                    raise RuntimeError(
                        "FlashInfer JIT build returned without its exact library artifact"
                    )

            artifact = library_path.stat()
            if artifact.st_size <= 0:
                raise RuntimeError("FlashInfer JIT artifact is empty")
            load_started_ns = time.perf_counter_ns()
            module = spec.load(library_path)
            load_ns = time.perf_counter_ns() - load_started_ns
            if module is None:
                raise RuntimeError("FlashInfer JIT loader returned no module")

            entry = _ProcessCacheEntry(module, origin, artifact.st_size)
            # Publish only after build, artifact verification, and load have
            # all succeeded. Exceptions above leave no successful cache entry.
            _store_process_cache(key, entry)

        provenance = FlashInferMaterializationProvenance(
            module_name=module_name,
            origin=origin,
            producer_origin=origin,
            library_path=library_path,
            lock_path=lock_path,
            artifact_bytes=artifact.st_size,
            process_id=os.getpid(),
            process_wait_ns=process_wait_ns,
            lock_wait_ns=lock_wait_ns,
            build_ns=build_ns,
            load_ns=load_ns,
            total_ns=time.perf_counter_ns() - started_ns,
        )
        return MaterializedFlashInferModule(module, provenance)


__all__ = [
    "FlashInferJitSpec",
    "FlashInferMaterializationOrigin",
    "FlashInferMaterializationProvenance",
    "MaterializedFlashInferModule",
    "materialize_typed_flashinfer_module",
]
