"""Small crash-safe writers for experiment artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_write_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` only after the complete payload reaches the filesystem."""

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
        temporary_path = None
        # fsync the directory as well as the payload: after a power loss the
        # rename itself must be durable, not only the temporary file's bytes.
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize a stable, human-readable JSON artifact atomically."""

    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
