#!/usr/bin/env python3
"""Fetch and verify the public anonymized Qwen-Bailian trace objects.

The repository stores these files with Git LFS.  This helper uses the public
media endpoint directly, verifies the immutable object size and SHA-256, and
never replaces an existing file unless ``--replace`` is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


SOURCE_REPOSITORY = "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon"
_MEDIA_ROOT = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main"

DATASETS = {
    "qwen_traceA_blksz_16.jsonl": {
        "size": 56_354_493,
        "sha256": "07cedc9ed8aff301994ac68ed4aede8123b7603673575eeba9dd677de663db17",
    },
    "qwen_traceB_blksz_16.jsonl": {
        "size": 96_209_982,
        "sha256": "68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59",
    },
    "qwen_thinking_blksz_16.jsonl": {
        "size": 27_901_454,
        "sha256": "41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906",
    },
    "qwen_coder_blksz_16.jsonl": {
        "size": 132_054_902,
        "sha256": "3d74974cebb7bccb69b1ca6ea210a44d9a3033443c0271ed015327284dea1aff",
    },
}


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _verify(path: Path, metadata: dict[str, object]) -> None:
    size, digest = _digest(path)
    if size != metadata["size"] or digest != metadata["sha256"]:
        raise ValueError(
            f"checksum mismatch for {path}: size={size}, sha256={digest}"
        )


def fetch(name: str, output_dir: Path, *, replace: bool = False) -> Path:
    if name not in DATASETS:
        raise ValueError(f"unknown Bailian trace: {name}")
    metadata = DATASETS[name]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    if destination.exists():
        try:
            _verify(destination, metadata)
        except ValueError:
            if not replace:
                raise ValueError(
                    f"existing file is not the expected object: {destination}; "
                    "use --replace to download it again"
                )
        else:
            return destination

    partial = destination.with_name(destination.name + ".partial")
    request = Request(
        f"{_MEDIA_ROOT}/{name}",
        headers={"User-Agent": "sched-pass-bailian-fetch/1"},
    )
    with urlopen(request, timeout=60) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    _verify(partial, metadata)
    partial.replace(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        choices=sorted(DATASETS),
        help="fetch only this object; repeat for multiple objects (default: all)",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    names = args.files or list(DATASETS)
    try:
        paths = [fetch(name, args.output_dir.resolve(), replace=args.replace) for name in names]
    except (OSError, ValueError) as error:
        print(f"fetch_bailian failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": 1,
                "classification": "bailian-source-fetch",
                "source_repository": SOURCE_REPOSITORY,
                "files": [str(path) for path in paths],
                "objects": {name: DATASETS[name] for name in names},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
