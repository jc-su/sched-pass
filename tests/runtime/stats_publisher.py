#!/usr/bin/env python3
"""Test asynchronous statistics publication and shutdown ownership."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from nta_runtime.engines.sglang_state import _StatsPublisher  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nta-stats-publisher-") as directory:
        root = Path(directory)
        output = root / "stats.json"
        publisher = _StatsPublisher(output)
        publisher.publish({"sequence": 1})
        publisher.publish({"sequence": 2})
        publisher.close()
        assert json.loads(output.read_text(encoding="utf-8")) == {"sequence": 2}
        publisher.close()
        try:
            publisher.publish({"sequence": 3})
        except RuntimeError:
            pass
        else:
            raise AssertionError("closed statistics publisher accepted a report")

        blocked_parent = root / "blocked"
        blocked_parent.write_text("not a directory", encoding="utf-8")
        failed = _StatsPublisher(blocked_parent / "stats.json")
        try:
            failed.publish({"sequence": 1}, wait=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("statistics write failure was not reported")
        failed.close()
    print("stats_publisher=pass")


if __name__ == "__main__":
    main()
