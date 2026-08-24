#!/usr/bin/env python3
"""Write a read-only physical-tier capability inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .hardware import write_inventory
except ImportError:
    from hardware import write_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_inventory(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
