#!/usr/bin/env python3
"""Download every required NAIF SPICE kernel into the configured data root.

Usage:
    python scripts/fetch_kernels.py [--force]

Idempotent: skips files already on disk unless --force is given.
"""

from __future__ import annotations

import argparse
import logging
import sys

from reflectors.kernels import (
    KERNEL_DIR,
    META_KERNEL_PATH,
    REQUIRED_KERNELS,
    download_kernels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download kernels even if files already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Kernel directory: {KERNEL_DIR}")
    for spec in REQUIRED_KERNELS:
        print(f"  - {spec.filename}: {spec.description}")

    paths = download_kernels(force=args.force)

    print()
    print(f"Meta-kernel: {META_KERNEL_PATH}")
    for p in paths:
        print(f"  {p.name}: {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
