#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.release_archives import build_release_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/release-assets"))
    parser.add_argument("--release", default="v3.0.0-research.1")
    args = parser.parse_args()
    receipts = []
    for package in args.package:
        output = args.output_dir / f"{package.name}-{args.release}.tar.gz"
        receipts.append(build_release_archive(package_dir=package, output_path=output))
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
