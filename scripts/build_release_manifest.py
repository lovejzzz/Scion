#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.release_manifest import build_release_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/orpo/dataset-manifest.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release/scion-3.0.0-research/release-manifest.json"),
    )
    args = parser.parse_args()
    result = build_release_manifest(
        repo_root=args.repo_root,
        package_dirs=args.package,
        comparison_paths=args.comparison,
        dataset_manifest_path=args.dataset_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
