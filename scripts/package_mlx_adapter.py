#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.packaging import build_mlx_adapter_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("lite", "pro"), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/orpo/dataset-manifest.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_mlx_adapter_manifest(
        tier=args.tier,
        adapter_dir=args.adapter_dir,
        dataset_manifest_path=args.dataset_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
