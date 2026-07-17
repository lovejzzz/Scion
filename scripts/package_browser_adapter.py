#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.browser_package import build_browser_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/orpo/dataset-manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, default=Path(".cache/llama.cpp"))
    parser.add_argument("--bridge", type=Path, default=Path("scripts/convert_mlx_lora_to_peft.py"))
    parser.add_argument("--inference-scale", type=float, default=16)
    args = parser.parse_args()
    result = build_browser_adapter(
        source_manifest_path=args.source_manifest,
        dataset_manifest_path=args.dataset_manifest,
        output_dir=args.output_dir,
        base_dir=args.base_dir,
        llama_cpp_dir=args.llama_cpp,
        bridge_path=args.bridge,
        inference_scale=args.inference_scale,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
