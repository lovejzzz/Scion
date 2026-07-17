#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.gguf_evaluation import evaluate_gguf_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8798")
    parser.add_argument("--fixtures", type=Path, default=Path("data/seeds/heldout.jsonl"))
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--runtime-base", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--adapter-manifest", type=Path)
    parser.add_argument("--adapter-id", type=int, default=0)
    parser.add_argument("--adapter-scale", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_gguf_runtime(
                endpoint=args.endpoint,
                fixture_path=args.fixtures,
                runtime_base_path=args.runtime_base,
                output_dir=args.output,
                variant=args.variant,
                split=args.split,
                adapter_manifest_path=args.adapter_manifest,
                adapter_id=args.adapter_id,
                adapter_scale=args.adapter_scale,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
