#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.local_evaluation import evaluate_locked_tasks
from scion.local_inference import MlxGenerator, snapshot_path
from scion.model_registry import student_pin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("lite", "pro"), required=True)
    parser.add_argument("--variant", choices=("base", "adapter"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path("data/seeds/heldout.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.variant == "adapter" and args.adapter is None:
        parser.error("--adapter is required for the adapter variant")
    if args.variant == "base" and args.adapter is not None:
        parser.error("--adapter is only valid for the adapter variant")
    pin = student_pin(args.tier)
    generator = MlxGenerator(
        snapshot_path(args.cache_dir, pin),
        pin,
        adapter_path=args.adapter,
    )
    output = args.output or Path(f"runs/evaluation/{args.tier}/{args.variant}")
    print(
        json.dumps(
            evaluate_locked_tasks(
                generator=generator,
                fixture_path=args.fixtures,
                output_dir=output,
                tier=args.tier,
                variant=args.variant,
                limit=args.limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
