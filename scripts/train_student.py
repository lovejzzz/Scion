#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.training import train_student


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("lite", "pro"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/orpo"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    iterations = 10 if args.smoke else args.iterations
    output = args.output or Path(f"artifacts/scion-{args.tier}-mlx")
    run_dir = args.run_dir or Path(f"runs/training/{args.tier}")
    print(
        json.dumps(
            train_student(
                tier=args.tier,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                output_dir=output,
                run_dir=run_dir,
                iterations=iterations,
                local_files_only=args.local_files_only,
                dry_run=args.dry_run,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
