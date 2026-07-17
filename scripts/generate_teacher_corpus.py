#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.local_inference import MlxGenerator, snapshot_path
from scion.model_registry import MODEL_PINS
from scion.teacher_corpus import generate_teacher_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "preference-test"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--seed-dir", type=Path, default=Path("data/seeds"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/teacher"))
    parser.add_argument("--attempts-dir", type=Path, default=Path("runs/teacher-attempts"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    pin = MODEL_PINS["teacher"]
    generator = MlxGenerator(snapshot_path(args.cache_dir, pin), pin)
    result = generate_teacher_split(
        generator=generator,
        seed_path=args.seed_dir / f"{args.split.replace('-', '_')}.jsonl",
        output_path=args.output_dir / f"{args.split.replace('-', '_')}.jsonl",
        attempts_path=args.attempts_dir / f"{args.split.replace('-', '_')}.jsonl",
        split=args.split,
        limit=args.limit,
        max_attempts=args.max_attempts,
    )
    manifest = args.output_dir / f"{args.split.replace('-', '_')}.manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
