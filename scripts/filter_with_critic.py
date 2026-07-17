#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.critic_filter import filter_with_critic
from scion.local_inference import MlxGenerator, snapshot_path
from scion.model_registry import MODEL_PINS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "preference-test"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--teacher-dir", type=Path, default=Path("data/teacher"))
    parser.add_argument("--critic-dir", type=Path, default=Path("data/critic"))
    parser.add_argument("--orpo-dir", type=Path, default=Path("data/orpo"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pin = MODEL_PINS["critic"]
    generator = MlxGenerator(snapshot_path(args.cache_dir, pin), pin)
    result = filter_with_critic(
        generator=generator,
        teacher_path=args.teacher_dir / f"{args.split.replace('-', '_')}.jsonl",
        evidence_path=args.critic_dir / f"{args.split.replace('-', '_')}.jsonl",
        orpo_path=args.orpo_dir
        / ("test.jsonl" if args.split == "preference-test" else f"{args.split}.jsonl"),
        split=args.split,
        limit=args.limit,
    )
    manifest = args.orpo_dir / f"{args.split.replace('-', '_')}.manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
