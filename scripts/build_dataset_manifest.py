#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scion.corpus_manifest import build_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--orpo-dir", type=Path, default=Path("data/orpo"))
    parser.add_argument("--critic-dir", type=Path, default=Path("data/critic"))
    parser.add_argument("--teacher-dir", type=Path, default=Path("data/teacher"))
    parser.add_argument(
        "--heldout-benchmark",
        type=Path,
        default=Path("data/heldout/coursemapper-heldout-benchmark-v1.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/orpo/dataset-manifest.json"))
    args = parser.parse_args()
    result = build_dataset_manifest(
        repo_root=args.repo_root,
        orpo_dir=args.orpo_dir,
        critic_dir=args.critic_dir,
        teacher_dir=args.teacher_dir,
        heldout_benchmark_path=args.heldout_benchmark,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
