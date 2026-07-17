"""Scion's local data, distillation, training, and evaluation command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scion", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="print the immutable teacher, critic, and student registry")

    seed = commands.add_parser("seed", help="build license-clean synthetic tasks and locked tests")
    seed.add_argument("--output", type=Path, default=Path("data/seeds"))

    teacher = commands.add_parser("teacher", help="generate oracle-admitted preferences with local Qwen")
    teacher.add_argument("--split", choices=("train", "validation", "preference-test"), required=True)
    teacher.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    teacher.add_argument("--seed-dir", type=Path, default=Path("data/seeds"))
    teacher.add_argument("--output-dir", type=Path, default=Path("data/teacher"))
    teacher.add_argument("--attempts-dir", type=Path, default=Path("runs/teacher-attempts"))
    teacher.add_argument("--limit", type=int)

    critic = commands.add_parser("critic", help="blind-filter preferences with local Gemma 4 31B")
    critic.add_argument("--split", choices=("train", "validation", "preference-test"), required=True)
    critic.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    critic.add_argument("--teacher-dir", type=Path, default=Path("data/teacher"))
    critic.add_argument("--critic-dir", type=Path, default=Path("data/critic"))
    critic.add_argument("--orpo-dir", type=Path, default=Path("data/orpo"))
    critic.add_argument("--limit", type=int)

    manifest = commands.add_parser("manifest", help="build the locked research-corpus receipt")
    manifest.add_argument("--repo-root", type=Path, default=Path("."))
    manifest.add_argument("--orpo-dir", type=Path, default=Path("data/orpo"))
    manifest.add_argument("--critic-dir", type=Path, default=Path("data/critic"))
    manifest.add_argument("--teacher-dir", type=Path, default=Path("data/teacher"))
    manifest.add_argument(
        "--heldout-benchmark",
        type=Path,
        default=Path("data/heldout/coursemapper-heldout-benchmark-v1.json"),
    )
    manifest.add_argument("--output", type=Path, default=Path("data/orpo/dataset-manifest.json"))

    train = commands.add_parser("train", help="train a receipt-bound Gemma 4 student adapter")
    train.add_argument("--tier", choices=("lite", "pro"), required=True)
    train.add_argument("--data-dir", type=Path, default=Path("data/orpo"))
    train.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    train.add_argument("--output", type=Path)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--iterations", type=int, default=400)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--local-files-only", action="store_true")
    train.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("evaluate", help="run the locked 32-task local evaluation")
    evaluate.add_argument("--tier", choices=("lite", "pro"), required=True)
    evaluate.add_argument("--variant", choices=("base", "adapter"), required=True)
    evaluate.add_argument("--adapter", type=Path)
    evaluate.add_argument("--fixtures", type=Path, default=Path("data/seeds/heldout.jsonl"))
    evaluate.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--limit", type=int)

    compare = commands.add_parser("compare", help="apply paired promotion gates")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--adapter", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        from .model_registry import registry_json

        _print(registry_json())
    elif args.command == "seed":
        from .seed_tasks import write_seed_tasks

        _print(write_seed_tasks(args.output))
    elif args.command == "teacher":
        from .local_inference import MlxGenerator, snapshot_path
        from .model_registry import MODEL_PINS
        from .teacher_corpus import generate_teacher_split

        pin = MODEL_PINS["teacher"]
        generator = MlxGenerator(snapshot_path(args.cache_dir, pin), pin)
        _print(
            generate_teacher_split(
                generator=generator,
                seed_path=args.seed_dir / f"{args.split.replace('-', '_')}.jsonl",
                output_path=args.output_dir / f"{args.split.replace('-', '_')}.jsonl",
                attempts_path=args.attempts_dir / f"{args.split.replace('-', '_')}.jsonl",
                split=args.split,
                limit=args.limit,
            )
        )
    elif args.command == "critic":
        from .critic_filter import filter_with_critic
        from .local_inference import MlxGenerator, snapshot_path
        from .model_registry import MODEL_PINS

        pin = MODEL_PINS["critic"]
        generator = MlxGenerator(snapshot_path(args.cache_dir, pin), pin)
        _print(
            filter_with_critic(
                generator=generator,
                teacher_path=args.teacher_dir / f"{args.split.replace('-', '_')}.jsonl",
                evidence_path=args.critic_dir / f"{args.split.replace('-', '_')}.jsonl",
                orpo_path=args.orpo_dir
                / ("test.jsonl" if args.split == "preference-test" else f"{args.split}.jsonl"),
                split=args.split,
                limit=args.limit,
            )
        )
    elif args.command == "manifest":
        from .corpus_manifest import build_dataset_manifest

        _print(
            build_dataset_manifest(
                repo_root=args.repo_root,
                orpo_dir=args.orpo_dir,
                critic_dir=args.critic_dir,
                teacher_dir=args.teacher_dir,
                heldout_benchmark_path=args.heldout_benchmark,
                output_path=args.output,
            )
        )
    elif args.command == "train":
        from .training import train_student

        iterations = 10 if args.smoke else args.iterations
        output = args.output or Path(f"artifacts/scion-{args.tier}-mlx")
        run_dir = args.run_dir or Path(f"runs/training/{args.tier}")
        _print(
            train_student(
                tier=args.tier,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                output_dir=output,
                run_dir=run_dir,
                iterations=iterations,
                local_files_only=args.local_files_only,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "evaluate":
        from .local_evaluation import evaluate_locked_tasks
        from .local_inference import MlxGenerator, snapshot_path
        from .model_registry import student_pin

        if args.variant == "adapter" and args.adapter is None:
            raise SystemExit("--adapter is required for adapter evaluation")
        pin = student_pin(args.tier)
        generator = MlxGenerator(snapshot_path(args.cache_dir, pin), pin, adapter_path=args.adapter)
        output = args.output or Path(f"runs/evaluation/{args.tier}/{args.variant}")
        _print(
            evaluate_locked_tasks(
                generator=generator,
                fixture_path=args.fixtures,
                output_dir=output,
                tier=args.tier,
                variant=args.variant,
                limit=args.limit,
            )
        )
    elif args.command == "compare":
        from .local_evaluation import compare_evaluations

        _print(compare_evaluations(args.base, args.adapter, args.output))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
