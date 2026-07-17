"""Scion training, conversion, evaluation, and serving command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scion", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="build the provenance-bound CourseMapper corpus")
    dataset.add_argument("--legacy-jsonl", type=Path, required=True)
    dataset.add_argument("--approved-jsonl", type=Path, required=True)
    dataset.add_argument("--source-capture-dir", type=Path, action="append", required=True)
    dataset.add_argument("--output", type=Path, default=Path("data"))
    dataset.add_argument("--eval-output", type=Path, default=Path("eval/fixtures.jsonl"))

    prepare = commands.add_parser("prepare", help="prepare an exact 4-bit MLX QLoRA base")
    prepare.add_argument("--output", type=Path, default=Path(".cache/bonsai-27b-mlx-4bit"))
    _paths(prepare)
    prepare.add_argument("--local-files-only", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")

    train = commands.add_parser("train", help="train the Scion MLX LoRA")
    train.add_argument("--config", type=Path, default=Path("configs/train-bonsai-27b.json"))
    train.add_argument("--model", type=Path, default=Path(".cache/bonsai-27b-mlx-4bit"))
    train.add_argument("--data", type=Path, default=Path("data"))
    train.add_argument("--output", type=Path, default=Path("artifacts/scion-bonsai-27b-mlx"))
    train.add_argument("--run-dir", type=Path, default=Path("runs/bonsai-27b"))
    train.add_argument("--iters", type=int)
    train.add_argument("--dry-run", action="store_true")

    runtime = commands.add_parser("runtime", help="manage the pinned PrismML llama.cpp runtime")
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_build = runtime_commands.add_parser("build")
    runtime_build.add_argument("--output", type=Path, default=Path(".cache/PrismML-llama.cpp"))
    runtime_build.add_argument("--jobs", type=int, default=8)
    runtime_build.add_argument("--dry-run", action="store_true")

    convert = commands.add_parser("convert", help="convert MLX LoRA to a deployment GGUF adapter")
    convert.add_argument("--mlx-adapter", type=Path, default=Path("artifacts/scion-bonsai-27b-mlx"))
    convert.add_argument("--output", type=Path, default=Path("artifacts/scion-bonsai-27b.gguf"))
    convert.add_argument("--runtime", type=Path, default=Path(".cache/PrismML-llama.cpp"))
    convert.add_argument("--work-dir", type=Path, default=Path("runs/bonsai-27b/conversion"))
    _paths(convert)
    convert.add_argument("--local-files-only", action="store_true")
    convert.add_argument("--dry-run", action="store_true")

    serve = commands.add_parser("serve", help="serve Bonsai with the Scion adapter on the CourseMapper port")
    serve.add_argument("--adapter", type=Path)
    serve.add_argument("--base", type=Path)
    serve.add_argument("--llama-server", type=Path)
    _paths(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8799)
    serve.add_argument("--context-size", type=int, default=32768)
    serve.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("evaluate", help="run held-out CourseMapper fixtures against an endpoint")
    evaluate.add_argument("--endpoint", default="http://127.0.0.1:8799")
    evaluate.add_argument("--fixtures", type=Path, default=Path("eval/fixtures.jsonl"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--model", default="scion-1")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--max-tokens", type=int, default=4096)

    compare = commands.add_parser("compare", help="apply base-versus-adapter promotion thresholds")
    compare.add_argument("--base", type=Path, required=True)
    compare.add_argument("--adapter", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("runs/bonsai-27b/evaluation-comparison.json"))

    smoke = commands.add_parser("smoke", help="exercise the exact CourseMapper OpenAI-compatible contract")
    smoke.add_argument("--endpoint", default="http://127.0.0.1:8799")
    smoke.add_argument("--output", type=Path, default=Path("runs/bonsai-27b/coursemapper-smoke.json"))
    smoke.add_argument("--model", default="scion-1")

    release = commands.add_parser("release", help="promote a verified adapter and write its release manifest")
    release.add_argument("--adapter", type=Path, default=Path("artifacts/scion-bonsai-27b.gguf"))
    release.add_argument("--dataset-manifest", type=Path, default=Path("data/manifest.json"))
    release.add_argument("--training-result", type=Path, default=Path("runs/bonsai-27b/training-result.json"))
    release.add_argument(
        "--conversion-receipt", type=Path, default=Path("runs/bonsai-27b/conversion/conversion-receipt.json")
    )
    release.add_argument(
        "--comparison", type=Path, default=Path("runs/bonsai-27b/evaluation-comparison.json")
    )
    release.add_argument("--smoke", type=Path, default=Path("runs/bonsai-27b/coursemapper-smoke.json"))
    release.add_argument("--output", type=Path, default=Path("artifacts/scion-bonsai-27b.manifest.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "dataset":
        from .dataset import build_dataset

        paths = []
        for directory in args.source_capture_dir:
            paths.extend(directory.glob("*-reference.json"))
        _print(
            build_dataset(
                legacy_jsonl=args.legacy_jsonl,
                approved_jsonl=args.approved_jsonl,
                source_capture_paths=paths,
                output_dir=args.output,
                eval_output=args.eval_output,
            )
        )
    elif args.command == "prepare":
        from .training import prepare_training_base

        _print(
            prepare_training_base(
                output=args.output,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "train":
        from .training import train

        _print(
            train(
                config_path=args.config,
                model_path=args.model,
                data_dir=args.data,
                output_dir=args.output,
                run_dir=args.run_dir,
                iters=args.iters,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "runtime" and args.runtime_command == "build":
        from .runtime import build_runtime

        _print(build_runtime(args.output, jobs=args.jobs, dry_run=args.dry_run))
    elif args.command == "convert":
        from .convert import convert_adapter

        _print(
            convert_adapter(
                mlx_adapter=args.mlx_adapter,
                output=args.output,
                runtime=args.runtime,
                cache_dir=args.cache_dir,
                work_dir=args.work_dir,
                local_files_only=args.local_files_only,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "serve":
        from .server import serve

        _print(
            serve(
                adapter=args.adapter,
                llama_server=args.llama_server,
                base=args.base,
                cache_dir=args.cache_dir,
                host=args.host,
                port=args.port,
                context_size=args.context_size,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate_endpoint

        _print(
            evaluate_endpoint(
                endpoint=args.endpoint,
                fixtures_path=args.fixtures,
                output=args.output,
                model=args.model,
                limit=args.limit,
                max_tokens=args.max_tokens,
            )
        )
    elif args.command == "compare":
        from .evaluate import compare_reports

        _print(compare_reports(args.base, args.adapter, args.output))
    elif args.command == "smoke":
        from .evaluate import coursemapper_smoke

        _print(coursemapper_smoke(args.endpoint, args.output, model=args.model))
    elif args.command == "release":
        from .manifest import build_release_manifest

        _print(
            build_release_manifest(
                adapter=args.adapter,
                dataset_manifest=args.dataset_manifest,
                training_result=args.training_result,
                conversion_receipt=args.conversion_receipt,
                comparison=args.comparison,
                smoke=args.smoke,
                output=args.output,
            )
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
