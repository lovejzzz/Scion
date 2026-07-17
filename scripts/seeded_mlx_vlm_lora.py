#!/usr/bin/env python3
"""Seed MLX-VLM ORPO and inject a real validation split.

MLX-VLM 0.6.3 exposes neither a seed option nor a validation-file option.
This narrow wrapper sets NumPy and MLX RNGs before trainer import and supplies
the manifest-bound validation split without changing upstream trainer code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import runpy
import sys
from pathlib import Path

PACKAGE_NAMES = (
    "mlx",
    "mlx-lm",
    "mlx-vlm",
    "numpy",
    "transformers",
    "huggingface-hub",
    "safetensors",
    "datasets",
    "pyarrow",
    "tokenizers",
)
MODULE_NAMES = (
    "mlx_vlm.lora",
    "mlx_vlm.trainer.lora",
    "mlx_vlm.trainer.orpo_trainer",
    "mlx_vlm.trainer.datasets",
    "mlx_vlm.prompt_utils",
    "mlx_vlm.models.gemma4.processing_gemma4",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_toolchain() -> dict[str, object]:
    modules = {}
    for name in MODULE_NAMES:
        module = __import__(name, fromlist=["*"])
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"toolchain module has no source: {name}")
        modules[name] = {"sha256": sha256_file(source)}
    return {
        "schemaVersion": 1,
        "protocol": "scion-mlx-orpo-toolchain-receipt-v1",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {name: importlib.metadata.version(name) for name in PACKAGE_NAMES},
        "modules": modules,
    }


def _pop(argv: list[str], flag: str) -> tuple[str, list[str]]:
    if flag not in argv:
        raise ValueError(f"{flag} is required")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} requires a value")
    return argv[index + 1], argv[:index] + argv[index + 2 :]


def _forwarded_value(argv: list[str], flag: str) -> str:
    if flag not in argv or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"forwarded trainer arguments require {flag}")
    return argv[argv.index(flag) + 1]


def launch(argv: list[str]) -> None:
    seed_text, remaining = _pop(argv, "--scion-seed")
    validation_split, forwarded = _pop(remaining, "--scion-validation-split")
    seed = int(seed_text)
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("--scion-seed is out of range")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    import mlx.core as mx
    import numpy as np
    from datasets import load_dataset
    from mlx_vlm.trainer import orpo_trainer

    np.random.seed(seed)
    mx.random.seed(seed)
    dataset_path = _forwarded_value(forwarded, "--dataset")
    original = orpo_trainer.train_orpo

    def with_validation(*, train_dataset, val_dataset=None, **kwargs):
        if val_dataset is not None:
            raise RuntimeError("upstream unexpectedly provided validation data")
        raw = load_dataset(dataset_path, split=validation_split)
        if len(raw) == 0:
            raise RuntimeError(f"validation split is empty: {validation_split}")
        validation = train_dataset.__class__(
            raw,
            train_dataset.config,
            train_dataset.processor,
            image_resize_shape=train_dataset.image_resize_shape,
        )
        return original(train_dataset=train_dataset, val_dataset=validation, **kwargs)

    orpo_trainer.train_orpo = with_validation
    sys.argv = ["mlx_vlm.lora", *forwarded]
    try:
        runpy.run_module("mlx_vlm.lora", run_name="__main__", alter_sys=True)
    finally:
        orpo_trainer.train_orpo = original


def main(argv: list[str]) -> None:
    if argv == ["--inspect-toolchain"]:
        print(json.dumps(inspect_toolchain(), separators=(",", ":"), sort_keys=True))
        return
    launch(argv)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"REFUSING: {error}") from error
