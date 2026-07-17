#!/usr/bin/env python3
"""Convert the pinned Gemma 4 E2B MLX LoRA to llama.cpp's PEFT input.

This narrow, fail-closed bridge is derived from CourseMapper's validated
``convert_mlx_lora_to_peft.py`` path. It never merges the base model and admits
only complete language-model LoRA A/B pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scion.constants import LITE_TRAIN_BASE_ID, LITE_TRAIN_BASE_REVISION

MLX_PREFIX = "language_model.model."
PEFT_PREFIX = "base_model.model.model."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def map_mlx_lora_name(name: str) -> tuple[str, str]:
    if name.endswith(".lora_a"):
        leaf, side, stem = "lora_A.weight", "A", name[: -len(".lora_a")]
    elif name.endswith(".lora_b"):
        leaf, side, stem = "lora_B.weight", "B", name[: -len(".lora_b")]
    else:
        raise ValueError(f"not a LoRA tensor: {name}")
    if not stem.startswith(MLX_PREFIX):
        raise ValueError(f"LoRA tensor is outside the Gemma text model: {name}")
    mapped = stem[len(MLX_PREFIX) :]
    if not mapped or mapped.startswith(".") or ".." in mapped:
        raise ValueError(f"invalid LoRA tensor path: {name}")
    return f"{PEFT_PREFIX}{mapped}.{leaf}", side


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_source_manifest(manifest_path: Path, adapter_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(manifest_path)
    adapter = manifest.get("adapter") or {}
    base = manifest.get("base") or {}
    if adapter.get("format") != "mlx-lora-safetensors":
        raise ValueError("source adapter must use mlx-lora-safetensors")
    if base.get("modelId") != LITE_TRAIN_BASE_ID or base.get("revision") != LITE_TRAIN_BASE_REVISION:
        raise ValueError("source adapter is not bound to the pinned Scion Lite base")
    if base.get("exactRevisionRequired") is not True:
        raise ValueError("source adapter does not require the exact base revision")
    expected = {entry.get("path"): entry for entry in manifest.get("files") or []}
    verified = []
    for name in ("adapter_config.json", "adapters.safetensors"):
        descriptor = expected.get(name)
        path = adapter_dir / name
        if not descriptor or not path.is_file():
            raise ValueError(f"source manifest is missing {name}")
        actual = file_record(path, adapter_dir)
        if actual["bytes"] != descriptor.get("bytes") or actual["sha256"] != descriptor.get("sha256"):
            raise ValueError(f"source adapter integrity mismatch: {name}")
        verified.append(actual)
    return manifest, {"manifestSha256": sha256_file(manifest_path), "files": verified}


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from safetensors.numpy import save_file

    adapter_dir = args.mlx_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.source_manifest.resolve()
    manifest, source = require_source_manifest(manifest_path, adapter_dir)
    mlx_config = load_json(adapter_dir / "adapter_config.json")
    params = mlx_config.get("lora_parameters") or {}
    rank, scale, dropout = params.get("rank"), params.get("scale"), params.get("dropout")
    declared_keys = params.get("keys")
    if mlx_config.get("fine_tune_type") != "lora":
        raise ValueError("MLX adapter is not a LoRA")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("MLX adapter rank must be a positive integer")
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("MLX adapter scale must be positive")
    if not isinstance(declared_keys, list) or not declared_keys:
        raise ValueError("MLX adapter must declare its LoRA keys")

    tensors = mx.load(str(adapter_dir / "adapters.safetensors"))
    lora_names = sorted(name for name in tensors if name.endswith((".lora_a", ".lora_b")))
    ignored_names = sorted(name for name in tensors if name not in lora_names)
    if not lora_names:
        raise ValueError("MLX adapter contains no LoRA tensors")

    by_stem: dict[str, set[str]] = {}
    output_tensors: dict[str, Any] = {}
    mapped_sources = []
    for name in lora_names:
        output_name, side = map_mlx_lora_name(name)
        stem = name.rsplit(".lora_", 1)[0]
        by_stem.setdefault(stem, set()).add(side)
        value = tensors[name]
        if len(value.shape) != 2:
            raise ValueError(f"LoRA tensor must be rank 2: {name}")
        if value.shape[1 if side == "A" else 0] != rank:
            raise ValueError(f"LoRA rank mismatch for {name}: {tuple(value.shape)}")
        converted = np.asarray(value.T.astype(mx.float32))
        if output_name in output_tensors:
            raise ValueError(f"duplicate converted tensor name: {output_name}")
        output_tensors[output_name] = converted
        mapped_sources.append(
            {
                "source": name,
                "target": output_name,
                "side": side,
                "sourceShape": list(value.shape),
                "targetShape": list(converted.shape),
            }
        )

    incomplete = sorted(stem for stem, sides in by_stem.items() if sides != {"A", "B"})
    if incomplete:
        raise ValueError(f"incomplete LoRA A/B pairs: {', '.join(incomplete[:5])}")
    declared, observed = {str(value) for value in declared_keys}, set(by_stem)
    if declared != observed:
        raise ValueError(
            f"MLX config/tensor key mismatch: missing={sorted(declared - observed)[:3]} "
            f"unexpected={sorted(observed - declared)[:3]}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    weights_path = output_dir / "adapter_model.safetensors"
    config_path = output_dir / "adapter_config.json"
    receipt_path = output_dir / "mlx-to-peft-receipt.json"
    save_file(output_tensors, str(weights_path), metadata={"format": "pt"})
    target_modules = sorted({stem.rsplit(".", 1)[-1] for stem in observed})
    peft_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": LITE_TRAIN_BASE_ID,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": float(rank * scale),
        "lora_bias": False,
        "lora_dropout": float(dropout or 0),
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "qalora_group_size": 16,
        "r": rank,
        "rank_pattern": {},
        "revision": LITE_TRAIN_BASE_REVISION,
        "target_modules": target_modules,
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    config_path.write_text(json.dumps(peft_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schemaVersion": 1,
        "conversion": "scion-mlx-lora-to-peft",
        "source": {
            "adapterId": manifest.get("adapter", {}).get("id"),
            "format": manifest.get("adapter", {}).get("format"),
            "manifestSha256": source["manifestSha256"],
            "files": source["files"],
            "totalTensorCount": len(tensors),
            "ignoredNonLoraTensorCount": len(ignored_names),
        },
        "base": {
            "modelId": LITE_TRAIN_BASE_ID,
            "revision": LITE_TRAIN_BASE_REVISION,
            "exactRevisionRequired": True,
        },
        "lora": {
            "rank": rank,
            "scale": float(scale),
            "alpha": float(rank * scale),
            "pairCount": len(by_stem),
            "tensorCount": len(output_tensors),
            "dtype": "float32",
            "transpose": "mlx-input-rank-to-peft-rank-input",
            "targetModules": target_modules,
        },
        "output": {
            "format": "peft-safetensors",
            "files": [file_record(config_path, output_dir), file_record(weights_path, output_dir)],
        },
        "mappingSha256": hashlib.sha256(
            json.dumps(mapped_sources, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def self_test() -> None:
    assert map_mlx_lora_name("language_model.model.layers.0.self_attn.q_proj.lora_a") == (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "A",
    )
    try:
        map_mlx_lora_name("audio_tower.layers.0.q_proj.lora_a")
    except ValueError:
        pass
    else:
        raise AssertionError("audio LoRA must be rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-dir", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and not all((args.mlx_dir, args.source_manifest, args.output_dir)):
        parser.error("--mlx-dir, --source-manifest, and --output-dir are required")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
        print(json.dumps({"status": "pass", "test": "scion-mlx-lora-name-contract"}))
    else:
        print(json.dumps(convert(arguments), indent=2, sort_keys=True))
