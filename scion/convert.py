"""Convert an MLX LoRA checkpoint into a llama.cpp-compatible GGUF adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import (
    MAX_SCION_ARTIFACT_BYTES,
    PRISM_LLAMA_CPP_REVISION,
    PRISM_LORA_CONVERTER_SHA256,
    TRAIN_BASE_ID,
    TRAIN_BASE_REVISION,
)
from .dataset import sha256_file
from .training import resolve_training_snapshot


def _converter(runtime: Path) -> Path:
    converter = runtime.resolve() / "convert_lora_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"pinned LoRA converter not found: {converter}")
    if sha256_file(converter) != PRISM_LORA_CONVERTER_SHA256:
        raise RuntimeError("LoRA converter does not match the pinned PrismML revision")
    revision = subprocess.check_output(
        ["git", "-C", str(runtime.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PRISM_LLAMA_CPP_REVISION:
        raise RuntimeError(f"llama.cpp must be at {PRISM_LLAMA_CPP_REVISION}, found {revision}")
    return converter


def _mlx_to_hf_name(name: str) -> str:
    prefix = "language_model.model."
    if not name.startswith(prefix):
        raise ValueError(f"unsupported MLX Bonsai LoRA tensor path: {name}")
    return "model.language_model." + name.removeprefix(prefix)


def _converter_base_config(snapshot: Path, work_dir: Path) -> tuple[Path, dict[str, str]]:
    """Normalize the nested architecture field expected by the pinned converter.

    The upstream Bonsai config declares ``architectures`` at the root while its
    nested ``text_config.architectures`` is null. The pinned PrismML converter
    prefers that nested value without checking for null. A minimal temporary
    config fixes only that representation mismatch; the immutable snapshot is
    never edited.
    """
    original = snapshot / "config.json"
    config = json.loads(original.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    text_config = config.get("text_config")
    if architectures != ["Qwen3_5ForConditionalGeneration"] or not isinstance(text_config, dict):
        raise RuntimeError("unexpected Bonsai config while preparing LoRA conversion metadata")
    normalized = json.loads(json.dumps(config))
    normalized["text_config"]["architectures"] = architectures
    output = work_dir.resolve() / "converter-base-config"
    output.mkdir(parents=True, exist_ok=True)
    normalized_path = output / "config.json"
    normalized_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, {
        "sourceConfigSha256": sha256_file(original),
        "normalizedConfigSha256": sha256_file(normalized_path),
        "normalization": "copied root architectures into null text_config.architectures",
    }


def export_peft_adapter(mlx_adapter: Path, output: Path) -> dict[str, Any]:
    """Transpose MLX LoRA matrices and write the PEFT interchange layout.

    MLX stores A as ``[input, rank]`` and B as ``[rank, output]``. PEFT and
    llama.cpp expect A as ``[rank, input]`` and B as ``[output, rank]``.
    Non-LoRA trainable tensors are deliberately excluded from the deployment
    adapter so the result has an unambiguous LoRA-only contract.
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as error:  # pragma: no cover - exercised by CLI environment check
        raise RuntimeError("conversion requires the `convert` dependency group") from error

    config_path = mlx_adapter / "adapter_config.json"
    weights_path = mlx_adapter / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("MLX adapter must contain adapter_config.json and adapters.safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = config.get("lora_parameters") or {}
    rank = int(parameters.get("rank", 0))
    scale = float(parameters.get("scale", 0))
    if rank <= 0 or scale <= 0:
        raise RuntimeError("MLX adapter config has invalid LoRA rank or scale")

    source = load_file(str(weights_path), device="cpu")
    prefixes = sorted(name.removesuffix(".lora_a") for name in source if name.endswith(".lora_a"))
    if not prefixes:
        raise RuntimeError("MLX checkpoint contains no LoRA matrices")
    converted: dict[str, torch.Tensor] = {}
    modules: set[str] = set()
    for prefix in prefixes:
        a_name = prefix + ".lora_a"
        b_name = prefix + ".lora_b"
        if b_name not in source:
            raise RuntimeError(f"missing LoRA B matrix for {prefix}")
        a = source[a_name]
        b = source[b_name]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != rank or b.shape[0] != rank:
            raise RuntimeError(
                f"unexpected LoRA matrix shapes for {prefix}: A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        hf_name = _mlx_to_hf_name(prefix)
        peft_prefix = "base_model.model." + hf_name
        converted[peft_prefix + ".lora_A.weight"] = a.transpose(0, 1).contiguous()
        converted[peft_prefix + ".lora_B.weight"] = b.transpose(0, 1).contiguous()
        modules.add(hf_name.rsplit(".", 1)[-1])

    output.mkdir(parents=True, exist_ok=True)
    peft_config = {
        "base_model_name_or_path": TRAIN_BASE_ID,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": rank * scale,
        "lora_dropout": float(parameters.get("dropout", 0)),
        "peft_type": "LORA",
        "r": rank,
        "revision": TRAIN_BASE_REVISION,
        "target_modules": sorted(modules),
        "task_type": "CAUSAL_LM",
    }
    (output / "adapter_config.json").write_text(
        json.dumps(peft_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_file(converted, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    return {
        "rank": rank,
        "scale": scale,
        "tensorPairs": len(prefixes),
        "modules": sorted(modules),
        "weightsSha256": sha256_file(output / "adapter_model.safetensors"),
    }


def convert_adapter(
    *,
    mlx_adapter: Path,
    output: Path,
    runtime: Path,
    cache_dir: Path,
    work_dir: Path,
    python: str = sys.executable,
    local_files_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    converter = _converter(runtime)
    base_snapshot = resolve_training_snapshot(cache_dir, local_files_only=local_files_only)
    peft_dir = work_dir.resolve() / "peft-adapter"
    base = work_dir.resolve() / "converter-base-config"
    command = [
        python,
        str(converter),
        "--base",
        str(base),
        "--outfile",
        str(output.resolve()),
        "--outtype",
        "f16",
        str(peft_dir),
    ]
    if dry_run:
        return {"status": "dry-run", "command": command, "peftDirectory": str(peft_dir)}
    base, config_normalization = _converter_base_config(base_snapshot, work_dir)
    command[3] = str(base)
    interchange = export_peft_adapter(mlx_adapter.resolve(), peft_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    if not output.is_file() or output.read_bytes()[:4] != b"GGUF":
        raise RuntimeError("LoRA conversion did not produce a valid GGUF file")
    artifact_bytes = output.stat().st_size
    if artifact_bytes >= MAX_SCION_ARTIFACT_BYTES:
        raise RuntimeError(f"Scion GGUF adapter exceeds the 1 GB cap: {artifact_bytes}")
    receipt = {
        "schemaVersion": 1,
        "status": "converted-unpromoted",
        "base": {"modelId": TRAIN_BASE_ID, "revision": TRAIN_BASE_REVISION},
        "converterConfigNormalization": config_normalization,
        "converter": {"revision": PRISM_LLAMA_CPP_REVISION, "sha256": sha256_file(converter)},
        "interchange": interchange,
        "artifact": {
            "path": str(output.resolve()),
            "bytes": artifact_bytes,
            "sha256": sha256_file(output),
            "capBytes": MAX_SCION_ARTIFACT_BYTES,
        },
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = work_dir / "conversion-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
