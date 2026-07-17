"""Prepare the exact Bonsai checkpoint and launch receipt-bound MLX QLoRA."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .constants import (
    MAX_SCION_ARTIFACT_BYTES,
    MLX_LM_REVISION,
    MLX_LM_VERSION,
    SCION_VERSION,
    TRAIN_BASE_ARCHITECTURE,
    TRAIN_BASE_ID,
    TRAIN_BASE_REVISION,
)
from .dataset import sha256_file


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX QLoRA preparation and training require an Apple Silicon Mac")


def resolve_training_snapshot(cache_dir: Path, *, local_files_only: bool = False) -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=TRAIN_BASE_ID,
            revision=TRAIN_BASE_REVISION,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    ).resolve()
    if snapshot.name != TRAIN_BASE_REVISION:
        raise RuntimeError(f"resolved mutable or unexpected base snapshot: {snapshot}")
    config = _json(snapshot / "config.json")
    if config.get("architectures") != [TRAIN_BASE_ARCHITECTURE] or config.get("model_type") != "qwen3_5":
        raise RuntimeError("resolved checkpoint is not the pinned Bonsai Qwen3.5 architecture")
    return snapshot


def verify_quantized_training_base(path: Path) -> dict[str, Any]:
    config = _json(path / "config.json")
    quantization = (
        (config.get("text_config") or config).get("quantization") or config.get("quantization") or {}
    )
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError("training base model_type must be qwen3_5")
    if quantization.get("bits") != 4 or quantization.get("group_size") != 64:
        raise RuntimeError(f"training base must be MLX affine 4-bit g64, got {quantization}")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("quantized training base has no safetensors weights")
    return {
        "path": str(path.resolve()),
        "modelType": config.get("model_type"),
        "quantization": quantization,
        "weightBytes": sum(weight.stat().st_size for weight in weights),
        "configSha256": sha256_file(path / "config.json"),
    }


def prepare_training_base(
    *,
    output: Path,
    cache_dir: Path,
    python: str = sys.executable,
    local_files_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    _require_apple_silicon()
    if output.exists() and (output / "config.json").is_file():
        return verify_quantized_training_base(output)
    command = [
        python,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        f"{TRAIN_BASE_ID}@{TRAIN_BASE_REVISION}",
        "--mlx-path",
        str(output),
        "--quantize",
        "--q-bits",
        "4",
        "--q-group-size",
        "64",
    ]
    if dry_run:
        return {"status": "dry-run", "command": command}
    # Resolve first so the immutable snapshot identity is verified before the
    # large conversion starts. Passing the local path also prevents a mutable
    # Hub lookup inside mlx_lm.convert.
    snapshot = resolve_training_snapshot(cache_dir, local_files_only=local_files_only)
    command[4] = str(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return verify_quantized_training_base(output)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_training_plan(
    *,
    config_path: Path,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    run_dir: Path,
    iters: int | None = None,
) -> tuple[dict[str, Any], Path]:
    config = _json(config_path)
    config["model"] = str(model_path.resolve())
    config["data"] = str(data_dir.resolve())
    config["adapter_path"] = str(output_dir.resolve())
    if iters is not None:
        if iters <= 0:
            raise ValueError("iterations must be positive")
        config["iters"] = iters
    verify_quantized_training_base(model_path)
    dataset_manifest = _json(data_dir / "manifest.json")
    expected_base = dataset_manifest.get("base") or {}
    if expected_base.get("modelId") != TRAIN_BASE_ID or expected_base.get("revision") != TRAIN_BASE_REVISION:
        raise RuntimeError("dataset manifest is not bound to the pinned Bonsai training base")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "config.resolved.json"
    resolved.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = {
        "schemaVersion": 1,
        "runId": hashlib.sha256(
            (sha256_file(resolved) + sha256_file(data_dir / "manifest.json") + TRAIN_BASE_REVISION).encode()
        ).hexdigest()[:24],
        "createdAt": datetime.now(UTC).isoformat(),
        "scionVersion": SCION_VERSION,
        "base": {
            "modelId": TRAIN_BASE_ID,
            "revision": TRAIN_BASE_REVISION,
            "exactRevisionRequired": True,
            "prepared": verify_quantized_training_base(model_path),
        },
        "dataset": {
            "manifest": str((data_dir / "manifest.json").resolve()),
            "manifestSha256": sha256_file(data_dir / "manifest.json"),
            "counts": dataset_manifest.get("counts"),
        },
        "toolchain": {
            "python": platform.python_version(),
            "mlxLm": _package_version("mlx-lm"),
            "mlxLmExpected": MLX_LM_VERSION,
            "mlxLmRevision": MLX_LM_REVISION,
            "mlx": _package_version("mlx"),
        },
        "config": {"path": str(resolved.resolve()), "sha256": sha256_file(resolved)},
        "output": str(output_dir.resolve()),
        "artifactCapBytes": MAX_SCION_ARTIFACT_BYTES,
    }
    plan_path = run_dir / "training-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan, resolved


def train(
    *,
    config_path: Path,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    run_dir: Path,
    python: str = sys.executable,
    iters: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    _require_apple_silicon()
    installed = _package_version("mlx-lm")
    if installed != MLX_LM_VERSION:
        raise RuntimeError(f"mlx-lm {MLX_LM_VERSION} is required, found {installed or 'not installed'}")
    plan, resolved = build_training_plan(
        config_path=config_path,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        run_dir=run_dir,
        iters=iters,
    )
    command = [python, "-m", "mlx_lm.lora", "--config", str(resolved)]
    if dry_run:
        return {"status": "dry-run", "plan": plan, "command": command}
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"mlx_lm.lora exited with status {return_code}; see {log_path}")
    required = [output_dir / "adapter_config.json", output_dir / "adapters.safetensors"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("training completed without the required adapter files")
    artifact_bytes = sum(path.stat().st_size for path in required)
    if artifact_bytes >= MAX_SCION_ARTIFACT_BYTES:
        raise RuntimeError(f"MLX adapter exceeds the 1 GB Scion cap: {artifact_bytes}")
    result = {
        "schemaVersion": 1,
        "status": "trained-unpromoted",
        "completedAt": datetime.now(UTC).isoformat(),
        "runId": plan["runId"],
        "trainingPlanSha256": sha256_file(run_dir / "training-plan.json"),
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in required
        ],
        "artifactBytes": artifact_bytes,
        "artifactCapBytes": MAX_SCION_ARTIFACT_BYTES,
        "promotion": "requires-base-vs-adapter held-out evaluation and CourseMapper smoke test",
    }
    (run_dir / "training-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
