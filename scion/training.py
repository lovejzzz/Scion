"""Receipt-bound Gemma 4 ORPO training for Scion Lite and Pro."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .constants import (
    MAX_LITE_BROWSER_ARTIFACT_BYTES,
    MAX_SCION_ARTIFACT_BYTES,
    MLX_LM_VERSION,
    MLX_VERSION,
    MLX_VLM_VERSION,
    TRANSFORMERS_VERSION,
)
from .model_registry import student_pin
from .token_audit import audit_orpo_lengths


def _stable_json(value: Any) -> str:
    """Match CourseMapper's canonical JSON, including JavaScript number spelling."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical training identities require finite numbers")
        if value == 0:
            return "0"
        raw = repr(value).lower()
        decimal = Decimal(raw)
        magnitude = abs(decimal)
        if Decimal("1e-6") <= magnitude < Decimal("1e21"):
            rendered = format(decimal, "f")
            return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
        if "e" not in raw:
            raw = format(decimal.normalize(), "e")
        mantissa, exponent = raw.split("e", 1)
        if "." in mantissa:
            mantissa = mantissa.rstrip("0").rstrip(".")
        exponent_value = int(exponent)
        exponent_sign = "+" if exponent_value >= 0 else ""
        return f"{mantissa}e{exponent_sign}{exponent_value}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical training identities require string object keys")
        return "{" + ",".join(
            f"{_stable_json(key)}:{_stable_json(value[key])}" for key in sorted(value)
        ) + "}"
    raise TypeError(f"unsupported canonical identity value: {type(value).__name__}")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _repository_identity(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("training repository must be clean and committed before a run")
    return {
        "commit": git("rev-parse", "HEAD^{commit}"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def student_model_type(tier: str) -> str:
    try:
        return {"lite": "gemma4", "pro": "gemma4_unified"}[tier]
    except KeyError as error:
        raise ValueError("tier must be lite or pro") from error


def resolve_student_snapshot(tier: str, cache_dir: Path, *, local_files_only: bool = False) -> Path:
    pin = student_pin(tier)
    path = Path(
        snapshot_download(
            pin.model_id,
            revision=pin.revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    ).resolve()
    if path.name != pin.revision:
        raise RuntimeError(f"resolved mutable snapshot: {path}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != student_model_type(tier):
        raise RuntimeError(f"student base is not Gemma 4: {path}")
    return path


def _versions() -> dict[str, str]:
    names = (
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "transformers",
        "datasets",
        "numpy",
        "huggingface-hub",
        "safetensors",
        "pyarrow",
        "tokenizers",
        "torch",
    )
    return {name: importlib.metadata.version(name) for name in names}


def verify_toolchain() -> dict[str, str]:
    found = _versions()
    expected = {
        "mlx": MLX_VERSION,
        "mlx-lm": MLX_LM_VERSION,
        "mlx-vlm": MLX_VLM_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "datasets": "5.0.0",
        "numpy": "2.5.1",
        "huggingface-hub": "1.22.0",
        "safetensors": "0.8.0",
        "pyarrow": "25.0.0",
        "tokenizers": "0.22.2",
        "torch": "2.10.0",
    }
    mismatches = {
        name: (expected[name], found.get(name)) for name in expected if found.get(name) != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"training toolchain mismatch: {mismatches}")
    return found


def _dataset_identity(data_dir: Path) -> dict[str, Any]:
    files = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"missing nonempty ORPO split: {path}")
        files[split] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
        }
    return files


def build_training_plan(
    *,
    tier: str,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    run_dir: Path,
    iterations: int,
    seed: int = 16031,
) -> dict[str, Any]:
    if tier not in {"lite", "pro"}:
        raise ValueError("tier must be lite or pro")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    pin = student_pin(tier)
    if model_path.resolve().name != pin.revision:
        raise RuntimeError("training model does not match pinned student base")
    dataset = _dataset_identity(data_dir)
    manifest_path = data_dir / "dataset-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing dataset manifest: {manifest_path}")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("status") != "research-ready":
        raise RuntimeError("training requires a research-ready dataset manifest")
    repository = _repository_identity(Path.cwd())
    hyperparameters = {
        "trainingMode": "orpo",
        "split": "train",
        "validationSplit": "validation",
        "iterations": iterations,
        "batchSize": 1,
        "learningRate": 0.00002,
        "stepsPerReport": 1 if iterations <= 10 else 20,
        "stepsPerEval": 5 if iterations <= 10 else 100,
        "stepsPerSave": max(5, min(100, iterations)),
        "validationBatches": 4,
        "maxSequenceLength": 2048,
        "gradientCheckpointing": True,
        "gradientAccumulationSteps": 2,
        "loraRank": 8 if tier == "lite" else 16,
        "loraAlpha": 16,
        "loraDropout": 0,
        "beta": 0.1,
        "epsilon": 1e-8,
    }
    verify_toolchain()
    toolchain_receipt = json.loads(
        subprocess.check_output(
            [sys.executable, "scripts/seeded_mlx_vlm_lora.py", "--inspect-toolchain"],
            cwd=Path.cwd(),
            text=True,
        )
    )
    toolchain_policy_sha256 = _sha256_value(
        {
            "protocol": "scion-mlx-orpo-toolchain-v1",
            "platform": toolchain_receipt["platform"],
            "packages": toolchain_receipt["packages"],
        }
    )
    plan = {
        "schemaVersion": 1,
        "protocol": "scion-adapter-training-run-v1",
        "status": "planned",
        "lane": "research",
        "scionVersion": "3.0.0",
        "startedAt": datetime.now(UTC).isoformat(),
        "repository": repository,
        "base": {
            "modelId": pin.model_id,
            "revision": pin.revision,
            "architecture": student_model_type(tier),
            "role": "instruction",
            "exactRevisionRequired": True,
            "snapshotRevision": model_path.resolve().name,
        },
        "dataset": {
            "manifestSha256": sha256_file(manifest_path),
            "identitySha256": dataset_manifest["identity"]["sha256"],
            "status": dataset_manifest["status"],
            "primaryPreferenceEvidence": dataset_manifest["primaryPreferenceEvidence"],
            "counts": {
                key: dataset_manifest["counts"][key]
                for key in ("total", "domains", "groups", "train", "valid", "test")
            },
            "files": dataset,
        },
        "toolchain": {
            "policySha256": toolchain_policy_sha256,
            "receipt": toolchain_receipt,
        },
        "trainer": {
            "entrypoint": "scripts/seeded_mlx_vlm_lora.py",
            "seed": seed,
            "hyperparameters": hyperparameters,
        },
        "output": str(output_dir.resolve()),
        "artifactCapBytes": MAX_LITE_BROWSER_ARTIFACT_BYTES if tier == "lite" else MAX_SCION_ARTIFACT_BYTES,
    }
    identity = _sha256_value(
        {
            key: plan[key]
            for key in (
                "protocol",
                "lane",
                "scionVersion",
                "repository",
                "base",
                "dataset",
                "toolchain",
                "trainer",
            )
        }
    )
    prefix = "scion-g4e2b" if tier == "lite" else "scion-g4-12b"
    adapter_id = f"{prefix}-research-{identity[:16]}"
    plan["identity"] = {
        "algorithm": "sha256-canonical-training-plan-v1",
        "sha256": identity,
    }
    plan["adapter"] = {"id": adapter_id, "promotionStatus": "research"}
    plan["runId"] = adapter_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def train_student(
    *,
    tier: str,
    data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    run_dir: Path,
    iterations: int,
    python: str = sys.executable,
    local_files_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Scion training requires Apple Silicon macOS")
    model_path = resolve_student_snapshot(tier, cache_dir, local_files_only=local_files_only)
    token_audit_path = output_dir / "token-audit.json"
    audit_orpo_lengths(
        model_path=model_path,
        data_dir=data_dir,
        output_path=token_audit_path,
        max_sequence_length=2048,
    )
    plan = build_training_plan(
        tier=tier,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        run_dir=run_dir,
        iterations=iterations,
    )
    plan["trainer"]["tokenAudit"] = {
        "path": "token-audit.json",
        "sha256": sha256_file(token_audit_path),
        "status": "pass",
    }
    plan["identity"]["sha256"] = _sha256_value(
        {
            key: plan[key]
            for key in (
                "protocol",
                "lane",
                "scionVersion",
                "repository",
                "base",
                "dataset",
                "toolchain",
                "trainer",
            )
        }
    )
    prefix = "scion-g4e2b" if tier == "lite" else "scion-g4-12b"
    plan["adapter"]["id"] = f"{prefix}-research-{plan['identity']['sha256'][:16]}"
    plan["runId"] = plan["adapter"]["id"]
    (output_dir / "training-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = plan["trainer"]["hyperparameters"]
    command = [
        python,
        "scripts/seeded_mlx_vlm_lora.py",
        "--scion-seed",
        str(plan["trainer"]["seed"]),
        "--scion-validation-split",
        "validation",
        "--",
        "--model-path",
        str(model_path),
        "--dataset",
        str(data_dir.resolve()),
        "--split",
        "train",
        "--train-mode",
        "orpo",
        "--iters",
        str(iterations),
        "--batch-size",
        str(config["batchSize"]),
        "--learning-rate",
        str(config["learningRate"]),
        "--steps-per-report",
        str(config["stepsPerReport"]),
        "--steps-per-eval",
        str(config["stepsPerEval"]),
        "--steps-per-save",
        str(config["stepsPerSave"]),
        "--val-batches",
        str(config["validationBatches"]),
        "--max-seq-length",
        str(config["maxSequenceLength"]),
        "--grad-checkpoint",
        "--gradient-accumulation-steps",
        str(config["gradientAccumulationSteps"]),
        "--lora-rank",
        str(config["loraRank"]),
        "--lora-alpha",
        str(config["loraAlpha"]),
        "--lora-dropout",
        str(config["loraDropout"]),
        "--beta",
        str(config["beta"]),
        "--eps",
        str(config["epsilon"]),
        "--output-path",
        str(output_dir.resolve()),
    ]
    if dry_run:
        return {"status": "dry-run", "plan": plan, "command": command}
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"training failed with exit {code}; see {log_path}")
    files = [output_dir / "adapter_config.json", output_dir / "adapters.safetensors"]
    if any(not path.is_file() for path in files):
        raise RuntimeError("trainer did not create a complete adapter")
    total = sum(path.stat().st_size for path in files)
    if total > plan["artifactCapBytes"]:
        raise RuntimeError(f"adapter exceeds tier cap: {total} > {plan['artifactCapBytes']}")
    plan_path = output_dir / "training-plan.json"
    plan_sha256 = sha256_file(plan_path)
    result = {
        "schemaVersion": 1,
        "protocol": "scion-adapter-training-result-v1",
        "status": "completed",
        "adapterId": plan["adapter"]["id"],
        "planSha256": plan_sha256,
        "planIdentitySha256": plan["identity"]["sha256"],
        "completedAt": datetime.now(UTC).isoformat(),
        "log": {
            "path": "training.log",
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
            "retainedLocally": True,
            "includedInPackage": False,
        },
        "artifactBytes": total,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files
        ],
    }
    result["identity"] = {
        "algorithm": "sha256-canonical-training-result-v1",
        "sha256": _sha256_value(
            {
                "protocol": result["protocol"],
                "planSha256": result["planSha256"],
                "planIdentitySha256": result["planIdentitySha256"],
                "adapterId": result["adapterId"],
                "files": result["files"],
                "log": result["log"],
            }
        ),
    }
    (output_dir / "training-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
