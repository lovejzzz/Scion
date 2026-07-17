"""Convert the Scion Lite MLX LoRA into CourseMapper's browser GGUF package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .constants import (
    BROWSER_CONVERSION_PIPELINE,
    COURSEMAPPER_SOURCE_REVISION,
    LITE_RUNTIME_BASE_BYTES,
    LITE_TRAIN_BASE_ID,
    LITE_TRAIN_BASE_REVISION,
    LLAMA_CPP_LORA_CONVERTER_SHA256,
    LLAMA_CPP_REVISION,
    MAX_LITE_BROWSER_ARTIFACT_BYTES,
    SCION_VERSION,
)
from .packaging import file_record, sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require_pinned_llama_cpp(llama_cpp_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=llama_cpp_dir, text=True).strip()

    converter = llama_cpp_dir / "convert_lora_to_gguf.py"
    revision = git("rev-parse", "HEAD")
    if revision != LLAMA_CPP_REVISION:
        raise RuntimeError(f"llama.cpp revision mismatch: {revision}")
    if git("status", "--porcelain"):
        raise RuntimeError("pinned llama.cpp checkout is dirty")
    if sha256_file(converter) != LLAMA_CPP_LORA_CONVERTER_SHA256:
        raise RuntimeError("llama.cpp converter digest mismatch")
    return {
        "converter": converter,
        "dump": llama_cpp_dir / "gguf-py/gguf/scripts/gguf_dump.py",
        "pythonPath": llama_cpp_dir / "gguf-py",
    }


def _audit_gguf(
    *, gguf_path: Path, llama: dict[str, Any], expected_tensor_count: int, expected_alpha: float
) -> dict[str, Any]:
    environment = {**os.environ, "PYTHONPATH": str(llama["pythonPath"])}
    result = subprocess.run(
        [sys.executable, str(llama["dump"]), "--json", str(gguf_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    dump = json.loads(result.stdout)
    metadata, tensors = dump.get("metadata", {}), dump.get("tensors", {})

    def value(key: str) -> Any:
        item = metadata.get(key, {})
        return item.get("value") if isinstance(item, dict) else None

    names = sorted(tensors)
    stems: dict[str, set[str]] = {}
    for name in names:
        if name.endswith(".lora_a"):
            stem, side = name[: -len(".lora_a")], "a"
        elif name.endswith(".lora_b"):
            stem, side = name[: -len(".lora_b")], "b"
        else:
            raise RuntimeError(f"GGUF contains a non-LoRA tensor: {name}")
        if tensors[name].get("type") != "F16":
            raise RuntimeError(f"GGUF tensor is not F16: {name}")
        stems.setdefault(stem, set()).add(side)
    checks = {
        "version": value("GGUF.version") == 3,
        "tensorCount": value("GGUF.tensor_count") == expected_tensor_count == len(names),
        "architecture": value("general.architecture") == "gemma4",
        "type": value("general.type") == "adapter",
        "adapterType": value("adapter.type") == "lora",
        "alpha": value("adapter.lora.alpha") == expected_alpha,
        "completePairs": all(sides == {"a", "b"} for sides in stems.values()),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"GGUF semantic audit failed: {', '.join(failed)}")
    return {
        "status": "pass",
        "metadata": {
            "version": value("GGUF.version"),
            "architecture": value("general.architecture"),
            "type": value("general.type"),
            "adapterType": value("adapter.type"),
            "alpha": value("adapter.lora.alpha"),
        },
        "tensorCount": len(names),
        "pairCount": len(stems),
        "tensorType": "F16",
    }


def build_browser_adapter(
    *,
    source_manifest_path: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    base_dir: Path,
    llama_cpp_dir: Path,
    bridge_path: Path = Path("scripts/convert_mlx_lora_to_peft.py"),
    inference_scale: float = 16,
) -> dict[str, Any]:
    if not 0.05 <= inference_scale <= 16:
        raise ValueError("inference_scale must be between 0.05 and 16")
    source_manifest_path = source_manifest_path.resolve()
    dataset_manifest_path = dataset_manifest_path.resolve()
    output_dir = output_dir.resolve()
    base_dir = base_dir.resolve()
    llama_cpp_dir = llama_cpp_dir.resolve()
    bridge_path = bridge_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"browser output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _read_json(source_manifest_path)
    dataset = _read_json(dataset_manifest_path)
    if source.get("adapter", {}).get("format") != "mlx-lora-safetensors":
        raise RuntimeError("browser conversion requires an MLX LoRA source")
    if (
        source.get("base", {}).get("modelId") != LITE_TRAIN_BASE_ID
        or source.get("base", {}).get("revision") != LITE_TRAIN_BASE_REVISION
    ):
        raise RuntimeError("browser conversion requires the exact Scion Lite base")
    if source.get("training", {}).get("datasetManifestSha256") != sha256_file(dataset_manifest_path):
        raise RuntimeError("source adapter and dataset manifest do not match")
    base_config = _read_json(base_dir / "config.json")
    if base_dir.name != LITE_TRAIN_BASE_REVISION or base_config.get("model_type") != "gemma4":
        raise RuntimeError("conversion base is not the pinned Gemma 4 E2B snapshot")
    llama = _require_pinned_llama_cpp(llama_cpp_dir)

    source_id = source["adapter"]["id"]
    adapter_id = f"{source_id}-browser"
    gguf_path = output_dir / f"{adapter_id}.gguf"
    with tempfile.TemporaryDirectory(prefix="scion-browser-") as temporary:
        peft_dir = Path(temporary) / "peft"
        subprocess.run(
            [
                sys.executable,
                str(bridge_path),
                "--mlx-dir",
                str(source_manifest_path.parent),
                "--source-manifest",
                str(source_manifest_path),
                "--output-dir",
                str(peft_dir),
            ],
            check=True,
        )
        peft_receipt_path = peft_dir / "mlx-to-peft-receipt.json"
        peft = _read_json(peft_receipt_path)
        if peft.get("source", {}).get("manifestSha256") != sha256_file(source_manifest_path):
            raise RuntimeError("PEFT bridge receipt is not bound to the source adapter")
        subprocess.run(
            [
                sys.executable,
                str(llama["converter"]),
                "--base",
                str(base_dir),
                "--outfile",
                str(gguf_path),
                "--outtype",
                "f16",
                str(peft_dir),
            ],
            check=True,
        )
        with gguf_path.open("rb") as handle:
            prefix = handle.read(4)
        if prefix != b"GGUF":
            raise RuntimeError("llama.cpp output is not a GGUF file")
        gguf_audit = _audit_gguf(
            gguf_path=gguf_path,
            llama=llama,
            expected_tensor_count=peft["lora"]["tensorCount"],
            expected_alpha=peft["lora"]["alpha"],
        )
        bridge_receipt = {
            "bytes": peft_receipt_path.stat().st_size,
            "sha256": sha256_file(peft_receipt_path),
        }

    source_manifest_sha = sha256_file(source_manifest_path)
    conversion_receipt_path = output_dir / "conversion-receipt.json"
    conversion_receipt = {
        "schemaVersion": 1,
        "conversion": BROWSER_CONVERSION_PIPELINE,
        "source": {
            "adapterId": source_id,
            "adapterFormat": source["adapter"]["format"],
            "adapterManifestSha256": source_manifest_sha,
            "datasetManifestSha256": sha256_file(dataset_manifest_path),
            "promotionStatus": source["promotion"]["status"],
        },
        "base": {
            "modelId": LITE_TRAIN_BASE_ID,
            "revision": LITE_TRAIN_BASE_REVISION,
            "architecture": "gemma4",
            "role": "instruction",
            "exactRevisionRequired": True,
        },
        "bridge": {
            "id": "scion-mlx-lora-to-peft",
            "receipt": bridge_receipt,
            "lora": peft["lora"],
            "mappingSha256": peft["mappingSha256"],
        },
        "converter": {
            "id": "ggml-org/llama.cpp/convert_lora_to_gguf.py",
            "revision": LLAMA_CPP_REVISION,
            "sha256": LLAMA_CPP_LORA_CONVERTER_SHA256,
            "outputType": "f16",
        },
        "output": {
            "format": "gguf-lora",
            "file": file_record(gguf_path, output_dir),
            "audit": gguf_audit,
        },
        "inference": {"scale": inference_scale},
    }
    conversion_receipt_path.write_text(
        json.dumps(conversion_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for source_path, name in (
        (source_manifest_path.parent / "training-plan.json", "training-plan.json"),
        (source_manifest_path.parent / "training-result.json", "training-result.json"),
        (source_manifest_path.parent / "token-audit.json", "token-audit.json"),
        (source_manifest_path, "source-adapter-manifest.json"),
    ):
        if not source_path.is_file():
            raise RuntimeError(f"missing inherited training receipt: {source_path}")
        shutil.copy2(source_path, output_dir / name)

    package_paths = [
        gguf_path,
        conversion_receipt_path,
        output_dir / "training-plan.json",
        output_dir / "training-result.json",
        output_dir / "token-audit.json",
        output_dir / "source-adapter-manifest.json",
    ]
    files = [file_record(path, output_dir) for path in package_paths]
    total_bytes = sum(entry["bytes"] for entry in files)
    if total_bytes > MAX_LITE_BROWSER_ARTIFACT_BYTES:
        raise RuntimeError(f"browser package exceeds 64 MiB: {total_bytes}")
    if total_bytes / LITE_RUNTIME_BASE_BYTES > 0.02:
        raise RuntimeError("browser package exceeds two percent of the pinned runtime base")
    counts = dataset["counts"]
    plan, result = (
        _read_json(output_dir / "training-plan.json"),
        _read_json(output_dir / "training-result.json"),
    )
    manifest = {
        "schemaVersion": 3,
        "adapter": {
            "id": adapter_id,
            "scionVersion": SCION_VERSION,
            "format": "gguf-lora",
            "scale": inference_scale,
        },
        "base": {
            "modelId": LITE_TRAIN_BASE_ID,
            "revision": LITE_TRAIN_BASE_REVISION,
            "architecture": "gemma4",
            "role": "instruction",
            "exactRevisionRequired": True,
        },
        "training": {
            "method": "orpo-lora",
            "datasetManifestSha256": sha256_file(dataset_manifest_path),
            "datasetIdentitySha256": dataset["identity"]["sha256"],
            "datasetStatus": dataset["status"],
            "primaryPreferenceEvidence": dataset["primaryPreferenceEvidence"],
            "pairCount": counts["total"],
            "domainCount": counts["domains"],
            "groupCount": counts["groups"],
            "instructorPairCount": counts["blindInstructorPairs"],
            "instructorDomainCount": counts["blindInstructorDomains"],
            "modelJudgePairCount": counts["singleModelJudgePairs"],
            "modelJudgeDomainCount": counts["singleModelJudgeDomains"],
            "domainGroupCounts": dataset["domainGroupCounts"],
            "instructorDomainCounts": dataset["instructorDomainCounts"],
            "modelJudgeDomainCounts": dataset["modelJudgeDomainCounts"],
            "splitCounts": {key: counts[key] for key in ("train", "valid", "test")},
            "splitDomainCounts": {
                "train": counts["trainDomains"],
                "valid": counts["validDomains"],
                "test": counts["testDomains"],
            },
            "run": {
                "protocol": plan["protocol"],
                "lane": plan["lane"],
                "seed": plan["trainer"]["seed"],
                "planPath": "training-plan.json",
                "planSha256": sha256_file(output_dir / "training-plan.json"),
                "planIdentitySha256": plan["identity"]["sha256"],
                "resultPath": "training-result.json",
                "resultSha256": sha256_file(output_dir / "training-result.json"),
                "resultIdentitySha256": result["identity"]["sha256"],
                "datasetIdentitySha256": dataset["identity"]["sha256"],
                "toolchainPolicySha256": plan["toolchain"]["policySha256"],
                "repositoryCommit": plan["repository"]["commit"],
                "repositoryTree": plan["repository"]["tree"],
                "repositoryDirty": False,
                "sourceAdapterId": source_id,
                "sourceManifestPath": "source-adapter-manifest.json",
                "sourceManifestSha256": source_manifest_sha,
            },
        },
        "files": files,
        "runtime": {"supported": ["scion-wllama-webgpu-jspi-v1"]},
        "promotion": {
            "status": "research",
            "promotable": False,
            "evidence": [
                {
                    "type": "conversion-receipt",
                    "status": "pass",
                    "sha256": sha256_file(conversion_receipt_path),
                }
            ],
        },
        "conversion": {
            "pipeline": BROWSER_CONVERSION_PIPELINE,
            "sourceAdapterId": source_id,
            "sourceManifestSha256": source_manifest_sha,
            "receiptPath": "conversion-receipt.json",
            "converter": conversion_receipt["converter"],
        },
        "limits": {
            "totalBytes": total_bytes,
            "browserCapBytes": MAX_LITE_BROWSER_ARTIFACT_BYTES,
            "runtimeBaseFraction": total_bytes / LITE_RUNTIME_BASE_BYTES,
        },
        "compatibility": {"courseMapperRevision": COURSEMAPPER_SOURCE_REVISION},
    }
    manifest_path = output_dir / "scion-adapter.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
