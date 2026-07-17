"""Create a release manifest only after all Scion promotion gates pass."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    MAX_SCION_ARTIFACT_BYTES,
    PRISM_LLAMA_CPP_REVISION,
    SCION_MODEL_ID,
    SCION_VERSION,
    SERVE_BASE_FILE,
    SERVE_BASE_ID,
    SERVE_BASE_REVISION,
    SERVE_BASE_SHA256,
    TRAIN_BASE_ID,
    TRAIN_BASE_REVISION,
)
from .dataset import sha256_file
from .server import validate_adapter


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _portable_values(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _portable_values(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_values(item, root) for item in value]
    if isinstance(value, str) and value.startswith(str(root)):
        return Path(value).relative_to(root).as_posix()
    return value


def build_release_manifest(
    *,
    adapter: Path,
    dataset_manifest: Path,
    training_result: Path,
    conversion_receipt: Path,
    comparison: Path,
    smoke: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    adapter_record = validate_adapter(adapter)
    adapter_record["path"] = _portable(adapter, root)
    records = {
        "dataset": _load(dataset_manifest),
        "training": _load(training_result),
        "conversion": _load(conversion_receipt),
        "comparison": _load(comparison),
        "smoke": _load(smoke),
    }
    if records["training"].get("status") != "trained-unpromoted":
        raise RuntimeError("training receipt is not eligible for promotion")
    if records["conversion"].get("status") != "converted-unpromoted":
        raise RuntimeError("conversion receipt is not eligible for promotion")
    if records["comparison"].get("status") != "pass" or records["smoke"].get("status") != "pass":
        raise RuntimeError("evaluation and CourseMapper smoke gates must pass before release")
    if records["conversion"]["artifact"]["sha256"] != adapter_record["sha256"]:
        raise RuntimeError("adapter identity does not match the conversion receipt")
    if int(adapter_record["bytes"]) >= MAX_SCION_ARTIFACT_BYTES:
        raise RuntimeError("adapter violates the Scion artifact cap")

    receipt_paths = {
        "datasetManifest": dataset_manifest,
        "trainingResult": training_result,
        "conversionReceipt": conversion_receipt,
        "evaluationComparison": comparison,
        "courseMapperSmoke": smoke,
    }
    receipt_output = output.parent / "receipts"
    receipt_output.mkdir(parents=True, exist_ok=True)
    published_receipts = {}
    for name, source in receipt_paths.items():
        destination = receipt_output / f"{name}.json"
        destination.write_text(
            json.dumps(_portable_values(_load(source), root), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published_receipts[name] = destination
    manifest = {
        "schemaVersion": 1,
        "status": "promoted",
        "createdAt": datetime.now(UTC).isoformat(),
        "model": {"id": SCION_MODEL_ID, "name": "Scion Bonsai 27B", "version": SCION_VERSION},
        "delivery": {
            "type": "GGUF LoRA adapter",
            "artifact": adapter_record,
            "capBytes": MAX_SCION_ARTIFACT_BYTES,
            "packageBytes": 0,
            "baseDownloadedSeparately": True,
        },
        "trainingBase": {"modelId": TRAIN_BASE_ID, "revision": TRAIN_BASE_REVISION},
        "servingBase": {
            "modelId": SERVE_BASE_ID,
            "revision": SERVE_BASE_REVISION,
            "file": SERVE_BASE_FILE,
            "sha256": SERVE_BASE_SHA256,
        },
        "runtime": {"repository": "PrismML-Eng/llama.cpp", "revision": PRISM_LLAMA_CPP_REVISION},
        "receipts": {
            name: {"path": _portable(path, root), "sha256": sha256_file(path)}
            for name, path in published_receipts.items()
        },
        "evaluation": _portable_values(records["comparison"]["adapter"], root),
        "limitations": records["dataset"].get("limitations", []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_bytes = sum(path.stat().st_size for path in published_receipts.values())
    for _ in range(3):
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["delivery"]["packageBytes"] = (
            int(adapter_record["bytes"]) + receipt_bytes + output.stat().st_size
        )
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    actual_package_bytes = int(adapter_record["bytes"]) + receipt_bytes + output.stat().st_size
    if actual_package_bytes != manifest["delivery"]["packageBytes"]:
        raise RuntimeError("release package size did not stabilize")
    if actual_package_bytes >= MAX_SCION_ARTIFACT_BYTES:
        raise RuntimeError(f"complete Scion release package exceeds the 1 GB cap: {actual_package_bytes}")
    return manifest
