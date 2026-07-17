"""Assemble a portable, checksum-only inventory for a Scion research release."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import MAX_SCION_ARTIFACT_BYTES, SCION_VERSION
from .model_registry import registry_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _package(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "scion-adapter.json"
    manifest = _json(manifest_path)
    verified = []
    for expected in manifest.get("files", []):
        path = directory / expected["path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular package file: {path}")
        actual = {"path": expected["path"], "bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual != expected:
            raise RuntimeError(f"package integrity mismatch: {path}")
        verified.append(actual)
    adapter_files = [
        record for record in verified if record["path"].endswith(("adapters.safetensors", ".gguf"))
    ]
    if not adapter_files or any(record["bytes"] >= MAX_SCION_ARTIFACT_BYTES for record in adapter_files):
        raise RuntimeError(f"package violates the adapter delivery cap: {directory}")
    return {
        "adapterId": manifest["adapter"]["id"],
        "format": manifest["adapter"]["format"],
        "base": manifest["base"],
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "files": verified,
    }


def build_release_manifest(
    *,
    repo_root: Path,
    package_dirs: list[Path],
    comparison_paths: list[Path],
    dataset_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    comparisons = []
    for path in comparison_paths:
        report = _json(path)
        if report.get("status") != "pass":
            raise RuntimeError(f"cannot release a failed paired comparison: {path}")
        comparisons.append(
            {
                "tier": report["adapter"]["tier"],
                "sha256": _sha256(path),
                "checks": report["checks"],
                "deltas": report["deltas"],
            }
        )
    dataset = _json(dataset_manifest_path)
    if dataset.get("status") != "research-ready":
        raise RuntimeError("release requires a research-ready dataset")
    cards = []
    for name in ("README.md", "MODEL_CARD.md", "DATASET_CARD.md", "THIRD_PARTY_NOTICES.md", "LICENSE"):
        path = repo_root / name
        cards.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest = {
        "schemaVersion": 1,
        "protocol": "scion-research-release-v1",
        "version": SCION_VERSION,
        "status": "research",
        "promotable": False,
        "generatedAt": datetime.now(UTC).isoformat(),
        "externalSpendUsd": 0,
        "closedApiOutputUsed": False,
        "optional122BTeacherUsed": False,
        "dataset": {
            "status": dataset["status"],
            "identitySha256": dataset["identity"]["sha256"],
            "manifestSha256": _sha256(dataset_manifest_path),
            "counts": dataset["counts"],
        },
        "models": registry_json(),
        "packages": [_package(path) for path in package_dirs],
        "comparisons": comparisons,
        "documentation": cards,
        "limits": {"adapterBytesLessThan": MAX_SCION_ARTIFACT_BYTES},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
