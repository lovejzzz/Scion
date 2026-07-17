"""Checksum-bound release packages for Scion's MLX student adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .constants import COURSEMAPPER_SOURCE_REVISION, MAX_SCION_ARTIFACT_BYTES, SCION_VERSION
from .model_registry import student_pin


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"release artifact must be a nonempty regular file: {path}")
    return {"path": relative, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require_training_receipts(
    *, tier: str, adapter_dir: Path, dataset_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_path = adapter_dir / "training-plan.json"
    result_path = adapter_dir / "training-result.json"
    audit_path = adapter_dir / "token-audit.json"
    plan, result, audit = _read_json(plan_path), _read_json(result_path), _read_json(audit_path)
    dataset = _read_json(dataset_manifest_path)
    pin = student_pin(tier)
    issues = []
    if plan.get("protocol") != "scion-adapter-training-run-v1" or plan.get("lane") != "research":
        issues.append("training-plan-protocol")
    if plan.get("repository", {}).get("dirty") is not False:
        issues.append("training-repository-dirty")
    if (
        plan.get("base", {}).get("modelId") != pin.model_id
        or plan.get("base", {}).get("revision") != pin.revision
    ):
        issues.append("training-base")
    if plan.get("dataset", {}).get("manifestSha256") != sha256_file(dataset_manifest_path):
        issues.append("training-dataset-manifest")
    if plan.get("dataset", {}).get("identitySha256") != dataset.get("identity", {}).get("sha256"):
        issues.append("training-dataset-identity")
    if result.get("protocol") != "scion-adapter-training-result-v1" or result.get("status") != "completed":
        issues.append("training-result-protocol")
    if result.get("adapterId") != plan.get("adapter", {}).get("id"):
        issues.append("training-adapter-id")
    if result.get("planSha256") != sha256_file(plan_path):
        issues.append("training-plan-sha256")
    if result.get("planIdentitySha256") != plan.get("identity", {}).get("sha256"):
        issues.append("training-plan-identity")
    if audit.get("status") != "pass":
        issues.append("token-audit-status")
    if plan.get("trainer", {}).get("tokenAudit", {}).get("sha256") != sha256_file(audit_path):
        issues.append("token-audit-sha256")
    for split in ("train", "validation", "test"):
        dataset_path = dataset_manifest_path.parent / f"{split}.jsonl"
        if audit.get("files", {}).get(split, {}).get("sha256") != sha256_file(dataset_path):
            issues.append(f"token-audit-dataset:{split}")
    expected_files = {entry.get("path"): entry for entry in result.get("files", [])}
    for name in ("adapter_config.json", "adapters.safetensors"):
        path = adapter_dir / name
        expected = expected_files.get(name)
        if not expected or not path.is_file():
            issues.append(f"training-file:{name}")
        elif expected.get("bytes") != path.stat().st_size or expected.get("sha256") != sha256_file(path):
            issues.append(f"training-file-integrity:{name}")
    if issues:
        raise RuntimeError(f"cannot package invalid training run: {', '.join(issues)}")
    return plan, result, dataset


def build_mlx_adapter_manifest(
    *, tier: str, adapter_dir: Path, dataset_manifest_path: Path, output_path: Path | None = None
) -> dict[str, Any]:
    """Create the schema-v3 package consumed by CourseMapper's adapter registry."""

    adapter_dir = adapter_dir.resolve()
    dataset_manifest_path = dataset_manifest_path.resolve()
    plan, result, dataset = _require_training_receipts(
        tier=tier, adapter_dir=adapter_dir, dataset_manifest_path=dataset_manifest_path
    )
    pin = student_pin(tier)
    package_paths = [
        adapter_dir / "adapter_config.json",
        adapter_dir / "adapters.safetensors",
        adapter_dir / "training-plan.json",
        adapter_dir / "training-result.json",
        adapter_dir / "token-audit.json",
    ]
    files = [file_record(path, adapter_dir) for path in package_paths]
    adapter_bytes = sum(
        record["bytes"]
        for record in files
        if record["path"] in {"adapter_config.json", "adapters.safetensors"}
    )
    if adapter_bytes > MAX_SCION_ARTIFACT_BYTES:
        raise RuntimeError(f"adapter exceeds the one-gigabyte delivery cap: {adapter_bytes}")
    counts = dataset["counts"]
    manifest = {
        "schemaVersion": 3,
        "adapter": {
            "id": plan["adapter"]["id"],
            "scionVersion": SCION_VERSION,
            "format": "mlx-lora-safetensors",
        },
        "base": {
            "modelId": pin.model_id,
            "revision": pin.revision,
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
            "splitCounts": {
                "train": counts["train"],
                "valid": counts["valid"],
                "test": counts["test"],
            },
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
                "planSha256": sha256_file(adapter_dir / "training-plan.json"),
                "planIdentitySha256": plan["identity"]["sha256"],
                "resultPath": "training-result.json",
                "resultSha256": sha256_file(adapter_dir / "training-result.json"),
                "resultIdentitySha256": result["identity"]["sha256"],
                "datasetIdentitySha256": dataset["identity"]["sha256"],
                "toolchainPolicySha256": plan["toolchain"]["policySha256"],
                "repositoryCommit": plan["repository"]["commit"],
                "repositoryTree": plan["repository"]["tree"],
                "repositoryDirty": False,
            },
        },
        "files": files,
        "runtime": {"supported": ["mlx-vlm"]},
        "promotion": {
            "status": "research",
            "promotable": False,
            "evidence": [],
        },
        "limits": {
            "adapterBytes": adapter_bytes,
            "deliveryCapBytes": MAX_SCION_ARTIFACT_BYTES,
        },
        "compatibility": {"courseMapperRevision": COURSEMAPPER_SOURCE_REVISION},
    }
    target = (output_path or adapter_dir / "scion-adapter.json").resolve()
    if target.parent != adapter_dir:
        raise RuntimeError("adapter manifest must stay inside its package directory")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
