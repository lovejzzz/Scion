"""CourseMapper-compatible identities and research-corpus receipts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import COURSEMAPPER_SOURCE_REVISION

TRAINING_FORMAT = {
    "protocol": "scion-orpo-conversations-v1",
    "columns": ["chosen", "rejected", "provenance"],
    "sequence": ["user", "assistant"],
    "promptIncludedInBothSequences": True,
}

CATEGORY_DOMAINS = {
    "prerequisite-reasoning": "course-planning",
    "schedule-constraints": "course-planning",
    "degree-audit": "academic-operations",
    "tool-use": "academic-operations",
    "tutoring": "education-pedagogy",
    "coursemapper-kernel": "education-pedagogy",
    "uncertainty-grounding": "responsible-guidance",
    "safe-education": "responsible-guidance",
}

SPLIT_FILES = {
    "train": "train.jsonl",
    "valid": "validation.jsonl",
    "test": "test.jsonl",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(_canonical(value), separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coursemapper_split(split: str) -> str:
    return {"train": "train", "validation": "valid", "preference-test": "test"}[split]


def preference_provenance(*, task_id: str, category: str, split: str) -> dict[str, str]:
    """Assign a stable capability domain and a group that never crosses a split."""

    domain = CATEGORY_DOMAINS.get(category)
    if domain is None:
        raise ValueError(f"unregistered corpus category: {category}")
    normalized_split = coursemapper_split(split)
    try:
        ordinal = int(task_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"task id has no numeric ordinal: {task_id}") from error
    group = f"scion-{normalized_split}-{category}-{ordinal % 4}"
    return {"split": normalized_split, "domain": domain, "courseGroupId": group}


def _dataset_identity(manifest: dict[str, Any]) -> str:
    source_receipts = [
        {
            "status": entry.get("status"),
            **(
                {"bytes": entry.get("bytes"), "sha256": entry.get("sha256")}
                if entry.get("status") == "verified"
                else {}
            ),
        }
        for entry in manifest.get("sourceReceipts", [])
    ]
    payload = {
        "protocol": "scion-adapter-dataset-identity-v2",
        "schemaVersion": manifest.get("schemaVersion"),
        "status": manifest.get("status"),
        "promotable": manifest.get("promotable"),
        "primaryPreferenceEvidence": manifest.get("primaryPreferenceEvidence"),
        "sourceReceipts": source_receipts,
        "domainMap": {
            "entries": manifest.get("domainMap", {}).get("entries"),
            "sha256": manifest.get("domainMap", {}).get("sha256"),
        },
        "holdoutBoundary": manifest.get("holdoutBoundary"),
        "counts": manifest.get("counts"),
        "domains": manifest.get("domains"),
        "evidenceCounts": manifest.get("evidenceCounts"),
        "instructorDomainCounts": manifest.get("instructorDomainCounts"),
        "modelJudgeDomainCounts": manifest.get("modelJudgeDomainCounts"),
        "domainGroupCounts": manifest.get("domainGroupCounts"),
        "groupIdentity": manifest.get("groupIdentity"),
        "splitIdentity": manifest.get("splitIdentity"),
        "trainingFormat": manifest.get("trainingFormat"),
        "gate": manifest.get("gate"),
        "leakage": manifest.get("leakage"),
        "files": manifest.get("files"),
    }
    return _sha256_text(_stable_json(payload))


def _gate(
    *,
    pair_count: int,
    domains: list[str],
    domain_group_counts: dict[str, int],
    judge_domain_counts: dict[str, int],
    split_counts: dict[str, int],
    group_overlap_count: int,
    minimum_pairs: int,
    minimum_domains: int,
    minimum_groups_per_domain: int,
    minimum_judge_pairs: int,
    minimum_judge_domains: int,
    minimum_judge_pairs_per_domain: int,
) -> dict[str, Any]:
    issues: list[str] = []
    if pair_count < minimum_pairs:
        issues.append(f"verified-pairs:{pair_count}<{minimum_pairs}")
    if len(domains) < minimum_domains:
        issues.append(f"domains:{len(domains)}<{minimum_domains}")
    if pair_count < minimum_judge_pairs:
        issues.append(f"single-model-judge-pairs:{pair_count}<{minimum_judge_pairs}")
    qualified = sum(count >= minimum_judge_pairs_per_domain for count in judge_domain_counts.values())
    if qualified < minimum_judge_domains:
        issues.append(f"single-model-judge-qualified-domains:{qualified}<{minimum_judge_domains}")
    for domain, count in sorted(domain_group_counts.items()):
        if count < minimum_groups_per_domain:
            issues.append(f"domain-groups:{domain}:{count}<{minimum_groups_per_domain}")
    for split in ("train", "valid", "test"):
        if split_counts.get(split, 0) <= 0:
            issues.append(f"{split}-empty")
    if group_overlap_count:
        issues.append("group-leakage")
    return {
        "minimumPairs": minimum_pairs,
        "minimumDomains": minimum_domains,
        "minimumGroupsPerDomain": minimum_groups_per_domain,
        "minimumModelJudgePairs": minimum_judge_pairs,
        "minimumModelJudgeDomains": minimum_judge_domains,
        "minimumModelJudgePairsPerDomain": minimum_judge_pairs_per_domain,
        "qualifiedModelJudgeDomains": qualified,
        "issues": issues,
    }


def build_dataset_manifest(
    *,
    repo_root: Path,
    orpo_dir: Path,
    critic_dir: Path,
    teacher_dir: Path,
    heldout_benchmark_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build a fail-closed schema-v4 receipt from the three admitted ORPO splits."""

    repo_root = repo_root.resolve()
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, Any] = {}
    sources: list[str] = []
    source_receipts: list[dict[str, Any]] = []
    critic_rejections: list[dict[str, Any]] = []
    oracle_exclusions: list[dict[str, Any]] = []

    for split, file_name in SPLIT_FILES.items():
        path = (orpo_dir / file_name).resolve()
        if not path.is_file():
            raise RuntimeError(f"missing ORPO split: {path}")
        rows = _read_jsonl(path)
        if not rows:
            raise RuntimeError(f"empty ORPO split: {path}")
        rows_by_split[split] = rows
        relative = path.relative_to(repo_root).as_posix()
        files[split] = {
            "path": path.relative_to(output_path.parent.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "rows": len(rows),
        }
        sources.append(relative)
        source_receipts.append(
            {
                "path": relative,
                "status": "verified",
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

        source_split = "preference_test" if split == "test" else ("validation" if split == "valid" else split)
        evidence_path = (critic_dir / f"{source_split}.jsonl").resolve()
        teacher_path = (teacher_dir / f"{source_split}.jsonl").resolve()
        for evidence_source in (evidence_path, teacher_path):
            if not evidence_source.is_file():
                raise RuntimeError(f"missing corpus evidence: {evidence_source}")
            relative_evidence = evidence_source.relative_to(repo_root).as_posix()
            sources.append(relative_evidence)
            source_receipts.append(
                {
                    "path": relative_evidence,
                    "status": "verified",
                    "bytes": evidence_source.stat().st_size,
                    "sha256": _sha256_file(evidence_source),
                }
            )
        for teacher_row in _read_jsonl(teacher_path):
            if teacher_row.get("status") == "excluded":
                oracle_exclusions.append(
                    {
                        "id": teacher_row.get("id"),
                        "split": split,
                        "stage": "deterministic-oracle",
                        "issues": teacher_row.get("issues", []),
                    }
                )
        for evidence in _read_jsonl(evidence_path):
            if evidence.get("status") == "critic-rejected":
                critic_rejections.append(
                    {
                        "id": evidence.get("id"),
                        "split": split,
                        "stage": "independent-critic",
                        "issues": evidence.get("judgment", {}).get("issues", []),
                    }
                )

    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    for split, rows in rows_by_split.items():
        for row in rows:
            provenance = row.get("provenance", {})
            if provenance.get("split") != split:
                raise RuntimeError(f"row has invalid split provenance: {row.get('id')}")
            if provenance.get("domain") not in set(CATEGORY_DOMAINS.values()):
                raise RuntimeError(f"row has invalid domain provenance: {row.get('id')}")
            if not provenance.get("courseGroupId"):
                raise RuntimeError(f"row has no course-group provenance: {row.get('id')}")
            if provenance.get("preferenceEvidenceKind") != "single-model-judge-preference":
                raise RuntimeError(f"row has no admitted critic evidence: {row.get('id')}")

    domains = sorted({row["provenance"]["domain"] for row in all_rows})
    groups = sorted({row["provenance"]["courseGroupId"] for row in all_rows})
    domain_counts = Counter(row["provenance"]["domain"] for row in all_rows)
    groups_by_domain: dict[str, set[str]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_domains: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            provenance = row["provenance"]
            groups_by_domain[provenance["domain"]].add(provenance["courseGroupId"])
            split_groups[split].add(provenance["courseGroupId"])
            split_domains[split].add(provenance["domain"])
    overlaps = []
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for group in sorted(split_groups[left] & split_groups[right]):
            overlaps.append({"group": group, "splits": [left, right]})
    domain_group_counts = {domain: len(groups_by_domain[domain]) for domain in domains}
    split_counts = {split: len(rows_by_split[split]) for split in ("train", "valid", "test")}

    production = _gate(
        pair_count=len(all_rows),
        domains=domains,
        domain_group_counts=domain_group_counts,
        judge_domain_counts=dict(domain_counts),
        split_counts=split_counts,
        group_overlap_count=len(overlaps),
        minimum_pairs=3000,
        minimum_domains=5,
        minimum_groups_per_domain=3,
        minimum_judge_pairs=100,
        minimum_judge_domains=5,
        minimum_judge_pairs_per_domain=20,
    )
    research = _gate(
        pair_count=len(all_rows),
        domains=domains,
        domain_group_counts=domain_group_counts,
        judge_domain_counts=dict(domain_counts),
        split_counts=split_counts,
        group_overlap_count=len(overlaps),
        minimum_pairs=100,
        minimum_domains=4,
        minimum_groups_per_domain=3,
        minimum_judge_pairs=100,
        minimum_judge_domains=4,
        minimum_judge_pairs_per_domain=20,
    )
    status = "research-ready" if not research["issues"] else "blocked"

    benchmark = json.loads(heldout_benchmark_path.read_text(encoding="utf-8"))
    heldout_domains = sorted({course["domain"] for course in benchmark["courses"]})
    heldout_groups = sorted({course["courseId"] for course in benchmark["courses"]})
    overlap_domains = sorted(set(domains) & set(heldout_domains))
    overlap_groups = sorted(set(groups) & set(heldout_groups))
    if overlap_domains or overlap_groups:
        raise RuntimeError("CourseMapper held-out benchmark overlaps the training corpus")
    benchmark_relative = heldout_benchmark_path.resolve().relative_to(repo_root).as_posix()

    domain_map = {category: domain for category, domain in sorted(CATEGORY_DOMAINS.items())}
    domain_map_sha = _sha256_text(_stable_json(domain_map))
    manifest = {
        "schemaVersion": 4,
        "status": status,
        "promotable": False,
        "primaryPreferenceEvidence": "single-model-judge",
        "courseMapperCompatibility": {
            "sourceRevision": COURSEMAPPER_SOURCE_REVISION,
            "datasetSchemaVersion": 4,
            "adapterManifestSchemaVersion": 3,
        },
        "generatedAt": datetime.now(UTC).isoformat(),
        "sources": sources,
        "sourceReceipts": source_receipts,
        "domainMap": {"entries": len(domain_map), "sha256": domain_map_sha, "mapping": domain_map},
        "holdoutBoundary": {
            "protocol": "scion-training-holdout-firewall-v1",
            "status": "pass",
            "manifestPath": benchmark_relative,
            "manifestSha256": _sha256_file(heldout_benchmark_path),
            "benchmarkId": benchmark["id"],
            "frozenAt": benchmark["frozenAt"],
            "domainDisjointRequired": True,
            "courseGroupDisjointRequired": True,
            "domains": heldout_domains,
            "domainCount": len(heldout_domains),
            "courseGroupCount": len(heldout_groups),
            "admittedDomainOverlapCount": 0,
            "admittedCourseGroupOverlapCount": 0,
            "excludedPairCount": 0,
            "excludedDomainPairCount": 0,
            "excludedCourseGroupPairCount": 0,
        },
        "counts": {
            "loaded": len(all_rows) + len(critic_rejections) + len(oracle_exclusions),
            "total": len(all_rows),
            "quarantined": len(critic_rejections) + len(oracle_exclusions),
            "domains": len(domains),
            "groups": len(groups),
            "train": split_counts["train"],
            "valid": split_counts["valid"],
            "test": split_counts["test"],
            "trainDomains": len(split_domains["train"]),
            "validDomains": len(split_domains["valid"]),
            "testDomains": len(split_domains["test"]),
            "blindInstructorPairs": 0,
            "blindInstructorDomains": 0,
            "singleModelJudgePairs": len(all_rows),
            "singleModelJudgeDomains": len(domains),
        },
        "domains": domains,
        "evidenceCounts": {"single-model-judge-preference": len(all_rows)},
        "instructorDomainCounts": {domain: 0 for domain in domains},
        "modelJudgeDomainCounts": dict(sorted(domain_counts.items())),
        "domainGroupCounts": domain_group_counts,
        "groupIdentity": {
            "algorithm": "sha256-domain-colon-course-id",
            "hashes": sorted(
                {
                    _sha256_text(f"{row['provenance']['domain']}:{row['provenance']['courseGroupId']}")
                    for row in all_rows
                }
            ),
            "courseIdAlgorithm": "sha256-course-id",
            "courseIdHashes": sorted(_sha256_text(group) for group in groups),
        },
        "splitIdentity": {
            "strategy": "explicit-locked-splits-v1",
            "groups": {
                split: sorted(_sha256_text(group) for group in split_groups[split])
                for split in ("train", "valid", "test")
            },
            "domains": {split: sorted(split_domains[split]) for split in ("train", "valid", "test")},
        },
        "trainingFormat": TRAINING_FORMAT,
        "gate": {
            "minimumPairs": 3000,
            "minimumDomains": 5,
            "minimumGroupsPerDomain": 3,
            "primaryPreferenceEvidence": "single-model-judge",
            "minimumModelJudgePairs": 100,
            "minimumModelJudgeDomains": 5,
            "minimumModelJudgePairsPerDomain": 20,
            "issues": research["issues"] if status == "research-ready" else production["issues"],
            "profiles": {"production": production, "research": research},
        },
        "leakage": {"groupOverlapCount": len(overlaps), "overlaps": overlaps},
        "files": files,
        "quarantine": [*oracle_exclusions, *critic_rejections],
    }
    manifest["identity"] = {
        "protocol": "scion-adapter-dataset-identity-v2",
        "sha256": _dataset_identity(manifest),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
