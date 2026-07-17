"""Locked, deterministic-oracle evaluation for Gemma 4 base and adapters."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_inference import GenerationSettings, MlxGenerator
from .task_contracts import validate_task_response


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _coursemapper_json_text(text: str) -> str:
    """Apply CourseMapper's conservative Markdown-fence transport repair.

    This intentionally mirrors the first, lossless stage of
    ``repairPublicScionJson`` at the pinned CourseMapper revision. It does not
    repair malformed JSON or change semantic fields.
    """

    repaired = re.sub(r"^```json\s*", "", str(text or ""), count=1, flags=re.IGNORECASE)
    repaired = re.sub(r"^```\s*", "", repaired, count=1, flags=re.IGNORECASE)
    repaired = re.sub(r"```\s*$", "", repaired, count=1, flags=re.IGNORECASE)
    return repaired.strip()


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["category"]].append(row)
    for category, rows in sorted(grouped.items()):
        passed = sum(row["status"] == "pass" for row in rows)
        by_category[category] = {
            "count": len(rows),
            "passCount": passed,
            "passRate": passed / len(rows),
            "issueCount": sum(len(row["issues"]) for row in rows),
        }
    pass_count = sum(row["status"] == "pass" for row in results)
    return {
        "passCount": pass_count,
        "passRate": pass_count / len(results) if results else 0,
        "issueCount": sum(len(row["issues"]) for row in results),
        "issueKinds": dict(sorted(Counter(issue for row in results for issue in row["issues"]).items())),
        "byCategory": by_category,
    }


def evaluate_locked_tasks(
    *,
    generator: MlxGenerator,
    fixture_path: Path,
    output_dir: Path,
    tier: str,
    variant: str,
    limit: int | None = None,
) -> dict[str, Any]:
    fixtures = [row for row in _rows(fixture_path) if row.get("split") == "heldout"]
    if limit is not None:
        fixtures = fixtures[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for position, fixture in enumerate(fixtures):
        max_tokens = 1900 if fixture["contract"] == "coursemapper-kernel-json-v1" else 700
        generation = generator.complete(
            fixture["messages"],
            GenerationSettings(
                max_tokens=max_tokens,
                temperature=0,
                top_p=1,
                seed=32003 + position,
            ),
        )
        parsed, issues = validate_task_response(fixture["contract"], generation["text"], fixture["oracle"])
        row = {
            "id": fixture["id"],
            "category": fixture["category"],
            "contract": fixture["contract"],
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "response": parsed,
            "rawText": generation["text"] if parsed is None else None,
            "metrics": generation["metrics"],
            "generationReceipt": generation["receipt"],
        }
        results.append(row)
        print(
            f"[{position + 1}/{len(fixtures)}] {fixture['id']} {row['status']} issues={len(issues)}",
            flush=True,
        )
    result_path = output_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = _summarize_results(results)
    report = {
        "schemaVersion": 1,
        "protocol": "scion-locked-local-evaluation-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "tier": tier,
        "variant": variant,
        "model": {
            "id": generator.pin.model_id,
            "revision": generator.pin.revision,
            "adapter": generator.adapter_identity,
        },
        "fixtures": {
            "path": str(fixture_path.resolve()),
            "sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "selectedIdsSha256": hashlib.sha256(
                json.dumps([row["id"] for row in fixtures], separators=(",", ":")).encode()
            ).hexdigest(),
            "count": len(fixtures),
            "trainingUseForbidden": True,
        },
        **summary,
        "resultsSha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "meanTokensPerSecond": (
            sum(row["metrics"]["generationTokensPerSecond"] for row in results) / len(results)
            if results
            else 0
        ),
        "peakMemoryGb": max((row["metrics"]["peakMemoryGb"] for row in results), default=0),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def rescore_coursemapper_runtime(
    *,
    fixture_path: Path,
    source_report_path: Path,
    source_results_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Rescore saved generations after CourseMapper's lossless fence repair."""

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("protocol") != "scion-locked-local-evaluation-v1":
        raise RuntimeError("runtime rescore requires a locked local evaluation")
    fixtures = [row for row in _rows(fixture_path) if row.get("split") == "heldout"]
    source_rows = _rows(source_results_path)
    if [row["id"] for row in source_rows] != [row["id"] for row in fixtures]:
        raise RuntimeError("runtime rescore fixture IDs do not match the locked source results")
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if (
        source_report.get("fixtures", {}).get("sha256") != fixture_sha256
        or source_report.get("fixtures", {}).get("count") != len(fixtures)
    ):
        raise RuntimeError("runtime rescore fixture receipt does not match")

    results = []
    for fixture, source in zip(fixtures, source_rows, strict=True):
        if source.get("response") is not None:
            candidate: str | dict[str, Any] = source["response"]
            normalization = "already-parsed"
        else:
            candidate = _coursemapper_json_text(source.get("rawText") or "")
            normalization = "coursemapper-fence-strip"
        parsed, issues = validate_task_response(fixture["contract"], candidate, fixture["oracle"])
        results.append(
            {
                "id": fixture["id"],
                "category": fixture["category"],
                "contract": fixture["contract"],
                "status": "pass" if not issues else "fail",
                "issues": issues,
                "normalization": normalization,
                "response": parsed,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    report = {
        "schemaVersion": 1,
        "protocol": "scion-coursemapper-runtime-normalized-evaluation-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "tier": source_report["tier"],
        "variant": source_report["variant"],
        "model": source_report["model"],
        "normalization": {
            "id": "coursemapper-lossless-fence-repair-v1",
            "scope": "transport-only",
            "courseMapperRevision": "4f5bed3833f72494917e67c1a0c878af8c2b9a70",
            "malformedJsonRepairUsed": False,
            "semanticRepairUsed": False,
        },
        "fixtures": source_report["fixtures"],
        **_summarize_results(results),
        "resultsSha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "sourceEvaluation": {
            "reportSha256": hashlib.sha256(source_report_path.read_bytes()).hexdigest(),
            "resultsSha256": hashlib.sha256(source_results_path.read_bytes()).hexdigest(),
            "protocol": source_report["protocol"],
        },
        "meanTokensPerSecond": source_report["meanTokensPerSecond"],
        "peakMemoryGb": source_report["peakMemoryGb"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def compare_evaluations(base_path: Path, adapter_path: Path, output: Path) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    same_set = (
        base.get("fixtures", {}).get("sha256") == adapter.get("fixtures", {}).get("sha256")
        and base.get("fixtures", {}).get("selectedIdsSha256")
        == adapter.get("fixtures", {}).get("selectedIdsSha256")
        and base.get("fixtures", {}).get("count") == adapter.get("fixtures", {}).get("count") == 32
    )
    categories = sorted(set(base.get("byCategory", {})) | set(adapter.get("byCategory", {})))
    regressions = [
        category
        for category in categories
        if adapter.get("byCategory", {}).get(category, {}).get("passRate", 0)
        < base.get("byCategory", {}).get(category, {}).get("passRate", 0)
    ]
    checks = {
        "sameLocked32Fixtures": same_set,
        "overallPassRateImproved": adapter.get("passRate", 0) > base.get("passRate", 0),
        "totalIssuesReduced": adapter.get("issueCount", 10**9) < base.get("issueCount", 10**9),
        "noCategoryPassRateRegression": not regressions,
        "noHallucinatedCitationRegression": adapter.get("issueKinds", {}).get("hallucinated-citation", 0)
        <= base.get("issueKinds", {}).get("hallucinated-citation", 0),
    }
    comparison = {
        "schemaVersion": 1,
        "protocol": "scion-paired-promotion-comparison-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "categoryRegressions": regressions,
        "deltas": {
            "passRate": adapter.get("passRate", 0) - base.get("passRate", 0),
            "issueCount": adapter.get("issueCount", 0) - base.get("issueCount", 0),
        },
        "base": base,
        "adapter": adapter,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison
