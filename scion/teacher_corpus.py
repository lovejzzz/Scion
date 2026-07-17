"""Generate locally taught, oracle-admitted ORPO preferences."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .local_inference import GenerationSettings, MlxGenerator, canonical_sha256
from .task_contracts import construct_rejected, preference_messages, validate_task_response


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def generate_teacher_split(
    *,
    generator: MlxGenerator,
    seed_path: Path,
    output_path: Path,
    attempts_path: Path,
    split: str,
    limit: int | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    all_seed_rows = [row for row in _load_jsonl(seed_path) if row.get("split") == split]
    rows = list(all_seed_rows)
    if limit is not None:
        selected: list[dict[str, Any]] = []
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(str(row["category"]), []).append(row)
        while len(selected) < limit and any(buckets.values()):
            for category in sorted(buckets):
                if buckets[category] and len(selected) < limit:
                    selected.append(buckets[category].pop(0))
        rows = selected
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    current_hashes = {row["id"]: canonical_sha256(row) for row in all_seed_rows}
    prior_rows = _load_jsonl(output_path) if output_path.exists() else []
    retained_rows = [
        row
        for row in prior_rows
        if row.get("id") in current_hashes and row.get("seedTaskSha256") == current_hashes[row["id"]]
    ]
    if len(retained_rows) != len(prior_rows):
        output_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained_rows),
            encoding="utf-8",
        )
    completed = {row["id"] for row in retained_rows}
    accepted = 0
    rejected = 0
    category_counts: Counter[str] = Counter()
    metrics: list[dict[str, Any]] = []

    for position, task in enumerate(rows):
        if task["id"] in completed:
            continue
        base_messages = task["messages"]
        messages = list(base_messages)
        final: dict[str, Any] | None = None
        final_generation: dict[str, Any] | None = None
        final_issues: list[str] = []
        for attempt in range(max_attempts):
            if task["contract"] == "coursemapper-kernel-json-v1":
                max_tokens = 1900
            elif task["contract"] == "schedule-json-v1" and attempt == 0:
                max_tokens = 400
            else:
                max_tokens = 700
            settings = GenerationSettings(
                max_tokens=max_tokens,
                temperature=0 if attempt == 0 else 0.15,
                top_p=1 if attempt == 0 else 0.9,
                seed=16031 + position * 7 + attempt,
            )
            generation = generator.complete(messages, settings)
            parsed, issues = validate_task_response(task["contract"], generation["text"], task["oracle"])
            attempt_row = {
                "id": task["id"],
                "attempt": attempt + 1,
                "issues": issues,
                "generation": generation,
            }
            with attempts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt_row, sort_keys=True) + "\n")
            metrics.append(generation["metrics"])
            if parsed is not None and not issues:
                final = parsed
                final_generation = generation
                break
            final_issues = issues
            messages = [
                *base_messages,
                {"role": "assistant", "content": generation["text"]},
                {
                    "role": "user",
                    "content": (
                        "The response failed deterministic admission for these reasons: "
                        f"{json.dumps(issues)}. Return a complete corrected JSON object only."
                    ),
                },
            ]

        if final is None or final_generation is None:
            failure = {
                "id": task["id"],
                "status": "excluded",
                "issues": final_issues,
                "category": task["category"],
                "seedTaskSha256": canonical_sha256(task),
            }
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
            rejected += 1
            continue

        negative = construct_rejected(task["contract"], final, task["oracle"])
        negative_issues = validate_task_response(task["contract"], negative, task["oracle"])[1]
        row = {
            "id": task["id"],
            "status": "oracle-admitted-awaiting-critic",
            "split": split,
            "category": task["category"],
            "contract": task["contract"],
            "chosen": preference_messages(base_messages, final),
            "rejected": preference_messages(base_messages, negative),
            "admission": {
                "validator": "scion.task_contracts.validate_task_response",
                "validatorProtocol": "scion-deterministic-oracle-v1",
                "chosenIssues": [],
                "rejectedIssues": negative_issues,
            },
            "teacher": final_generation["receipt"],
            "provenance": task["provenance"],
            "seedTaskSha256": canonical_sha256(task),
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        accepted += 1
        category_counts[task["category"]] += 1
        print(
            f"[{position + 1}/{len(rows)}] {task['id']} accepted "
            f"tokens={final_generation['metrics']['generationTokens']} "
            f"memory={final_generation['metrics']['peakMemoryGb']:.2f}GB",
            flush=True,
        )

    all_rows = _load_jsonl(output_path) if output_path.exists() else []
    admitted = [row for row in all_rows if row.get("status") == "oracle-admitted-awaiting-critic"]
    summary = {
        "schemaVersion": 1,
        "protocol": "scion-local-teacher-corpus-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "split": split,
        "source": str(seed_path.resolve()),
        "sourceSha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "output": str(output_path.resolve()),
        "outputSha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "requestedRows": len(rows),
        "admittedRows": len(admitted),
        "excludedRows": len(all_rows) - len(admitted),
        "newlyAdmittedRows": accepted,
        "newlyExcludedRows": rejected,
        "byCategory": dict(sorted(Counter(row["category"] for row in admitted).items())),
        "teacher": {
            "modelId": generator.pin.model_id,
            "revision": generator.pin.revision,
            "license": generator.pin.license,
            "closedApiOutputUsed": False,
        },
        "meanGenerationTokensThisRun": (
            sum(metric["generationTokens"] for metric in metrics) / len(metrics) if metrics else 0
        ),
        "peakMemoryGbThisRun": max((metric["peakMemoryGb"] for metric in metrics), default=0),
    }
    return summary
