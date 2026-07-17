"""Blind, local Gemma 4 critic for oracle-admitted teacher preferences."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .corpus_manifest import preference_provenance
from .local_inference import GenerationSettings, MlxGenerator


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("critic response is not an object")
    return value


def _candidate_content(messages: list[dict[str, str]]) -> str:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("preference candidate is missing assistant response")
    return messages[-1]["content"]


def _valid_judgment_shape(judgment: Any) -> bool:
    if not isinstance(judgment, dict):
        return False
    issues = judgment.get("issues")
    return (
        judgment.get("preferred") in {"A", "B", "tie"}
        and isinstance(judgment.get("preferredScore"), int)
        and not isinstance(judgment.get("preferredScore"), bool)
        and 1 <= judgment["preferredScore"] <= 5
        and isinstance(judgment.get("groundingPass"), bool)
        and isinstance(judgment.get("pedagogyPass"), bool)
        and isinstance(issues, list)
        and all(isinstance(issue, str) for issue in issues)
    )


def _judge_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    chosen = _candidate_content(row["chosen"])
    rejected = _candidate_content(row["rejected"])
    chosen_label = "A" if int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % 2 == 0 else "B"
    candidates = {chosen_label: chosen, "B" if chosen_label == "A" else "A": rejected}
    task = row["chosen"][:-1]
    user = {
        "taskMessages": task,
        "contract": row["contract"],
        "candidateA": json.loads(candidates["A"]),
        "candidateB": json.loads(candidates["B"]),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are an independent education-data critic. Compare two blind candidates against the supplied "
                "task. Reward exact factual grounding, correct constraints, honest uncertainty, useful pedagogy, "
                "safe behavior, and exact JSON instructions. Penalize even one invented catalog fact or citation. "
                "For planning and tool tasks, pedagogyPass means task-appropriate clarity and usefulness; it does "
                "not require a tutoring lesson. "
                "Return JSON only with exactly these keys: preferred (A, B, or tie), preferredScore (integer 1-5), "
                "groundingPass (boolean), pedagogyPass (boolean), issues (array of short strings). Do not reveal "
                "chain-of-thought."
            ),
        },
        {"role": "user", "content": json.dumps(user, separators=(",", ":"), sort_keys=True)},
    ]
    return messages, chosen_label


def filter_with_critic(
    *,
    generator: MlxGenerator,
    teacher_path: Path,
    evidence_path: Path,
    orpo_path: Path,
    split: str,
    limit: int | None = None,
) -> dict[str, Any]:
    teacher_rows = [
        row for row in _load_jsonl(teacher_path) if row.get("status") == "oracle-admitted-awaiting-critic"
    ]
    if limit is not None:
        teacher_rows = teacher_rows[:limit]
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    orpo_path.parent.mkdir(parents=True, exist_ok=True)
    previous = _load_jsonl(evidence_path) if evidence_path.exists() else []
    current_hashes = {
        row["id"]: hashlib.sha256(json.dumps(row, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        for row in teacher_rows
    }
    previous = [
        row
        for row in previous
        if row.get("id") in current_hashes
        and row.get("teacherRowSha256") == current_hashes[row["id"]]
        and row.get("parseIssue") is None
        and _valid_judgment_shape(row.get("judgment"))
    ]
    if evidence_path.exists():
        evidence_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in previous),
            encoding="utf-8",
        )
    complete = {row["id"] for row in previous}
    for position, row in enumerate(teacher_rows):
        if row["id"] in complete:
            continue
        messages, chosen_label = _judge_messages(row)
        attempts: list[dict[str, Any]] = []
        parse_issue = None
        judgment: dict[str, Any] = {}
        generation: dict[str, Any] = {}
        valid_shape = False
        for critic_attempt in range(2):
            attempt_messages = list(messages)
            if critic_attempt:
                attempt_messages.extend(
                    [
                        {"role": "assistant", "content": generation["text"]},
                        {
                            "role": "user",
                            "content": (
                                "Your response did not match the exact JSON contract. Return a corrected JSON "
                                "object now. preferred must be exactly A, B, or tie; include all five required "
                                "keys and no prose."
                            ),
                        },
                    ]
                )
            generation = generator.complete(
                attempt_messages,
                GenerationSettings(
                    max_tokens=256,
                    temperature=0,
                    top_p=1,
                    seed=24017 + position + (100_000 * critic_attempt),
                    repetition_penalty=1,
                ),
            )
            parse_issue = None
            try:
                judgment = _extract_object(generation["text"])
            except (json.JSONDecodeError, ValueError) as error:
                judgment = {}
                parse_issue = str(error)
            valid_shape = parse_issue is None and _valid_judgment_shape(judgment)
            attempts.append(
                {
                    "attempt": critic_attempt + 1,
                    "generation": generation,
                    "judgment": judgment,
                    "parseIssue": parse_issue,
                    "validShape": valid_shape,
                }
            )
            if valid_shape:
                break
        admitted = (
            valid_shape
            and judgment["preferred"] == chosen_label
            and judgment["preferredScore"] >= 4
            and judgment["groundingPass"] is True
            and judgment["pedagogyPass"] is True
        )
        evidence = {
            "id": row["id"],
            "status": "critic-admitted" if admitted else "critic-rejected",
            "category": row["category"],
            "chosenBlindLabel": chosen_label,
            "judgment": judgment,
            "parseIssue": parse_issue,
            "generation": generation,
            "criticAttempts": attempts,
            "teacherRowSha256": hashlib.sha256(
                json.dumps(row, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
        }
        with evidence_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evidence, sort_keys=True) + "\n")
        print(
            f"[{position + 1}/{len(teacher_rows)}] {row['id']} {evidence['status']} "
            f"preferred={judgment.get('preferred')} expected={chosen_label}",
            flush=True,
        )

    evidence_rows = _load_jsonl(evidence_path)
    by_id = {row["id"]: row for row in evidence_rows}
    admitted_rows = [
        row for row in teacher_rows if by_id.get(row["id"], {}).get("status") == "critic-admitted"
    ]
    orpo_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "provenance": {
                **preference_provenance(task_id=row["id"], category=row["category"], split=split),
                "preferenceEvidenceKind": "single-model-judge-preference",
                "criticProtocol": "scion-blind-local-critic-v1",
                "criticEvidenceSha256": hashlib.sha256(
                    json.dumps(by_id[row["id"]], separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest(),
                "teacherRowSha256": by_id[row["id"]]["teacherRowSha256"],
            },
        }
        for row in admitted_rows
    ]
    orpo_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in orpo_rows),
        encoding="utf-8",
    )
    summary = {
        "schemaVersion": 1,
        "protocol": "scion-blind-local-critic-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "split": split,
        "teacherRows": len(teacher_rows),
        "admittedRows": len(admitted_rows),
        "rejectedRows": len(teacher_rows) - len(admitted_rows),
        "admissionRate": len(admitted_rows) / len(teacher_rows) if teacher_rows else 0,
        "byCategory": dict(sorted(Counter(row["category"] for row in admitted_rows).items())),
        "critic": {
            "modelId": generator.pin.model_id,
            "revision": generator.pin.revision,
            "license": generator.pin.license,
            "closedApiOutputUsed": False,
        },
        "teacherSha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "evidenceSha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "orpoSha256": hashlib.sha256(orpo_path.read_bytes()).hexdigest(),
    }
    return summary
