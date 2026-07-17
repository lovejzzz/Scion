"""Structural and pedagogical admission checks shared by data and evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_META_PHRASES = (
    "this lesson",
    "this course",
    "success criteria",
    "evidence moves",
    "weekly check",
)


def words(value: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(value) if len(token) > 2}


def parse_json_object(value: str | Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(value, Mapping):
        return dict(value), []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None, ["invalid-json"]
    if not isinstance(parsed, dict):
        return None, ["root-not-object"]
    return parsed, []


def _text(value: Any, minimum: int, issue: str, issues: list[str]) -> str:
    text = str(value or "").strip()
    if len(text) < minimum:
        issues.append(issue)
    return text


def validate_key_term(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["key-term-not-object"]
    issues: list[str] = []
    fields = {
        "tr": _text(value.get("tr"), 2, "key-term-tr", issues),
        "df": _text(value.get("df"), 20, "key-term-df", issues),
        "eg": _text(value.get("eg"), 12, "key-term-eg", issues),
        "mi": _text(value.get("mi"), 12, "key-term-mi", issues),
        "cx": _text(value.get("cx"), 18, "key-term-cx", issues),
    }
    normalized = {name: " ".join(text.lower().split()) for name, text in fields.items()}
    if normalized["mi"] == normalized["cx"]:
        issues.append("key-term-mi-equals-cx")
    if words(fields["mi"]) and words(fields["mi"]) == words(fields["df"]):
        issues.append("key-term-mi-repeats-df")
    return issues


def validate_mc_item(
    value: Any, *, fact_count: int | None = None, require_fact_indexes: bool = False
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["mc-not-object"]
    issues: list[str] = []
    _text(value.get("q"), 12, "mc-question", issues)
    options = value.get("op")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or len(options) != 4:
        issues.append("mc-options-count")
        options = []
    else:
        cleaned = [str(option or "").strip() for option in options]
        if any(len(option) < 1 for option in cleaned):
            issues.append("mc-option-empty")
        if len({option.casefold() for option in cleaned}) != 4:
            issues.append("mc-options-duplicate")
    answer = value.get("ai")
    if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer <= 3:
        issues.append("mc-answer-index")
    _text(value.get("ex"), 18, "mc-explanation", issues)
    fact_indexes = value.get("fi", value.get("sourceFactIndexes"))
    if require_fact_indexes or fact_indexes is not None:
        if not isinstance(fact_indexes, list) or not 1 <= len(fact_indexes) <= 2:
            issues.append("mc-fact-index-count")
        elif len(set(fact_indexes)) != len(fact_indexes) or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or (fact_count is not None and index >= fact_count)
            for index in fact_indexes
        ):
            issues.append("mc-fact-index-invalid")
    return issues


def validate_lesson(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["lesson-not-object"]
    issues: list[str] = []
    _text(value.get("lessonId"), 3, "lesson-id", issues)
    facts = value.get("facts")
    if not isinstance(facts, list) or len(facts) != 5:
        issues.append("facts-count")
        facts = []
    elif any(len(str(fact or "").strip()) < 20 for fact in facts):
        issues.append("fact-too-short")

    terms = value.get("keyTerms")
    if not isinstance(terms, list) or len(terms) != 3:
        issues.append("key-terms-count")
    else:
        for index, term in enumerate(terms):
            issues.extend(f"key-term-{index}:{issue}" for issue in validate_key_term(term))

    scenario = value.get("scenario")
    if not isinstance(scenario, Mapping):
        issues.append("scenario-not-object")
    else:
        _text(scenario.get("su"), 35, "scenario-setup", issues)
        _text(scenario.get("ma"), 20, "scenario-materials", issues)

    discussion = value.get("discussionPrompt")
    if not isinstance(discussion, Mapping):
        issues.append("discussion-not-object")
    else:
        _text(discussion.get("pr"), 20, "discussion-question", issues)
        _text(discussion.get("tn"), 20, "discussion-tension", issues)
        positions = discussion.get("po")
        if not isinstance(positions, list) or len(positions) != 3:
            issues.append("discussion-positions-count")

    assignment = value.get("assignmentCore")
    if not isinstance(assignment, Mapping):
        issues.append("assignment-not-object")
    else:
        _text(assignment.get("td"), 45, "assignment-description", issues)
        parameters = assignment.get("pa")
        if not isinstance(parameters, list) or len(parameters) != 4:
            issues.append("assignment-parameters-count")

    items = value.get("mc")
    if not isinstance(items, list) or len(items) != 4:
        issues.append("mc-count")
    else:
        for index, item in enumerate(items):
            issues.extend(
                f"mc-{index}:{issue}"
                for issue in validate_mc_item(item, fact_count=len(facts), require_fact_indexes=True)
            )

    guide = value.get("studyGuide")
    if not isinstance(guide, Mapping):
        issues.append("study-guide-not-object")
    else:
        _text(guide.get("sm"), 60, "study-guide-summary", issues)
        _text(guide.get("rs"), 30, "study-guide-strategy", issues)

    serialized = json.dumps(value, ensure_ascii=False).casefold()
    for phrase in _META_PHRASES:
        if phrase in serialized:
            issues.append(f"meta-language:{phrase.replace(' ', '-')}")
    return issues


def validate_response(kind: str, value: str | Mapping[str, Any]) -> list[str]:
    parsed, issues = parse_json_object(value)
    if issues or parsed is None:
        return issues
    if kind == "lesson":
        lessons = parsed.get("lessons")
        if not isinstance(lessons, list) or len(lessons) != 1:
            return ["lessons-count"]
        return validate_lesson(lessons[0])
    if kind == "mc-item":
        return validate_mc_item(parsed, require_fact_indexes="fi" in parsed)
    if kind == "key-term":
        return validate_key_term(parsed)
    if kind == "source-bundle":
        bundle_issues: list[str] = []
        items = parsed.get("mcItems")
        terms = parsed.get("keyTerms")
        if not isinstance(items, list) or len(items) < 1:
            bundle_issues.append("bundle-mc-count")
        else:
            for index, item in enumerate(items):
                bundle_issues.extend(f"mc-{index}:{issue}" for issue in validate_mc_item(item))
        if not isinstance(terms, list) or len(terms) < 1:
            bundle_issues.append("bundle-key-term-count")
        else:
            for index, term in enumerate(terms):
                bundle_issues.extend(f"key-term-{index}:{issue}" for issue in validate_key_term(term))
        return bundle_issues
    return ["unknown-response-kind"]


def quality_score(kind: str, value: str | Mapping[str, Any]) -> float:
    """Return a deterministic 0..1 structural/pedagogical score."""
    issues = validate_response(kind, value)
    if not issues:
        return 1.0
    fatal = sum(issue in {"invalid-json", "root-not-object", "unknown-response-kind"} for issue in issues)
    if fatal:
        return 0.0
    return max(0.0, 1.0 - min(1.0, len(issues) / 12.0))
