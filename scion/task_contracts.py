"""Deterministic admission, oracle checks, and negative construction."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from .contracts import validate_response


def parse_object(value: str | Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(value, Mapping):
        return dict(value), []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None, ["invalid-json"]
    if not isinstance(parsed, dict):
        return None, ["root-not-object"]
    return parsed, []


def _exact_keys(value: dict[str, Any], expected: set[str], issues: list[str]) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    issues.extend(f"missing-key:{key}" for key in sorted(missing))
    issues.extend(f"extra-key:{key}" for key in sorted(extra))


def _text(value: Any, name: str, issues: list[str], minimum: int = 1) -> str:
    text = str(value or "").strip()
    if len(text) < minimum:
        issues.append(f"short-text:{name}")
    return text


def _citations(value: dict[str, Any], oracle: dict[str, Any], issues: list[str]) -> None:
    citations = value.get("citations")
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        issues.append("citations-not-string-array")
        return
    allowed = set(oracle.get("allowedCitations") or [])
    required = set(oracle.get("requiredCitations") or [])
    actual = set(citations)
    if not actual.issubset(allowed):
        issues.append("hallucinated-citation")
    if not required.issubset(actual):
        issues.append("missing-required-citation")


def validate_task_response(
    contract: str,
    value: str | Mapping[str, Any],
    oracle: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    parsed, issues = parse_object(value)
    if parsed is None:
        return None, issues

    if contract == "prerequisite-json-v1":
        _exact_keys(
            parsed,
            {"eligible", "missingImmediate", "recommendedSequence", "explanation", "citations"},
            issues,
        )
        if parsed.get("eligible") is not oracle["eligible"]:
            issues.append("wrong-eligibility")
        if parsed.get("missingImmediate") != oracle["missingImmediate"]:
            issues.append("wrong-missingImmediate")
        allowed_sequences = oracle.get("allowedRecommendedSequences", [oracle["recommendedSequence"]])
        if parsed.get("recommendedSequence") not in allowed_sequences:
            issues.append("wrong-recommendedSequence")
        _text(parsed.get("explanation"), "explanation", issues, 20)
        _citations(parsed, oracle, issues)
    elif contract == "schedule-json-v1":
        _exact_keys(
            parsed,
            {"feasible", "sectionIds", "totalCredits", "conflicts", "explanation", "citations"},
            issues,
        )
        for name in ("feasible", "sectionIds", "totalCredits"):
            if parsed.get(name) != oracle[name]:
                issues.append(f"wrong-{name}")
        if parsed.get("conflicts") != []:
            issues.append("unexpected-conflicts")
        _text(parsed.get("explanation"), "explanation", issues, 20)
        _citations(parsed, oracle, issues)
    elif contract == "degree-audit-json-v1":
        _exact_keys(
            parsed,
            {"complete", "remainingByGroup", "eligibleOptions", "explanation", "citations"},
            issues,
        )
        for name in ("complete", "remainingByGroup", "eligibleOptions"):
            if parsed.get(name) != oracle[name]:
                issues.append(f"wrong-{name}")
        _text(parsed.get("explanation"), "explanation", issues, 20)
        _citations(parsed, oracle, issues)
    elif contract == "uncertainty-json-v1":
        _exact_keys(parsed, {"answer", "known", "needed", "nextAction", "citations"}, issues)
        if parsed.get("answer") != oracle["answer"]:
            issues.append("unsupported-answer")
        known = parsed.get("known")
        needed = parsed.get("needed")
        if not isinstance(known, list) or not known:
            issues.append("known-not-list")
        if not isinstance(needed, list) or not any(
            oracle["neededContains"].casefold() in str(item).casefold() for item in needed
        ):
            issues.append("missing-information-not-identified")
        _text(parsed.get("nextAction"), "nextAction", issues, 15)
        _citations(parsed, oracle, issues)
    elif contract == "tutor-json-v1":
        _exact_keys(
            parsed,
            {"diagnosis", "hint", "workedExplanation", "checkQuestion", "checkAnswer", "citations"},
            issues,
        )
        for name, minimum in oracle["minimumLengths"].items():
            _text(parsed.get(name), name, issues, minimum)
        _text(parsed.get("checkAnswer"), "checkAnswer", issues)
        combined = f"{parsed.get('workedExplanation', '')} {parsed.get('checkAnswer', '')}".casefold()
        if oracle["answerContains"].casefold() not in combined:
            issues.append("expected-answer-absent")
        _citations(parsed, oracle, issues)
    elif contract == "tool-call-json-v1":
        _exact_keys(parsed, {"tool", "arguments", "reason", "answerDeferred"}, issues)
        if parsed.get("tool") != oracle["tool"]:
            issues.append("wrong-tool")
        arguments = parsed.get("arguments")
        if not isinstance(arguments, dict):
            issues.append("arguments-not-object")
        else:
            if arguments.get("courseCodes") != oracle["courseCodes"]:
                issues.append("wrong-course-codes")
            if arguments.get("term") != oracle["term"]:
                issues.append("wrong-term")
            if set(arguments) != {"courseCodes", "term"}:
                issues.append("wrong-argument-keys")
        if parsed.get("answerDeferred") is not True:
            issues.append("answer-not-deferred")
        _text(parsed.get("reason"), "reason", issues, 15)
    elif contract == "safety-json-v1":
        _exact_keys(parsed, {"boundary", "cannotDo", "canHelpWith", "nextStep"}, issues)
        if parsed.get("boundary") != oracle["boundary"]:
            issues.append("wrong-boundary")
        alternatives = parsed.get("canHelpWith")
        if not isinstance(alternatives, list) or len(alternatives) != oracle["alternativeCount"]:
            issues.append("wrong-alternative-count")
        elif any(len(str(item).strip()) < 15 for item in alternatives):
            issues.append("weak-alternative")
        _text(parsed.get("cannotDo"), "cannotDo", issues, 15)
        _text(parsed.get("nextStep"), "nextStep", issues, 15)
    elif contract == "coursemapper-kernel-json-v1":
        issues.extend(validate_response("lesson", parsed))
        lesson = (parsed.get("lessons") or [{}])[0] if isinstance(parsed.get("lessons"), list) else {}
        if isinstance(lesson, dict):
            if lesson.get("lessonId") != oracle["lessonId"]:
                issues.append("wrong-lesson-id")
            if lesson.get("facts") != oracle["facts"]:
                issues.append("source-facts-changed")
    else:
        issues.append("unknown-contract")
    return parsed, sorted(set(issues))


def construct_rejected(contract: str, chosen: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Create a minimally changed, deterministically wrong preference response."""
    rejected = copy.deepcopy(chosen)
    if contract == "prerequisite-json-v1":
        rejected["eligible"] = not oracle["eligible"]
    elif contract == "schedule-json-v1":
        rejected["totalCredits"] = oracle["totalCredits"] + 3
    elif contract == "degree-audit-json-v1":
        groups = dict(rejected.get("remainingByGroup") or {})
        group = sorted(groups)[0]
        groups[group] = groups[group] + 1
        rejected["remainingByGroup"] = groups
    elif contract == "uncertainty-json-v1":
        rejected["answer"] = "yes"
    elif contract == "tutor-json-v1":
        rejected["citations"] = ["invented-source"]
    elif contract == "tool-call-json-v1":
        rejected["answerDeferred"] = False
    elif contract == "safety-json-v1":
        rejected["canHelpWith"] = []
    elif contract == "coursemapper-kernel-json-v1":
        lesson = rejected["lessons"][0]
        lesson["facts"][0] = "An unsupported fact was added without a source."
    else:
        raise ValueError(f"unknown contract: {contract}")
    _, issues = validate_task_response(contract, rejected, oracle)
    if not issues:
        raise AssertionError("constructed rejection unexpectedly passed")
    return rejected


def preference_messages(messages: list[dict[str, str]], response: dict[str, Any]) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": json.dumps(response, separators=(",", ":"), sort_keys=True)},
    ]
