"""Held-out HTTP evaluation and CourseMapper-compatible smoke tests."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import SCION_MODEL_ID
from .contracts import quality_score, validate_response

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 300) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"request failed for {url}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def wait_until_ready(endpoint: str, *, timeout: int = 600) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = _request_json(endpoint.rstrip("/") + "/v1/models", timeout=10)
            if value.get("data"):
                return value
        except RuntimeError as error:
            last_error = error
        time.sleep(2)
    raise RuntimeError(f"model server did not become ready in {timeout}s: {last_error}")


def _tokens(value: Any) -> Counter[str]:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return Counter(token.casefold() for token in _TOKEN.findall(serialized) if len(token) > 2)


def reference_f1(actual: Any, expected: Any) -> float:
    actual_tokens = _tokens(actual)
    expected_tokens = _tokens(expected)
    overlap = sum((actual_tokens & expected_tokens).values())
    if not actual_tokens or not expected_tokens or not overlap:
        return 0.0
    precision = overlap / sum(actual_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _assistant_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"invalid chat-completions response: {response}") from error
    if not isinstance(content, str):
        raise RuntimeError("chat-completions content is not text")
    return content.strip()


def _stratified(fixtures: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit >= len(fixtures):
        return fixtures
    if limit <= 0:
        raise ValueError("evaluation limit must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fixture in fixtures:
        buckets[str(fixture["kind"])].append(fixture)
    selected: list[dict[str, Any]] = []
    kinds = sorted(buckets)
    cursor = 0
    while len(selected) < limit and any(buckets.values()):
        kind = kinds[cursor % len(kinds)]
        if buckets[kind]:
            selected.append(buckets[kind].pop(0))
        cursor += 1
    return selected


def evaluate_endpoint(
    *,
    endpoint: str,
    fixtures_path: Path,
    output: Path,
    model: str = SCION_MODEL_ID,
    limit: int | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    fixtures = [json.loads(line) for line in fixtures_path.read_text(encoding="utf-8").splitlines() if line]
    selected = _stratified(fixtures, limit)
    wait_until_ready(endpoint)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, fixture in enumerate(selected, start=1):
        request = {
            "model": model,
            "messages": fixture["messages"],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        row: dict[str, Any] = {
            "id": fixture["id"],
            "kind": fixture["kind"],
            "domain": fixture["domain"],
            "courseGroup": fixture["courseGroup"],
        }
        call_started = time.monotonic()
        try:
            content = _assistant_content(
                _request_json(endpoint.rstrip("/") + "/v1/chat/completions", request, timeout=600)
            )
            parsed = json.loads(content)
            issues = validate_response(fixture["kind"], parsed)
            row.update(
                {
                    "status": "pass" if not issues else "fail",
                    "issues": issues,
                    "qualityScore": quality_score(fixture["kind"], parsed),
                    "referenceF1": reference_f1(parsed, fixture["expected"]),
                    "response": parsed,
                }
            )
        except (RuntimeError, json.JSONDecodeError) as error:
            row.update({"status": "error", "issues": [str(error)], "qualityScore": 0.0, "referenceF1": 0.0})
        row["latencySeconds"] = round(time.monotonic() - call_started, 3)
        row["position"] = index
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    passing = [row for row in rows if row["status"] == "pass"]
    report = {
        "schemaVersion": 1,
        "endpoint": endpoint,
        "model": model,
        "fixtures": str(fixtures_path.resolve()),
        "count": len(rows),
        "contractPassCount": len(passing),
        "contractPassRate": len(passing) / len(rows) if rows else 0.0,
        "meanQualityScore": sum(row["qualityScore"] for row in rows) / len(rows) if rows else 0.0,
        "meanReferenceF1": sum(row["referenceF1"] for row in rows) / len(rows) if rows else 0.0,
        "p95LatencySeconds": sorted(row["latencySeconds"] for row in rows)[
            min(len(rows) - 1, math.ceil(len(rows) * 0.95) - 1)
        ]
        if rows
        else 0.0,
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "results": str(results_path.resolve()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def compare_reports(base_path: Path, adapter_path: Path, output: Path) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    same_count = base.get("count") == adapter.get("count") and int(adapter.get("count", 0)) >= 20
    checks = {
        "sameFixtureCountAndAtLeast20": same_count,
        "adapterContractPassAtLeast90Percent": adapter.get("contractPassRate", 0) >= 0.9,
        "adapterContractNotWorse": adapter.get("contractPassRate", 0) >= base.get("contractPassRate", 0),
        "adapterReferenceF1NotWorse": adapter.get("meanReferenceF1", 0) >= base.get("meanReferenceF1", 0),
        "adapterQualityNotWorse": adapter.get("meanQualityScore", 0) >= base.get("meanQualityScore", 0),
    }
    result = {
        "schemaVersion": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "base": base,
        "adapter": adapter,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def coursemapper_smoke(endpoint: str, output: Path, *, model: str = SCION_MODEL_ID) -> dict[str, Any]:
    models = wait_until_ready(endpoint)
    messages = [
        {
            "role": "system",
            "content": "You are CourseMapper Scion. Return only accurate JSON and no commentary.",
        },
        {
            "role": "user",
            "content": (
                "Course domain: computer-science. Supplied fact: A Python dictionary maps "
                "unique keys to values. "
                "Return one evidence-bearing multiple-choice item as JSON with q, op (exactly four options), "
                "ai, ex, and fi containing [0]."
            ),
        },
    ]
    response = _request_json(
        endpoint.rstrip("/") + "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 768,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
        timeout=600,
    )
    content = _assistant_content(response)
    try:
        parsed = json.loads(content)
        issues = validate_response("mc-item", parsed)
    except json.JSONDecodeError:
        parsed = None
        issues = ["invalid-json"]
    result = {
        "schemaVersion": 1,
        "status": "pass" if not issues else "fail",
        "endpoint": endpoint,
        "model": model,
        "modelsResponse": models,
        "issues": issues,
        "response": parsed,
        "rawResponse": content if parsed is None else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
