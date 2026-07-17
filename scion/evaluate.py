"""Held-out HTTP evaluation and CourseMapper-compatible smoke tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .constants import SCION_MODEL_ID
from .contracts import quality_score, validate_response
from .schemas import response_format

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
COURSEMAPPER_CONTRACT_DIRECTIVE = (
    "Follow CourseMapper kernel admission rules. Every multiple-choice option must be distinct, "
    "parallel, and plausible. For every key term, mi must be a plausible misconception, while cx "
    "must directly correct that misconception in distinct wording; neither may repeat df."
)


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


def validate_browser_preflight(
    status: int, headers: Mapping[str, str], *, origin: str
) -> list[str]:
    normalized = {key.casefold(): value for key, value in headers.items()}
    issues = []
    if not 200 <= status < 300:
        issues.append(f"cors-status:{status}")
    if normalized.get("access-control-allow-origin") not in {"*", origin}:
        issues.append("cors-origin")
    methods = {item.strip().upper() for item in normalized.get("access-control-allow-methods", "").split(",")}
    if "POST" not in methods:
        issues.append("cors-post-method")
    allowed_headers = normalized.get("access-control-allow-headers", "").casefold()
    if allowed_headers != "*" and "content-type" not in allowed_headers:
        issues.append("cors-content-type-header")
    return issues


def browser_preflight(endpoint: str, *, origin: str = "http://localhost:5173") -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        method="OPTIONS",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = dict(response.headers.items())
            status = response.status
    except urllib.error.URLError as error:
        return {"status": None, "headers": {}, "origin": origin, "issues": [f"cors-request:{error}"]}
    return {
        "status": status,
        "headers": headers,
        "origin": origin,
        "issues": validate_browser_preflight(status, headers, origin=origin),
    }


def parse_sse_chat(lines: list[bytes]) -> tuple[str, bool]:
    chunks: list[str] = []
    saw_done = False
    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            saw_done = True
            continue
        try:
            event = json.loads(data)
            content = event["choices"][0]["delta"].get("content")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"invalid chat-completions SSE event: {data}") from error
        if isinstance(content, str):
            chunks.append(content)
    return "".join(chunks).strip(), saw_done


def stream_chat_completion(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            content_type = response.headers.get("Content-Type", "")
            content, saw_done = parse_sse_chat(list(response))
            status = response.status
    except urllib.error.URLError as error:
        raise RuntimeError(f"streaming request failed: {error}") from error
    if "text/event-stream" not in content_type:
        raise RuntimeError(f"streaming response has unexpected content type: {content_type}")
    if not saw_done:
        raise RuntimeError("streaming response ended without [DONE]")
    if not content:
        raise RuntimeError("streaming response contained no assistant content")
    return {"status": status, "contentType": content_type, "sawDone": saw_done, "content": content}


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


def coursemapper_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bound = [dict(message) for message in messages]
    if not bound or bound[0].get("role") != "system":
        bound.insert(0, {"role": "system", "content": COURSEMAPPER_CONTRACT_DIRECTIVE})
    else:
        bound[0]["content"] = f"{bound[0].get('content', '').rstrip()}\n\n{COURSEMAPPER_CONTRACT_DIRECTIVE}"
    return bound


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
    guided: bool = True,
) -> dict[str, Any]:
    fixtures = [json.loads(line) for line in fixtures_path.read_text(encoding="utf-8").splitlines() if line]
    selected = _stratified(fixtures, limit)
    fixture_bytes = fixtures_path.read_bytes()
    fixture_set = {
        "sourceSha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "selectedIdsSha256": hashlib.sha256(
            json.dumps([fixture["id"] for fixture in selected], separators=(",", ":")).encode()
        ).hexdigest(),
        "splits": dict(sorted(Counter(str(fixture.get("split")) for fixture in selected).items())),
        "byKind": dict(sorted(Counter(str(fixture["kind"]) for fixture in selected).items())),
        "byDomain": dict(sorted(Counter(str(fixture["domain"]) for fixture in selected).items())),
    }
    wait_until_ready(endpoint)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, fixture in enumerate(selected, start=1):
        request = {
            "model": model,
            "messages": coursemapper_messages(fixture["messages"]),
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": response_format(fixture["kind"]) if guided else {"type": "json_object"},
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
        print(
            f"[{index}/{len(selected)}] {row['id']} {row['kind']} {row['status']} "
            f"quality={row['qualityScore']:.3f} f1={row['referenceF1']:.3f} "
            f"latency={row['latencySeconds']:.3f}s",
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    passing = [row for row in rows if row["status"] == "pass"]
    report = {
        "schemaVersion": 1,
        "endpoint": endpoint,
        "model": model,
        "responseMode": "coursemapper-json-schema" if guided else "unguided-json-object",
        "contractDirectiveSha256": hashlib.sha256(COURSEMAPPER_CONTRACT_DIRECTIVE.encode()).hexdigest(),
        "fixtures": str(fixtures_path.resolve()),
        "count": len(rows),
        "fixtureSet": fixture_set,
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
    expected_kinds = {"key-term": 12, "lesson": 12, "mc-item": 12, "source-bundle": 12}
    base_set = base.get("fixtureSet") or {}
    adapter_set = adapter.get("fixtureSet") or {}
    same_fixture_set = (
        base.get("count") == adapter.get("count") == 48
        and base_set.get("sourceSha256") == adapter_set.get("sourceSha256")
        and base_set.get("selectedIdsSha256") == adapter_set.get("selectedIdsSha256")
    )
    checks = {
        "samePinned48FixtureSet": same_fixture_set,
        "sameCourseMapperSchemaMode": (
            base.get("responseMode") == adapter.get("responseMode") == "coursemapper-json-schema"
        ),
        "sameCourseMapperContractDirective": (
            base.get("contractDirectiveSha256")
            == adapter.get("contractDirectiveSha256")
            == hashlib.sha256(COURSEMAPPER_CONTRACT_DIRECTIVE.encode()).hexdigest()
        ),
        "adapterFixturesTestOnlyAndKindBalanced": (
            adapter_set.get("splits") == {"test": 48} and adapter_set.get("byKind") == expected_kinds
        ),
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
    cors = browser_preflight(endpoint)
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
    stream = stream_chat_completion(
        endpoint,
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 768,
            "response_format": response_format("mc-item"),
            "stream": True,
        },
    )
    content = stream["content"]
    try:
        parsed = json.loads(content)
        issues = cors["issues"] + validate_response("mc-item", parsed)
    except json.JSONDecodeError:
        parsed = None
        issues = cors["issues"] + ["invalid-json"]
    result = {
        "schemaVersion": 1,
        "status": "pass" if not issues else "fail",
        "endpoint": endpoint,
        "model": model,
        "modelsResponse": models,
        "browserPreflight": cors,
        "streamingResponse": {key: value for key, value in stream.items() if key != "content"},
        "issues": issues,
        "response": parsed,
        "rawResponse": content if parsed is None else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
