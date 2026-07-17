"""Locked schema-constrained evaluation through the pinned llama.cpp server."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    COURSEMAPPER_SOURCE_REVISION,
    LITE_RUNTIME_BASE_BYTES,
    LITE_RUNTIME_BASE_FILE,
    LITE_RUNTIME_BASE_ID,
    LITE_RUNTIME_BASE_REVISION,
    LITE_RUNTIME_BASE_SHA256,
    LLAMA_CPP_REVISION,
)
from .local_evaluation import _rows, _summarize_results
from .schemas import contract_response_schema
from .task_contracts import validate_task_response

_RESERVED_MARKERS = (
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coursemapper_gemma4_prompt(messages: list[dict[str, str]]) -> str:
    """Match CourseMapper's pinned text-only Gemma 4 browser formatter."""

    turns = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        content = str(message["content"]).strip()
        for marker in _RESERVED_MARKERS:
            content = content.replace(marker, marker.replace("<|", "< |").replace("|>", "| >"))
        turns.append(f"<|turn>{role}\n{content}<turn|>\n")
    turns.append("<|turn>model\n")
    return "".join(turns)


def _completion(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise RuntimeError("llama.cpp returned an invalid completion response")
    return result


def evaluate_gguf_runtime(
    *,
    endpoint: str,
    fixture_path: Path,
    runtime_base_path: Path,
    output_dir: Path,
    variant: str,
    split: str = "heldout",
    adapter_manifest_path: Path | None = None,
    adapter_id: int = 0,
    adapter_scale: float = 0,
    allow_scale_tuning: bool = False,
) -> dict[str, Any]:
    if variant not in {"base", "adapter"}:
        raise ValueError("variant must be base or adapter")
    if variant == "adapter" and adapter_manifest_path is None:
        raise ValueError("adapter evaluation requires a manifest")
    if runtime_base_path.name != LITE_RUNTIME_BASE_FILE:
        raise RuntimeError("GGUF evaluation base filename does not match the pin")
    if (
        runtime_base_path.stat().st_size != LITE_RUNTIME_BASE_BYTES
        or _sha256(runtime_base_path) != LITE_RUNTIME_BASE_SHA256
    ):
        raise RuntimeError("GGUF evaluation base bytes do not match the pin")

    adapter = None
    if adapter_manifest_path is not None:
        adapter = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
        if adapter.get("adapter", {}).get("format") != "gguf-lora":
            raise RuntimeError("GGUF evaluation adapter manifest has the wrong format")
        if (
            variant == "adapter"
            and not allow_scale_tuning
            and float(adapter["adapter"]["scale"]) != float(adapter_scale)
        ):
            raise RuntimeError("GGUF evaluation scale does not match the package manifest")

    fixtures = [row for row in _rows(fixture_path) if row.get("split") == split]
    if not fixtures:
        raise RuntimeError(f"GGUF evaluation found no {split} fixtures")
    results = []
    for position, fixture in enumerate(fixtures):
        max_tokens = 1900 if fixture["contract"] == "coursemapper-kernel-json-v1" else 700
        payload = {
            "prompt": coursemapper_gemma4_prompt(fixture["messages"]),
            "n_predict": max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 32003 + position,
            "json_schema": contract_response_schema(fixture["contract"]),
            "lora": [{"id": adapter_id, "scale": adapter_scale if variant == "adapter" else 0}],
        }
        generation = _completion(endpoint, payload)
        parsed, issues = validate_task_response(
            fixture["contract"], generation["content"], fixture["oracle"]
        )
        timing = generation.get("timings") or {}
        row = {
            "id": fixture["id"],
            "category": fixture["category"],
            "contract": fixture["contract"],
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "response": parsed,
            "rawText": generation["content"] if parsed is None else None,
            "metrics": {
                "promptTokens": timing.get("prompt_n", 0),
                "generationTokens": timing.get("predicted_n", 0),
                "generationTokensPerSecond": timing.get("predicted_per_second", 0),
            },
        }
        results.append(row)
        print(
            f"[{position + 1}/{len(fixtures)}] {fixture['id']} {row['status']} issues={len(issues)}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    selected_ids = [row["id"] for row in fixtures]
    report = {
        "schemaVersion": 1,
        "protocol": "scion-locked-gguf-schema-evaluation-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "tier": "lite",
        "variant": variant,
        "model": {
            "id": LITE_RUNTIME_BASE_ID,
            "revision": LITE_RUNTIME_BASE_REVISION,
            "file": LITE_RUNTIME_BASE_FILE,
            "bytes": LITE_RUNTIME_BASE_BYTES,
            "sha256": LITE_RUNTIME_BASE_SHA256,
            "adapter": (
                {
                    "id": adapter["adapter"]["id"],
                    "manifestSha256": _sha256(adapter_manifest_path),
                    "scale": adapter_scale,
                }
                if variant == "adapter" and adapter is not None and adapter_manifest_path is not None
                else None
            ),
        },
        "runtime": {
            "id": "llama.cpp-server-json-schema",
            "revision": LLAMA_CPP_REVISION,
            "courseMapperPromptRevision": COURSEMAPPER_SOURCE_REVISION,
            "schemaConstrained": True,
            "validationScaleTuning": allow_scale_tuning,
        },
        "fixtures": {
            "path": str(fixture_path.resolve()),
            "sha256": _sha256(fixture_path),
            "selectedIdsSha256": hashlib.sha256(
                json.dumps(selected_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "count": len(fixtures),
            "split": split,
            "trainingUseForbidden": split == "heldout",
        },
        **_summarize_results(results),
        "resultsSha256": _sha256(result_path),
        "meanTokensPerSecond": (
            sum(row["metrics"]["generationTokensPerSecond"] for row in results) / len(results)
            if results
            else 0
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
