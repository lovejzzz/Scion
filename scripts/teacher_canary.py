#!/usr/bin/env python3
"""Run the primary local teacher through Scion's factual and JSON canary."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scion.local_inference import GenerationSettings, MlxGenerator, snapshot_path
from scion.model_registry import MODEL_PINS

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are Scion's local education-data teacher. Use only the supplied synthetic catalog. "
            "Return one valid JSON object, with no Markdown or hidden reasoning. Never invent a course."
        ),
    },
    {
        "role": "user",
        "content": (
            "Synthetic catalog (authored for this test): CSC 110 has no prerequisites; CSC 210 requires "
            "CSC 110; CSC 310 requires CSC 210 and MAT 120. Dana completed CSC 110 and MAT 120. "
            "Can Dana enroll in CSC 310 now? Return exactly these keys: eligible (boolean), missing "
            "(array of course codes), nextSteps (array), citations (array of catalog course codes)."
        ),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--output", type=Path, default=Path("runs/canaries/qwen36-27b.json"))
    args = parser.parse_args()
    pin = MODEL_PINS["teacher"]
    generator = MlxGenerator(snapshot_path(args.cache_dir, pin), pin)
    result = generator.complete(MESSAGES, GenerationSettings(max_tokens=256, temperature=0, top_p=1))
    issues: list[str] = []
    try:
        value = json.loads(result["text"])
        if value.get("eligible") is not False:
            issues.append("eligibility")
        if value.get("missing") != ["CSC 210"]:
            issues.append("missing-prerequisite")
        if not {"CSC 210", "CSC 310"}.issubset(set(value.get("citations") or [])):
            issues.append("catalog-citations")
    except (json.JSONDecodeError, AttributeError) as error:
        issues.append(f"json:{error}")
    result.update(
        {
            "schemaVersion": 1,
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "generatedAt": datetime.now(UTC).isoformat(),
            "messages": MESSAGES,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
