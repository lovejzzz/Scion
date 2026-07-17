"""Audit ORPO sequence lengths exactly as the pinned MLX-VLM trainer formats them."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1)]


def shared_preference_prompt(row: dict[str, Any]) -> bool:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    return (
        isinstance(chosen, list)
        and isinstance(rejected, list)
        and len(chosen) >= 2
        and len(chosen) == len(rejected)
        and chosen[:-1] == rejected[:-1]
        and chosen[-1].get("role") == "assistant"
        and rejected[-1].get("role") == "assistant"
    )


def audit_orpo_lengths(
    *, model_path: Path, data_dir: Path, output_path: Path, max_sequence_length: int
) -> dict[str, Any]:
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config, load_processor

    config = load_config(model_path, trust_remote_code=True)
    processor = load_processor(model_path, add_detokenizer=False, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    records: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    category_maxima: Counter[str] = Counter()

    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing ORPO split for token audit: {path}")
        rows = _read_jsonl(path)
        files[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for row in rows:
            if not shared_preference_prompt(row):
                raise RuntimeError(f"preference prompt mismatch: {row.get('id')}")
            for candidate in ("chosen", "rejected"):
                prompt = apply_chat_template(
                    processor,
                    config,
                    row[candidate],
                    add_generation_prompt=False,
                    num_images=0,
                )
                encoded = tokenizer(
                    [prompt], add_special_tokens=False, padding=False, return_attention_mask=False
                )
                token_ids = encoded["input_ids"][0]
                token_count = len(token_ids)
                record = {
                    "id": row.get("id"),
                    "split": split,
                    "category": row.get("category"),
                    "candidate": candidate,
                    "tokens": token_count,
                }
                records.append(record)
                category_maxima[row.get("category", "unknown")] = max(
                    category_maxima[row.get("category", "unknown")], token_count
                )
                if token_count > max_sequence_length:
                    violations.append(record)

    lengths = [record["tokens"] for record in records]
    receipt = {
        "schemaVersion": 1,
        "protocol": "scion-mlx-vlm-orpo-token-audit-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "status": "pass" if not violations else "fail",
        "modelSnapshotRevision": model_path.resolve().name,
        "modelType": config.get("model_type"),
        "maxSequenceLength": max_sequence_length,
        "sequenceCount": len(records),
        "statistics": {
            "minimum": min(lengths, default=0),
            "median": _percentile(lengths, 0.5),
            "p95": _percentile(lengths, 0.95),
            "maximum": max(lengths, default=0),
            "categoryMaxima": dict(sorted(category_maxima.items())),
        },
        "violations": violations,
        "files": files,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if violations:
        raise RuntimeError(
            f"{len(violations)} ORPO sequences exceed {max_sequence_length} tokens; see {output_path}"
        )
    return receipt
