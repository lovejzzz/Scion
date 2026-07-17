from __future__ import annotations

import json
from pathlib import Path

from scion.contracts import quality_score, validate_response

ROOT = Path(__file__).parents[1]


def test_all_checked_in_training_outputs_pass_their_contract() -> None:
    for split in ("train", "valid", "test"):
        rows = (ROOT / "data" / "metadata" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        training = (ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == len(training)
        for raw_training, raw_metadata in zip(training, rows, strict=True):
            row = json.loads(raw_training)
            identity = json.loads(raw_metadata)
            response = row["messages"][-1]["content"]
            assert not validate_response(identity["kind"], response)


def test_invalid_json_scores_zero() -> None:
    assert quality_score("lesson", "not JSON") == 0.0


def test_fixture_references_are_admitted() -> None:
    for raw in (ROOT / "eval" / "fixtures.jsonl").read_text(encoding="utf-8").splitlines():
        fixture = json.loads(raw)
        assert not validate_response(fixture["kind"], fixture["expected"])
