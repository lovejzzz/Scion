from __future__ import annotations

import json
from pathlib import Path

from scion.local_evaluation import compare_evaluations


def test_paired_comparison_requires_a_real_locked_improvement(tmp_path: Path) -> None:
    fixture = {
        "sha256": "a" * 64,
        "selectedIdsSha256": "b" * 64,
        "count": 32,
    }
    base = {
        "fixtures": fixture,
        "passRate": 0.5,
        "issueCount": 20,
        "issueKinds": {"hallucinated-citation": 2},
        "byCategory": {"tool-use": {"passRate": 0.5}},
    }
    adapter = {
        "fixtures": fixture,
        "passRate": 0.75,
        "issueCount": 10,
        "issueKinds": {"hallucinated-citation": 1},
        "byCategory": {"tool-use": {"passRate": 0.75}},
    }
    base_path = tmp_path / "base.json"
    adapter_path = tmp_path / "adapter.json"
    base_path.write_text(json.dumps(base))
    adapter_path.write_text(json.dumps(adapter))
    result = compare_evaluations(base_path, adapter_path, tmp_path / "comparison.json")
    assert result["status"] == "pass"


def test_paired_comparison_rejects_category_regression(tmp_path: Path) -> None:
    fixture = {"sha256": "a" * 64, "selectedIdsSha256": "b" * 64, "count": 32}
    base = {
        "fixtures": fixture,
        "passRate": 0.5,
        "issueCount": 20,
        "issueKinds": {},
        "byCategory": {"safety": {"passRate": 1.0}},
    }
    adapter = {
        "fixtures": fixture,
        "passRate": 0.75,
        "issueCount": 10,
        "issueKinds": {},
        "byCategory": {"safety": {"passRate": 0.5}},
    }
    base_path = tmp_path / "base.json"
    adapter_path = tmp_path / "adapter.json"
    base_path.write_text(json.dumps(base))
    adapter_path.write_text(json.dumps(adapter))
    assert compare_evaluations(base_path, adapter_path, tmp_path / "out.json")["status"] == "fail"
