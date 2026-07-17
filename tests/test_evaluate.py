from __future__ import annotations

import json
from pathlib import Path

from scion.evaluate import compare_reports, reference_f1


def test_reference_f1_is_bounded_and_exact_is_one() -> None:
    value = {"q": "What is a Python dictionary?", "op": ["A mapping", "A list"]}
    assert reference_f1(value, value) == 1.0
    assert 0.0 <= reference_f1(value, {"q": "Explain a mapping"}) <= 1.0


def test_comparison_passes_only_non_regressing_adapter(tmp_path: Path) -> None:
    base = {
        "count": 20,
        "contractPassRate": 0.9,
        "meanReferenceF1": 0.5,
        "meanQualityScore": 0.9,
    }
    adapter = {
        "count": 20,
        "contractPassRate": 0.95,
        "meanReferenceF1": 0.51,
        "meanQualityScore": 0.95,
    }
    base_path = tmp_path / "base.json"
    adapter_path = tmp_path / "adapter.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
    result = compare_reports(base_path, adapter_path, tmp_path / "comparison.json")
    assert result["status"] == "pass"


def test_comparison_rejects_tiny_or_regressing_run(tmp_path: Path) -> None:
    report = {
        "count": 3,
        "contractPassRate": 1.0,
        "meanReferenceF1": 1.0,
        "meanQualityScore": 1.0,
    }
    base = tmp_path / "base.json"
    adapter = tmp_path / "adapter.json"
    base.write_text(json.dumps(report), encoding="utf-8")
    adapter.write_text(json.dumps(report), encoding="utf-8")
    assert compare_reports(base, adapter, tmp_path / "out.json")["status"] == "fail"
