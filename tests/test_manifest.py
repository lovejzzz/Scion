from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scion.manifest import build_release_manifest


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_release_manifest_publishes_portable_receipts_and_package_size(tmp_path: Path) -> None:
    adapter = tmp_path / "scion.gguf"
    adapter.write_bytes(b"GGUF" + bytes(1020))
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    dataset = _write(tmp_path / "dataset.json", {"limitations": ["test limitation"]})
    training = _write(tmp_path / "training.json", {"status": "trained-unpromoted"})
    conversion = _write(
        tmp_path / "conversion.json",
        {"status": "converted-unpromoted", "artifact": {"sha256": digest}},
    )
    comparison = _write(
        tmp_path / "comparison.json",
        {"status": "pass", "adapter": {"count": 20, "contractPassRate": 1.0}},
    )
    smoke = _write(tmp_path / "smoke.json", {"status": "pass"})
    output = tmp_path / "scion.manifest.json"
    result = build_release_manifest(
        adapter=adapter,
        dataset_manifest=dataset,
        training_result=training,
        conversion_receipt=conversion,
        comparison=comparison,
        smoke=smoke,
        output=output,
    )
    receipt_bytes = sum(path.stat().st_size for path in (tmp_path / "receipts").glob("*.json"))
    assert result["status"] == "promoted"
    assert (
        result["delivery"]["packageBytes"] == adapter.stat().st_size + receipt_bytes + output.stat().st_size
    )
    assert all(not item["path"].startswith("/") for item in result["receipts"].values())
