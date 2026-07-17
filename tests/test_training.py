from __future__ import annotations

import json
from pathlib import Path

import pytest

from scion.training import _dataset_identity


def test_dataset_identity_requires_all_three_nonempty_splits(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text(json.dumps({"chosen": [], "rejected": []}) + "\n")
    with pytest.raises(RuntimeError, match="validation"):
        _dataset_identity(tmp_path)
    (tmp_path / "validation.jsonl").write_text(json.dumps({"chosen": [], "rejected": []}) + "\n")
    with pytest.raises(RuntimeError, match="test"):
        _dataset_identity(tmp_path)
    (tmp_path / "test.jsonl").write_text(json.dumps({"chosen": [], "rejected": []}) + "\n")
    identity = _dataset_identity(tmp_path)
    assert identity["train"]["rows"] == 1
    assert identity["validation"]["rows"] == 1
    assert identity["test"]["rows"] == 1
