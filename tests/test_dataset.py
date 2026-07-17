from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_manifest_has_disjoint_course_groups_and_reproducible_files() -> None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    groups = {name: set(values) for name, values in manifest["courseGroups"].items()}
    assert not groups["train"] & groups["valid"]
    assert not groups["train"] & groups["test"]
    assert not groups["valid"] & groups["test"]
    assert manifest["counts"]["train"]["total"] == 711
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        assert path.stat().st_size == expected["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]


def test_published_provenance_has_no_local_absolute_paths() -> None:
    for path in [ROOT / "data" / "manifest.json", ROOT / "eval" / "fixtures.jsonl"]:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "CodexWorkSpace" not in text


def test_evaluation_fixtures_are_test_only_and_kind_balanced() -> None:
    fixtures = [
        json.loads(line)
        for line in (ROOT / "eval" / "heldout-fixtures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert len(fixtures) == 48
    assert {fixture["split"] for fixture in fixtures} == {"test"}
    assert Counter(fixture["kind"] for fixture in fixtures) == {
        "lesson": 12,
        "mc-item": 12,
        "key-term": 12,
        "source-bundle": 12,
    }
    assert {fixture["courseGroup"] for fixture in fixtures} <= set(manifest["courseGroups"]["test"])

    heldout_manifest = json.loads((ROOT / "eval" / "heldout-manifest.json").read_text(encoding="utf-8"))
    assert heldout_manifest["fixtures"]["count"] == 48
    assert heldout_manifest["fixtures"]["splits"] == {"test": 48}
    assert heldout_manifest["datasetManifest"]["sha256"] == hashlib.sha256(
        (ROOT / "data" / "manifest.json").read_bytes()
    ).hexdigest()
