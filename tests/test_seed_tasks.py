from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scion.seed_tasks import build_seed_tasks, write_seed_tasks


def test_seed_tasks_are_balanced_and_split_isolated() -> None:
    tasks = build_seed_tasks()
    assert Counter(task.split for task in tasks) == {
        "train": 160,
        "validation": 32,
        "preference-test": 32,
        "heldout": 32,
    }
    assert all(task.provenance["containsRealStudentData"] is False for task in tasks)
    assert all(task.provenance["containsRealCatalogData"] is False for task in tasks)
    per_category = {"train": 20, "validation": 4, "preference-test": 4, "heldout": 4}
    for split in per_category:
        assert Counter(task.category for task in tasks if task.split == split) == {
            category: per_category[split]
            for category in (
                "coursemapper-kernel",
                "degree-audit",
                "prerequisite-reasoning",
                "safe-education",
                "schedule-constraints",
                "tool-use",
                "tutoring",
                "uncertainty-grounding",
            )
        }


def test_written_seed_manifest_binds_every_file(tmp_path: Path) -> None:
    manifest = write_seed_tasks(tmp_path)
    for record in manifest["files"]:
        path = tmp_path / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert manifest["origin"].endswith("no closed-model output")
    assert not any("/Users/" in path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json*"))


def test_checked_in_heldout_split_is_never_a_teacher_split() -> None:
    root = Path(__file__).parents[1]
    rows = [json.loads(line) for line in (root / "data/seeds/heldout.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    assert {row["split"] for row in rows} == {"heldout"}


def test_schedule_prompt_has_a_unique_tie_break() -> None:
    root = Path(__file__).parents[1]
    rows = [json.loads(line) for line in (root / "data/seeds/train.jsonl").read_text().splitlines()]
    schedule = next(row for row in rows if row["category"] == "schedule-constraints")
    assert "lexicographically smallest sectionIds array" in schedule["messages"][-1]["content"]
