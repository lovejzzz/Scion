from __future__ import annotations

from scion.corpus_manifest import preference_provenance


def test_preference_provenance_is_domain_bound_and_split_disjoint() -> None:
    train = preference_provenance(task_id="train-tutoring-007", category="tutoring", split="train")
    valid = preference_provenance(task_id="validation-tutoring-007", category="tutoring", split="validation")
    assert train["domain"] == "education-pedagogy"
    assert train["split"] == "train"
    assert valid["split"] == "valid"
    assert train["courseGroupId"] != valid["courseGroupId"]
