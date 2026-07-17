from __future__ import annotations

from scion.token_audit import shared_preference_prompt


def test_shared_preference_prompt_rejects_prompt_drift() -> None:
    base = [
        {"role": "system", "content": "grounded"},
        {"role": "user", "content": "question"},
    ]
    valid = {
        "chosen": [*base, {"role": "assistant", "content": "right"}],
        "rejected": [*base, {"role": "assistant", "content": "wrong"}],
    }
    assert shared_preference_prompt(valid)
    invalid = {**valid, "rejected": [{"role": "user", "content": "different"}, valid["rejected"][-1]]}
    assert not shared_preference_prompt(invalid)
