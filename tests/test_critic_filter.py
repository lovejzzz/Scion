import hashlib
import json
from types import SimpleNamespace

from scion.critic_filter import filter_with_critic


def _teacher_row(task_id: str = "train-prerequisite-reasoning-000") -> dict:
    prompt = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Choose the grounded result."},
    ]
    return {
        "id": task_id,
        "category": "prerequisite-reasoning",
        "contract": "prerequisite-reasoning-json-v1",
        "status": "oracle-admitted-awaiting-critic",
        "chosen": [*prompt, {"role": "assistant", "content": '{"eligible":true}'}],
        "rejected": [*prompt, {"role": "assistant", "content": '{"eligible":false}'}],
    }


def _chosen_label(task_id: str) -> str:
    return "A" if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


class _FakeGenerator:
    def __init__(self, responses: list[dict]):
        self.responses = iter(responses)
        self.calls = 0
        self.pin = SimpleNamespace(model_id="local-critic", revision="abc123", license="Apache-2.0")

    def complete(self, messages, settings):
        self.calls += 1
        response = next(self.responses)
        return {
            "text": json.dumps(response),
            "metrics": {"peakMemoryGb": 1},
            "receipt": {"messages": len(messages), "seed": settings.seed},
        }


def _judgment(preferred: str) -> dict:
    return {
        "preferred": preferred,
        "preferredScore": 5,
        "groundingPass": True,
        "pedagogyPass": True,
        "issues": [],
    }


def test_critic_retries_invalid_shape_once(tmp_path):
    row = _teacher_row()
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(row) + "\n")
    generator = _FakeGenerator([_judgment("candidateA"), _judgment(_chosen_label(row["id"]))])

    result = filter_with_critic(
        generator=generator,
        teacher_path=teacher,
        evidence_path=tmp_path / "evidence.jsonl",
        orpo_path=tmp_path / "train.jsonl",
        split="train",
    )

    assert generator.calls == 2
    assert result["admittedRows"] == 1
    evidence = json.loads((tmp_path / "evidence.jsonl").read_text().strip())
    assert [attempt["validShape"] for attempt in evidence["criticAttempts"]] == [False, True]


def test_critic_does_not_retry_valid_content_rejection(tmp_path):
    row = _teacher_row()
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps(row) + "\n")
    wrong = "B" if _chosen_label(row["id"]) == "A" else "A"
    generator = _FakeGenerator([_judgment(wrong)])

    result = filter_with_critic(
        generator=generator,
        teacher_path=teacher,
        evidence_path=tmp_path / "evidence.jsonl",
        orpo_path=tmp_path / "train.jsonl",
        split="train",
    )

    assert generator.calls == 1
    assert result["rejectedRows"] == 1
