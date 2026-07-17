from __future__ import annotations

from scion.seed_tasks import build_seed_tasks
from scion.task_contracts import construct_rejected, validate_task_response


def test_tool_contract_and_constructed_negative() -> None:
    task = next(task for task in build_seed_tasks() if task.category == "tool-use")
    oracle = task.oracle
    chosen = {
        "tool": oracle["tool"],
        "arguments": {"courseCodes": oracle["courseCodes"], "term": oracle["term"]},
        "reason": "Current catalog facts are required before answering.",
        "answerDeferred": True,
    }
    parsed, issues = validate_task_response(task.contract, chosen, oracle)
    assert parsed == chosen
    assert issues == []
    rejected = construct_rejected(task.contract, chosen, oracle)
    assert "answer-not-deferred" in validate_task_response(task.contract, rejected, oracle)[1]


def test_hallucinated_citation_is_rejected() -> None:
    task = next(task for task in build_seed_tasks() if task.category == "uncertainty-grounding")
    value = {
        "answer": "insufficient_information",
        "known": ["The course is offered in autumn."],
        "needed": [task.oracle["neededContains"]],
        "nextAction": "Look up the missing official field.",
        "citations": ["invented-source"],
    }
    issues = validate_task_response(task.contract, value, task.oracle)[1]
    assert "hallucinated-citation" in issues
    assert "missing-required-citation" in issues
