"""JSON Schemas forwarded by CourseMapper for constrained local generation."""

from __future__ import annotations

from typing import Any


def _text(minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def key_term_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tr": _text(2, 64),
            "df": _text(20, 256),
            "eg": _text(12, 192),
            "mi": _text(12, 160),
            "cx": _text(18, 192),
        },
        "required": ["tr", "df", "eg", "mi", "cx"],
        "additionalProperties": False,
    }


def mc_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "q": _text(12, 256),
            "op": {
                "type": "array",
                "items": _text(1, 160),
                "minItems": 4,
                "maxItems": 4,
                "uniqueItems": True,
            },
            "ai": {"type": "integer", "minimum": 0, "maximum": 3},
            "ex": _text(18, 400),
            "fi": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": 1,
                "maxItems": 2,
            },
        },
        "required": ["q", "op", "ai", "ex", "fi"],
        "additionalProperties": False,
    }


def lesson_schema() -> dict[str, Any]:
    term = key_term_schema()
    item = mc_item_schema()
    lesson = {
        "type": "object",
        "properties": {
            "lessonId": _text(3, 64),
            "facts": {"type": "array", "items": _text(20, 192), "minItems": 5, "maxItems": 5},
            "keyTerms": {"type": "array", "items": term, "minItems": 3, "maxItems": 3},
            "scenario": {
                "type": "object",
                "properties": {"su": _text(35, 480), "ma": _text(20, 224)},
                "required": ["su", "ma"],
                "additionalProperties": False,
            },
            "discussionPrompt": {
                "type": "object",
                "properties": {
                    "pr": _text(20, 224),
                    "tn": _text(20, 224),
                    "po": {
                        "type": "array",
                        "items": _text(1, 160),
                        "minItems": 3,
                        "maxItems": 3,
                    },
                },
                "required": ["pr", "tn", "po"],
                "additionalProperties": False,
            },
            "assignmentCore": {
                "type": "object",
                "properties": {
                    "td": _text(45, 400),
                    "pa": {
                        "type": "array",
                        "items": _text(1, 128),
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["td", "pa"],
                "additionalProperties": False,
            },
            "workedExample": {
                "type": "object",
                "properties": {
                    "wp": _text(20, 256),
                    "ws": {"type": "array", "items": _text(8, 128), "minItems": 3, "maxItems": 3},
                    "wr": _text(1, 128),
                },
                "required": ["wp", "ws", "wr"],
                "additionalProperties": False,
            },
            "mc": {"type": "array", "items": item, "minItems": 4, "maxItems": 4},
            "studyGuide": {
                "type": "object",
                "properties": {"sm": _text(60, 480), "rs": _text(30, 288)},
                "required": ["sm", "rs"],
                "additionalProperties": False,
            },
        },
        "required": [
            "lessonId",
            "facts",
            "keyTerms",
            "scenario",
            "discussionPrompt",
            "assignmentCore",
            "mc",
            "studyGuide",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"lessons": {"type": "array", "items": lesson, "minItems": 1, "maxItems": 1}},
        "required": ["lessons"],
        "additionalProperties": False,
    }


def response_schema(kind: str) -> dict[str, Any]:
    if kind == "lesson":
        return lesson_schema()
    if kind == "mc-item":
        return mc_item_schema()
    if kind == "key-term":
        return key_term_schema()
    if kind == "source-bundle":
        return {
            "type": "object",
            "properties": {
                "mcItems": {"type": "array", "items": mc_item_schema(), "minItems": 1},
                "keyTerms": {"type": "array", "items": key_term_schema(), "minItems": 1},
            },
            "required": ["mcItems", "keyTerms"],
            "additionalProperties": False,
        }
    raise ValueError(f"unknown CourseMapper response kind: {kind}")


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _strings() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def contract_response_schema(contract: str) -> dict[str, Any]:
    """Return the structural decoder contract for a locked evaluation task."""

    text = {"type": "string"}
    strings = _strings()
    if contract == "prerequisite-json-v1":
        return _object(
            {
                "eligible": {"type": "boolean"},
                "missingImmediate": strings,
                "recommendedSequence": strings,
                "explanation": text,
                "citations": strings,
            }
        )
    if contract == "schedule-json-v1":
        return _object(
            {
                "feasible": {"type": "boolean"},
                "sectionIds": strings,
                "totalCredits": {"type": "integer"},
                "conflicts": strings,
                "explanation": text,
                "citations": strings,
            }
        )
    if contract == "degree-audit-json-v1":
        return _object(
            {
                "complete": {"type": "boolean"},
                "remainingByGroup": {"type": "object", "additionalProperties": {"type": "integer"}},
                "eligibleOptions": {"type": "object", "additionalProperties": strings},
                "explanation": text,
                "citations": strings,
            }
        )
    if contract == "uncertainty-json-v1":
        return _object(
            {"answer": text, "known": strings, "needed": strings, "nextAction": text, "citations": strings}
        )
    if contract == "tutor-json-v1":
        return _object(
            {
                "diagnosis": text,
                "hint": text,
                "workedExplanation": text,
                "checkQuestion": text,
                "checkAnswer": text,
                "citations": strings,
            }
        )
    if contract == "tool-call-json-v1":
        return _object(
            {
                "tool": text,
                "arguments": _object({"courseCodes": strings, "term": text}),
                "reason": text,
                "answerDeferred": {"type": "boolean"},
            }
        )
    if contract == "safety-json-v1":
        return _object(
            {"boundary": text, "cannotDo": text, "canHelpWith": strings, "nextStep": text}
        )
    if contract == "coursemapper-kernel-json-v1":
        return response_schema("lesson")
    raise ValueError(f"unknown locked evaluation contract: {contract}")


def response_format(kind: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"scion_{kind.replace('-', '_')}",
            "strict": True,
            "schema": response_schema(kind),
        },
    }
