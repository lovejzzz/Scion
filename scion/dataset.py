"""Build a deterministic, provenance-bound CourseMapper education corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    COURSEMAPPER_SOURCE_REVISION,
    LEGACY_SCION_SOURCE_REVISION,
    TRAIN_BASE_ID,
    TRAIN_BASE_REVISION,
)
from .contracts import validate_response, words

SYSTEM_FULL = (
    "You are CourseMapper Scion, a precise university subject-matter expert and assessment writer. "
    "Return the final JSON immediately. Use only the supplied topic and source context. "
    "Never invent citations, URLs, page numbers, statistics, or named studies. "
    "Return only valid JSON with no Markdown or commentary."
)
SYSTEM_ATOM = (
    "You are CourseMapper Scion, a university subject-matter and assessment writer. Return one accurate, "
    "learner-ready JSON object and no other text."
)

LEGACY_SPLITS = {
    "geology": "valid",
    "nutrition-101": "valid",
    "world-lit": "test",
    "nursing-fundamentals": "test",
    "stats-intro": "test",
}

SOURCE_SPLITS = {
    "igneous-volcanic-processes": "train",
    "python-control-flow-workshop": "train",
    "harmony-form-analysis": "train",
    "ux-prototyping-accessibility-lab": "train",
    "tectonics-seismology-field-lab": "valid",
    "python-data-structures-lab": "valid",
    "music-notation-ear-training": "valid",
    "ux-research-synthesis-studio": "valid",
    "earth-materials-history-lab": "test",
    "python-program-architecture-studio": "test",
    "tonal-analysis-integration-studio": "test",
    "ux-evidence-to-prototype-capstone": "test",
}

LEGACY_DOMAINS = {
    "astro-101": "astronomy",
    "cs-python": "computer-science",
    "econ-intro": "economics",
    "geology": "geology",
    "mandarin": "language-learning",
    "music-theory": "music-theory",
    "nursing-fundamentals": "nursing",
    "nutrition-101": "nutrition",
    "psych-101": "psychology",
    "stats-intro": "statistics",
    "world-lit": "world-literature",
}


@dataclass(frozen=True)
class Example:
    split: str
    kind: str
    domain: str
    course_group: str
    source: str
    messages: tuple[dict[str, str], ...]
    provenance: dict[str, Any]

    @property
    def identity(self) -> str:
        canonical = json.dumps(
            {"kind": self.kind, "messages": self.messages},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def training_row(self) -> dict[str, Any]:
        return {"messages": list(self.messages)}

    def metadata_row(self) -> dict[str, Any]:
        return {
            "id": f"scion-{self.identity[:20]}",
            "split": self.split,
            "kind": self.kind,
            "domain": self.domain,
            "courseGroup": self.course_group,
            "source": self.source,
            "messagesSha256": self.identity,
            "provenance": self.provenance,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coursemapper_path(path: Path) -> str:
    parts = path.parts
    if "evaluation" in parts:
        return Path(*parts[parts.index("evaluation") :]).as_posix()
    return path.name


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def _clean_text(value: Any, minimum: int, suffix: str) -> str:
    text = str(value or "").strip()
    if len(text) < minimum:
        text = f"{text.rstrip('.')} {suffix}".strip()
    return text


def _best_fact_indexes(item: dict[str, Any], facts: list[str]) -> list[int]:
    answer = item.get("ai")
    options = item.get("op") if isinstance(item.get("op"), list) else []
    keyed = options[answer] if isinstance(answer, int) and 0 <= answer < len(options) else ""
    evidence_words = words(f"{item.get('q', '')} {keyed} {item.get('ex', '')}")
    scored = sorted(
        ((len(evidence_words & words(fact)), index) for index, fact in enumerate(facts)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    if not scored:
        return [0]
    selected = [scored[0][1]]
    if len(scored) > 1 and scored[1][0] > 0 and scored[1][0] >= max(1, scored[0][0] * 0.65):
        selected.append(scored[1][1])
    return sorted(selected)


def _normalize_term(term: dict[str, Any]) -> dict[str, str]:
    return {
        "tr": _clean_text(term.get("tr"), 2, "term"),
        "df": _clean_text(term.get("df"), 20, "in this disciplinary context."),
        "eg": _clean_text(term.get("eg"), 12, "in a concrete case."),
        "mi": _clean_text(term.get("mi"), 12, "is a common learner misconception."),
        "cx": _clean_text(term.get("cx"), 18, "is the accurate correction."),
    }


def _normalize_mc(item: dict[str, Any], facts: list[str]) -> dict[str, Any]:
    normalized = {
        "q": _clean_text(item.get("q"), 12, "Which answer is most accurate?"),
        "op": [str(option).strip() for option in (item.get("op") or [])[:4]],
        "ai": item.get("ai"),
        "ex": _clean_text(item.get("ex"), 18, "This choice best follows the supplied subject facts."),
    }
    normalized["fi"] = _best_fact_indexes(normalized, facts)
    return normalized


def _normalize_positions(values: Any) -> list[str]:
    positions = [str(value).strip() for value in values or [] if str(value).strip()][:3]
    defaults = [
        "A main position follows the strongest disciplinary evidence.",
        "A contrasting position gives greater weight to the competing constraint.",
        "A conditional position changes when the evidence or context changes.",
    ]
    while len(positions) < 3:
        positions.append(defaults[len(positions)])
    return positions


def _normalize_parameters(values: Any) -> list[str]:
    parameters = [str(value).strip() for value in values or [] if str(value).strip()][:4]
    defaults = [
        "Analyze one clearly bounded case.",
        "Submit a concise written response.",
        "Use the supplied subject evidence.",
        "Complete the response within 500 words.",
    ]
    while len(parameters) < 4:
        parameters.append(defaults[len(parameters)])
    return parameters


def normalize_lesson(raw: dict[str, Any]) -> dict[str, Any]:
    facts = [_clean_text(fact, 20, "is a specific subject claim.") for fact in (raw.get("facts") or [])[:5]]
    if len(facts) != 5:
        raise ValueError(f"lesson {raw.get('lessonId')} does not contain five usable facts")
    terms = [_normalize_term(term) for term in (raw.get("keyTerms") or [])[:3]]
    if len(terms) != 3:
        raise ValueError(f"lesson {raw.get('lessonId')} does not contain three usable key terms")
    items = [_normalize_mc(item, facts) for item in (raw.get("mc") or [])[:4]]
    if len(items) != 4:
        raise ValueError(f"lesson {raw.get('lessonId')} does not contain four usable MC items")

    scenario = raw.get("scenario") or {}
    discussion = raw.get("discussionPrompt") or {}
    assignment = raw.get("assignmentCore") or {}
    question = _clean_text(discussion.get("pr"), 20, "What position is best supported")
    if not question.endswith("?"):
        question += "?"
    guide = raw.get("studyGuide") or {}
    summary = str(guide.get("sm") or "").strip()
    if len(summary) < 60:
        summary = (
            f"Connect {terms[0]['tr']}, {terms[1]['tr']}, and {terms[2]['tr']} by explaining these claims: "
            f"{facts[0]} {facts[1]}"
        )[:300]
    strategy = str(guide.get("rs") or "").strip()
    if len(strategy) < 30:
        strategy = " ".join(
            [
                "Retrieve each definition and correction,",
                "then apply the three terms to the scenario without notes.",
            ]
        )

    normalized: dict[str, Any] = {
        "lessonId": _clean_text(raw.get("lessonId"), 3, "lesson-1"),
        "facts": facts,
        "keyTerms": terms,
        "scenario": {
            "su": _clean_text(
                scenario.get("su"), 35, "The case creates a concrete decision with competing constraints."
            ),
            "ma": _clean_text(scenario.get("ma"), 20, "The supplied case record and observations."),
        },
        "discussionPrompt": {
            "pr": question,
            "tn": _clean_text(
                discussion.get("tn"), 20, "The evidence supports more than one defensible priority."
            ),
            "po": _normalize_positions(discussion.get("po")),
        },
        "assignmentCore": {
            "td": _clean_text(
                assignment.get("td"),
                45,
                "Analyze the supplied case and produce a concise evidence-based disciplinary response.",
            ),
            "pa": _normalize_parameters(assignment.get("pa")),
        },
        "mc": items,
        "studyGuide": {"sm": summary, "rs": strategy},
    }
    worked = raw.get("workedExample")
    if isinstance(worked, dict) and worked:
        normalized["workedExample"] = worked
    return normalized


def _legacy_context(prompt: str, course_id: str, lesson_id: str) -> tuple[str, str, str]:
    course = re.search(r"^Course:\s*(.+)$", prompt, re.MULTILINE)
    title = re.search(rf'"lessonId":"{re.escape(lesson_id)}","title":"([^"]+)"', prompt)
    topics = re.search(rf'"lessonId":"{re.escape(lesson_id)}".*?"topics":"([^"]*)"', prompt)
    return (
        course.group(1).strip() if course else course_id.replace("-", " ").title(),
        title.group(1).strip() if title else lesson_id.replace("-", " ").title(),
        topics.group(1).strip() if topics else "",
    )


def _full_prompt(course: str, title: str, topics: str, lesson_id: str, *, compact: bool) -> str:
    context = {"lessonId": lesson_id, "title": title, "topics": topics}
    if compact:
        preamble = "Author one complete CourseMapper knowledge kernel for this lesson."
    else:
        preamble = (
            "Author the lesson substance used by CourseMapper to compile plans, assessments, "
            "assignments, and study materials. Facts must be subject claims, not descriptions "
            "of teaching process."
        )
    return f"""COURSE: {course}
LESSON TO AUTHOR: {json.dumps(context, ensure_ascii=False, separators=(",", ":"))}

TASK: {preamble}

Return {json.dumps({"lessons": [{"lessonId": lesson_id}]}, separators=(",", ":"))} with exactly one lesson.
The lesson must contain exactly 5 facts, 3 keyTerms, one scenario, one discussionPrompt, one assignmentCore,
exactly 4 mc items, and one studyGuide. Use abbreviated keys tr/df/eg/mi/cx and q/op/ai/ex/fi.
Each mc item has four options, one valid ai, a teaching explanation, and fi with 1-2 zero-based fact indexes.
discussionPrompt.po has exactly 3 positions. assignmentCore.pa has exactly 4 constraints.
studyGuide has sm (60-300 characters) and rs (30-200 characters).
Return only the JSON object."""


def _atom_messages(
    kind: str, domain: str, title: str, facts: list[str], output: dict[str, Any]
) -> tuple[dict[str, str], ...]:
    if kind == "mc-item":
        instruction = (
            "Write one evidence-bearing multiple-choice item as JSON with q, op "
            "(exactly four options), ai, ex, and fi (one or two zero-based indexes "
            "into the supplied facts)."
        )
    else:
        instruction = "Write one learner-ready key term as JSON with tr, df, eg, mi, and cx."
    prompt = (
        f"Course domain: {domain}. Lesson focus: {title}.\n"
        f"Supplied facts: {json.dumps(facts, ensure_ascii=False)}\n{instruction}"
    )
    return (
        {"role": "system", "content": SYSTEM_ATOM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(output, ensure_ascii=False, separators=(",", ":"))},
    )


def legacy_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    for row_number, row in enumerate(_load_jsonl(path), start=1):
        if row.get("kind") != "lesson":
            continue
        course_group = str(row.get("courseId") or "unknown")
        domain = LEGACY_DOMAINS.get(course_group, course_group)
        split = LEGACY_SPLITS.get(course_group, "train")
        chosen = json.loads(row["chosen"])
        lessons = chosen.get("lessons") if isinstance(chosen, dict) else None
        if not isinstance(lessons, list) or len(lessons) != 1:
            raise ValueError(f"legacy row {row_number} must contain exactly one lesson")
        lesson = normalize_lesson(lessons[0])
        response = json.dumps({"lessons": [lesson]}, ensure_ascii=False, separators=(",", ":"))
        issues = validate_response("lesson", response)
        if issues:
            raise ValueError(f"legacy row {row_number} failed normalized contract: {issues[:5]}")
        course, title, topics = _legacy_context(
            str(row.get("prompt") or ""), course_group, lesson["lessonId"]
        )
        provenance = {
            "repository": "lovejzzz/Scion",
            "revision": LEGACY_SCION_SOURCE_REVISION,
            "sourcePath": "data/preference-pairs-full.jsonl",
            "sourceRow": row_number,
            "selection": "teacher-chosen-side-normalized-to-current-contract",
        }
        for compact in (False, True):
            messages = (
                {"role": "system", "content": SYSTEM_FULL},
                {
                    "role": "user",
                    "content": _full_prompt(course, title, topics, lesson["lessonId"], compact=compact),
                },
                {"role": "assistant", "content": response},
            )
            examples.append(
                Example(split, "lesson", domain, course_group, "teacher-corpus", messages, provenance)
            )
        for item in lesson["mc"]:
            examples.append(
                Example(
                    split,
                    "mc-item",
                    domain,
                    course_group,
                    "teacher-corpus",
                    _atom_messages("mc-item", domain, title, lesson["facts"], item),
                    provenance,
                )
            )
        for term in lesson["keyTerms"]:
            examples.append(
                Example(
                    split,
                    "key-term",
                    domain,
                    course_group,
                    "teacher-corpus",
                    _atom_messages("key-term", domain, title, lesson["facts"], term),
                    provenance,
                )
            )
    return examples


def approved_examples(path: Path) -> list[Example]:
    examples: list[Example] = []
    for row_number, row in enumerate(_load_jsonl(path), start=1):
        kind = str(row.get("kind") or "")
        if kind not in {"mc-item", "key-term"}:
            continue
        response = str(row.get("chosen") or "")
        issues = validate_response(kind, response)
        if issues:
            raise ValueError(f"approved row {row_number} failed contract: {issues[:5]}")
        course_group = str(row.get("courseGroupId") or row.get("courseId") or "unknown")
        split = SOURCE_SPLITS.get(course_group)
        if split is None:
            raise ValueError(f"approved row {row_number} uses an unregistered course group: {course_group}")
        domain = str(row.get("domain") or "unknown")
        evidence = row.get("preferenceEvidence") or {}
        provenance = {
            "repository": "lovejzzz/CourseMapper",
            "revision": COURSEMAPPER_SOURCE_REVISION,
            "sourcePath": "evaluation/scion-adapters/evidence/codex-approved-preferences-v0.16.42.jsonl",
            "sourceRow": row_number,
            "selection": "stable-order-swapped-judge-preference",
            "trainingPairSha256": evidence.get("trainingPairSha256"),
            "humanEvidence": False,
        }
        messages = (
            {"role": "system", "content": SYSTEM_ATOM},
            {"role": "user", "content": str(row.get("prompt") or "").strip()},
            {"role": "assistant", "content": response},
        )
        examples.append(
            Example(split, kind, domain, course_group, "reviewed-preference", messages, provenance)
        )
    return examples


def _source_prompt(domain: str, kernel: dict[str, Any]) -> str:
    facts = [fact.get("text") for fact in kernel.get("facts") or []]
    return f"""Course domain: {domain}
Lesson focus: {kernel.get("term")}
Definition: {kernel.get("definition")}
Source facts: {json.dumps(facts, ensure_ascii=False)}

Using only the definition and facts, return one JSON object with mcItems and keyTerms.
Each mc item has q, exactly four op, ai, ex, and fi with 1-2 zero-based source-fact indexes.
Each key term has tr, df, eg, mi, and cx. Return only JSON."""


def _normalize_source_response(response: dict[str, Any], fact_count: int) -> dict[str, Any]:
    items = []
    for raw in response.get("mcItems") or []:
        item = {key: raw.get(key) for key in ("q", "op", "ai", "ex")}
        indexes = raw.get("fi", raw.get("sourceFactIndexes")) or []
        valid = []
        for index in indexes:
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < fact_count
                and index not in valid
            ):
                valid.append(index)
        item["fi"] = valid[:2] or [0]
        items.append(item)
    terms = [_normalize_term(term) for term in response.get("keyTerms") or []]
    return {"mcItems": items, "keyTerms": terms}


def source_capture_examples(paths: Iterable[Path]) -> tuple[list[Example], list[dict[str, Any]]]:
    examples: list[Example] = []
    fixtures: list[dict[str, Any]] = []
    for path in sorted(paths):
        capture = json.loads(path.read_text(encoding="utf-8"))
        metadata = capture.get("scionSourceCapture") or {}
        course_group = str(metadata.get("courseGroupId") or path.stem.removesuffix("-reference"))
        split = SOURCE_SPLITS.get(course_group)
        if split is None:
            raise ValueError(f"source capture uses an unregistered course group: {course_group}")
        domain = str((metadata.get("sourcePacket") or {}).get("domain") or "")
        if not domain:
            domain = {
                "cs": "computer-science",
                "ux": "user-experience-design",
                "geo": "geology",
                "music": "music-theory",
            }.get(
                str((metadata.get("sourcePacket") or {}).get("kernels", [{}])[0].get("id", "")).split("/")[0],
                "unknown",
            )
        kernels = {
            kernel.get("id"): kernel for kernel in (metadata.get("sourcePacket") or {}).get("kernels") or []
        }
        calls = (metadata.get("compilerRecovery") or {}).get("rawCalls") or []
        for call_index, call in enumerate(calls):
            kernel_id = call.get("kernelId")
            kernel = kernels.get(kernel_id)
            response = call.get("admittedResponse") or (call.get("response") or {})
            if not kernel or not response:
                raise ValueError(f"{path}: incomplete source call {call_index}")
            fact_count = len(kernel.get("facts") or [])
            normalized = _normalize_source_response(response, fact_count)
            prompt = _source_prompt(domain, kernel)
            provenance = {
                "repository": "lovejzzz/CourseMapper",
                "revision": COURSEMAPPER_SOURCE_REVISION,
                "sourcePath": _coursemapper_path(path),
                "promptId": call.get("promptId"),
                "promptSha256": call.get("promptSha256"),
                "selection": "source-grounded-compiler-admitted-reference",
                "sourceLicense": kernel.get("license"),
                "attribution": kernel.get("attribution"),
            }
            if not normalized["mcItems"] and not normalized["keyTerms"]:
                raise ValueError(f"{path}: source call {call_index} has no admitted atoms")
            if normalized["mcItems"] and normalized["keyTerms"]:
                response_text = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
                issues = validate_response("source-bundle", response_text)
                if issues:
                    raise ValueError(f"{path}: source call {call_index} failed contract: {issues[:5]}")
                bundle_messages = (
                    {"role": "system", "content": SYSTEM_ATOM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                )
                examples.append(
                    Example(
                        split,
                        "source-bundle",
                        domain,
                        course_group,
                        "source-grounded-reference",
                        bundle_messages,
                        provenance,
                    )
                )
                fixtures.append(
                    {
                        "id": (
                            "source-"
                            + hashlib.sha256((course_group + str(kernel_id)).encode()).hexdigest()[:16]
                        ),
                        "split": split,
                        "kind": "source-bundle",
                        "domain": domain,
                        "courseGroup": course_group,
                        "messages": list(bundle_messages[:-1]),
                        "expected": normalized,
                        "provenance": provenance,
                    }
                )
            for item in normalized["mcItems"]:
                examples.append(
                    Example(
                        split,
                        "mc-item",
                        domain,
                        course_group,
                        "source-grounded-reference",
                        _atom_messages(
                            "mc-item",
                            domain,
                            str(kernel.get("term") or ""),
                            [str(fact.get("text") or "") for fact in kernel.get("facts") or []],
                            item,
                        ),
                        provenance,
                    )
                )
            for term in normalized["keyTerms"]:
                examples.append(
                    Example(
                        split,
                        "key-term",
                        domain,
                        course_group,
                        "source-grounded-reference",
                        _atom_messages(
                            "key-term",
                            domain,
                            str(kernel.get("term") or ""),
                            [str(fact.get("text") or "") for fact in kernel.get("facts") or []],
                            term,
                        ),
                        provenance,
                    )
                )
            if not (normalized["mcItems"] and normalized["keyTerms"]):
                kind = "mc-item" if normalized["mcItems"] else "key-term"
                output = normalized["mcItems"][0] if normalized["mcItems"] else normalized["keyTerms"][0]
                messages = _atom_messages(
                    kind,
                    domain,
                    str(kernel.get("term") or ""),
                    [str(fact.get("text") or "") for fact in kernel.get("facts") or []],
                    output,
                )
                fixtures.append(
                    {
                        "id": (
                            "source-"
                            + hashlib.sha256((course_group + str(kernel_id)).encode()).hexdigest()[:16]
                        ),
                        "split": split,
                        "kind": kind,
                        "domain": domain,
                        "courseGroup": course_group,
                        "messages": list(messages[:-1]),
                        "expected": output,
                        "provenance": provenance,
                    }
                )
    return examples, fixtures


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
    )
    _atomic_write(path, content)


def heldout_fixtures(examples: Iterable[Example], *, per_kind: int = 12) -> list[dict[str, Any]]:
    """Select a deterministic, kind-balanced test-only evaluation set."""
    buckets: dict[str, dict[str, list[Example]]] = defaultdict(lambda: defaultdict(list))
    for example in sorted(examples, key=lambda item: item.identity):
        if example.split != "test":
            raise ValueError("held-out fixtures may only be selected from the test split")
        buckets[example.kind][example.domain].append(example)

    expected_kinds = {"lesson", "mc-item", "key-term", "source-bundle"}
    if set(buckets) != expected_kinds:
        raise ValueError(f"held-out fixture kinds do not match the CourseMapper contracts: {set(buckets)}")
    selected: list[Example] = []
    for kind in sorted(expected_kinds):
        domain_buckets = buckets[kind]
        domains = sorted(domain_buckets)
        chosen: list[Example] = []
        cursor = 0
        while len(chosen) < per_kind and any(domain_buckets.values()):
            domain = domains[cursor % len(domains)]
            if domain_buckets[domain]:
                chosen.append(domain_buckets[domain].pop(0))
            cursor += 1
        if len(chosen) != per_kind:
            raise ValueError(f"test split has only {len(chosen)} usable {kind} fixtures")
        selected.extend(chosen)

    return [
        {
            "id": f"heldout-{example.identity[:16]}",
            "split": example.split,
            "kind": example.kind,
            "domain": example.domain,
            "courseGroup": example.course_group,
            "messages": list(example.messages[:-1]),
            "expected": json.loads(example.messages[-1]["content"]),
            "provenance": example.provenance,
        }
        for example in selected
    ]


def build_dataset(
    *,
    legacy_jsonl: Path,
    approved_jsonl: Path,
    source_capture_paths: Iterable[Path],
    output_dir: Path,
    eval_output: Path,
    heldout_output: Path,
) -> dict[str, Any]:
    source_paths = tuple(sorted(source_capture_paths))
    source_examples, source_fixtures = source_capture_examples(source_paths)
    examples = legacy_examples(legacy_jsonl) + approved_examples(approved_jsonl) + source_examples
    deduplicated: dict[str, Example] = {}
    for example in examples:
        prior = deduplicated.get(example.identity)
        if prior and prior.split != example.split:
            raise ValueError(f"cross-split duplicate: {example.identity}")
        deduplicated.setdefault(example.identity, example)
    examples = sorted(deduplicated.values(), key=lambda example: (example.split, example.identity))

    split_rows: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        response = example.messages[-1]["content"]
        issues = validate_response(example.kind, response)
        if issues:
            raise ValueError(f"generated example {example.identity} failed contract: {issues[:5]}")
        split_rows[example.split].append(example)

    metadata_dir = output_dir / "metadata"
    for split in ("train", "valid", "test"):
        _write_jsonl(output_dir / f"{split}.jsonl", (example.training_row() for example in split_rows[split]))
        _write_jsonl(
            metadata_dir / f"{split}.jsonl", (example.metadata_row() for example in split_rows[split])
        )
    _write_jsonl(eval_output, sorted(source_fixtures, key=lambda fixture: fixture["id"]))
    heldout = heldout_fixtures(split_rows["test"])
    _write_jsonl(heldout_output, sorted(heldout, key=lambda fixture: fixture["id"]))

    groups_by_split = {
        split: sorted({example.course_group for example in split_rows[split]})
        for split in ("train", "valid", "test")
    }
    if set(groups_by_split["train"]) & set(groups_by_split["valid"]):
        raise ValueError("train/valid course-group leakage")
    if set(groups_by_split["train"]) & set(groups_by_split["test"]):
        raise ValueError("train/test course-group leakage")
    if set(groups_by_split["valid"]) & set(groups_by_split["test"]):
        raise ValueError("valid/test course-group leakage")

    file_records = {}
    for path in [
        *(output_dir / f"{split}.jsonl" for split in ("train", "valid", "test")),
        *(metadata_dir / f"{split}.jsonl" for split in ("train", "valid", "test")),
        eval_output,
    ]:
        file_records[path.relative_to(output_dir.parent).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    counts = {}
    for split in ("train", "valid", "test"):
        counts[split] = {
            "total": len(split_rows[split]),
            "byKind": dict(sorted(Counter(example.kind for example in split_rows[split]).items())),
            "byDomain": dict(sorted(Counter(example.domain for example in split_rows[split]).items())),
        }
    manifest = {
        "schemaVersion": 1,
        "dataset": "scion-coursemapping-education-v2",
        "purpose": "CourseMapper lesson-kernel, assessment, and source-grounding adapter training",
        "base": {"modelId": TRAIN_BASE_ID, "revision": TRAIN_BASE_REVISION, "exactRevisionRequired": True},
        "counts": counts,
        "splitPolicy": "course-group-disjoint",
        "courseGroups": groups_by_split,
        "sources": [
            {
                "repository": "lovejzzz/Scion",
                "revision": LEGACY_SCION_SOURCE_REVISION,
                "path": "data/preference-pairs-full.jsonl",
                "sha256": sha256_file(legacy_jsonl),
                "use": "teacher-chosen full kernels normalized to the current CourseMapper contract",
            },
            {
                "repository": "lovejzzz/CourseMapper",
                "revision": COURSEMAPPER_SOURCE_REVISION,
                "path": "evaluation/scion-adapters/evidence/codex-approved-preferences-v0.16.42.jsonl",
                "sha256": sha256_file(approved_jsonl),
                "use": "stable order-swapped preference winners",
            },
            {
                "repository": "lovejzzz/CourseMapper",
                "revision": COURSEMAPPER_SOURCE_REVISION,
                "paths": [_coursemapper_path(path) for path in source_paths],
                "sha256": {path.name: sha256_file(path) for path in source_paths},
                "use": "source-grounded, compiler-admitted reference atoms",
            },
        ],
        "limitations": [
            "Teacher-chosen examples are machine-authored and are not classroom outcome evidence.",
            "Stable Codex preference review is not a substitute for independent instructor review.",
            "The source-grounded subset covers computer science, geology, music theory, and UX design.",
        ],
        "files": file_records,
    }
    _atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    heldout_manifest = {
        "schemaVersion": 1,
        "purpose": "test-only base-versus-adapter CourseMapper promotion evaluation",
        "selection": "12 deterministic domain-round-robin examples per response kind",
        "datasetManifest": {
            "path": (output_dir / "manifest.json").relative_to(output_dir.parent).as_posix(),
            "sha256": sha256_file(output_dir / "manifest.json"),
        },
        "fixtures": {
            "path": heldout_output.relative_to(output_dir.parent).as_posix(),
            "bytes": heldout_output.stat().st_size,
            "sha256": sha256_file(heldout_output),
            "count": len(heldout),
            "splits": dict(sorted(Counter(item["split"] for item in heldout).items())),
            "byKind": dict(sorted(Counter(item["kind"] for item in heldout).items())),
            "byDomain": dict(sorted(Counter(item["domain"] for item in heldout).items())),
        },
    }
    _atomic_write(
        heldout_output.with_name("heldout-manifest.json"),
        json.dumps(heldout_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-jsonl", type=Path, required=True)
    parser.add_argument("--approved-jsonl", type=Path, required=True)
    parser.add_argument("--source-capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--eval-output", type=Path, default=Path("eval/fixtures.jsonl"))
    parser.add_argument("--heldout-output", type=Path, default=Path("eval/heldout-fixtures.jsonl"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = []
    for directory in args.source_capture_dir:
        paths.extend(directory.glob("*-reference.json"))
    manifest = build_dataset(
        legacy_jsonl=args.legacy_jsonl,
        approved_jsonl=args.approved_jsonl,
        source_capture_paths=paths,
        output_dir=args.output,
        eval_output=args.eval_output,
        heldout_output=args.heldout_output,
    )
    print(json.dumps({"status": "pass", "counts": manifest["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
