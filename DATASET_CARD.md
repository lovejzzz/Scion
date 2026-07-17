# Dataset card: Scion Education preferences

## Purpose and composition

The corpus teaches grounded educational behavior and CourseMapper-compatible structured output. The deterministic
seed generator creates 256 fictional tasks:

| Split | Tasks | Teacher/critic use | Student training use |
|---|---:|---|---|
| Train | 160 | yes | yes, after admission |
| Validation | 32 | yes | validation only, after admission |
| Preference test | 32 | yes | ORPO test only, never optimization |
| Locked evaluation | 32 | **never** | **never** |

Every split contains equal representation of prerequisite reasoning, schedule constraints, degree audits,
uncertainty grounding, tutoring, safe education, tool use, and CourseMapper lesson kernels. Final admitted counts
and per-domain coverage are recorded in `data/orpo/dataset-manifest.json`; rejected rows remain quarantined with
their admission stage and issues.

The completed corpus admits all 224 generated preference pairs: 160 train, 32 validation, and 32 preference test,
across 96 disjoint groups and four capability domains. The locked dataset identity is
`dcf75cfdf30c57ecc7ebf3084c3b63042fa686926942563246f669eb9b0b45b8`. All preference judgments are from the
single local Gemma critic; there are no blind human-instructor labels.

## Sources and licensing

All prompts, catalogs, policies, student records, and source packets are deterministic fictional examples authored
in this repository and released under Apache-2.0. They contain no real student data and no real institutional
catalog data. Chosen responses come only from the pinned local Qwen teacher. Preference validation comes only
from deterministic repository code and the pinned local Gemma critic.

No OpenAI, Qwen Cloud, Alibaba Cloud, or other closed API output is included.

## Admission and leakage controls

- Each task has an exact response contract and deterministic semantic oracle.
- A minimally changed rejected response is constructed only after the chosen response passes.
- The rejected response must fail the oracle for a recorded reason.
- The independent critic receives blind, hash-randomized A/B labels and must select the oracle-admitted candidate
  with score at least 4/5 while passing grounding and pedagogy.
- A prompt change changes the task hash; cached teacher or critic output then becomes stale and is regenerated.
- Course-group IDs are split-disjoint. Dataset identity includes all source and split hashes.
- The locked 32-task benchmark is never available to teacher generation, critic filtering, or training.
- A second frozen CourseMapper five-domain benchmark is hash-bound as a disjoint integration boundary.
- Every final chosen and rejected sequence is formatted with the actual Gemma 4 chat template and must fit within
  2,048 tokens; silent truncation is forbidden.

The four CourseMapper research domains are `course-planning`, `academic-operations`, `education-pedagogy`, and
`responsible-guidance`. These are capability domains, not claims of comprehensive subject-matter coverage.

## Known limitations

The corpus is small and synthetic. It is suitable for a measured research adapter, not CourseMapper's production
promotion threshold. The production gate fails because 224 is below 3,000 verified pairs and four domains is
below five; it also lacks the required fifth model-judge-qualified domain. Automated teacher/critic agreement is
not independent instructor review. The task families are deliberately structured and may overestimate
performance on free-form dialogue. The corpus contains no student outcomes, demographic attributes, longitudinal
records, or evidence of instructional effectiveness.
