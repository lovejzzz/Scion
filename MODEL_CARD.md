# Model card: Scion Education 3

## Summary

Scion Education 3 is a pair of LoRA adapters for Gemma 4 E2B and Gemma 4 12B. The adapters specialize strict,
source-grounded educational JSON behavior for CourseMapper while leaving changing catalog facts to retrieval and
tools. Scion Lite is also converted to a separate GGUF LoRA for CourseMapper's WebGPU-JSPI browser runtime.

These are research adapters, not standalone models. They require the exact base revision named in each package
manifest.

## Intended use

- Draft course plans and explain prerequisite or scheduling constraints from supplied facts.
- Produce strict CourseMapper lesson-kernel and tool-call JSON.
- Tutor with a diagnosis, hint, explanation, and check question grounded in a supplied source.
- Identify missing evidence instead of inventing course facts.

Scion is not intended to grade students autonomously, decide admissions or degree eligibility without an official
system of record, replace an instructor, facilitate academic dishonesty, or provide medical, legal, or other
high-stakes professional advice.

## Training

- Primary teacher: local Qwen3.6 27B 8-bit, deterministic or low-temperature generation.
- Independent critic: local Gemma 4 31B Q4 with blind A/B label randomization.
- Students: exact Gemma 4 E2B and 12B QAT checkpoints.
- Method: ORPO LoRA, rank 16, alpha 16, learning rate 2e-5, beta 0.1, batch size 1, gradient accumulation 2,
  gradient checkpointing, and a 2,048-token limit.
- Training data: only pairs that pass both a deterministic task oracle and the blind independent critic.
- Reproducibility: exact model revisions, file hashes, clean Git tree, package versions, seed, dataset identity,
  token-length audit, training log, and adapter checksums are recorded in release receipts.

No closed API output is present in the training corpus. The pinned 122B escalation model is not used unless a
recorded gate demonstrates that it is necessary.

## Evaluation

Each base and adapter is evaluated with deterministic decoding on the same locked 32-task benchmark, balanced
across all eight capabilities. Promotion logic requires higher pass rate, fewer issues, no capability regression,
and no citation-hallucination regression. The package's `training-result.json`, evaluation reports, and model
manifest are the authority for completed-run measurements.

This benchmark measures contract adherence and bounded reasoning on fictional tasks. It does not measure student
learning outcomes, broad subject expertise, fairness across populations, or correctness on a live university
catalog.

## Limitations and risks

- A small adapter cannot transfer all reasoning ability of a 27B teacher or 31B critic into an E2B student.
- Synthetic templates improve coverage and determinism but do not substitute for instructor review or authentic
  classroom evaluation.
- The model can still emit plausible but incorrect explanations or JSON that fails downstream validation.
- Subject facts are intentionally sparse; using the model without CourseMapper retrieval increases hallucination
  risk.
- The Lite quantized browser base trades quality for memory and download size.
- Safety examples are bounded academic scenarios, not a comprehensive safety evaluation.

CourseMapper must retain schema validation, citation checks, tool boundaries, visible editing, and a base-only
rollback path. Human review remains required before publishing educational material or acting on a degree audit.

## Size and license

Each Scion-specific adapter must be below 1,000,000,000 bytes. The Lite browser package must additionally stay
below 64 MiB and two percent of the 3,349,514,112-byte pinned runtime base. Base weights are not bundled.

Scion code and synthetic data are Apache-2.0. Model and dependency terms remain those of their upstream projects;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
