# Model card: Scion Bonsai 27B

## Summary

Scion Bonsai 27B is a LoRA adaptation of PrismML Bonsai 27B for CourseMapper. Its intended tasks
are structured university lesson kernels, evidence-bearing multiple-choice questions, key-term
explanations with misconception corrections, and source-grounded atom bundles. The deployment
artifact is an adapter; it cannot run without the separately downloaded pinned Bonsai 27B base.

## Intended use

- Local CourseMapper generation through model ID `scion-1`.
- Drafting learner-ready educational material that an instructor can inspect and edit.
- Producing strict JSON for the CourseMapper compilation pipeline.

It is not intended to grade students autonomously, replace qualified instructors, make high-stakes
educational decisions, or provide medical, legal, or safety-critical instruction.

## Training

- Base: `prism-ml/Bonsai-27B-unpacked`, exact revision
  `d619b27283ac02b4199ced97a89419529dc0bfac`.
- Method: 4-bit MLX QLoRA, rank 8, MLP projections in the final 24 transformer layers plus their
  standard full-attention projections, prompt masking, gradient checkpointing, and
  course-group-disjoint validation/test sets. Fused linear-attention projections are excluded
  because the pinned llama.cpp GGUF LoRA converter cannot losslessly represent their head reorder.
- Corpus: 711 train / 297 validation / 457 test examples over lesson, assessment, terminology,
  and grounded-source response types.
- Deployment: LoRA converted to F16 GGUF and applied to PrismML's 1-bit Bonsai GGUF at runtime.

The generated release manifest is the authority for the exact trained artifact, run receipt,
evaluation metrics, and SHA-256. No checkpoint is promoted merely because training completed.

## Evaluation

Promotion compares the unadapted base and adapter on the same held-out fixtures. It measures strict
contract pass rate, deterministic structural/pedagogical checks, reference-token F1, and a live
CourseMapper protocol smoke test. These automated measures test integration and content retention;
they are not evidence of student learning outcomes or broad academic correctness.

## Limitations and risks

- Much of the corpus is machine-authored and teacher-*style* chosen, not independently instructor
  reviewed.
- Grounded examples cover computer science, geology, music theory, and UX more deeply than other
  fields. Quality may vary by subject and level.
- A model can return plausible but incorrect facts, misleading distractors, or fabricated details.
- Deterministic decoding improves reproducibility but does not guarantee truth.
- The 1-bit serving base can lose quality relative to a higher-precision checkpoint.

Human review remains required before publication. CourseMapper should retain source constraints,
schema validation, and visible editing controls around the model.

## Size and license

The release gate limits the Scion-specific adapter to fewer than 1,000,000,000 bytes. The pinned
base is about 3.8 GB and is not bundled. This repository is Apache-2.0; consult the upstream Bonsai
model card and third-party notices for upstream terms and attribution.
