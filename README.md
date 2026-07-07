# Scion

**The house model of the CourseMapper / Trellis lab stack — a customized,
$0-per-call course-authoring model that runs on your own machine.**

Scion is not a base model you download. It is a *graft*: in horticulture a
scion is the cultivated cutting grafted onto wild rootstock. Here the
rootstock is Google's open-weights **Gemma 4 E2B** (~4B params, Apache-2.0,
Apple-Silicon via MLX); the scion — the part we cultivated — is a harness of
grammar-constrained decoding, per-lesson contracts, self-verification, and now
a preference-trained LoRA adapter. The first trained cut ships as **Scion-1**.

This repo is the **training + serving package**, split out from the main
CourseMapper app so it can be trained on a machine with enough unified memory.

---

## Why Scion exists

CourseMapper compiles a full course (syllabus → lesson plans, slide decks,
rubrics, quizzes, assignments, study guides) from an LLM. Every paid compile on
`gpt-5.4-mini` costs ~$0.07 and recurs forever. **Scion's promise is the same
course at $0 — offline, private, unlimited regeneration** — and it improves
with every verified example it banks, while the paid price never moves.

## The honest quality band (measured, not claimed)

Everything below was decided by frozen instruments — deterministic gates, a
blind cross-family solver, and pooled multi-seat judge panels. See
`docs/BAKEOFF.md` and `docs/MODEL_CARD.md` for the full record.

| Where | Scion (untrained harness) | gpt-5.4-mini |
| --- | --- | --- |
| Structural grade | **98–99/A** | 99/A |
| Study guides (judge) | **6.0–6.8 — WINS** | 5.0–5.3 |
| Lesson plans (judge) | 5.7–6.2 — **parity** | 5.33 |
| Quiz items (judge) | **3.6–4.7 — LOSES** | 6.0–6.3 |
| **Pooled compiler-seat mean** | **~5.2 (best draw 5.83)** | **6.08** |
| Cost | **$0.00** | $0.07/compile |

**The whole gap is quiz items.** Scion ties or beats mini everywhere else. The
17-round harness campaign (grammar decoding, per-lesson chunking, self-verify,
polish) took the seat from **3.33 → 5.83** but hit a ceiling — the last ~0.9
points live in the weights, not the scaffolding.

## The plan to beat mini

Train Scion to write quiz items as well as mini writes them, using a signal
that exists **by construction**: on the identical prompt, mini's quiz item
judges higher than Scion's, so `(mini item = chosen, Scion item = rejected)` is
a preference pair with no labeling needed. **841 such atom pairs are already
built and shipped in `data/`** (353 quiz-item pairs — the direct signal — plus
488 key-term pairs). ORPO-training on "prefer mini-quality over your own draft"
moves Scion's output distribution toward mini's on exactly the weak artifact.
Preference, not imitation — the rule that survived two SFT collapses.

Full step-by-step in **`PLAN.md`**. Measured outcomes (incl. the honest Round-1 result — it did NOT beat mini, it over-narrowed and broke structural output) are logged in **`RESULTS.md`**; the round-1 adapter ships in `adapters-round1/` as a reference.

---

## Quick start (Apple Silicon Mac)

```bash
# 1. Environment (Python 3.13, MLX — Apple Silicon only)
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. Train (memory-safe default: 841 atom pairs, batch 1, rank 8, grad-ckpt)
bash train.sh
#    → checkpoints land in adapters-scion/
#    On a big box (48 GB+): DATASET=full BATCH=2 RANK=16 bash train.sh

# 3. Serve the trained adapter + verify it still emits valid JSON
G4_ADAPTERS=adapters-scion ./.venv/bin/python runtime/serve_scion.py
#    (JSONL stdin protocol; or wire the OpenAI shim — see runtime/)

# 4. Measure vs mini — see gauntlet/README.md (runs in the CourseMapper app)
```

**Memory note:** training OOM'd on a 32 GB M-series with batch-size 2 + full
whole-lesson pairs. The default `train.sh` config (atom pairs, batch 1, rank 8,
gradient checkpointing, 1024-token cap) fits ~24–32 GB. Scale up on more.

## Repo layout

```
data/          the preference corpus (the expensive artifact — $0.084 to build)
  preference-pairs-atoms.jsonl   841 short mc-item + key-term pairs (train these)
  preference-pairs-full.jsonl    974 pairs incl. whole-lesson JSON (big machines)
  app-flywheel.jsonl             pairs banked live by the app's own generations
runtime/       serve_scion.py (grammar-constrained MLX server) + the OpenAI shim
contracts/     kernelSchemas.mjs — the grammar contract the server enforces
train.sh       ORPO training entry point (memory-safe)
docs/          MODEL_CARD, BAKEOFF (measured results), ROADMAP
gauntlet/      how to measure trained-Scion vs mini
```

## Relationship to CourseMapper

Scion is consumed by the CourseMapper app as its "Local" provider (Provider:
Local · API key: Free · Model: Scion-1). The app talks to `runtime/`'s
OpenAI-compatible server. Corpus-building and the vs-mini gauntlet run inside
the CourseMapper repo (they need the app's compiler + judge); this package
carries the pre-built corpus so you can train standalone, then drop the adapter
back into CourseMapper via `G4_ADAPTERS`.

## Provenance & license

Gemma 4 E2B weights: Apache-2.0 (Google), untouched — Scion is a LoRA adapter
on top, so the base model is never modified and rollback is deleting the
adapter. The preference corpus is machine-built (mini authored the "chosen"
side); every claim here is SIMULATED-verified pending a two-human read, per the
lab constitution.
