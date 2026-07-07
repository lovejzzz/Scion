# Scion training campaign — the plan to beat gpt-5.4-mini

The goal is precise: **make Scion's pooled compiler-seat judge mean exceed
mini's 6.08**, at $0. Everything below is the pre-registered path, with the
decision gates that decide adoption.

---

## The diagnosis (why we lose today)

Scion ties mini on structure and WINS study guides; it loses on **quiz items
only** (judge ~4.5 vs mini ~6.0). That one artifact drags the pooled mean to
5.2. So "beat mini" = "write quiz items as well as mini." The 17-round harness
campaign proved prompt/decoding tricks cap at ~5.83 — the fix is in the weights.

## The mechanism (ORPO preference training)

On the identical kernel prompt, mini's atom judges higher than Scion's greedy
draft. So `(mini = chosen, Scion = rejected)` pairs exist for free. Training
Scion to PREFER the chosen side moves its distribution toward mini's on exactly
the weak artifact. ORPO (odds-ratio preference optimization) needs no separate
reward model and no reference model — ideal for a LoRA on a 4B base.

**Why not SFT?** Two SFT collapses (26.7% → 13.3%) proved it: the corpus
targets are near-copies of inputs, so SFT-on-near-identity teaches the model to
COPY. DPO round 1 also collapsed at 105 pairs. Preference-only + ≥ hundreds of
quality-DIFFERENTIATED pairs is the rule. This corpus is 841 differentiated
atom pairs (chosen and rejected are genuinely different-quality).

---

## Step 1 — Train (this repo, ~15–40 min)

```bash
bash train.sh                              # atoms, batch 1, rank 8, grad-ckpt
# bigger machine:
DATASET=full BATCH=2 RANK=16 SEQ=2048 bash train.sh
```

Watch the loss fall in the log. Checkpoints save every 200 iters to
`adapters-scion/`. **Keep every checkpoint** — the best one is chosen by the
gates below, not by loss. (Loss falling ≠ better output; the DPO-r1 collapse
"ranked well, wrote badly.")

**Kill condition:** if the loss diverges or the model stops emitting valid
JSON at every checkpoint, the run is a collapse — stop, and the corpus needs
more/cleaner pairs before retrying. Do not adopt a collapsed adapter.

## Step 2 — Gate each checkpoint (does it still work at all?)

Before measuring quality, confirm the adapter didn't break the base
capabilities. Serve each checkpoint and check it still emits **valid,
grammar-constrained JSON** at length (the whole point of Scion):

```bash
G4_ADAPTERS=adapters-scion/0000200_adapters.safetensors \
  ./.venv/bin/python runtime/serve_scion.py
# feed it a kernel prompt via the JSONL protocol; the reply MUST parse as JSON.
```

A checkpoint that regressed JSON validity, or that produces degenerate/looping
text, is **rejected** regardless of any quiz gain. This is the frozen-bench
discipline that caught the past collapses.

## Step 3 — The gauntlet: measure vs mini (in the CourseMapper repo)

Copy the winning `adapters-scion/` into CourseMapper and run the pre-built
gauntlet (`gauntlet/README.md` has the exact commands). It:

1. serves Scion + adapter through the app's Local provider,
2. compiles a full course (`crucible --provider local`),
3. runs **pooled ≥12-seat judge panels** and prints the mean.

**Decision:** adopt the adapter (set `G4_ADAPTERS` permanently) **only if**
the pooled mean **> 6.08** AND the frozen benches held. Base weights are never
touched, so a losing adapter is discarded by deleting the file.

## Step 4 — If one round isn't enough

A single round on 841 pairs may not cross 6.08 — that outcome is expected-
possible, not failure. The levers, in order:

1. **More quiz pairs.** The app's on-device flywheel (`data/app-flywheel.jsonl`)
   banks a `{rejected, chosen}` pair every time it regenerates a bad quiz item —
   grow the corpus by just *using* Scion in CourseMapper, then retrain.
2. **Up-weight mc-item pairs.** Duplicate the 353 quiz pairs 2–3× in the train
   set so the gradient leans harder on the artifact that actually loses.
3. **A second gated round** from the best round-1 checkpoint (resume-adapter).
4. **Rank / iters / β sweep** — rank 16, β {0.05, 0.1, 0.3}, 1–2 epochs.

The kill condition stands: **two consecutive ruler-rejected rounds re-retire
the weights for this version** and the honest report is "harness ties mini;
training did not cross it this campaign" — never a fabricated win.

---

## What success looks like

A single `adapters-scion/<checkpoint>.safetensors` file that, dropped into
CourseMapper via `G4_ADAPTERS`, makes a full course compile at **$0** with a
pooled judge mean **above mini's 6.08** and every frozen bench green. That file
IS Scion-1. Ship it in CourseMapper's `serve_g4.py` default and the local
provider serves a model that beats the paid tier at zero marginal cost.
