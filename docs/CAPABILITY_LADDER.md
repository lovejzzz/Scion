# Gemma 4 E2B — Capability Ladder (the pipeline's eval standard)

A level-based evaluation standard for the local model that powers the
zero-cost pipeline. Each level is a **pre-registered bar decided by a frozen
instrument** — the gate bench, the blind cross-family solver, or the classroom
battery. **No level is granted by argument.** A level is _Achieved_ only when a
ruler says so; _Partial_ when it is earned on some ground but not all; _Locked_
when the quest is named but not yet cleared.

Shareable character-sheet view: published as an Artifact (see session).

**Character:** `google/gemma-4-e2b-it` · ~4B · Apache-2.0 · runs on-device
(Apple Silicon, mlx-vlm). **Class:** local item-author, zero-cost tier.
**Current level: 6 / 10.**

> Honest ceiling: every figure below is machine-judged and stamped
> **SIMULATED**. Level 10 (the two-human anchor) is the only verdict the
> constitution accepts, and it is still pending — so the whole ladder is
> provisional by construction.

## Stats (measured)

| Stat                          | Value                            | Note                                                                                    |
| ----------------------------- | -------------------------------- | --------------------------------------------------------------------------------------- |
| Cost                          | **$0.0000**                      | the whole point — $0 authoring, offline                                                 |
| Honesty                       | **100%**                         | gate + solver catch every miss; nothing bad ships; strict-$0 DISCLOSES registry kernels |
| Acceptance (adaptive, stable) | **18/24 ×2 runs**                | pooled /96: E2B 67 · ds 80 · mini 55; zero drift flags on the scoreboard                |
| Routed system (production)    | **41/48 ≥ ds 40/48**             | E2B + 2 registry kernels routed paid — edges the best paid author                       |
| Unseen hard set               | **19/27 (70%)**                  | incl. 7/9 notation-dense linear algebra; noise floor ±1 (control)                       |
| Teachability (battery)        | **0.69 vs ds 0.67**              | classroom sim can't tell $0 items from paid; judge polish −1.0 remains                  |
| Speed                         | **7.9s greedy / ~21s escalated** | adaptive: compute goes where failure is (ds 27–29s)                                     |
| Trainability                  | **brittle**                      | SFT ×2 + DPO r1 all ruler-rejected; flywheel at 102/300 pairs for DPO r2                |

## The ladder

| Lv    | Name                      | Status                            | The bar / the evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----- | ------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Loads & Speaks            | **Achieved**                      | Runs on-device via mlx-vlm, ~6.5s/sample. Only official weights load (community 4-bit quants broken; mlx-lm can't load it at all).                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 2     | Holds the Format          | **Achieved\***                    | Emits structured items — _with a crutch_: it doubles the closing brace on every object, so a whole-array parse returned nothing. `parseItemArray` (balanced per-object slice) recovers it; one kernel went 0→3 on the fix.                                                                                                                                                                                                                                                                                                                             |
| 3     | Clears the Gate           | **Achieved**                      | Output survives the deterministic gates (length/punct/lexical catch/dedupe): 87% gate-pass across 10 disciplines.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 4     | Survives the Blind Solver | **Achieved**                      | An independent, different-family model can actually answer the item: 77% end-to-end. Failure signature = vague under-specified stems, all caught.                                                                                                                                                                                                                                                                                                                                                                                                      |
| 5     | Matches the Paid Author   | **Partial**                       | Parity is domain-dependent: **WON** diverse (26/30 vs ds 22/30), **TRAILS** dense poetry (18/24 & 13/24 vs 19/24 & 20/24).                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **6** | **Ships as a Default**    | **Achieved ← current**            | Default item-author for Researcher-Zero — the $0 path had **no items at all** before, so this is a capability unlock, not just a saving. (Paid runs keep it opt-in.) **Moat widened (L1, July 6):** the zero-deposit runner filled 7 poetry-form kernels (61 surfaces + 13 E2B items, $0.02 of solver seat) and the coverage proof went **refusal → 7/7 shipped** (0 → 3 segments + 3–4 verified items each).                                                                                                                                          |
| 7     | Wins Everywhere           | **Locked — but the map is drawn** | Adaptive E2B-MAX (deployed) took acceptance 15→16→18 across three same-protocol runs, WON one dense half outright, briefly broke the rhyme-scheme blind spot (2/3), and **beats GPT-5.4-mini pooled 49v42 /72**. ds still leads pooled (60). The two stable blind spots (`rhyme-scheme`, `psych`) are now ROUTED to the paid author via `author-registry.json` — so the SYSTEM wins everywhere even where the model doesn't. Full-model Level 7 waits on DPO r2 (flywheel at 102/300 pairs).                                                           |
| 8     | Runs in the Browser       | **Locked**                        | Web runtime ≈ native. Currently fp32 **65%** / q8 **43%** vs **83%** native — an 18-pt runtime-parity gap even unquantized.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 9     | Learns From Its Mistakes  | **Locked**                        | Toolchain UNBLOCKED (July 6): mlx-lm-lora 2.1.0 in `.venv-dpo` (transformers-5 shim; note the package's `PreferenceDataset` bug — encodes the literal string "rejected" — but `--train-mode dpo` uses the correct `DPODataset`). **Round 1 REJECTED by the frozen gate bench:** DPO from s3-800 on 105 pairs hit val pref-accuracy 0.764 but collapsed deployment acceptance to 37.5% (train loss 0.002 = overtrained; it learned to rank, drifted off the writing distribution). Deployed pair stands. Round 2 waits for a 3–5× corpus + fewer iters. |
| 10    | Human-Anchored            | **Locked — STAGED**               | Two human readers confirm the output teaches. **Packet sealed (July 6):** `verification-output/trellis/item-author-packet-v3` — 4 blind kernel-pairs, E2B vs DeepSeek quizzes (all solver-verified), X/Y hash-shuffled, key sealed. ~15-minute read. This level is granted by humans or not at all.                                                                                                                                                                                                                                                    |

## How to advance a level

1. **Pre-register the bar** before running anything (a number, not a vibe).
2. **Freeze the instrument** (gate bench / blind solver / classroom battery /
   the 2-human packet). The ruler decides; anecdotes stay in the residual.
3. **Ship-only-if-better.** A tie or a regression does not advance a level —
   even a plausible one. Levels 7 and 9 above are Locked precisely because a
   ruler said the cheap path (prompt tricks, thin-data DPO) did not clear them.

## Reading the current rank

Gemma 4 E2B sits at **Level 6**: it ships, for free, and never ships something
broken. Levels 7–9 are engineering the lab is actively building — and the
measured lesson so far is that they are **not cheap**: prompt-hardening failed
its A/B (Level 7), browser parity is an 18-pt gap (Level 8), and DPO is
tooling- and data-blocked (Level 9). Level 10 is the one only humans can grant,
and it caps everything above it.
