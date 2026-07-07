# E2B-MAX — Model Card

_The zero-cost item author of the CourseMapper lab stack. A fixed open-weights
model inside a measured harness: every capability claim below was decided by a
frozen instrument, and every number has a bench file behind it._

**Status: SIMULATED-verified, human anchor pending.** All figures are
machine-judged (deterministic gates, a blind cross-family solver, simulated
classrooms, two-family judge panels). The sealed blind packet
(`verification-output/trellis/item-author-packet-v3`) awaits two human
readers; per house constitution, that is the only verdict that upgrades this
card from SIMULATED to confirmed.

## 1. What it is

|              |                                                                                                                                                                                                                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base model   | `google/gemma-4-e2b-it` (~4B, Apache-2.0) — **weights untouched**                                                                                                                                                                                                                                                           |
| Runtime      | on-device, Apple Silicon via mlx-vlm (`trellis/tendril/.venv-g4`); JSONL server `trellis/tendril/distill/serve_g4.py`                                                                                                                                                                                                       |
| Harness      | **adaptive test-time compute** (`authorItemsE2BMax`, `trellis/researcher/shape.mjs`): greedy first → if the deterministic gate passes, stop; on failure, escalate to 3-candidate sampling (T=0/0.7/0.9) + one accepted-item exemplar + per-slot gate argmax + one feedback resample quoting the gate's own rejection reason |
| Prompts      | generic v1 everywhere except `ADOPTED_GENRES` (history — evidence-genre line, won +3 on its A/B); math/lang genre lines were REJECTED by the same protocol                                                                                                                                                                  |
| Routing      | `author-registry.json`: kernels measured at e2b ≤3/9 AND ds ≥7/9 pooled (currently `lit/rhyme-scheme-and-internal-rhyme`, `psych/operant-conditioning`) route to the paid author when a ledger is present; strict-$0 mode DISCLOSES instead of shipping weak items                                                          |
| Verification | every item passes `gapItemRejection` (catch/confront/aesthetics/dedupe) + a **blind cross-family solver** (paid by design, ~$0.001/item). The harness selects on the local gate only — run 2 proved the solver can't be gamed (local wins pushed solver rejects 2→5).                                                       |
| Cost         | authoring $0.0000; verification ≈ $0.01–0.03 per course-sized batch                                                                                                                                                                                                                                                         |

## 2. Measured capability (all same-protocol, 8 frozen kernels unless noted)

| Instrument                                                                                        | Result                                                                         | Source                  |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------- |
| Acceptance, 3 config runs                                                                         | plain 15 → MAX 16 → **adaptive 18** /24 (ds 22/18/20, mini 16/15/11)           | `author-showdown*.json` |
| Pooled vs paid authors /72                                                                        | **E2B 49 · GPT-5.4-mini 42 · DeepSeek 60**                                     | runs 1–3                |
| Unseen hard set (9 kernels: notation LA, Korean pragmatics, Reconstruction history, abstract lit) | **19/27 (70%)**, incl. 7/9 notation-dense linear algebra                       | `hard-set.json`         |
| Classroom battery (deterministic, seed 1)                                                         | realistic mastery **0.69 vs ds 0.67** — parity; item health/catching identical | `edu-bar.json`          |
| Blind two-family judge on quiz pairs                                                              | 6.9 vs ds 7.9 (−1.0 polish gap; ds-judges-ds bias disclosed)                   | `edu-bar.json`          |
| Latency                                                                                           | 7.9s/kernel greedy path; ~21s fully escalated (ds: 27–29s)                     | showdown runs           |
| Noise floor                                                                                       | ±1/kernel (identical-prompt control)                                           | `hard-set.json`         |

## 3. Honest limits

1. **DeepSeek leads pooled acceptance** (60 v 49). E2B-MAX beats the paid
   mini tier, ties or beats ds on individual runs' dense halves, but has not
   met the ≥ds same-run bar (−2 twice). The registry covers its two stable
   blind spots for ~a cent per course.
2. **Judge polish gap (−1.0).** Items teach identically (battery parity) but
   read rougher. Candidate fix: gated explanation polish; not yet benched.
3. **Weight tuning is retired by evidence**: SFT collapsed twice (copies),
   DPO round 1 collapsed on the frozen bench (ranks well, writes badly at 105
   pairs). The item-verdict flywheel (102 pairs and growing) reopens DPO at
   ~300+.
4. **Day-one course quality on virgin ground is below the mature band** —
   judge 4.33 (extractive surfaces) → 5.0 with the $0.17 paid surface pass;
   covered courses with mature deposits read 6.67. The items are never the
   gap (three measurements); first-touch PROSE is, and it matures with
   deposits while the replay price stays $0.
5. **Every claim is SIMULATED** until the two-human packet reads.

## 4. Reproduce

```bash
# suite (24 tendril tests incl. routing + parser regressions)
npx vitest run trellis/__tests__/tendril.test.mjs

# the 3-author showdown, E2B in deployed (adaptive) config
SHOWDOWN=run npx vite-node trellis/researcher/authorShowdownBench.mjs max

# unseen-hard-set A/B (generic vs discipline prompts)
HARD_SET=run npx vite-node trellis/researcher/hardSetBench.mjs

# teachability A/B (battery + blind judge)
EDU_BAR=run npx vite-node trellis/researcher/eduBar.mjs

# per-kernel scoreboard + drift alarm
SCOREBOARD=run npx vite-node trellis/researcher/scoreboard.mjs
```

## 5. Provenance & license

Gemma 4 E2B weights: Apache-2.0 (Google). Items deposited by this author
carry `provenance.model: "e2b"` (or the routed paid author, flagged
`routed: true`); every deposited item is solver-verified unless explicitly
disclosed `solverVerified: false` (strict-$0 mode). Kernel knowledge is
CC-BY-SA (Wikipedia) with per-fact span anchors, plus OpenAlex-cited
misconceptions.
