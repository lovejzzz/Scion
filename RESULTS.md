# Scion — training results log

Honest, measured outcomes of each training round. No fabricated wins — a round
that doesn't beat mini is recorded as such, with the lesson for the next round.

---

## Baseline (untrained harness)

- Compiler seat: **98–99/A structural**, pooled judge **~5.2** (best draw 5.83).
- vs **gpt-5.4-mini pooled 6.08** → **loses by ~0.9**, entirely on quiz items.
- 17 harness rounds proved the prompt/decoding ceiling; the gap is in the weights.

## Round 1 — atom-only ORPO (2026-07-07, on a 32 GB M-series)

- **Config:** ORPO, LoRA rank 8, β 0.1, batch 1, grad-checkpoint, seq 768,
  600 iters. Dataset: `data/preference-pairs-atoms.jsonl` (841 pairs — 353
  mc-item + 488 key-term only; the whole-lesson pairs were dropped for memory).
- **Training:** healthy — loss 6.69 → 2.19, peak mem 13 GB, ~5 min. Adapter in
  `adapters-round1/adapters.safetensors` (52 MB).
- **Capability gate:** PASS — still emits valid grammar-constrained JSON on a
  single mc-item prompt (though that greedy sample keyed G→C as a fifth, which
  is a fourth — a factual slip the compiler's blind-solve pass would catch).
- **Gauntlet (full compile vs mini): FAILED — did NOT beat mini.**
  The compile broke at Pass A: `skeleton-unparseable` and a kernel chunk
  truncated `Unterminated string at 9869 chars`. The adapter made Scion
  **more verbose**, overflowing the token budgets on the structural contracts
  (skeleton, full kernel) so the JSON truncated before closing → no shippable
  course → finalize timeout.

### The lesson (→ Round 2)

Atom-only training **over-narrowed** the model: optimizing "write richer quiz
items than your greedy draft" (chosen = mini's fuller items) taught it to write
*more* everywhere, which broke the length-bounded skeleton and full-kernel
contracts. The LoRA shifts the WHOLE distribution, not just the quiz artifact.

**Round-2 levers, in priority order:**
1. **Include the whole-lesson pairs** (`preference-pairs-full.jsonl`) so the
   model stays coherent on the full contracts — train the mix, not atoms alone.
   This needs a bigger machine (the full pairs are 5–8 K chars; `DATASET=full
   BATCH=2 SEQ=2048` per `train.sh`).
2. **Fewer iters / lower LR** — 600 iters on 841 pairs (~0.7 epoch) already
   over-shifted; try 200–300 iters and gate each checkpoint.
3. **Budget-aware pairs** — the "chosen" items should respect the same length
   floors/ceilings the compiler enforces, so richer ≠ longer-than-the-contract.
4. **Gate on the FULL compile, not a single item** — capability-verify each
   checkpoint by compiling a course, not just generating one mc-item.

Verdict of record: **Round 1 did not beat mini** (broke structural generation).
The harness still ties mini; the weights campaign continues.

## Round 2 — full-mix ORPO (2026-07-07, on the 32 GB M-series)

- **Config:** ORPO, LoRA rank 8, β 0.1, batch 1, grad-checkpoint, seq 3072,
  400 iters. Dataset: `data/preference-pairs-full.jsonl` (974 pairs — 133
  whole-lesson + 353 mc-item + 488 key-term; the mix, not atoms alone, per the
  Round-1 lesson). Adapter in `adapters-round2/adapters.safetensors` (52 MB);
  the 200-iter checkpoint is also kept (`adapters-round2/checkpoint-0000200`).
- **Training:** healthy, no OOM (the memory-safe config fits — peak 27.7 GB, the
  full pairs at seq 3072). The 32 GB box CAN train the mix; the OOM note in
  README/train.sh was the batch-2 config only.
- **Capability gate: PASSED — decisively better than Round 1.** A direct
  grammar-constrained probe (OpenAI `response_format: json_schema`, the real
  course-skeleton schema, `max_tokens: 6000`) produced a **VALID, CLOSED,
  schema-compliant skeleton — 7 sessions, 7 assessments, all closed — in 28.3
  seconds.** On a kernel prompt it wrote genuinely good content: facts, key
  terms *with misconception + correction fields*, and a worked example. Round 1
  ran away and truncated; Round 2 closes cleanly and writes well. The adapter
  is NOT slow (28 s) and NOT structurally broken.
- **Gauntlet (full CourseMapper compile vs mini): BLOCKED — but NOT by the
  weights.** The app-path skeleton call falls back to prose
  (`skeleton-unparseable`). Root cause isolated to the SERVING path, not the
  adapter: the app requests the skeleton at `max_tokens: 16000`; serve_g4
  returns the whole grammar-constrained response as one buffered chunk after
  ~6-7 min; the CourseMapper/crucible local-provider stream has a
  `STREAM_INACTIVITY_TIMEOUT_MS` / `DEFAULT_PROVIDER_TIMEOUT_MS` that fires
  before that single chunk lands, so the client reads a partial (dangling-comma)
  body and falls back. Proof it is the timeout, not the model: the byte-identical
  request at `max_tokens: 6000` over a client with a long (870 s) read timeout
  **closes valid in 28 s** (above). The instrumented shim confirmed the schema
  arrived intact (`rfType=json_schema, hasInnerSchema=true, contractSchema=true,
  maxTok=16000`) — so the grammar contract was built; the generation was simply
  cut before completion.

### Serving note (important, corrects a scare)

Grammar-constrained decoding on the g4 route **does work.** In `mlx_vlm` 0.6.3,
`generate()` has no `logits_processors` kwarg, but it forwards `**kwargs`
through `stream_generate` → `generate_step` (dispatch), which DOES accept and
apply `logits_processors`. A short probe confirmed enforcement: prompted for a
`{"lessons":...}` object under the skeleton schema, the grammar FORCED
`{"course":...}` instead. (An earlier "grammar is a silent no-op" hypothesis
was wrong — it came from misattributing a leftover kernel-call body, which
legitimately uses a lessons-array contract, to a skeleton probe.)

### The lesson (→ Round 3, on the bigger machine)

Round 2's adapter is **proven capable in isolation and writes good content.**
Beating mini is now gated on a SERVING fix, not more training. Do these in order:

1. **Unblock the compile (highest value, no retraining needed).** Any one of:
   (a) lower the local skeleton budget to ~4-6 k tokens (it closes there),
   (b) raise the crucible/local-provider stream timeout, or
   (c) make serve_g4 STREAM tokens so the inactivity timer never fires.
   Then the mix adapter should compile a full course.
2. **Only then measure vs mini** — pooled ≥12-seat judge panel; adopt iff mean
   > 6.08 AND frozen benches hold (PLAN.md Step 3).
3. **If a quiz gap remains after a clean compile,** the Round-1 corpus levers
   still apply (more mc-item pairs, budget-aware "chosen" lengths).

Verdict of record: **Round 2 has NOT yet produced a full vs-mini judged number**
— the compile is blocked by a client-timeout serving bug, not the weights. The
adapter itself is the strongest artifact so far: valid structured output + good
content at $0. The next machine should fix the serving path first, then judge.
