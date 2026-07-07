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
