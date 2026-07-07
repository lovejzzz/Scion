#!/bin/zsh
# Scion — ORPO preference training (the beat-mini attempt).
#
# Trains Gemma 4 E2B (rootstock) with a LoRA adapter (the graft) to PREFER
# gpt-5.4-mini-quality atoms over its own greedy drafts. Preference, not
# imitation — the rule that survived two SFT collapses (see docs/MODEL_CARD.md).
#
# MEMORY: this OOM'd on a 32 GB M-series with batch-size 2 + full-length
# whole-lesson pairs. The safe config below (batch 1, rank 8, grad-checkpoint,
# short atom pairs) fits comfortably; scale UP on a bigger machine.
#
#   ATOMS (default, memory-safe): 841 short mc-item + key-term pairs — the
#          quiz-artifact signal (where Scion loses to mini). Fits ~24-32 GB.
#   FULL  : 974 pairs incl. whole-lesson JSON (~5-8 K chars) — needs ~48+ GB
#          and a larger --max-seq-length. Set DATASET=full BATCH=2 on a big box.
set -e
cd "$(dirname "$0")"

DATASET="${DATASET:-atoms}"   # atoms | full
BATCH="${BATCH:-1}"
RANK="${RANK:-8}"
ITERS="${ITERS:-800}"
SEQ="${SEQ:-1024}"
VENV="${VENV:-.venv}"

if [ "$DATASET" = "full" ]; then DATA=data-full; SEQ="${SEQ:-2048}"; else DATA=data-atoms; fi
# mlx_vlm.lora loads a directory expecting train.jsonl inside it.
mkdir -p "$DATA"
if [ "$DATASET" = "full" ]; then cp data/preference-pairs-full.jsonl "$DATA/train.jsonl";
else cp data/preference-pairs-atoms.jsonl "$DATA/train.jsonl"; fi

echo "=== Scion ORPO: dataset=$DATASET batch=$BATCH rank=$RANK iters=$ITERS seq=$SEQ ==="
"$VENV/bin/python" -m mlx_vlm.lora \
  --model-path google/gemma-4-e2b-it \
  --dataset "$DATA" \
  --split train --train-mode orpo \
  --iters "$ITERS" --batch-size "$BATCH" --lora-rank "$RANK" --beta 0.1 \
  --grad-checkpoint --max-seq-length "$SEQ" \
  --steps-per-report 25 --steps-per-save 200 \
  --output-path adapters-scion

echo ""
echo "=== training done — checkpoints in adapters-scion/ ==="
echo "verify it serves + still emits valid JSON:"
echo "  G4_ADAPTERS=adapters-scion python runtime/serve_scion.py   (then poke it)"
echo "then measure vs mini — see gauntlet/README.md"
