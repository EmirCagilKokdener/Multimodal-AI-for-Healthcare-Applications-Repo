#!/usr/bin/env bash
# Fine-tune MedGemma 1.5 on CT-RATE (Turkish) with QLoRA.
# Usage: ./scripts/train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

: "${HF_TOKEN:?export HF_TOKEN before running}"

python "$ROOT/src/train.py" \
    --reports    "${TRAIN_REPORTS:-$DATA_ROOT/train_reports_tr.xlsx}" \
    --slices-dir "${TRAIN_SLICES_DIR:-$DATA_ROOT/slices}" \
    --out-dir    "${CHECKPOINT_DIR:-$ROOT/checkpoints}" \
    --epochs     "${EPOCHS:-3}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --grad-accum "${GRAD_ACCUM:-4}" \
    --lr         "${LR:-2e-4}"
