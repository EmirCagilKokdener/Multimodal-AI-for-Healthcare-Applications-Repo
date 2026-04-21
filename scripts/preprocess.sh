#!/usr/bin/env bash
# Preprocess a CT-RATE split into .npz slice files.
# Usage: ./scripts/preprocess.sh <train|val>
set -euo pipefail

SPLIT="${1:-train}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

if [[ "$SPLIT" == "train" ]]; then
    EXCEL="${TRAIN_REPORTS:-$DATA_ROOT/train_reports_tr.xlsx}"
    OUT="${TRAIN_SLICES_DIR:-$DATA_ROOT/slices}"
elif [[ "$SPLIT" == "val" ]]; then
    EXCEL="${VAL_REPORTS:-$DATA_ROOT/validation_reports_tr.xlsx}"
    OUT="${VAL_SLICES_DIR:-$DATA_ROOT/val_slices}"
else
    echo "usage: $0 <train|val>" >&2
    exit 1
fi

: "${HF_TOKEN:?export HF_TOKEN before running}"

python "$ROOT/src/preprocess.py" \
    --split "$SPLIT" \
    --excel "$EXCEL" \
    --out-dir "$OUT" \
    --tmp-dir "${TMP_DL_DIR:-$ROOT/tmp_dl}" \
    --workers "${NUM_WORKERS:-8}"
