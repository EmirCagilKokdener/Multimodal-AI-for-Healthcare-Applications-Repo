#!/usr/bin/env bash
# Run inference with LoRA adapter (default: HF Hub enderorman/medgemma-1.5-ct-rate-tr).
# Usage: ./scripts/inference.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

: "${HF_TOKEN:?export HF_TOKEN before running}"

python "$ROOT/src/inference.py" \
    --reports    "${VAL_REPORTS:-$DATA_ROOT/validation_reports_tr.xlsx}" \
    --slices-dir "${VAL_SLICES_DIR:-$DATA_ROOT/val_slices}" \
    --out-dir    "${RESULTS_DIR:-$ROOT/results}" \
    --lora       "${LORA:-enderorman/medgemma-1.5-ct-rate-tr}"
