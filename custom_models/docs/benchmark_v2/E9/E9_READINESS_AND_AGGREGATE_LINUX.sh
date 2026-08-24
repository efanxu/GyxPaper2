#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHON="$PYTHON"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

ORIGINAL26_XLSX=${1:-}
test -n "$ORIGINAL26_XLSX" || { echo "usage: $0 /absolute/path/original26.xlsx"; exit 2; }
test -f "$ORIGINAL26_XLSX" || { echo "missing workbook: $ORIGINAL26_XLSX"; exit 2; }
cd "$PROJECT_ROOT" || exit 2

OUTPUT_ROOT="$PROJECT_ROOT/custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026"
"$PYTHON" -m benchmark_v2.e9 --analysis-root "$OUTPUT_ROOT" readiness --original26-xlsx "$ORIGINAL26_XLSX" || exit $?
"$PYTHON" -m benchmark_v2.e9 --analysis-root "$OUTPUT_ROOT" aggregate --original26-xlsx "$ORIGINAL26_XLSX" --require-complete
