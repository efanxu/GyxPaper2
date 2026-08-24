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

MANIFEST="$PROJECT_ROOT/custom_models/docs/benchmark_v2/E9/E9_VARIANT_MANIFEST.json"
OUTPUT_ROOT="$PROJECT_ROOT/custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026"
PREFLIGHT_ROOT="$PROJECT_ROOT/custom_models/logs/benchmark_v2/e9/preflight"
LOG_ROOT="$PROJECT_ROOT/custom_models/logs/benchmark_v2/e9/formal"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
MAIN_LOG="$LOG_ROOT/run_all_6.log"
FAILED="$LOG_ROOT/failed_models.csv"
COMPLETED="$LOG_ROOT/completed_models.csv"
EXIT_FILE="$LOG_ROOT/run_all_exit_code.txt"
PID_FILE="$LOG_ROOT/run_all.pid"
printf '%s\n' "$$" > "$PID_FILE"
printf '%s\n' "model_id,run_id,exit_code" > "$FAILED"
printf '%s\n' "model_id,run_id,status" > "$COMPLETED"

test -d "$PROJECT_ROOT" || { echo "missing project root" | tee -a "$MAIN_LOG"; exit 2; }
test -x "$PYTHON" || { echo "missing Python" | tee -a "$MAIN_LOG"; exit 2; }
test -f "$MANIFEST" || { echo "missing variant manifest" | tee -a "$MAIN_LOG"; exit 2; }
cd "$PROJECT_ROOT" || exit 2

git rev-parse HEAD | tee -a "$MAIN_LOG"
"$PYTHON" -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")' | tee -a "$MAIN_LOG"

overall=0
while IFS=$'\t' read -r model_id run_id; do
  model_log="$LOG_ROOT/${run_id}.log"
  echo "START $model_id $run_id" | tee -a "$MAIN_LOG"
  "$PYTHON" -m benchmark_v2.e9 \
    --analysis-root "$OUTPUT_ROOT" \
    train-one \
    --model "$model_id" \
    --device cuda \
    --preflight-root "$PREFLIGHT_ROOT" \
    > "$model_log" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    printf '%s,%s,%s\n' "$model_id" "$run_id" "COMPLETED_OR_SAFE_SKIP" >> "$COMPLETED"
  else
    printf '%s,%s,%s\n' "$model_id" "$run_id" "$code" >> "$FAILED"
    overall=1
  fi
  echo "END $model_id $run_id exit=$code" | tee -a "$MAIN_LOG"
done < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); [print(v["model_id"]+"\t"+v["transfer_run_id"]) for v in p["variants"]]' "$MANIFEST")

"$PYTHON" -m benchmark_v2.e9 --analysis-root "$OUTPUT_ROOT" transfer-readiness >> "$MAIN_LOG" 2>&1
transfer_readiness_code=$?
if [ "$transfer_readiness_code" -ne 0 ]; then overall=1; fi
if [ -n "${ORIGINAL26_XLSX:-}" ]; then
  "$PYTHON" -m benchmark_v2.e9 --analysis-root "$OUTPUT_ROOT" readiness --original26-xlsx "$ORIGINAL26_XLSX" >> "$MAIN_LOG" 2>&1
  readiness_code=$?
  if [ "$readiness_code" -ne 0 ]; then overall=1; fi
fi
printf '%s\n' "$overall" > "$EXIT_FILE"
exit "$overall"
