#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
LOG_ROOT="$PROJECT_ROOT/custom_models/logs/benchmark_v2/e9/formal"
WRAPPER_LOG="$LOG_ROOT/autoshutdown.log"
mkdir -p "$LOG_ROOT"

bash "$PROJECT_ROOT/custom_models/docs/benchmark_v2/E9/E9_RUN_ALL_6_LINUX.sh" >> "$WRAPPER_LOG" 2>&1
run_code=$?
printf '%s\n' "$run_code" > "$LOG_ROOT/autoshutdown_run_all_exit_code.txt"
test -f "$LOG_ROOT/failed_models.csv" && cp "$LOG_ROOT/failed_models.csv" "$LOG_ROOT/autoshutdown_failed_models.csv"
test -f "$LOG_ROOT/completed_models.csv" && cp "$LOG_ROOT/completed_models.csv" "$LOG_ROOT/autoshutdown_completed_models.csv"
test -f "$PROJECT_ROOT/custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026/E9_READINESS.json" && cp "$PROJECT_ROOT/custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026/E9_READINESS.json" "$LOG_ROOT/autoshutdown_final_readiness.json"
test -f "$PROJECT_ROOT/custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026/E9_TRANSFER_READINESS.json" && cp "$PROJECT_ROOT/custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026/E9_TRANSFER_READINESS.json" "$LOG_ROOT/autoshutdown_transfer_readiness.json"
sync
/usr/bin/shutdown -h now
