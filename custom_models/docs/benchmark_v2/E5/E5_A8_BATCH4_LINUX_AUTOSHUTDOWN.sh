#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/GyxPaper2}
LOG_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4"
RUNNER="$PROJECT_ROOT/custom_models/docs/benchmark_v2/E5/E5_A8_BATCH4_LINUX.sh"
SHUTDOWN_COMMAND=${SHUTDOWN_COMMAND:-/usr/bin/shutdown}
SYNC_COMMAND=${SYNC_COMMAND:-sync}
BASH_EXECUTABLE=${BASH_EXECUTABLE:-bash}

mkdir -p "$LOG_ROOT"
set +e
"$BASH_EXECUTABLE" "$RUNNER" 2>&1 | tee "$LOG_ROOT/a8_batch4_autoshutdown.log"
run_code=${PIPESTATUS[0]}
printf 'exit_code=%s\n' "$run_code" > "$LOG_ROOT/a8_batch4_autoshutdown.exitcode"
"$SYNC_COMMAND"
"$SHUTDOWN_COMMAND" -h now
exit "$run_code"
