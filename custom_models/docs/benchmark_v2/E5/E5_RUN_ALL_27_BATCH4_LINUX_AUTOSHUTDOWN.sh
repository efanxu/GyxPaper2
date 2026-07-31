#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT" || exit 64
LOG_ROOT="$PROJECT_ROOT/logs/benchmark_v2/e5_batch4_scope27"
MAIN_LOG="$LOG_ROOT/e5_27_batch4_seed2026.log"
EXIT_CODE_FILE="$LOG_ROOT/e5_27_batch4_seed2026.exitcode"
RUN_ALL="$PROJECT_ROOT/custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX.sh"
mkdir -p "$LOG_ROOT"

set +e
bash "$RUN_ALL" "$MAIN_LOG"
main_code=$?
printf 'exit_code=%s\n' "$main_code" > "$EXIT_CODE_FILE"
sync
/usr/bin/shutdown -h now
exit "$main_code"
