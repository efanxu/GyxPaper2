#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHON
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT" || exit 64
LOG_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/e5_scope27"
mkdir -p "$LOG_ROOT"
MAIN_LOG="$LOG_ROOT/e5_scope27_autoshutdown.log"
RUN_ALL="$PROJECT_ROOT/custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX.sh"

set +e
bash "$RUN_ALL" 2>&1 | tee "$MAIN_LOG"
main_code=${PIPESTATUS[0]}
printf 'exit_code=%s\n' "$main_code" > "$LOG_ROOT/e5_scope27_autoshutdown.exitcode"
sync
/usr/bin/shutdown -h now
exit "$main_code"
