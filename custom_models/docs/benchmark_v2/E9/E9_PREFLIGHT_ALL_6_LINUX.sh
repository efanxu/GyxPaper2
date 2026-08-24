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

cd "$PROJECT_ROOT" || exit 2
"$PYTHON" -m benchmark_v2.e9 preflight-all \
  --device cuda \
  --preflight-root "$PROJECT_ROOT/custom_models/logs/benchmark_v2/e9/preflight"
