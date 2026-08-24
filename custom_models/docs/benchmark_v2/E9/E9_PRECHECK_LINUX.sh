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

test -d "$PROJECT_ROOT" || { echo "missing project root: $PROJECT_ROOT"; exit 2; }
test -x "$PYTHON" || { echo "missing Python: $PYTHON"; exit 2; }
cd "$PROJECT_ROOT" || exit 2

git rev-parse HEAD
"$PYTHON" -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")'
"$PYTHON" -m compileall -q custom_models/src/benchmark_v2/e9
if [ -n "${ORIGINAL26_XLSX:-}" ]; then
  "$PYTHON" -m benchmark_v2.e9 precheck --original26-xlsx "$ORIGINAL26_XLSX"
else
  "$PYTHON" -m benchmark_v2.e9 precheck
fi
