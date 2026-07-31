#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON:-python}
PROFILE="uniform_train_batch4_v1"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"

set +e
bash "$PROJECT_ROOT/custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_E5_CORE_28_LINUX.sh"
code=$?
sync
/usr/bin/shutdown -h now
exit $code
