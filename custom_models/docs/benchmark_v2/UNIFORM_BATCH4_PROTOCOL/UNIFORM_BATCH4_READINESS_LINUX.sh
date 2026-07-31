#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON:-python}
PROFILE="uniform_train_batch4_v1"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"

echo "Deprecated partial readiness entry. Use UNIFORM_BATCH4_FINALIZE_E5_LINUX.sh after base28 and batch4 A8 complete." >&2
exit 64
