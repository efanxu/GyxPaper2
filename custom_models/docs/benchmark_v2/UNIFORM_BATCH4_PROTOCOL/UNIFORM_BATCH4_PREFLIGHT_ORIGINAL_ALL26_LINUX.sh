#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON:-python}
PROFILE="uniform_train_batch4_v1"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"

"$PYTHON_BIN" scripts/uniform_batch4_machine_gate.py preflight-inventory --suite original --python "$PYTHON_BIN"
exit $?
