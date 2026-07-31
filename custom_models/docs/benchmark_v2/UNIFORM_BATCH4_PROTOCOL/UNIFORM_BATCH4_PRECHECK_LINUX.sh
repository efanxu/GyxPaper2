#!/usr/bin/env bash
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON:-python}
PROFILE="uniform_train_batch4_v1"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"

if pgrep -af 'run_benchmark.py (train|evaluate-only)|hardware[-_]preflight|run_st_mgprompt.py.*--full|UNIFORM_BATCH4_RUN_' >/tmp/uniform_batch4_active.txt; then
  cat /tmp/uniform_batch4_active.txt
  exit 20
fi
"$PYTHON_BIN" custom_models/src/benchmark_v2/run_benchmark.py protocol-check

"$PYTHON_BIN" scripts/uniform_batch4_machine_gate.py precheck
exit $?
