#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PATH="$(dirname "$PYTHON"):$PATH"

cd "$PROJECT_ROOT"
GATE="$PROJECT_ROOT/scripts/st_mgprompt_a8_batch4_gate.py"
AUDIT_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4"
PREFLIGHT_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/preflight/st_mgprompt_a8_batch4"
PREFLIGHT_REPORT="$AUDIT_ROOT/a8_batch4_preflight_summary.json"
CHILD_LOG_ROOT="$AUDIT_ROOT/preflight"
READINESS_REPORT="$AUDIT_ROOT/a8_batch4_readiness.json"
REFERENCE="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/e5_scope27/E5_A8_BATCH4_REFERENCE.json"

mkdir -p "$AUDIT_ROOT" "$PREFLIGHT_ROOT"
"$PYTHON" "$GATE" validate-contract
"$PYTHON" "$GATE" lock-status
"$PYTHON" "$GATE" freeze-plan > "$AUDIT_ROOT/a8_batch4_freeze_plan.json"
"$PYTHON" "$GATE" preflight-plan > "$AUDIT_ROOT/a8_batch4_preflight_plan.json"
set +e
"$PYTHON" "$GATE" dry-run > "$AUDIT_ROOT/a8_batch4_dry_run.json"
dry_run_code=$?
set -e
if [[ "$dry_run_code" -ne 0 ]]; then
  echo "A8 dry-run is blocked; inspect the current artifact before preflight." >&2
  exit "$dry_run_code"
fi

set +e
"$PYTHON" "$GATE" preflight \
  --report-path "$PREFLIGHT_REPORT" \
  --child-log-root "$CHILD_LOG_ROOT" \
  --source-revision "$(git rev-parse HEAD)"
preflight_code=$?
set -e
if [[ "$preflight_code" -ne 0 ]]; then
  echo "A8 exact Batch4 preflight did not pass; formal A8 training is stopped." >&2
  exit "$preflight_code"
fi

set +e
"$PYTHON" "$GATE" run \
  --log-root "$AUDIT_ROOT/formal" \
  --source-revision "$(git rev-parse HEAD)"
run_code=$?
set -e
if [[ "$run_code" -ne 0 ]]; then
  echo "A8 formal run failed; E5 consumption and reference publication are stopped." >&2
  exit "$run_code"
fi

"$PYTHON" "$GATE" readiness \
  --report-path "$READINESS_REPORT" \
  --reference-path "$REFERENCE"
echo "A8 Batch4 prerequisite is READY; E5 may now consume it read-only."
