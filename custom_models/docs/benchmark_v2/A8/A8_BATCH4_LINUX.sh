#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/GyxPaper2}
PYTHON=${PYTHON:-/root/miniconda3/envs/env_tslib/bin/python}
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
REFERENCE="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/a8/A8_BATCH4_REFERENCE.json"

mkdir -p "$AUDIT_ROOT" "$PREFLIGHT_ROOT"
"$PYTHON" "$GATE" validate-contract
lock_status=$("$PYTHON" "$GATE" lock-status | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
if [[ "$lock_status" != "ABSENT" ]]; then
  echo "A8 lock must be ABSENT; ACTIVE/MALFORMED fail closed and STALE requires explicit clear-stale-lock: $lock_status" >&2
  exit 73
fi
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
  --child-log-root "$CHILD_LOG_ROOT"
preflight_code=$?
set -e
if [[ "$preflight_code" -ne 0 ]]; then
  echo "A8 exact Batch4 preflight did not pass; formal A8 training is stopped." >&2
  exit "$preflight_code"
fi

set +e
"$PYTHON" "$GATE" run \
  --log-root "$AUDIT_ROOT/formal"
run_code=$?
set -e
if [[ "$run_code" -ne 0 ]]; then
  echo "A8 formal run failed; A8 reference publication is stopped." >&2
  exit "$run_code"
fi

"$PYTHON" "$GATE" readiness \
  --report-path "$READINESS_REPORT" \
  --reference-path "$REFERENCE"
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p.get("status")=="READY"' "$READINESS_REPORT"
echo "A8 Batch4 ablation result and reference are READY."
