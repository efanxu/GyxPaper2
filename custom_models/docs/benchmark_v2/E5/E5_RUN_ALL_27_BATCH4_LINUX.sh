#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHON
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT"
MANIFEST="$PROJECT_ROOT/custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
GATE="$PROJECT_ROOT/scripts/e5_batch4_scope27_gate.py"
A8_GATE="$PROJECT_ROOT/scripts/st_mgprompt_a8_batch4_gate.py"
INPUT="$PROJECT_ROOT/dataset/sdwpf_model_input_base.parquet"
TARGET="$PROJECT_ROOT/dataset/sdwpf_eval_target.parquet"
OUTPUT_ROOT="$PROJECT_ROOT/custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026"
AUDIT_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/e5_scope27"
A8_AUDIT_ROOT="$PROJECT_ROOT/custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4"
PREFLIGHT_ROOT="$AUDIT_ROOT/preflight"
RUN_LOG_ROOT="$AUDIT_ROOT/runs"
PRECHECK_LOG="$AUDIT_ROOT/e5_scope27_precheck.log"
PREFLIGHT_LOG="$AUDIT_ROOT/e5_scope27_preflight.log"
PREFLIGHT_REPORT="$AUDIT_ROOT/e5_scope27_preflight_summary.json"
PREFLIGHT_CHILD_LOG_ROOT="$AUDIT_ROOT/preflight/child_logs"
RUN_LOG="$AUDIT_ROOT/e5_scope27_run.log"
READINESS_LOG="$AUDIT_ROOT/e5_scope27_readiness.log"
AGGREGATE_LOG="$AUDIT_ROOT/e5_scope27_aggregate.log"
REPORT="$AUDIT_ROOT/e5_scope27_readiness.json"
EVIDENCE="$AUDIT_ROOT/e5_scope27_evidence.json"
A8_READINESS_REPORT="$A8_AUDIT_ROOT/a8_batch4_readiness.json"
A8_PREFLIGHT_REPORT="$A8_AUDIT_ROOT/a8_batch4_preflight_summary.json"
A8_REFERENCE="$AUDIT_ROOT/E5_A8_BATCH4_REFERENCE.json"

mkdir -p "$AUDIT_ROOT" "$A8_AUDIT_ROOT" "$RUN_LOG_ROOT"
if [[ -e "$AUDIT_ROOT/e5_scope27_started" ]]; then
  echo "ERROR: legacy launch sentinel exists; inspect before retrying" >&2
  exit 72
fi
for required in "$PYTHON" "$MANIFEST" "$GATE" "$INPUT" "$TARGET"; do
  [[ -e "$required" ]] || { echo "ERROR: missing required path: $required" >&2; exit 66; }
done
[[ -d "$(dirname "$OUTPUT_ROOT")" ]] || { echo "ERROR: output-root parent is missing" >&2; exit 67; }

exec > >(tee -a "$PRECHECK_LOG") 2>&1
set +e
"$PYTHON" "$A8_GATE" validate-contract
a8_contract_code=$?
"$PYTHON" "$A8_GATE" readiness --report-path "$A8_READINESS_REPORT" --reference-path "$A8_REFERENCE"
a8_readiness_code=$?
set -e
if [[ "$a8_contract_code" -ne 0 || "$a8_readiness_code" -ne 0 ]]; then
  echo "A8 Batch4 prerequisite is not READY; E5 scope27 will not start." >&2
  if [[ "$a8_contract_code" -ne 0 ]]; then
    exit "$a8_contract_code"
  fi
  exit "$a8_readiness_code"
fi
"$PYTHON" "$GATE" --manifest "$MANIFEST" lock-status
"$PYTHON" "$GATE" --manifest "$MANIFEST" validate-manifest
"$PYTHON" "$GATE" --manifest "$MANIFEST" freeze > "$AUDIT_ROOT/e5_scope27_freeze.json"
"$PYTHON" "$GATE" --manifest "$MANIFEST" preflight-plan > "$AUDIT_ROOT/e5_scope27_preflight_plan.json"
"$PYTHON" "$GATE" --manifest "$MANIFEST" dry-run --output-root "$OUTPUT_ROOT" > "$AUDIT_ROOT/e5_scope27_dry_run.json"

HEAD="$(git rev-parse HEAD)"
echo "Static E5 scope27 plan complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Current output root: $OUTPUT_ROOT"
echo "Legacy root is read-only and is never passed to the gate."

set +e
"$PYTHON" "$GATE" --manifest "$MANIFEST" preflight \
  --preflight-root "$PREFLIGHT_ROOT" \
  --source-revision "$HEAD" \
  --report-path "$PREFLIGHT_REPORT" \
  --child-log-root "$PREFLIGHT_CHILD_LOG_ROOT" 2>&1 | tee "$PREFLIGHT_LOG"
preflight_code=${PIPESTATUS[0]}
set -e
if [[ "$preflight_code" -ne 0 ]]; then
  echo "E5 exact preflight was not 24/24 PASS; formal run and aggregate are stopped." >&2
  exit "$preflight_code"
fi
"$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p.get("status")=="PASS" and p.get("counts",{}).get("pass")==24' "$PREFLIGHT_REPORT"

set +e
"$PYTHON" "$GATE" --manifest "$MANIFEST" run \
  --input-path "$INPUT" \
  --target-path "$TARGET" \
  --log-root "$RUN_LOG_ROOT" \
  --preflight-root "$PREFLIGHT_ROOT" \
  --source-revision "$HEAD" 2>&1 | tee "$RUN_LOG"
run_code=${PIPESTATUS[0]}
set -e
if [[ "$run_code" -ne 0 ]]; then
  echo "E5 run did not finish successfully; aggregate is stopped." >&2
  exit "$run_code"
fi

set +e
"$PYTHON" "$GATE" --manifest "$MANIFEST" readiness \
  --output-root "$OUTPUT_ROOT" \
  --report-path "$REPORT" \
  --evidence-path "$EVIDENCE" 2>&1 | tee "$READINESS_LOG"
readiness_code=${PIPESTATUS[0]}
set -e
if [[ "$readiness_code" -ne 0 ]]; then
  echo "Readiness is not COMPLETED_READY_27_OF_27; aggregate is stopped." >&2
  exit "$readiness_code"
fi

"$PYTHON" "$GATE" --manifest "$MANIFEST" aggregate \
  --output-root "$OUTPUT_ROOT" \
  --report-path "$REPORT" \
  --evidence-path "$EVIDENCE" \
  --require-complete 2>&1 | tee "$AGGREGATE_LOG"

echo "E5 scope27 Batch4 completed with exact 24/24 preflight and 27/27 readiness."
