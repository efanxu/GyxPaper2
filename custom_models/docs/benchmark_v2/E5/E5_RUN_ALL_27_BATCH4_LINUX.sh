#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/root/autodl-tmp/GyxPaper2
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: project directory does not exist: $PROJECT_ROOT" >&2
  exit 64
fi
cd "$PROJECT_ROOT" || exit 64
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 65
fi

MANIFEST="$PROJECT_ROOT/custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json"
GATE="$PROJECT_ROOT/scripts/e5_batch4_scope27_gate.py"
RUNNER="$PROJECT_ROOT/custom_models/src/benchmark_v2/run_benchmark.py"
INPUT="$PROJECT_ROOT/dataset/sdwpf_model_input_base.parquet"
TARGET="$PROJECT_ROOT/dataset/sdwpf_eval_target.parquet"
OUTPUT_ROOT="$PROJECT_ROOT/custom_models/results/benchmark_v2/common_loss_architecture_seed2026"
DEFAULT_LOG="$PROJECT_ROOT/logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.log"
TOTAL_LOG="${1:-$DEFAULT_LOG}"
if [[ "$TOTAL_LOG" != /* ]]; then
  TOTAL_LOG="$PROJECT_ROOT/$TOTAL_LOG"
fi
LOG_ROOT="$(dirname "$TOTAL_LOG")"
MODEL_LOG_ROOT="$LOG_ROOT/models"
FAILED_MODELS="$LOG_ROOT/failed_models.txt"
EXIT_CODES="$LOG_ROOT/model_exit_codes.tsv"
PLAN_FILE="$LOG_ROOT/run_plan.tsv"
READINESS_REPORT="$LOG_ROOT/e5_scope27_readiness.json"
EVIDENCE_MANIFEST="$LOG_ROOT/e5_scope27_evidence_manifest.json"
READINESS_LOG="$LOG_ROOT/e5_scope27_readiness.log"
AGGREGATE_LOG="$LOG_ROOT/e5_scope27_aggregate.log"
FINAL_STATUS="$LOG_ROOT/e5_scope27_final_status.env"
STATE_SENTINEL="$LOG_ROOT/e5_scope27_started"

mkdir -p "$LOG_ROOT" "$MODEL_LOG_ROOT"
if [[ -e "$STATE_SENTINEL" ]]; then
  echo "ERROR: refusing to overwrite prior E5 scope27 launch state: $STATE_SENTINEL" >&2
  exit 73
fi
printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATE_SENTINEL"
printf 'model_id\texit_code\tstatus\trun_id\tlog_path\n' > "$EXIT_CODES"
printf '' > "$FAILED_MODELS"
exec > >(tee -a "$TOTAL_LOG") 2>&1

echo "E5 scope27 batch4 formal run started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHON=$PYTHON"
"$PYTHON" - <<'PY'
import sys
import torch
print(f"sys.executable={sys.executable}")
print(f"torch.__version__={torch.__version__}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(
    "torch.cuda.get_device_name(0)="
    + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE")
)
PY
printf 'git_commit='
git rev-parse HEAD 2>/dev/null || printf 'UNAVAILABLE\n'

for required in "$MANIFEST" "$GATE" "$RUNNER" "$INPUT" "$TARGET"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required file missing: $required" >&2
    exit 66
  fi
done

set +e
"$PYTHON" "$GATE" --manifest "$MANIFEST" plan-runs \
  --output-root "$OUTPUT_ROOT" > "$PLAN_FILE"
plan_code=$?
set -e
if [[ "$plan_code" -ne 0 ]]; then
  echo "ERROR: scope27 manifest/run planning failed with code $plan_code" >&2
  printf 'manifest_or_plan\t%s\tFAILED\t-\t%s\n' \
    "$plan_code" "$PLAN_FILE" >> "$EXIT_CODES"
  printf 'manifest_or_plan\texit_code=%s\n' "$plan_code" >> "$FAILED_MODELS"
  printf 'run_status=FAILED\nreadiness_status=NOT_RUN\naggregate_status=NOT_RUN\nexit_code=%s\n' \
    "$plan_code" > "$FINAL_STATUS"
  exit "$plan_code"
fi

mapfile -t RUN_ROWS < "$PLAN_FILE"
if [[ "${#RUN_ROWS[@]}" -ne 26 ]]; then
  echo "ERROR: expected 26 runnable manifest rows, got ${#RUN_ROWS[@]}" >&2
  printf 'manifest_count\t67\tFAILED\t-\t%s\n' "$PLAN_FILE" >> "$EXIT_CODES"
  printf 'manifest_count\texpected=26\tactual=%s\n' "${#RUN_ROWS[@]}" >> "$FAILED_MODELS"
  printf 'run_status=FAILED\nreadiness_status=NOT_RUN\naggregate_status=NOT_RUN\nexit_code=67\n' \
    > "$FINAL_STATUS"
  exit 67
fi

echo "Manifest-driven runnable order:"
cut -f1-4 "$PLAN_FILE"

formal_failures=0
for row in "${RUN_ROWS[@]}"; do
  IFS=$'\t' read -r model_id command_name run_id device action reason <<< "$row"
  model_log="$MODEL_LOG_ROOT/${model_id}.log"
  if [[ "$action" == "SKIP_COMPLETED" ]]; then
    echo "SKIP completed identity-matching run: $model_id / $run_id"
    printf '%s\t0\tSKIPPED_COMPLETED_IDENTITY_MATCH\t%s\t%s\n' \
      "$model_id" "$run_id" "$model_log" >> "$EXIT_CODES"
    continue
  fi
  if [[ "$action" != "RUN" ]]; then
    echo "BLOCK existing non-reusable run: $model_id / $run_id / $reason"
    printf '%s\t90\tBLOCKED_EXISTING_PRESERVED\t%s\t%s\n' \
      "$model_id" "$run_id" "$model_log" >> "$EXIT_CODES"
    printf '%s\texit_code=90\treason=%s\n' "$model_id" "$reason" >> "$FAILED_MODELS"
    formal_failures=$((formal_failures + 1))
    continue
  fi

  echo "START $model_id / $run_id / $command_name / $device"
  set +e
  "$PYTHON" "$RUNNER" "$command_name" \
    --model "$model_id" \
    --experiment-profile e5_common_loss_v1 \
    --training-profile uniform_train_batch4_v1 \
    --formal-scope-id e5_batch4_scope27_seed2026 \
    --input-path "$INPUT" \
    --target-path "$TARGET" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$run_id" \
    --device "$device" 2>&1 | tee -a "$model_log"
  model_code=${PIPESTATUS[0]}
  set -e
  if [[ "$model_code" -eq 0 ]]; then
    model_status=COMPLETED
  else
    model_status=FAILED
    formal_failures=$((formal_failures + 1))
    printf '%s\texit_code=%s\n' "$model_id" "$model_code" >> "$FAILED_MODELS"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$model_id" "$model_code" "$model_status" "$run_id" "$model_log" >> "$EXIT_CODES"
  echo "END $model_id exit_code=$model_code"
done

echo "Validating the read-only A8 reference and full 27-entry readiness."
set +e
"$PYTHON" "$GATE" --manifest "$MANIFEST" readiness \
  --output-root "$OUTPUT_ROOT" \
  --report-path "$READINESS_REPORT" \
  --evidence-path "$EVIDENCE_MANIFEST" 2>&1 | tee -a "$READINESS_LOG"
readiness_code=${PIPESTATUS[0]}
set -e

aggregate_code=125
aggregate_status=SKIPPED_NOT_READY
if [[ "$formal_failures" -eq 0 && "$readiness_code" -eq 0 ]]; then
  echo "Readiness is 27/27; starting require-complete aggregate."
  set +e
  "$PYTHON" "$GATE" --manifest "$MANIFEST" aggregate \
    --output-root "$OUTPUT_ROOT" \
    --report-path "$READINESS_REPORT" \
    --evidence-path "$EVIDENCE_MANIFEST" \
    --require-complete 2>&1 | tee -a "$AGGREGATE_LOG"
  aggregate_code=${PIPESTATUS[0]}
  set -e
  if [[ "$aggregate_code" -eq 0 ]]; then
    aggregate_status=COMPLETED
  else
    aggregate_status=FAILED
  fi
else
  echo "Aggregate skipped: formal_failures=$formal_failures readiness_code=$readiness_code"
fi

final_code=0
if [[ "$formal_failures" -ne 0 || "$readiness_code" -ne 0 || "$aggregate_code" -ne 0 ]]; then
  final_code=1
fi
readiness_status=NOT_READY
if [[ "$readiness_code" -eq 0 ]]; then
  readiness_status=READY_27_OF_27
fi
run_status=COMPLETED
if [[ "$formal_failures" -ne 0 ]]; then
  run_status=COMPLETED_WITH_FAILURES
fi
printf 'run_status=%s\nformal_failures=%s\nreadiness_status=%s\nreadiness_exit_code=%s\naggregate_status=%s\naggregate_exit_code=%s\nexit_code=%s\nfinished_at=%s\n' \
  "$run_status" "$formal_failures" "$readiness_status" "$readiness_code" \
  "$aggregate_status" "$aggregate_code" "$final_code" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FINAL_STATUS"
printf 'exit_code=%s\n' "$final_code" > "$LOG_ROOT/e5_27_batch4_seed2026.run_all.exitcode"
echo "E5 scope27 run-all finished: exit_code=$final_code"
exit "$final_code"
