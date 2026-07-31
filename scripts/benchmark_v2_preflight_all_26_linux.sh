#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT" || exit 2
RUNNER="custom_models/src/benchmark_v2/run_benchmark.py"
INPUT="dataset/sdwpf_model_input_base.parquet"
TARGET="dataset/sdwpf_eval_target.parquet"
for p in "$RUNNER" "$INPUT" "$TARGET"; do [[ -f "$p" ]] || { echo "Missing: $PROJECT_ROOT/$p" >&2; exit 2; }; done
export PYTHONPATH="$PROJECT_ROOT/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/benchmark_v2
python "$RUNNER" protocol-check || exit $?
python "$RUNNER" graph-protocol-check || exit $?

MODELS=(
  "gru"
  "dlinear"
  "lightts"
  "tide"
  "segrnn"
  "transformer"
  "patchtst"
  "itransformer"
  "timexer"
  "timesnet"
  "micn"
  "wpmixer"
  "multipatchformer"
  "timemixer"
  "tsmixer"
  "frets"
  "crossformer"
  "msgnet"
  "timefilter"
  "gcn"
  "stgcn"
  "dcrnn"
  "graph_wavenet"
  "mtgnn"
  "agcrn"
  "stid"
)

for m in "${MODELS[@]}"; do
  echo "Preflight: $m"
  python "$RUNNER" hardware-preflight --model "$m"
  c=$?
  [[ $c -eq 0 ]] || { echo "Preflight failed: $m ($c)" >&2; exit "$c"; }
done
echo "All 26 trainable models passed preflight."
