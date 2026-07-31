#!/usr/bin/env bash
set -euo pipefail
test -f custom_models/src/benchmark_v2/run_benchmark.py
test -f dataset/sdwpf_model_input_base.parquet
test -f dataset/sdwpf_eval_target.parquet
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
runner=custom_models/src/benchmark_v2/run_benchmark.py
input=dataset/sdwpf_model_input_base.parquet
target=dataset/sdwpf_eval_target.parquet
output=custom_models/results/benchmark_v2/common_loss_architecture_seed2026

python "$runner" protocol-check
python "$runner" graph-protocol-check
python "$runner" loss-profile-check --experiment-profile e5_common_loss_v1
