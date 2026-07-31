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
python "$runner" hardware-preflight --model gru --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model dlinear --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model lightts --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model tide --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model segrnn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model transformer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model patchtst --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model itransformer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model timexer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model timesnet --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model micn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model wpmixer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model multipatchformer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model timemixer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model tsmixer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model frets --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model crossformer --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model msgnet --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model timefilter --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model gcn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model stgcn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model dcrnn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model graph_wavenet --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model mtgnn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model agcrn --experiment-profile e5_common_loss_v1
python "$runner" hardware-preflight --model stid --experiment-profile e5_common_loss_v1
