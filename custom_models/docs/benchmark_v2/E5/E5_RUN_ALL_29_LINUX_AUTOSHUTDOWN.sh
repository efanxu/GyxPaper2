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

mkdir -p logs/benchmark_v2
nohup bash -lc '
bash custom_models/docs/benchmark_v2/E5/E5_PRECHECK_LINUX.sh &&
bash custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_29_LINUX.sh;
code=$?
sync
/usr/bin/shutdown -h now
exit $code
' > logs/benchmark_v2/E5_RUN_ALL_29.log 2>&1 &
echo $! > logs/benchmark_v2/E5_RUN_ALL_29.pid
echo "tail -f logs/benchmark_v2/E5_RUN_ALL_29.log"
echo "ps -ef | grep '[r]un_benchmark.py'"
echo "nvidia-smi"
echo "wait $(cat logs/benchmark_v2/E5_RUN_ALL_29.pid); echo $?"
