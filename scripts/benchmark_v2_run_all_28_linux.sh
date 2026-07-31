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

run_job(){
  stage="$1"; mode="$2"; model="$3"; output="$4"; runid="$5"; device="$6"
  rundir="$output/$runid"
  [[ ! -e "$rundir" ]] || { echo "Refusing to overwrite: $rundir" >&2; return 90; }
  echo "[$stage] Starting $model / $runid"
  python "$RUNNER" "$mode" --model "$model" --input-path "$INPUT" --target-path "$TARGET" --output-root "$output" --run-id "$runid" --device "$device"
  c=$?
  [[ $c -eq 0 ]] || { echo "$model failed ($c). Suite stopped." >&2; return "$c"; }
  echo "Completed $model; process exited."
}
run_job "E1-A" "evaluate-only" "persistence" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "Persistence_seed2026" "cpu" || exit $?
run_job "E1-A" "evaluate-only" "moving_average" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "MovingAverage_w144_seed2026" "cpu" || exit $?
run_job "E1-A" "train" "gru" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "GRU_node_shared_h64_l1_seed2026" "cuda" || exit $?
run_job "E1-B" "train" "dlinear" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "DLinear_node_shared_ma25_seed2026" "cuda" || exit $?
run_job "E1-B" "train" "lightts" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "LightTS_node_shared_chunk8_seed2026" "cuda" || exit $?
run_job "E1-B" "train" "tide" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "TiDE_node_shared_d512_seed2026" "cuda" || exit $?
run_job "E1-B" "train" "segrnn" "custom_models/results/benchmark_v2/basic_lightweight_seed2026" "SegRNN_node_shared_seg2_d512_seed2026" "cuda" || exit $?
run_job "E2-A" "train" "transformer" "custom_models/results/benchmark_v2/e2_a_seed2026" "Transformer_node_shared_d512_seed2026" "cuda" || exit $?
run_job "E2-A" "train" "patchtst" "custom_models/results/benchmark_v2/e2_a_seed2026" "PatchTST_node_shared_p16_s8_seed2026" "cuda" || exit $?
run_job "E2-A" "train" "itransformer" "custom_models/results/benchmark_v2/e2_a_seed2026" "iTransformer_node_shared_d512_seed2026" "cuda" || exit $?
run_job "E2-A" "train" "timexer" "custom_models/results/benchmark_v2/e2_a_seed2026" "TimeXer_node_shared_ms_d512_seed2026" "cuda" || exit $?
run_job "E2-B" "train" "timesnet" "custom_models/results/benchmark_v2/e2_b_seed2026" "TimesNet_node_shared_d32_k5_seed2026" "cuda" || exit $?
run_job "E2-B" "train" "micn" "custom_models/results/benchmark_v2/e2_b_seed2026" "MICN_node_shared_k12_16_d32_seed2026" "cuda" || exit $?
run_job "E2-B" "train" "wpmixer" "custom_models/results/benchmark_v2/e2_b_seed2026" "WPMixer_node_shared_db2_l1_p16_s8_d256_seed2026" "cuda" || exit $?
run_job "E2-B" "train" "multipatchformer" "custom_models/results/benchmark_v2/e2_b_seed2026" "MultiPatchFormer_node_shared_p8_16_24_32_d256_seed2026" "cuda" || exit $?
run_job "E2-C" "train" "timemixer" "custom_models/results/benchmark_v2/e2_c_seed2026" "TimeMixer_node_shared_ci1_ds3_w2_avg_d16_seed2026" "cuda" || exit $?
run_job "E2-C" "train" "tsmixer" "custom_models/results/benchmark_v2/e2_c_seed2026" "TSMixer_node_shared_el2_d32_seed2026" "cuda" || exit $?
run_job "E2-C" "train" "frets" "custom_models/results/benchmark_v2/e2_c_seed2026" "FreTS_node_shared_tcfft_e128_h256_seed2026" "cuda" || exit $?
run_job "E2-D" "train" "crossformer" "custom_models/results/benchmark_v2/e2_d_seed2026" "Crossformer_node_shared_seg12_win2_d32_el2_seed2026" "cuda" || exit $?
run_job "E2-D" "train" "msgnet" "custom_models/results/benchmark_v2/e2_d_seed2026" "MSGNet_node_shared_k5_d512_el2_gdep2_seed2026" "cuda" || exit $?
run_job "E2-D" "train" "timefilter" "custom_models/results/benchmark_v2/e2_d_seed2026" "TimeFilter_node_shared_p16_d512_el2_tp0p5_seed2026" "cuda" || exit $?
run_job "E3-B" "train" "gcn" "custom_models/results/benchmark_v2/e3_b_seed2026" "GCN_native_agcn_h64_l2_seed2026" "cuda" || exit $?
run_job "E3-B" "train" "stgcn" "custom_models/results/benchmark_v2/e3_b_seed2026" "STGCN_native_cheb3_kt3_b2_seed2026" "cuda" || exit $?
run_job "E3-B" "train" "dcrnn" "custom_models/results/benchmark_v2/e3_b_seed2026" "DCRNN_native_dualrw_k2_h64_l2_ar_seed2026" "cuda" || exit $?
run_job "E3-C" "train" "graph_wavenet" "custom_models/results/benchmark_v2/e3_c_seed2026" "GraphWaveNet_native_pfpr_adp10_b4l2_seed2026" "cuda" || exit $?
run_job "E3-C" "train" "mtgnn" "custom_models/results/benchmark_v2/e3_c_seed2026" "MTGNN_native_adaptive_k20_gdep2_l3_seed2026" "cuda" || exit $?
run_job "E3-C" "train" "agcrn" "custom_models/results/benchmark_v2/e3_c_seed2026" "AGCRN_native_dagg_napl_e10_h64_l2_seed2026" "cuda" || exit $?
run_job "E3-C" "train" "stid" "custom_models/results/benchmark_v2/e3_c_seed2026" "STID_native_node32_tid32_diw32_mlp3_seed2026" "cuda" || exit $?

echo "All 28 formal requests completed."
