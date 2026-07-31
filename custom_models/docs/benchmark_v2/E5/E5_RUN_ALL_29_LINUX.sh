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
python "$runner" train --model gru --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id GRU_node_shared_h64_l1_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model dlinear --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id DLinear_node_shared_ma25_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model lightts --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id LightTS_node_shared_chunk8_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model tide --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id TiDE_node_shared_d512_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model segrnn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id SegRNN_node_shared_seg2_d512_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model transformer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id Transformer_node_shared_d512_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model patchtst --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id PatchTST_node_shared_p16_s8_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model itransformer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id iTransformer_node_shared_d512_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model timexer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id TimeXer_node_shared_ms_d512_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model timesnet --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id TimesNet_node_shared_d32_k5_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model micn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id MICN_node_shared_k12_16_d32_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model wpmixer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id WPMixer_node_shared_db2_l1_p16_s8_d256_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model multipatchformer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id MultiPatchFormer_node_shared_p8_16_24_32_d256_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model timemixer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id TimeMixer_node_shared_ci1_ds3_w2_avg_d16_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model tsmixer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id TSMixer_node_shared_el2_d32_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model frets --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id FreTS_node_shared_tcfft_e128_h256_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model crossformer --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id Crossformer_node_shared_seg12_win2_d32_el2_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model msgnet --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id MSGNet_node_shared_k5_d512_el2_gdep2_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model timefilter --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id TimeFilter_node_shared_p16_d512_el2_tp0p5_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model gcn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id GCN_native_agcn_h64_l2_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model stgcn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id STGCN_native_cheb3_kt3_b2_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model dcrnn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id DCRNN_native_dualrw_k2_h64_l2_ar_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model graph_wavenet --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id GraphWaveNet_native_pfpr_adp10_b4l2_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model mtgnn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id MTGNN_native_adaptive_k20_gdep2_l3_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model agcrn --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id AGCRN_native_dagg_napl_e10_h64_l2_loss_msa_hybrid_seed2026 --device cuda
python "$runner" train --model stid --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id STID_native_node32_tid32_diw32_mlp3_loss_msa_hybrid_seed2026 --device cuda
python "$runner" evaluate-only --model persistence --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id Persistence_loss_msa_hybrid_seed2026 --device cpu
python "$runner" evaluate-only --model moving_average --experiment-profile e5_common_loss_v1 --input-path "$input" --target-path "$target" --output-root "$output" --run-id MovingAverage_w144_loss_msa_hybrid_seed2026 --device cpu
python "$runner" e5-reference-a8
