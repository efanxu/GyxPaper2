$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES = '0'
$Python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$Runner = 'D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py'
$InputPath = 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet'
$TargetPath = 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'
$OutputRoot = 'D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\common_loss_architecture_seed2026'
foreach ($Path in @($Python, $Runner, $InputPath, $TargetPath)) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required path: $Path" }
}
& $Python $Runner train --model gru --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'GRU_node_shared_h64_l1_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model dlinear --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'DLinear_node_shared_ma25_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model lightts --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'LightTS_node_shared_chunk8_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model tide --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'TiDE_node_shared_d512_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model segrnn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'SegRNN_node_shared_seg2_d512_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model transformer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'Transformer_node_shared_d512_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model patchtst --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'PatchTST_node_shared_p16_s8_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model itransformer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'iTransformer_node_shared_d512_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model timexer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'TimeXer_node_shared_ms_d512_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model timesnet --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'TimesNet_node_shared_d32_k5_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model micn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'MICN_node_shared_k12_16_d32_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model wpmixer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'WPMixer_node_shared_db2_l1_p16_s8_d256_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model multipatchformer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'MultiPatchFormer_node_shared_p8_16_24_32_d256_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model timemixer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'TimeMixer_node_shared_ci1_ds3_w2_avg_d16_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model tsmixer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'TSMixer_node_shared_el2_d32_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model frets --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'FreTS_node_shared_tcfft_e128_h256_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model crossformer --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'Crossformer_node_shared_seg12_win2_d32_el2_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model msgnet --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'MSGNet_node_shared_k5_d512_el2_gdep2_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model timefilter --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'TimeFilter_node_shared_p16_d512_el2_tp0p5_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model gcn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'GCN_native_agcn_h64_l2_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model stgcn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'STGCN_native_cheb3_kt3_b2_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model dcrnn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'DCRNN_native_dualrw_k2_h64_l2_ar_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model graph_wavenet --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'GraphWaveNet_native_pfpr_adp10_b4l2_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model mtgnn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'MTGNN_native_adaptive_k20_gdep2_l3_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model agcrn --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'AGCRN_native_dagg_napl_e10_h64_l2_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner train --model stid --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'STID_native_node32_tid32_diw32_mlp3_loss_msa_hybrid_seed2026' --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner evaluate-only --model persistence --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'Persistence_loss_msa_hybrid_seed2026' --device cpu
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner evaluate-only --model moving_average --experiment-profile e5_common_loss_v1 --input-path $InputPath --target-path $TargetPath --output-root $OutputRoot --run-id 'MovingAverage_w144_loss_msa_hybrid_seed2026' --device cpu
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner e5-reference-a8
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
