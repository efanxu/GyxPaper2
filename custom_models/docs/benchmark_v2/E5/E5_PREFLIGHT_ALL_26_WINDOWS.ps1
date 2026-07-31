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
& $Python $Runner hardware-preflight --model gru --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model dlinear --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model lightts --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model tide --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model segrnn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model transformer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model patchtst --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model itransformer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model timexer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model timesnet --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model micn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model wpmixer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model multipatchformer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model timemixer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model tsmixer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model frets --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model crossformer --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model msgnet --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model timefilter --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model gcn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model stgcn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model dcrnn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model graph_wavenet --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model mtgnn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model agcrn --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner hardware-preflight --model stid --experiment-profile e5_common_loss_v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
