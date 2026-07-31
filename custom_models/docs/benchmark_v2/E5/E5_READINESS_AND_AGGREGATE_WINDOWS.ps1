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

& $Python $Runner e5-readiness --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python $Runner e5-aggregate --output-root $OutputRoot --require-complete
exit $LASTEXITCODE
