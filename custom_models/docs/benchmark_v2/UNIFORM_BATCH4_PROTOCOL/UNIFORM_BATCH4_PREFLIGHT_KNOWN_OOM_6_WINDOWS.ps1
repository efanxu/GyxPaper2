$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "D:\Apps\Miniconda3\envs\env_tslib\python.exe"
$Profile = "uniform_train_batch4_v1"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\src"

$Failed = 0
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "segrnn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "transformer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "patchtst" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "frets" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "msgnet" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "timefilter" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "scripts/uniform_batch4_generate.py" collect-preflight --suite "known_oom6"
if ($Failed -ne 0) { exit 3 }
