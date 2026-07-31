$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$Python = "D:\Apps\Miniconda3\envs\env_tslib\python.exe"
$Profile = "uniform_train_batch4_v1"
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "custom_models\src"

$Failed = 0
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "gru" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "dlinear" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "lightts" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "tide" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "segrnn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "transformer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "patchtst" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "itransformer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "timexer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "timesnet" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "micn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "wpmixer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "multipatchformer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "timemixer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "tsmixer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "frets" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "crossformer" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "msgnet" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "timefilter" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "gcn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "stgcn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "dcrnn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "graph_wavenet" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "mtgnn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "agcrn" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "custom_models/src/benchmark_v2/run_benchmark.py" hardware-preflight --model "stid" --training-profile $Profile --preflight-root "custom_models/results_smoke/benchmark_v2_uniform_bs4/hardware_preflight"
if ($LASTEXITCODE -ne 0) { $Failed = 1 }
& $Python "scripts/uniform_batch4_generate.py" collect-preflight --suite "all26"
if ($Failed -ne 0) { exit 3 }
