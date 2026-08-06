# E3-C formal runbook

Commands are prepared only. None was executed during E3-C. Formal Full,
formal evaluation, target high-memory preflight, and shutdown commands remain
`NOT_RUN`.

Formal output root (not created):

`D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_c_seed2026`

Frozen run IDs:

- `GraphWaveNet_native_pfpr_adp10_b4l2_seed2026`
- `MTGNN_native_adaptive_k20_gdep2_l3_seed2026`
- `AGCRN_native_dagg_napl_e10_h64_l2_seed2026`
- `STID_native_node32_tid32_diw32_mlp3_seed2026`

Run one model in one Python process at a time.

## Windows PyCharm

- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script: `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment:
  `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8;PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;CUDA_VISIBLE_DEVICES=0`

Create four independent diagnostic configurations:

```text
hardware-preflight --model graph_wavenet
hardware-preflight --model mtgnn
hardware-preflight --model agcrn
hardware-preflight --model stid
```

Create four independent formal train configurations:

```text
train --model graph_wavenet --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_c_seed2026 --run-id GraphWaveNet_native_pfpr_adp10_b4l2_seed2026 --device cuda
train --model mtgnn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_c_seed2026 --run-id MTGNN_native_adaptive_k20_gdep2_l3_seed2026 --device cuda
train --model agcrn --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_c_seed2026 --run-id AGCRN_native_dagg_napl_e10_h64_l2_seed2026 --device cuda
train --model stid --input-path D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet --target-path D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet --output-root D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_c_seed2026 --run-id STID_native_node32_tid32_diw32_mlp3_seed2026 --device cuda
```

## Windows PowerShell

```powershell
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
$env:CUDA_VISIBLE_DEVICES='0'
$python='D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$runner='D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py'
$input='D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet'
$target='D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'
$output='D:\PaperProject\GyxPaper2\custom_models\results\benchmark_v2\e3_c_seed2026'
```

Standalone exact preflight:

```powershell
& $python $runner hardware-preflight --model graph_wavenet
& $python $runner hardware-preflight --model mtgnn
& $python $runner hardware-preflight --model agcrn
& $python $runner hardware-preflight --model stid
```

Independent formal requests:

```powershell
& $python $runner train --model graph_wavenet --input-path $input --target-path $target --output-root $output --run-id 'GraphWaveNet_native_pfpr_adp10_b4l2_seed2026' --device cuda
& $python $runner train --model mtgnn --input-path $input --target-path $target --output-root $output --run-id 'MTGNN_native_adaptive_k20_gdep2_l3_seed2026' --device cuda
& $python $runner train --model agcrn --input-path $input --target-path $target --output-root $output --run-id 'AGCRN_native_dagg_napl_e10_h64_l2_seed2026' --device cuda
& $python $runner train --model stid --input-path $input --target-path $target --output-root $output --run-id 'STID_native_node32_tid32_diw32_mlp3_seed2026' --device cuda
```

Inspect:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_benchmark.py' } | Select-Object ProcessId,ParentProcessId,CommandLine
nvidia-smi
$LASTEXITCODE
```

## Linux foreground

```bash
cd /path/to/GyxPaper2
test -f custom_models/src/benchmark_v2/run_benchmark.py
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/benchmark_v2

python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model graph_wavenet
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model mtgnn
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model agcrn
python custom_models/src/benchmark_v2/run_benchmark.py hardware-preflight --model stid

python custom_models/src/benchmark_v2/run_benchmark.py train --model graph_wavenet --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id GraphWaveNet_native_pfpr_adp10_b4l2_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model mtgnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id MTGNN_native_adaptive_k20_gdep2_l3_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model agcrn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id AGCRN_native_dagg_napl_e10_h64_l2_seed2026 --device cuda
python custom_models/src/benchmark_v2/run_benchmark.py train --model stid --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id STID_native_node32_tid32_diw32_mlp3_seed2026 --device cuda
```

## Linux nohup, exit code, sync, unconditional shutdown

Run exactly one block at a time. Each block shuts down the host regardless of
the train exit code after recording it and calling `sync`.

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model graph_wavenet --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id GraphWaveNet_native_pfpr_adp10_b4l2_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/GraphWaveNet_native_pfpr_adp10_b4l2_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model mtgnn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id MTGNN_native_adaptive_k20_gdep2_l3_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/MTGNN_native_adaptive_k20_gdep2_l3_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model agcrn --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id AGCRN_native_dagg_napl_e10_h64_l2_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/AGCRN_native_dagg_napl_e10_h64_l2_seed2026.log 2>&1 &
```

```bash
nohup bash -lc '
python custom_models/src/benchmark_v2/run_benchmark.py train --model stid --input-path dataset/sdwpf_model_input_base.parquet --target-path dataset/sdwpf_eval_target.parquet --output-root custom_models/results/benchmark_v2/e3_c_seed2026 --run-id STID_native_node32_tid32_diw32_mlp3_seed2026 --device cuda;
code=$?;
sync;
/usr/bin/shutdown -h now;
exit $code
' > logs/benchmark_v2/STID_native_node32_tid32_diw32_mlp3_seed2026.log 2>&1 &
```

Inspect:

```bash
tail -f logs/benchmark_v2/<run-id>.log
ps -ef | grep '[r]un_benchmark.py'
nvidia-smi
```

A reusable preflight PASS must match every benchmark, source listing, exact
shape, AMP, graph, support, adaptive graph, node identity, temporal identity,
and initialization policy field, and must contain completed forward/backward
with `status=PASS`.
