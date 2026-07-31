# E3-A graph protocol runbook

This runbook provides validation only. It contains no graph-model train, evaluation, or hardware-preflight command.

## Windows PyCharm

- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`
- Script: `D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py`
- Working directory: `D:\PaperProject\GyxPaper2`
- Environment: `PYTHONPATH=D:\PaperProject\GyxPaper2\custom_models\src;PYTHONUTF8=1;PYTHONIOENCODING=utf-8`
- Parameters: `graph-protocol-check`

No `CUDA_VISIBLE_DEVICES` or `PYTORCH_CUDA_ALLOC_CONF` is needed.

## Windows PowerShell

```powershell
$env:PYTHONPATH='D:\PaperProject\GyxPaper2\custom_models\src'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$python='D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$runner='D:\PaperProject\GyxPaper2\custom_models\src\benchmark_v2\run_benchmark.py'

& $python $runner protocol-check
& $python $runner graph-protocol-check
& $python -m unittest custom_models.tests.benchmark_v2.test_e3_a_graph_protocol -v
& $python -m unittest discover -s custom_models/tests/benchmark_v2 -p 'test_*.py'
```

The offline builder defaults to fail closed if graph_v1 exists:

```powershell
& $python 'D:\PaperProject\GyxPaper2\custom_models\docs\benchmark_v2\E3_A\build_graph_protocol_v1.py'
```

It must report that overwrite is forbidden. Do not use or create graph_v2 in E3-A.

## Linux

```bash
export PYTHONPATH="$(pwd)/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

python custom_models/src/benchmark_v2/run_benchmark.py protocol-check
python custom_models/src/benchmark_v2/run_benchmark.py graph-protocol-check
python -m unittest custom_models.tests.benchmark_v2.test_e3_a_graph_protocol -v
python -m unittest discover -s custom_models/tests/benchmark_v2 -p 'test_*.py'
```

No shutdown flow is applicable because E3-A performs no long training.
