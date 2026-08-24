# E8 Runbook

Run from `D:\PaperProject\GyxPaper2` with the fixed interpreter:

```powershell
$env:PYTHONPATH = (Resolve-Path '.\custom_models\src').Path
$env:PYTHONIOENCODING = 'utf-8'
$PY = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$XLSX = 'C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx'
```

The repository currently contains no unique `original26*.xlsx`, so pass `--original26-xlsx`. Never guess or hand-copy values.

## Discovery and readiness

A0 resolves through `custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026`. A4/A5/A6/A7 resolve under `custom_models/results/st_mgprompt_component_ablation/component_ablation_fixed_dual_seed2026/<variant>/STMGPrompt_ComponentAblation`.

```powershell
& $PY -m benchmark_v2.e8 readiness --original26-xlsx $XLSX
& $PY -m benchmark_v2.e8 audit --original26-xlsx $XLSX
```

Readiness/audit never train.

## Diagnostics

```powershell
& $PY -m benchmark_v2.e8 diagnostics
```

Diagnostics is evaluate-only. It writes only to `custom_models/results/benchmark_v2/e8_prompt_cross_fusion_analysis_seed2026/`. With current batch32-only internal artifacts it exits blocked and must not run a checkpoint export.

## Aggregate

For future complete evidence, configure the bundled artifact runtime before Excel creation:

```powershell
$env:E8_ARTIFACT_NODE = 'C:\Users\12811\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:E8_ARTIFACT_NODE_MODULES = 'C:\Users\12811\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
& $PY -m benchmark_v2.e8 aggregate --original26-xlsx $XLSX --require-complete
& $PY -m benchmark_v2.e8 all --original26-xlsx $XLSX --require-complete
```

Current strict aggregate is expected to exit nonzero at 6/11 and create no formal Excel/result tables.

E5/common-loss never participates. A8 never participates. Persistence, GRU, DLinear, Crossformer, GraphWaveNet, AGCRN, SegRNN, and MSGNet never participate. E8 never writes back to source runs, configs, checkpoints, metrics, protocol files, or run-status files.
