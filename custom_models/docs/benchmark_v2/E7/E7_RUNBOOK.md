# E7 Runbook

Run from `D:\PaperProject\GyxPaper2` with:

```powershell
$Py = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
$Xlsx = 'C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx'
```

The prompt-supplied path `D:\Apps\Miniconda3\envs\env\_tslib\python.exe` does not exist on the audited machine; `env_tslib\python.exe` is the installed fixed environment used for this implementation.

The CLI discovers A0/A1/A2/A3 only from the authoritative fixed-dual component manifest and its referenced formal directories. It rejects smoke, old batch32 as current evidence, E5/common-loss, A8, and excluded model paths.

Readiness and audit do not train:

```powershell
& $Py -m benchmark_v2.e7 readiness --original26-xlsx $Xlsx
& $Py -m benchmark_v2.e7 audit --original26-xlsx $Xlsx
```

Read-only diagnostics load frozen GraphBundle supports and the formal MTGNN best checkpoint. They write only to the E7 analysis root:

```powershell
& $Py -m benchmark_v2.e7 diagnostics
```

Formal aggregation fails closed unless readiness is 10/10:

```powershell
& $Py -m benchmark_v2.e7 aggregate --original26-xlsx $Xlsx --require-complete
& $Py -m benchmark_v2.e7 all --original26-xlsx $Xlsx --require-complete
```

For a future complete formal workbook, obtain the bundled paths from Codex workspace dependencies, then set `E7_ARTIFACT_NODE` and `E7_ARTIFACT_NODE_MODULES`. The aggregator uses `@oai/artifact-tool`; no workbook is created before core readiness passes.

E5 is cancelled and never participates. Persistence, GRU, DLinear, Crossformer, GraphWaveNet, AGCRN, SegRNN, and MSGNet never participate. Source results/checkpoints/configs are read-only and are never overwritten.
