# ST-MGPrompt formal component ablations

The current formal component-ablation IDs are A0–A8:

| ID | Definition |
| --- | --- |
| A0 | Canonical Full |
| A1 | w/o Spatial Graph |
| A2 | w/o Adaptive Graph |
| A3 | w/o Diffusion |
| A4 | Mean-Pooling Macro Prompt |
| A5 | Short-context Reverse Cross |
| A6 | Early-History Macro Cross Fusion |
| A7 | Shared-Projection Cross Fusion |
| A8 | w/o MS-MG-DWU |

A6 promotes the former A6-C1 result: Macro-to-Fine attention excludes the
latest 24 Fine positions (`macro_to_fine_exclude_recent_len=24`). A7 promotes
the former A6-C3 result and shares the two cross-attention projection modules
(`share_cross_attention_projections=true`). The former A6-C2 static-injection
candidate is removed.

A8 is the former A7 loss ablation (`w/o MS-MG-DWU`) under its new permanent ID.

On Windows, run A6 or A7 fresh from the repository without `--resume`:

```powershell
& '.\custom_models\docs\st_mgprompt_component_ablation\A6_WINDOWS_FULL.ps1' -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\st_mgprompt_component_ablation\A7_WINDOWS_FULL.ps1' -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
```

For a fresh Batch4 A8 loss-ablation artifact, use the A8 gate:

```powershell
& '.\custom_models\docs\benchmark_v2\A8\A8_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1' -Action StaticAudit -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\benchmark_v2\A8\A8_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1' -Action Preflight -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\benchmark_v2\A8\A8_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1' -Action Run -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
```
