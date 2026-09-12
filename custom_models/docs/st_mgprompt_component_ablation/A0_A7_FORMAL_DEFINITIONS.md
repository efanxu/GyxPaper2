# ST-MGPrompt formal component ablations

The only current formal component-ablation IDs are A0–A7:

| ID | Definition |
| --- | --- |
| A0 | Canonical Full |
| A1 | w/o Spatial Graph |
| A2 | w/o Adaptive Graph |
| A3 | w/o Diffusion |
| A4 | Mean-Pooling Macro Prompt |
| A5 | Short-context Reverse Cross |
| A6 | Fixed-Gate Cross Fusion |
| A7 | w/o MS-MG-DWU |

A6 keeps `fusion_mode=cross`, both cross-attention directions, residual connections, and LayerNorm. Its only formal semantic difference from A0 is `cross_fusion_gate_strategy: adaptive -> fixed_half`, which applies a constant 0.5 coefficient to both cross-attention residual injections and adds no learnable gate parameters.

A7 inherits the former A8 architecture/loss configuration. The checked-in `renumbered_reference.json` records the identity-only mapping and leaves the former checkpoint and metrics unchanged. The former Prompt A7, former A8 ID, and A9 are historical only and are rejected by the current runner.

On Windows, run A6 fresh from the repository without `--resume`:

```powershell
& '.\custom_models\docs\st_mgprompt_component_ablation\A6_WINDOWS_FULL.ps1' -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
```

If a fresh Batch4 A7 artifact is required instead of the explicit former-A8 reference, use the migrated A7 gate:

```powershell
& '.\custom_models\docs\benchmark_v2\A7\A7_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1' -Action StaticAudit -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\benchmark_v2\A7\A7_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1' -Action Preflight -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\benchmark_v2\A7\A7_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1' -Action Run -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
```
