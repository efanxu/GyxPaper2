# ST-MGPrompt Cross-Fusion candidate ablations

The formal A0-A7 component-ablation definitions remain unchanged. These three
opt-in candidates are derived independently from A0 and are excluded from the
default formal `metrics_summary.csv` / `metrics_summary.json`:

| Candidate | Definition | Only semantic config difference from A0 |
| --- | --- | --- |
| A6-C1 | Early-History Macro Cross Fusion | `macro_to_fine_exclude_recent_len: 0 -> 24` |
| A6-C2 | Static Macro Injection Cross Fusion | `macro_to_fine_mode: query_attention -> static_mean` |
| A6-C3 | Shared-Projection Cross Fusion | `share_cross_attention_projections: false -> true` |

A6-C1 keeps Macro-to-Fine attention on the first 120 Fine positions and removes
its increment from the latest 24 positions. Its Fine history remains length 144,
and its Fine-to-Coarse K/V context remains 24. This differs from A5, which
changes Fine-to-Coarse K/V from 24 to 6, and from formal A6, which disables
Macro-to-Fine attention entirely.

A6-C2 keeps A0 Macro Prompt generation, attention pooling, and four prompt
tokens, but consumes their node-specific mean as a static context at every Fine
position. A6-C3 assigns both directions to the same `MultiheadAttention` module,
so its projection parameters are shared by identity.

The runner writes each candidate beneath its own ID directory, including
`effective_config.json`, `effective_config_diff.json`, and
`candidate_manifest.json`. The root-level
`cross_fusion_candidate_metrics_summary.csv` / `.json` compares A0, formal A6,
A6-C1, A6-C2, and A6-C3 without changing the formal A0-A7 summary.

Run on Windows from the repository root:

```powershell
& '.\custom_models\docs\st_mgprompt_component_ablation\A6_CANDIDATES_WINDOWS_FULL.ps1' -Candidate A6-C1 -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\st_mgprompt_component_ablation\A6_CANDIDATES_WINDOWS_FULL.ps1' -Candidate A6-C2 -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\st_mgprompt_component_ablation\A6_CANDIDATES_WINDOWS_FULL.ps1' -Candidate A6-C3 -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& '.\custom_models\docs\st_mgprompt_component_ablation\A6_CANDIDATES_WINDOWS_FULL.ps1' -Candidate All -PythonExecutable 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
```
