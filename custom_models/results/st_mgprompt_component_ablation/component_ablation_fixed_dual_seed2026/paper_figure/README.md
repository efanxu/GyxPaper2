# ST-MGPrompt A0-A8 component-ablation paper figure

This directory contains the reproducible paper figure generated from the
formal A0-A8 checkpoints and their H3/H6/H10 metric files.

## Reproduce

Use the fixed project interpreter from the repository root:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' `
  'custom_models\src\st_mgprompt\component_ablation_paper_figure.py' all `
  --device cuda
```

The `export` command performs inference only. It does not train models or
modify source checkpoints, metrics, or experiment status files. Once the
prediction exports exist, the figure can be regenerated with:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' `
  'custom_models\src\st_mgprompt\component_ablation_paper_figure.py' plot
```

The second-round visual QA can be regenerated against the approved design
reference with:

```powershell
& 'D:\Apps\Miniconda3\envs\env_tslib\python.exe' `
  'custom_models\src\st_mgprompt\component_ablation_figure_visual_qa.py' `
  --reference 'C:\Users\12811\Desktop\ChatGPT Image 2026年9月18日 15_03_11.png' `
  --render 'custom_models\results\st_mgprompt_component_ablation\component_ablation_fixed_dual_seed2026\paper_figure\figures\st_mgprompt_component_ablation_A0_A8_600dpi.png' `
  --output-dir 'custom_models\results\st_mgprompt_component_ablation\component_ablation_fixed_dual_seed2026\paper_figure\visual_qa'
```

## Data and selection rules

- Target: `Patv_raw`; validity mask: `valid_target_mask`.
- Horizons: H=3, H=6, H=10, using the formal prefix-evaluation protocol.
- Forecast tracking: mean Patv over valid turbines on the first 400 common
  available target timestamps, aligned across all three horizons.
- Tracking methods: Actual plus A0-A8 in every horizon panel. Actual and A0
  use the highest drawing priority; A1-A8 use thinner, partially transparent,
  uniquely colored and styled lines so the complete comparison remains legible.
- Error distributions: all real valid absolute errors. Histogram density and
  box statistics are derived from the exported pointwise predictions; Mean
  labels are checked against the corresponding formal MAE. The plotted range
  keeps the complete 0-1500 kW histogram support. The violin body represents
  99.7% of the real histogram mass, while the remaining high-error support is
  represented by up to eight evenly spaced, light tail markers instead of a
  visually dominant needle-like line.
- Metric comparison: formal `metrics_eval_h3.json`, `metrics_eval_h6.json`,
  and `metrics_eval_h10.json` values for every A0-A8 variant, shown as four
  vertically stacked white-background charts with a compact 3 x 3 legend.
- Error distributions: three lightweight Three-step/Six-step/Ten-step panels
  with matching blue, orange, and coral title strips and no repeated blue
  card borders.

## Important audit note

The formal A1 checkpoint and `config.json` match the A1 definition. A later
incomplete launch overwrote A1's mutable `active_config.json` and
`requested_config.json` with A8 values and appended an orphan `TRAIN_STARTED`
event. The figure pipeline does not use those polluted mutable files; the
finding is recorded in `protocol_audit.json`.

## Outputs

- `figures/`: PDF, SVG, and 600 dpi PNG.
- `figure_data/metrics.csv`: the 27 formal variant-horizon metric rows.
- `figure_data/tracking.csv`: the 1,200 aligned tracking rows.
- `figure_data/error_distribution.npz`: full-data histograms and box/mean
  statistics used for the violin panels.
- `figure_data/mean_mae_audit.json`: Mean-versus-MAE checks.
- `predictions/`: shared real targets/mask/index data and one prediction array
  per variant.
- `protocol_audit.json` and `figure_manifest.json`: protocol and output
  manifests.
- `visual_qa/`: same-resolution reference, 50% overlay, difference image, and
  a dimension/aspect-ratio report for the pixel-level layout check.
