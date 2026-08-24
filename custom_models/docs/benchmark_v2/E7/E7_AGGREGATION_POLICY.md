# E7 Aggregation Policy

All formal numbers are loaded automatically. Internal metrics come from validated A0/A1/A2/A3 artifacts; external metrics come from the explicitly supplied original26 workbook and must match the associated formal artifacts.

For Score, MAE, and RMSE, internal degradation is `ablation - A0` and percent degradation is `(ablation - A0) / A0 * 100`. R2 drop is `R2_A0 - R2_ablation`.

For external Score, MAE, and RMSE, ST-MGPrompt improvement is `(baseline - A0) / baseline * 100`. R2 gain is `R2_A0 - R2_baseline`. Each metric is ranked separately at H3/H6/H10; official Score is the main ranking metric, with the arithmetic mean of the three Score ranks also reported. No mixed-metric composite is created.

The external table is original26 native-training-system context, not a unified-loss structure-controlled comparison. E5 is cancelled and no common-loss table is consumed or produced.

When `CORE_E7_READY != 10/10`, `--require-complete` exits nonzero before creating `GRAPH_MECHANISM_ANALYSIS.xlsx`, formal result CSV files, or the formal Markdown report. The non-strict path writes only a clearly blocked preview JSON.
