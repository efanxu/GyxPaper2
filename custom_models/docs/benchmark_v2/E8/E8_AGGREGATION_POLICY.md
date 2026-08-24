# E8 Aggregation Policy

1. Evidence is resolved only from the frozen A0/A4/A5/A6/A7 locations and the six frozen original26 model IDs.
2. Project discovery accepts exactly one `original26*.xlsx`; otherwise the CLI requires `--original26-xlsx <path>` and reports `MISSING_ORIGINAL26_XLSX` or ambiguity.
3. Workbook H3/H6/H10 Score/MAE/RMSE/R2 values are reconciled against formal run artifacts. Any difference is `ORIGINAL26_ARTIFACT_MISMATCH` and blocks that evidence.
4. NaN, infinity, None, blank values, invalid run status, invalid protocol status, smoke, old batch32, E5/common-loss, A8, quarantine, archived attempts, and excluded models are rejected. Missing values are never replaced by zero.
5. Internal Score/MAE/RMSE degradation is ablation minus A0, in absolute and percent form. R2 is A0 minus ablation.
6. External Score/MAE/RMSE improvement is `(baseline-A0)/baseline*100`; R2 is `A0-baseline`. Each horizon is ranked separately; Score mean rank is the mean of H3/H6/H10 Score ranks only.
7. No unregistered composite score combines Score/MAE/RMSE/R2.
8. Direction decomposition is omitted automatically when its config predicate fails.
9. Failed or blocked diagnostics do not contribute diagnostic numbers to the formal report.
10. Formal `PROMPT_CROSS_FUSION_ANALYSIS.xlsx`, Markdown, and result CSVs are created only at 11/11. Workbook creation uses the bundled `@oai/artifact-tool`, scans formula errors, and renders every sheet before export.
