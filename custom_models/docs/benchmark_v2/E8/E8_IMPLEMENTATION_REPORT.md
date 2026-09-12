# E8 Implementation Report

E8 is isolated under `custom_models/src/benchmark_v2/e8/`. It implements project-only workbook discovery plus explicit path override, strict workbook schema loading, frozen external whitelist and excluded-model rejection, workbook/artifact reconciliation, current batch4 protocol audit, internal config-diff and direction audit, evidence/readiness manifests, source-backed mechanism capability matrix, default-off diagnostic hooks, numerical representation/Cross-Fusion/Prompt analysis, paired-window identity and mask checks, grouped-definition gating, fail-closed aggregation, Markdown reporting, artifact inventory, and an artifact-tool Excel path.

No training entry point exists.

Current audit result is intentionally incomplete. The six external original26 artifacts are valid batch4 formal evidence and match the workbook. Available A0/A4/A5/A6/A7 artifacts record train batch32, lack `uniform_train_batch4_v1`, lack `resolved_config.json`, and do not expose the current formal run-status schema. They are retained as read-only historical component artifacts but rejected from current E8 readiness. Therefore current readiness is internal 0/5, external 6/6, core 6/11.

Static source audit confirms the intended A4/A5/A6/A7 differences: A4 changes only Macro Prompt temporal pooling, A5 shortens Reverse Cross context, A6 replaces the learned complementary gate with a fixed 0.5 gate, and A7 inherits the former A8 `w/o MS-MG-DWU` loss configuration. This static result does not override the formal protocol evidence gap.

Formal mechanism export and paired-window analysis are blocked. The existing group-analysis directory has training-fitted thresholds but lacks the complete frozen public group/window definition set.

Verification completed with compileall, import checks, and 13 E8 contract tests. The tests cover workbook schema fixtures, the exact external whitelist, excluded/E5/smoke/batch32 rejection, config diffs, direction decomposition and automatic disablement, missing-evidence readiness, strict fail-closed aggregation, metric directions, workbook/artifact mismatch detection, hook shapes, deterministic Fine/Coarse alignment, cosine/CKA numerics, attention `NOT_APPLICABLE`, window/mask identity, diagnostic side effects, and a 16-sheet artifact-tool workbook render/schema check.

The real diagnostics command returned exit code 2 without checkpoint evaluation. The real strict aggregate returned exit code 2 at `CORE_E8_READY=6/11`; checks confirmed that no formal workbook, Markdown report, internal/external result CSV, direction table, or representation CSV was created.
