# E9 Implementation Report

## Implemented

- E9-A dependency audit and portability classification.
- Model-independent `LossInputBundle` and backward-compatible engine dispatch.
- Exact `(B,N,H)` adapter around `st_mgprompt.losses.MSMGDWULoss`.
- Loss module train/eval lifecycle and loss-state checkpoint save/restore.
- Six-model variant/run-id manifest and isolated formal output root.
- Read-only original26 workbook/artifact resolver with loss-identity and metric reconciliation.
- Loss-only config diff, initialization pairing audit, evidence manifest, and fail-closed readiness.
- E9 full-shape preflight inventory and formal `train-one` orchestration.
- Paired metric/family/training/loss-component aggregation and 19-sheet Excel/Markdown framework.
- Linux precheck, all-six preflight, manifest-driven run-all, transfer-only readiness, autoshutdown, and full readiness/aggregate scripts.

## Current audit result

The workbook `C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx` has eight sheets. E9 uses `原始指标` and `模型明细`; the selected 18 metric rows reconcile exactly to the six formal artifacts.

All six controls are `COMPLETED`, protocol `PASS`, batch 4, seed 2026, and `loss=masked_mse`. Static proposed transfer config diff is 6/6 valid. Exact initial-state pairing is `NOT_VERIFIED` because the historical controls do not expose a dedicated initial-model-state identity; only same seed/source/config/protocol may be claimed.

Current readiness after implementation is controls 6/6, transfers 0/6, pairing 0/6, and core 6/12. This is the expected pre-training state.

## Local validation

The E9 module compiles/imports. Twenty-three E9 tests pass, covering portability outcomes/subset naming, control loss/model identity, E5 source rejection, loss forward/backward, partial/all-masked behavior, frozen eval state, loss-state checkpoint roundtrip, workbook resolver fixtures, config-diff failure cases, metric directions, both readiness gates, script scope/order, and the 19-sheet artifact-tool workbook fixture. The existing loss/metric and Trainer no-test-loader unittest modules also pass when run directly with the source path fixed.

No formal E9 training or GPU full-shape preflight was started locally.
