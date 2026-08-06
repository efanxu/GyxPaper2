# E1-A implementation report

## Outcome

Persistence, MovingAverage, and the independent node-shared GRU are implemented and available through the explicit 28-entry registry. Runtime counts are 3 available and 25 unavailable; one model is trainable and two are deterministic evaluate-only baselines. No formal training, full formal evaluation, or formal result directory was created.

## Architecture and integration

New `models/` and `configs/` packages isolate model-specific logic. Statistical scaler conversion is centralized in `BaselineScalerContext` and `StatisticalBaselineAdapter`; CLI code never fits a second scaler or reads a second dataset path. `NodeSharedGRUAdapter` preserves the frozen B/N reshape and records parameter-sharing semantics.

Registry factories are explicit `module:function` paths and imported only by `create_model`. Registry listing remains data-free and does not import ST-MGPrompt or TSLib. The first three entries retain their E0-A `audit_status` as history while using current E1-A runtime fields.

`model_cli.py` extends the sole existing CLI with `model-smoke`, `full-shape-model-smoke`, `real-data-model-smoke`, `evaluate-only`, and the E1-A `train` path. `ProviderBatchIterable` lazily reuses `SDWPFDataProvider.batches` without duplicating split/window/scaler logic. `run_benchmark.py` remains a thin wrapper and now works both as a module and as the documented PyCharm script.

The generic Trainer remains the only trainer. It now honors the frozen Adam weight decay and AMP settings, saves AMP scaler state, monitors validation H10 Score, and still exposes only `fit(train_loader, val_loader)`.

## NON_TRAINABLE path

Persistence and MovingAverage create `NON_TRAINABLE` artifacts containing resolved/effective configuration, protocol and data signatures, environment, model summary, baseline state, scaler records/metadata, H3/H6/H10 metrics, prediction metadata, and status. They create no optimizer, scheduler, epoch log, or checkpoint.

## Tests and smoke

Final unit result: 30/30 PASS, including all 17 E0-B tests. Protocol check and registry count checks PASS.

Ordinary synthetic smoke: all three PASS. Statistical artifacts validate as `NON_TRAINABLE`; GRU completes masked loss, backward, optimizer step, validation H10 checkpointing, strict reload, independent evaluation, and SMOKE artifact validation.

Full-shape smoke: all three PASS at `(32,144,134,16) -> (32,134,10)` on CUDA. GRU uses hidden size 64, AMP, and backward; loss is `5.408382415771484`, peak allocated GPU memory is `2305106944` bytes, and parameter count is 16,394.

Limited real SDWPF smoke: accepted Persistence retry2, MovingAverage, and GRU runs PASS. GRU uses at most two train batches, one validation batch, and one independent evaluation batch, then strict reloads the best checkpoint.

## E0-B framework defect fix

The first real Persistence attempt revealed that `official_score` could return NaN when raw invalid targets were NaN: NumPy preserves NaN in `NaN * False`. A regression test was added and failed before the fix (`nan != 0.0`). The minimal fix in `benchmark_v2/metrics.py` masks the error with `np.where(mask, error, 0.0)` before aggregation. The Score formula, valid mask, horizons, units, and protocol are unchanged. The initial and retry1 Persistence runs remain recorded as FAIL; retry2 is the accepted result.

## Actual code changes

Created: model/config packages; statistical and GRU adapters; `model_runtime.py`; `model_cli.py`; two E1-A test modules.

Modified: adapter exports, explicit registry and loader, Trainer AMP/optimizer handling, runtime lazy batch iterable, CLI and executable wrapper, plus the single documented framework defect fix in `metrics.py`. One existing metrics test file gained the regression test.

Protected ST-MGPrompt, Canonical, TSLib, protocol, data, losses, and schemas were not modified. `metrics.py` is the sole intentional frozen-scope difference and is covered by the pre-fix failure/post-fix pass evidence.

## Limitations and formal readiness

Smoke metrics are diagnostic only and must not be used in the paper. The limited real slice has low valid-target ratio and undefined R2, serialized as JSON `null`. Formal Persistence/MovingAverage evaluation and formal 20-epoch GRU training are NOT_RUN. The exact commands are in `E1_A_RUNBOOK.md`.
