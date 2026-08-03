# Current Original Batch4 scope26

Status: `ACTIVE`.

The only current Original execution source is
`CURRENT_BATCH4_SCOPE26_MANIFEST.json`, with the exact run-id map in
`ORIGINAL_SCOPE26_RUN_ID_MAP.json`. The active gate is
`scripts/original_batch4_scope26_gate.py` and the Windows entry point is
`ORIGINAL_RUN_ALL_26_BATCH4_WINDOWS.ps1`.

## Scope and frozen identity

- Scope: `benchmark_v2_batch4_scope26_seed2026`
- 26 active entries: 24 `TRAINABLE` and 2 `EVALUATE_ONLY`
- Train/validation/test batch: 4/4/4; gradient accumulation: 1; effective train batch: 4
- Seed: 2026; lookback: 144; prediction horizon: 10; evaluation horizons: 3/6/10
- `Transformer`: effective AMP `false`, precision `fp32`
- Other active trainable models: profile-default AMP; `Persistence` and
  `MovingAverage`: CPU-only `cpu_baseline`

`SegRNN` and `MSGNet` remain source-code and historical-artifact records only.
Their current status is `EXCLUDED_FROM_CURRENT_FORMAL_SCOPE` with reason
`RESOURCE_REQUIREMENT_EXCEEDS_AVAILABLE_FORMAL_HARDWARE`; they are absent from
the active manifest and denominator.

The Original freeze is independent of E5. It covers the current 26-entry
manifest, current run map, active model/config source closures, protocol,
Batch4 profile, dataset/graph identities, Original loss, per-model precision,
and the active gate/launcher revisions. It does not include the E5 manifest,
legacy E5 run maps, SegRNN, MSGNet, or Batch32 artifacts.

## Commands

All commands use the configured environment and do not enable autoshutdown:

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
$env:PYTHONPATH = 'D:\PaperProject\GyxPaper2\custom_models\src'
$python = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
$launcher = 'D:\PaperProject\GyxPaper2\custom_models\docs\benchmark_v2\BATCH4\ORIGINAL_RUN_ALL_26_BATCH4_WINDOWS.ps1'

# Safe dry-run: prints a 26-entry plan and inspects existing artifacts.
& $launcher -Action DryRun

# Formal readiness only; no training is started.
& $launcher -Action Readiness

# Static current preflight denominator; this does not run GPU preflight.
& $launcher -Action PreflightPlan

# Formal run. Each active entry is a child process; use explicit data paths when needed.
& $launcher -Action Run -InputPath 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' -TargetPath 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'

# Aggregate is fail-closed and requires all 26 identities to be ready.
& $launcher -Action Aggregate -RequireComplete
```

The direct gate commands are equivalent:

```powershell
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' dry-run
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' readiness
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' aggregate --require-complete
```

## Resume and failure behavior

Resume builds and prints the full plan before any child starts. A completed
result is skipped only after exact model/run/output-root/formal/profile/Batch4
protocol/dataset/config/source/precision/checkpoint/metric/receipt validation.
Incomplete canonical directories are planned as
`ARCHIVE_INCOMPLETE_THEN_RUN`; identity mismatches are blocked and never
overwritten. Noncanonical directories, renamed runs, smoke output, Batch32
output, and results without an execution or adoption receipt are not silently
accepted.

Each model runs in its own subprocess. A non-zero child exit preserves its
log and failure artifact, reports `failed_model`, `error_type`,
`error_message`, and `log_path`, then continues to later `RUN_MISSING` entries.
The final state distinguishes `COMPLETED_READY_26_OF_26`,
`COMPLETED_WITH_FAILURES`, `BLOCKED_GLOBAL_IDENTITY`, `NOT_READY`, and
`AGGREGATE_SKIPPED_NOT_COMPLETE`. Aggregate never runs unless the readiness
denominator is exactly 26/26.

## Existing result audit

The read-only inventory command writes machine-specific, untracked files to
`custom_models/logs/uniform_bs4/audit/`:

```powershell
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' inventory
```

The inventory separately records reusable, incomplete, failed, identity
mismatch, orphan/noncanonical, excluded historical, and unknown directories.
It validates H3/H6/H10 metrics fail-closed, including finite numeric values,
positive valid-target counts, and JSON/CSV consistency.

## Transformer retry2 decision

`Transformer_node_shared_d512_fp32_bs4_seed2026_retry2` is a formally completed
FP32 result with non-empty checkpoints and finite H3/H6/H10 metrics. It is not
adopted because the local artifact has no current Original source-closure
identity and no execution/adoption receipt. The decision is therefore
`RERUN_CANONICAL`; the exact canonical run-id remains
`Transformer_node_shared_d512_bs4_seed2026`. The retry2 directory, the older
AMP failure directory, and all other historical evidence remain read-only.

The old Original 27/28/29 launchers, including
`UNIFORM_BATCH4_RUN_ORIGINAL_27_SKIP_MSGNET_WINDOWS.ps1`, are superseded and
are not called by this entry point.
