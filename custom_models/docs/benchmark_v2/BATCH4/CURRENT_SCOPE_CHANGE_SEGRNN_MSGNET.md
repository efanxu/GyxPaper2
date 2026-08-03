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

# Explicit exact current-scope GPU preflight (runs 24 isolated GPU children).
# Do not use this command during static validation.
& $launcher -Action Preflight

# Formal run. Each active entry is a child process; use explicit data paths when needed.
& $launcher -Action Run -InputPath 'D:\PaperProject\GyxPaper2\dataset\sdwpf_model_input_base.parquet' -TargetPath 'D:\PaperProject\GyxPaper2\dataset\sdwpf_eval_target.parquet'

# Aggregate is fail-closed and requires all 26 identities to be ready.
& $launcher -Action Aggregate -RequireComplete
```

The direct gate commands are equivalent:

```powershell
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' dry-run
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' readiness
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' preflight-plan
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' lock-status
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' clear-stale-lock
# Identity-mismatch quarantine is never implicit:
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' quarantine-existing --model transformer
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' quarantine-existing --model transformer --apply
& $python 'D:\PaperProject\GyxPaper2\scripts\original_batch4_scope26_gate.py' aggregate --require-complete
```

## Resume and failure behavior

Resume builds and prints the full plan before any child starts. A completed
result is skipped only after exact model/run/output-root/formal/profile/Batch4
protocol/dataset/config/source/precision/checkpoint/metric/receipt validation.
Incomplete or FAILED identity-matched canonical directories are planned as
`ARCHIVE_INCOMPLETE_THEN_RUN`. The runner writes an `archive_receipt.json` with
the recursive file manifest, hashes, current Git/manifest/freeze identity, and
then uses an atomic `os.replace()` move to
`.original_scope26_archived_attempts/<run_id>/<timestamp>_<pid>/` before a new
canonical child starts. Archive failure leaves the source in place and only
blocks that model. Identity mismatches are never auto-archived; the explicit
`quarantine-existing --apply` command is required and moves only a fully
audited current canonical directory into `.original_scope26_quarantine/`.
Noncanonical directories, renamed runs, smoke output, Batch32 output, and
results without a valid receipt are not silently accepted.

Each model runs in its own subprocess. A trainable current-scope child is
started only after an exact same-machine, same-Git, same-precision, same-source
`PASS` preflight is found. Missing or mismatched preflight records
`PREFLIGHT_MISSING_OR_MISMATCH`, does not start that model, and continues to
the remaining entries. A non-zero child exit reads the per-model log tail and
records the real exception class/message, traceback tail, CUDA OOM,
NaN/Inf ContractError, collision, preflight, and identity classifications in
the failure artifact.

`PreflightPlan` is static and reports the 24/2 denominator without touching
GPU. `Preflight` is the explicit execution command and runs 24 independent
preflight workers; `Persistence` and `MovingAverage` never need GPU preflight.
`SegRNN` and `MSGNet` are excluded from both.

Every successful run has an execution receipt bound to scope, entry/model/run,
root, Git/source revision, manifest/run-map/freeze, source closure, model
config, precision, protocol, training profile, dataset, graph, checkpoint and
metrics hashes, command, timestamps, and exit code. `inspect_run()` also checks
the effective/artifact/prediction run and root bindings and fails closed on an
old or incomplete receipt; it never fills missing fields.

The lock payload includes scope, host, PID, process start time, Git,
manifest/freeze hashes, and creation time. `lock-status` distinguishes
`ACTIVE`, `STALE`, `UNKNOWN_REMOTE`, and `MALFORMED`; only
`clear-stale-lock` can remove a confirmed stale lock, while normal exit
releases its own lock in `finally`.

The readiness states are distinct: manifest/run-map/freeze errors are
`BLOCKED_GLOBAL_IDENTITY`; existing canonical/orphan identity conflicts are
`BLOCKED_EXISTING_ARTIFACTS`; missing-only results are
`NOT_READY_MISSING_RESULTS`; failed attempts are `COMPLETED_WITH_FAILURES`;
and a complete denominator is `COMPLETED_READY_26_OF_26`.
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

`Transformer_node_shared_d512_fp32_bs4_seed2026_retry2` is declared in the
active manifest's `historical_noncanonical_results` allowlist with policy
`READ_ONLY_NOT_ADOPTED_DO_NOT_BLOCK_CANONICAL_RERUN`. It remains inventory
orphan/history evidence, is outside the 26 denominator, creates no adoption
receipt, and cannot block `RUN_MISSING` for the exact canonical
`Transformer_node_shared_d512_bs4_seed2026`. The retry2 directory, the older
AMP failure directory, and all other historical evidence remain read-only and
are never moved, renamed, deleted, or adopted by a normal run.

The old Original 27/28/29 launchers, including
`UNIFORM_BATCH4_RUN_ORIGINAL_27_SKIP_MSGNET_WINDOWS.ps1`, are superseded and
are not called by this entry point.
