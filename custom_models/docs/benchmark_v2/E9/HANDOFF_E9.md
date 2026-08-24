# HANDOFF E9

## Current state

E9-A/E9-B design, portability, six frozen transfer variants, original26 controls, batch4 protocol, readiness, aggregation, and report framework remain unchanged. The current recommended formal execution platform is now Windows PowerShell with an NVIDIA GPU; Linux scripts remain compatibility entry points.

Portability remains `PORTABLE_WITH_TRAIN_STATE`. Full MS-MG-DWU transfer is allowed, with no portable subset. The original26 controls remain read-only `masked_mse`, batch4, seed2026 artifacts.

Current pre-training readiness remains:

- `PORTABILITY_AUDIT_READY=PASS`
- `ORIGINAL_MASKED_MSE_REFERENCE_READY=6/6`
- `MSMG_DWU_TRANSFER_READY=0/6`
- `LOSS_ONLY_PAIRING_READY=0/6`
- `CORE_E9_READY=6/12`
- `FULL_MS_MG_DWU_TRANSFER_READY=true`
- `PORTABLE_SUBSET_TRANSFER_READY=false`

## Current recommended resume sequence

On the Windows GPU computer:

1. Pull `origin/main`.
2. Run `E9_PRECHECK_WINDOWS.ps1`.
3. Run `E9_PREFLIGHT_ALL_6_WINDOWS.ps1` and require 6/6 PASS.
4. Run `E9_RUN_ALL_6_WINDOWS.ps1`, or its autoshutdown wrapper.
5. Inspect `E9_TRANSFER_READINESS.json`, completed/failed CSVs, and the overall exit code.
6. With an explicit original26 workbook, require core 12/12 and pairing 6/6.
7. Run `E9_READINESS_AND_AGGREGATE_WINDOWS.ps1` with an artifact-tool runtime to create the formal Excel/Markdown report.

The Windows run-all is manifest-driven and sequentially starts one independent Python process per transfer. Re-running the same command safely skips identity-matching COMPLETED runs. Matching checkpointed interruptions resume through the existing Trainer/checkpoint path; identity conflicts and non-resumable incomplete directories fail closed. No script deletes a run directory.

See `E9_RUNBOOK.md` for parameterized foreground, background, autoshutdown, monitoring, readiness, aggregate, and safe-continuation commands. Paths containing spaces are supported through the repository's Windows native command-line quoting helper.

## Integrity constraints

Controls are not retrained. A0/A8, E5/common-loss material, smoke artifacts, and excluded models do not enter E9 training or core evidence. Do not reduce batch/shape/model settings after a preflight failure. Exact historical control/transfer initial-state equality remains `NOT_VERIFIED`; claim only same seed/source/config/protocol unless future artifacts prove otherwise.

No formal E9 training was started during the Windows orchestration implementation.
