# Handoff: E5 batch4 scope27

Local engineering is complete when the static test report passes. No formal
training, GPU preflight, git operation, or shutdown is part of local handoff.

## User actions

1. Synchronize the current project to `/root/autodl-tmp/GyxPaper2`.
2. Confirm the cloud Python exists at
   `/root/miniconda3/envs/env_tslib/bin/python`.
3. Use the single background Full + automatic-shutdown command in
   `E5_SCOPE27_RUNBOOK.md`.
4. Monitor the main log, launcher log, PID, per-model processes, GPU, and
   failure list using the commands in the Runbook.

## Active deliverables

- `E5_ACTIVE_SCOPE.json`
- `E5_SCOPE27_VARIANT_MANIFEST.json`
- `E5_SCOPE27_RUN_ID_MAP.json`
- `E5_SCOPE27_READINESS_POLICY.json`
- `E5_RUN_ALL_27_BATCH4_LINUX.sh`
- `E5_RUN_ALL_27_BATCH4_LINUX_AUTOSHUTDOWN.sh`
- `scripts/e5_batch4_scope27_gate.py`

## Runtime outputs

- `logs/benchmark_v2/e5_batch4_scope27/failed_models.txt`
- `logs/benchmark_v2/e5_batch4_scope27/model_exit_codes.tsv`
- `logs/benchmark_v2/e5_batch4_scope27/e5_scope27_readiness.json`
- `logs/benchmark_v2/e5_batch4_scope27/e5_scope27_evidence_manifest.json`
- `logs/benchmark_v2/e5_batch4_scope27/e5_scope27_final_status.env`
- formal aggregate files under
  `custom_models/results/benchmark_v2/common_loss_architecture_seed2026`,
  only when readiness is 27/27.

The legacy E5 scope29 and old uniform batch4 scope28/29 launchers are retained
as read-only/superseded evidence. Do not use them for the current run.
