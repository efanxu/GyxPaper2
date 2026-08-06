# A8 prerequisite and E5 Batch4 scope27 runbook

The formal E5 denominator is fixed at 24 trainable models, 2 evaluate-only
baselines, and one read-only formal Batch4 A8 reference. SegRNN and MSGNet are
excluded. The benchmark output root is
`custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026`.
The legacy Batch32 and Original scope26 roots are historical read-only evidence.

The A8 runtime reference is written only to
`custom_models/logs/uniform_bs4/audit/e5_scope27/E5_A8_BATCH4_REFERENCE.json`.
No launcher writes a reference into tracked documentation.

## Linux order

Run A8 first. `lock-status` must be `ABSENT`. If it is `STALE`, inspect it and
explicitly run `clear-stale-lock`; `ACTIVE` and `MALFORMED` fail closed.

```bash
cd /root/autodl-tmp/GyxPaper2 || exit 1
PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHONPATH="$PWD/custom_models/src"

"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py validate-contract
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py lock-status
# Only after manual confirmation of STALE, if needed:
# "$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py clear-stale-lock
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py freeze-plan
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py preflight-plan
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py dry-run
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py preflight
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py run \
  --log-root custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4/formal
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py readiness \
  --report-path custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4/a8_batch4_readiness.json \
  --reference-path custom_models/logs/uniform_bs4/audit/e5_scope27/E5_A8_BATCH4_REFERENCE.json
```

The last command must report `READY`. It publishes the runtime reference only
after the checkpoint and finite H3/H6/H10 metrics from the same canonical A8
run directory have passed validation.

Then run E5 in this exact order:

```bash
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py readiness \
  --reference-path custom_models/logs/uniform_bs4/audit/e5_scope27/E5_A8_BATCH4_REFERENCE.json
"$PYTHON" scripts/e5_batch4_scope27_gate.py validate-manifest
"$PYTHON" scripts/e5_batch4_scope27_gate.py lock-status
# Only after manual confirmation of STALE, if needed:
# "$PYTHON" scripts/e5_batch4_scope27_gate.py clear-stale-lock
"$PYTHON" scripts/e5_batch4_scope27_gate.py freeze-plan
"$PYTHON" scripts/e5_batch4_scope27_gate.py dry-run
"$PYTHON" scripts/e5_batch4_scope27_gate.py preflight \
  --preflight-root custom_models/logs/uniform_bs4/audit/e5_scope27/preflight
"$PYTHON" scripts/e5_batch4_scope27_gate.py run \
  --input-path dataset/sdwpf_model_input_base.parquet \
  --target-path dataset/sdwpf_eval_target.parquet \
  --preflight-root custom_models/logs/uniform_bs4/audit/e5_scope27/preflight
"$PYTHON" scripts/e5_batch4_scope27_gate.py readiness \
  --output-root custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026
"$PYTHON" scripts/e5_batch4_scope27_gate.py aggregate \
  --output-root custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026 \
  --require-complete
```

E5 aggregate consumes the same static
`E5_SCOPE27_VARIANT_MANIFEST.json` used by validation, safe-resume planning,
and readiness. All 26 benchmark run IDs end in `_bs4_seed2026`; A8 appears once
as `STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference`. Aggregate is blocked
unless readiness is `READY` with 27/27 entries.

Safe resume actions are `RUN_MISSING`, `SKIP_COMPLETED`, `ARCHIVE_AND_RUN`,
`BLOCK_EXPLICIT_CONFIG_CONFLICT`, and `BLOCK_ACTIVE_PROCESS`. Incomplete runs
are archived without deleting historical attempts. Explicit configuration
conflicts and active workers are never archived. A8 is never trained, archived,
or copied into the E5 output root.

## Convenience launchers

Ordinary launchers never shut down the machine:

```bash
bash custom_models/docs/benchmark_v2/E5/E5_A8_BATCH4_LINUX.sh
bash custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX.sh
```

Shutdown occurs only when the user explicitly invokes one of these wrappers.
Each wrapper runs the ordinary launcher, records its real exit code, syncs, and
shuts down whether the run succeeded or failed:

```bash
bash custom_models/docs/benchmark_v2/E5/E5_A8_BATCH4_LINUX_AUTOSHUTDOWN.sh
bash custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX_AUTOSHUTDOWN.sh
```

## Windows PowerShell order

```powershell
$Launcher = '.\custom_models\docs\benchmark_v2\E5\E5_A8_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1'
& $Launcher -Action StaticAudit
# Only after manual confirmation that the reported lock is STALE:
# & $Launcher -Action ClearStaleLock
& $Launcher -Action Preflight
& $Launcher -Action Run
& $Launcher -Action Readiness
```

The Windows launcher uses only `READY`/`NOT_READY`, and its run action performs
post-run acceptance before returning success.
