# A8 prerequisite and E5 Batch4 scope27 runbook

The formal E5 denominator is fixed at 24 trainable models, 2 evaluate-only
baselines, and one read-only formal Batch4 A8 reference. SegRNN and MSGNet are
excluded. The benchmark output root is
`custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026`.
The legacy Batch32 and Original scope26 roots are historical read-only evidence.

All 24 E5 training runs and both evaluate-only baselines use FP32. The E5
overlay disables AMP uniformly so model comparisons cannot be interrupted or
biased by model-specific float16 overflow. This does not change the precision
policy of Original or other experiment profiles. The independent A8 reference
keeps its own frozen precision identity.

PatchTST uses encoder-layer activation checkpointing only under the E5 FP32
overlay. This keeps the exact Batch4/model/optimizer contract while reducing
saved activation memory; there is no automatic batch reduction or gradient
accumulation fallback.

Every E5 preflight PASS includes a full-shape FP32 forward, common-loss
backward, and Adam optimizer update. Outputs, loss, gradients, parameters, and
optimizer state must remain finite before a formal worker can start.

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
"$PYTHON" -m pip install -r custom_models/docs/benchmark_v2/E5/requirements.txt

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
$ProjectRoot = 'D:\PaperProject\GyxPaper2'
Set-Location -LiteralPath $ProjectRoot
$PythonExecutable = 'D:\Apps\Miniconda3\envs\env_tslib\python.exe'
& $PythonExecutable -m pip install -r custom_models\docs\benchmark_v2\E5\requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

$A8Launcher = '.\custom_models\docs\benchmark_v2\E5\E5_A8_BATCH4_WINDOWS_FORMAL_COMMANDS.ps1'
$E5Launcher = '.\custom_models\docs\benchmark_v2\E5\E5_SCOPE27_WINDOWS_FORMAL_COMMANDS.ps1'

# A8: StaticAudit returns 4 before the first completed A8 run; that means
# NOT_READY and is expected. Any other nonzero code is a hard failure.
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $A8Launcher `
  -Action StaticAudit -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -notin @(0, 4)) { throw "A8 StaticAudit failed: $LASTEXITCODE" }
# Only after manual confirmation that the reported lock is STALE:
# & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $A8Launcher -Action ClearStaleLock -PythonExecutable $PythonExecutable
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $A8Launcher `
  -Action Preflight -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "A8 Preflight failed: $LASTEXITCODE" }
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $A8Launcher `
  -Action Run -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "A8 Run failed: $LASTEXITCODE" }
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $A8Launcher `
  -Action Readiness -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "A8 Readiness failed: $LASTEXITCODE" }

# E5: the launcher independently rechecks the A8 reference before every GPU
# or result-producing action. All 24 trainable models run FP32 with AMP off.
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $E5Launcher `
  -Action StaticAudit -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "E5 StaticAudit failed: $LASTEXITCODE" }
# Only after manual confirmation that the reported lock is STALE:
# & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $E5Launcher -Action ClearStaleLock -PythonExecutable $PythonExecutable
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $E5Launcher `
  -Action Preflight -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "E5 Preflight failed: $LASTEXITCODE" }
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $E5Launcher `
  -Action Run -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "E5 Run failed: $LASTEXITCODE" }
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $E5Launcher `
  -Action Readiness -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "E5 Readiness failed: $LASTEXITCODE" }
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $E5Launcher `
  -Action Aggregate -PythonExecutable $PythonExecutable
if ($LASTEXITCODE -ne 0) { throw "E5 Aggregate failed: $LASTEXITCODE" }
```

Use `-InputPath` and `-TargetPath` on both `Run` commands only when the parquet
files are not in the default `dataset` directory. Each Windows launcher uses a
child `powershell.exe` process so `$LASTEXITCODE` is reliable. A8 and E5 use
only `READY`/`NOT_READY`; both run actions perform post-run acceptance before
returning success. The E5 preflight requires 24/24 full-shape FP32 forward,
backward, finite-gradient, and optimizer-step checks before training starts.
