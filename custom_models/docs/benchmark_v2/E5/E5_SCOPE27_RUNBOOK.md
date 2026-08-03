# E5 batch4 scope27 Linux Runbook

## Frozen scope

- Project root: `/root/autodl-tmp/GyxPaper2`
- Python: `/root/miniconda3/envs/env_tslib/bin/python`
- Training profile: `uniform_train_batch4_v1`
- Active E5 scope: `e5_batch4_scope27_seed2026`
- Formal output root:
  `/root/autodl-tmp/GyxPaper2/custom_models/results/benchmark_v2/common_loss_architecture_seed2026`
- Evidence denominator: trainable 24/24, evaluate-only 2/2, A8 1/1,
  total 27/27

The manifest is the sole execution-order source:
`custom_models/docs/benchmark_v2/E5/E5_SCOPE27_VARIANT_MANIFEST.json`.
The launcher contains no duplicate hard-coded model list.

## Safety behavior

- An absent exact run id is eligible to run.
- A completed run is skipped only when formal scope, batch profile, benchmark
  protocol, loss, model/config, dataset, metrics, and artifact identities match.
- An existing failed, incomplete, or identity-mismatched directory is preserved
  and blocks only that item; remaining items are still checked.
- A model failure is recorded and the launcher continues.
- A8 is validated from its existing read-only formal directory; it is never
  trained or copied.
- Final aggregation runs only after readiness is exactly 27/27 and always uses
  `--require-complete`.
- Missing, null, NaN, or infinite metrics fail closed and are never filled with
  zero.
- Smoke and historical results are not resolver candidates.

## Lock and safe restart procedure

The run-all launcher performs project/Python/input/output checks, imports the
gate, validates the active manifest, and generates the complete 26-row plan
before it creates an active lock. The lock is
`logs/benchmark_v2/e5_batch4_scope27/e5_scope27_lock.json` and records its
schema version, scope, PID, host, start time, Git commit, Python executable,
and manifest SHA256.

Inspect the lock without changing it:

```bash
cd /root/autodl-tmp/GyxPaper2 || exit 1
export PYTHON=/root/miniconda3/envs/env_tslib/bin/python
"$PYTHON" scripts/e5_scope27_lock.py inspect \
  --path logs/benchmark_v2/e5_batch4_scope27/e5_scope27_lock.json
```

An `ACTIVE` lock means the recorded PID is alive and a second run is refused.
An absent PID is reported as `STALE`; the launcher remains fail-closed until
the following explicit, single-lock cleanup is performed after inspection:

```bash
"$PYTHON" scripts/e5_scope27_lock.py clear-stale \
  --path logs/benchmark_v2/e5_batch4_scope27/e5_scope27_lock.json
```

This command removes only the specified stale lock JSON. It never removes or
rewrites a model result directory. Existing failed, incomplete, or identity
mismatched runs remain `BLOCK_EXISTING_PRESERVED`; create a new retry run-id or
quarantine directory through a separately reviewed change rather than
overwriting the original evidence.

Never delete result directories, copy checkpoints, relabel smoke output as
formal output, reuse an old scope29/A8 result, force-push Git, or run formal
training from the legacy CLI path.

## Full run with automatic shutdown

Paste this into the cloud JupyterLab Terminal:

```bash
cd /root/autodl-tmp/GyxPaper2 || exit 1
mkdir -p logs/benchmark_v2/e5_batch4_scope27
nohup bash -lc '
cd /root/autodl-tmp/GyxPaper2 || exit 1
export PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$PWD/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
set +e
bash custom_models/docs/benchmark_v2/E5/E5_RUN_ALL_27_BATCH4_LINUX_AUTOSHUTDOWN.sh
code=$?
exit "$code"
' >> logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.launcher.log 2>&1 &
launcher_pid=$!
printf "%s\n" "$launcher_pid" > logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.pid
echo "PID=$launcher_pid"
```

The automatic-shutdown wrapper waits until the run-all script has checked every
remaining item and completed final readiness/aggregate status handling. It then
saves the main exit code, calls `sync`, and invokes
`/usr/bin/shutdown -h now` whether the main run succeeded or failed.

## Monitoring

```bash
tail -f /root/autodl-tmp/GyxPaper2/logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.log
tail -f /root/autodl-tmp/GyxPaper2/logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.launcher.log
cat /root/autodl-tmp/GyxPaper2/logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.pid
cat /root/autodl-tmp/GyxPaper2/logs/benchmark_v2/e5_batch4_scope27/e5_27_batch4_seed2026.exitcode
ps -ef | grep '[r]un_benchmark.py'
ps -ef | grep '[E]5_RUN_ALL_27_BATCH4_LINUX'
watch -n 2 nvidia-smi
cat /root/autodl-tmp/GyxPaper2/logs/benchmark_v2/e5_batch4_scope27/failed_models.txt
```

Additional final state is saved in
`logs/benchmark_v2/e5_batch4_scope27/e5_scope27_final_status.env`; per-model
exit codes are saved in `model_exit_codes.tsv`; readiness and evidence hashes
are saved in `e5_scope27_readiness.json` and
`e5_scope27_evidence_manifest.json`.
