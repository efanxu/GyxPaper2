# Uniform Batch=4 machine-bound Runbook

## Status semantics

The GTX 1060 result is local evidence, not a global prohibition:

```text
LOCAL_DEVELOPMENT_MACHINE_STATUS = BLOCKED_MACHINE_OOM
LOCAL_DEVELOPMENT_MACHINE_ID = GTX1060_WINDOWS
TARGET_MACHINE_STATUS = PENDING_TARGET_MACHINE_PREFLIGHT
GLOBAL_ENGINEERING_STATUS = READY_FOR_TARGET_MACHINE_VALIDATION
```

A target changes to `READY_FOR_FORMAL_RUN` only for each suite whose exact,
current-machine artifacts all pass. OOM changes only that target to
`BLOCKED_ON_THIS_MACHINE_OOM`; another machine may still run its own preflight.
A non-OOM failure changes only that target to
`BLOCKED_ON_THIS_MACHINE_NON_OOM`.

The historical SegRNN and MSGNet GTX 1060 `FAIL_OOM` artifacts remain
read-only. They are never imported as a target result and never overwritten.

## Frozen identity and machine binding

Every benchmark preflight identity and ST-MGPrompt gate artifact records
`machine_id`, hostname, OS, Python, PyTorch, CUDA, GPU name, GPU UUID when
available, total GPU memory, and driver version when available. It also records
the frozen protocol/profile/loss/model/config/source/graph/node/shape/AMP/seed
identity.

Only a PASS produced by the current machine with a byte-for-byte matching
identity is accepted. Original-loss and E5 common-loss have distinct identities
and distinct summaries. Missing or mismatched rows are printed before a formal
script exits with a nonzero code.

Deployment archives use the same explicit scope, model, batch, loss, precision,
and dataset-path metadata as repository checkouts. If repository metadata
is present and the tree is dirty, every changed path is recorded.

## Preflight order

Run PRECHECK, then the preflight needed by the suite that machine will execute.
All inventory preflights run 26 isolated child processes sequentially and
continue after failures. Persistence and MovingAverage receive separate
evaluate-only prechecks. Full and A8 each use their exact formal structure and
formal loss.

Windows:

```powershell
Set-Location -LiteralPath 'D:\PaperProject\GyxPaper2'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_PRECHECK_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_PREFLIGHT_ORIGINAL_ALL26_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_PREFLIGHT_E5_ALL26_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_PREFLIGHT_FULL_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_PREFLIGHT_A8_WINDOWS.ps1'
```

Linux:

```bash
cd /path/to/GyxPaper2
export PYTHON=/absolute/path/to/env_tslib/bin/python
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_PRECHECK_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_PREFLIGHT_ORIGINAL_ALL26_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_PREFLIGHT_E5_ALL26_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_PREFLIGHT_FULL_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_PREFLIGHT_A8_LINUX.sh
```

Each machine writes
`UNIFORM_BATCH4_MACHINE_PREFLIGHT_<machine_id>.json` and `.md`. Preflight logs,
exit codes, full tracebacks, Git/dirty details, protocol/profile/dataset records,
and artifact paths are retained.

## Formal suite commands

Formal scripts verify their own current-machine summary before launching any
formal child. Formal suites run one child process at a time and stop at the
first failure while retaining the failure directory, log, and exit code.

Windows:

```powershell
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_RUN_ORIGINAL_28_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_RUN_FULL_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_RUN_A8_WINDOWS.ps1'
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_RUN_E5_CORE_28_WINDOWS.ps1'
```

Linux:

```bash
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_ORIGINAL_28_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_FULL_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_A8_LINUX.sh
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_RUN_E5_CORE_28_LINUX.sh
```

Linux auto-shutdown variants are available for Original 28, Full, A8, and E5
core. They always capture the suite code, run `sync`, call
`/usr/bin/shutdown -h now`, and then return the captured code.

## E5 core and finalize

`E5_CORE` is the 26 common-loss trainable models plus the two deterministic
evaluate-only entries. A matching E5 preflight releases it even if Original 28
is not yet frozen and even if the new batch4 A8 is not yet complete.

`E5_FINALIZE` is separate and fail-closed. It requires both the completed frozen
Original 28 and a valid newly completed batch4 A8 before creating the A8
reference, evaluating 29/29 readiness, or running require-complete aggregation.

```powershell
& '.\custom_models\docs\benchmark_v2\UNIFORM_BATCH4_PROTOCOL\UNIFORM_BATCH4_FINALIZE_E5_WINDOWS.ps1'
```

```bash
bash custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL/UNIFORM_BATCH4_FINALIZE_E5_LINUX.sh
```

## Two-machine parallel execution

Machine A may run Original 28 while Machine B runs E5 core. The manifests must
match on commit (or the explicit archive commit), benchmark and Batch4 protocol
records, profile record, model/config/source listing records, graph/node identity,
dataset records, run-id map, AMP, shape, and seed. The tasks use the frozen,
disjoint output roots already recorded in `UNIFORM_BATCH4_RUN_ID_MAP.json`;
they must never share a run-id directory.

Use the gate's `compare-manifests` command before importing artifacts between
machines. Any source/config/profile/data/graph/run-map mismatch rejects the
import.
