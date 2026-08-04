# E5 Batch4 scope27 runbook

The active E5 denominator is 24 trainable models, 2 evaluate-only baselines,
and one read-only formal Batch4 A8 reference: 27 evidence entries. The active
output root is:

`custom_models/results/benchmark_v2_uniform_bs4/common_loss_architecture_seed2026`

The legacy root
`custom_models/results/benchmark_v2/common_loss_architecture_seed2026` is
`LEGACY_OR_HISTORICAL_READ_ONLY_NOT_CURRENT_BATCH4_OUTPUT`. It is never read as
current evidence and is never copied, moved, or adopted.

The current manifest, run map, readiness policy, gate, and launchers are bound
by `E5_ACTIVE_SCOPE.json`. The source identity is per-model: excluded SegRNN
and MSGNet are not in the active closure or denominator. Batch32 artifacts,
old scope29 manifests, old A8 artifacts, and Transformer retry2 are historical
only.

## Required order

```bash
cd /root/autodl-tmp/GyxPaper2 || exit 1
export PYTHON=/root/miniconda3/envs/env_tslib/bin/python
export PYTHONPATH="$PWD/custom_models/src"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py validate-contract
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py lock-status
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py freeze-plan
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py preflight-plan
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py dry-run
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py preflight
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py run \
  --log-root custom_models/logs/uniform_bs4/audit/st_mgprompt_a8_batch4/formal
"$PYTHON" scripts/st_mgprompt_a8_batch4_gate.py readiness

# E5 is not allowed to start until the independent A8 readiness is READY.
"$PYTHON" scripts/e5_batch4_scope27_gate.py lock-status
"$PYTHON" scripts/e5_batch4_scope27_gate.py validate-manifest
"$PYTHON" scripts/e5_batch4_scope27_gate.py freeze
"$PYTHON" scripts/e5_batch4_scope27_gate.py preflight-plan
"$PYTHON" scripts/e5_batch4_scope27_gate.py dry-run
"$PYTHON" scripts/e5_batch4_scope27_gate.py preflight \
  --preflight-root custom_models/logs/uniform_bs4/audit/e5_scope27/preflight \
  --source-revision "$(git rev-parse HEAD)"
"$PYTHON" scripts/e5_batch4_scope27_gate.py run \
  --input-path dataset/sdwpf_model_input_base.parquet \
  --target-path dataset/sdwpf_eval_target.parquet \
  --preflight-root custom_models/logs/uniform_bs4/audit/e5_scope27/preflight \
  --source-revision "$(git rev-parse HEAD)"
"$PYTHON" scripts/e5_batch4_scope27_gate.py readiness
"$PYTHON" scripts/e5_batch4_scope27_gate.py aggregate --require-complete
```

The independent A8 preflight must pass before its formal run. E5 `preflight`
must then report exact 24/24 PASS. Formal execution never creates a
preflight implicitly. `run` uses only exact canonical run IDs and emits one
execution receipt per successful train/evaluate-only result. Its failure
artifacts retain the real per-model log tail and classify OOM, nonfinite,
identity, preflight, collision, archive, and lock failures.

`dry-run` actions are `SKIP_COMPLETED_IDENTITY_MATCH`, `RUN_MISSING`,
`ARCHIVE_INCOMPLETE_THEN_RUN`, `BLOCK_EXISTING_IDENTITY_MISMATCH`, and
`BLOCK_A8_BATCH4_REFERENCE`. Identity-mismatched or renamed directories are
never auto-archived. Use `quarantine-existing --model-id ... --apply` only
after an explicit per-model review. Failed archive moves use external intent
receipts, so a failed move leaves the canonical source unchanged and can be
retried safely. Symlinks, path traversal, active workers, and target
collisions are rejected.

The A8 prerequisite is trained independently once, then consumed by E5
strictly read-only. Its reference is
`STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference`, sourced from
`custom_models/results/st_mgprompt_uniform_bs4/component_ablation_a8_bs4_seed2026/STMGPrompt_ComponentAblation/`.
Batch4 config, protocol, nonempty checkpoint, finite H3/H6/H10 metrics,
JSON/CSV equality, and the three copied/retrained flags are checked. Missing
or invalid A8 evidence yields `BLOCKED_A8_BATCH4_PREREQUISITE`; no Batch32
fallback is allowed.

Readiness must be `COMPLETED_READY_27_OF_27` before aggregation. Otherwise no
CSV, XLSX, Markdown, or final audit aggregate is produced. Lock states are
`ABSENT`, `ACTIVE`, `STALE`, `UNKNOWN_REMOTE`, and `MALFORMED`; only an
explicit `clear-stale-lock` for confirmed `STALE` is permitted. The
autoshutdown wrapper is optional and must never be used as a substitute for
the preflight/readiness gates.
