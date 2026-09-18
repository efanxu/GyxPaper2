# Uniform Batch=4 Runbook

## Current experiment set

- Original26: `benchmark_v2_batch4_scope26_seed2026`, with 24 trainable and
  2 evaluate-only entries, `default_benchmark_v1`, and `masked_mse`.
- ST-MGPrompt Full: retained under its existing formal protocol.
- ST-MGPrompt A8: retained as the independent `w/o MS-MG-DWU` ablation using
  `masked_score_aligned_hybrid`.

The common-loss architecture experiment has been cancelled and has no launcher,
gate, manifest, readiness, aggregation, or CLI entry.

## Original26

Use the maintained launcher and gate:

```powershell
& '.\custom_models\docs\benchmark_v2\BATCH4\ORIGINAL_SCOPE26_WINDOWS_FORMAL_COMMANDS.ps1' -Action StaticAudit
```

```bash
python scripts/original_batch4_scope26_gate.py static-audit
```

Original26 remains fail-closed on its explicit manifest, run map, Batch4
profile, result identity, and hardware preflight.

## ST-MGPrompt A8

Use the launchers in `custom_models/docs/benchmark_v2/A8/`. A8 artifacts stay
under `custom_models/results/st_mgprompt_uniform_bs4`; its loss and result
protocol are unchanged.
