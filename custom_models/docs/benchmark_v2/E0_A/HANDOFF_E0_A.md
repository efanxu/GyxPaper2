# HANDOFF_E0_A

## Scope and protection

This handoff completes audit-only E0-A for `D:\PaperProject\GyxPaper2`. It read `HANDOFF(8).md` and `PLAN.md`, audited `custom_models/src/st_mgprompt`, `custom_models/results`, `custom_models/results_smoke`, `dataset`, and the local `Time-Series-Library` source/config/runner/layer/data paths. No model core, formal registry, runner, data protocol, Canonical checkpoint, P0-P5, A0-A8, dependency set, or formal result directory was modified. No training or tuning started; no files were deleted.

Git metadata is absent, so Git HEAD/branch/status are `null` rather than invented. Canonical is `custom_models/results/st_mgprompt_canonical/full_fixed_dual_keep_msmgdwu_seed2026` and its best checkpoint remains SHA256 `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. P0 and A0 `reference.json` point to this artifact with `trainable:false`; `run_st_mgprompt.py:489` and `train.py:441,589` use strict checkpoint loading.

## Findings

- Model matrix: {'MISSING': 8, 'PARTIAL_IMPLEMENTATION': 1, 'IMPLEMENTED_PROTOCOL_RISK': 18, 'NAME_CONFLICT': 1}.
- TSLib provides 18 importable upstream models through filename auto-scan/LazyModelDict, not a unified SDWPF benchmark. All 18 are `IMPLEMENTED_PROTOCOL_RISK`.
- Persistence and MovingAverage are `MISSING`.
- GRU is `PARTIAL_IMPLEMENTATION`: only an internal ST-MGPrompt GRU block exists; no standalone factory/runner/data/loss/metrics path.
- GCN, DCRNN, Graph WaveNet, MTGNN, AGCRN and STID are `MISSING`.
- STCN/STGCN is `NAME_CONFLICT` with Situation C: no source evidence for either identity; preserve the label and resolve in E3-A.
- Main protocol gap: TSLib generic `(B,T,C)`/`[B,H,C]`, MSE, generic target/data loader, no `valid_target_mask`, no official Score/clip/H10 checkpoint contract. Its test loss is observed every training epoch.
- Smoke pollution exists under `custom_models/results/st_mgprompt` for smoke-named runs; those are historical reference only, not benchmark artifacts. No local external SDWPF benchmark run was found.

## Deliverables

See all files in this directory. The required files are `BENCHMARK_IMPLEMENTATION_AUDIT.md`, `BENCHMARK_PROTOCOL_GAP_AUDIT.md`, `STCN_STGCN_NAMING_AUDIT.md`, `benchmark_implementation_matrix.csv`, `benchmark_dependency_report.md`, `benchmark_registry_inventory.json`, `E0_A_AUDIT_MANIFEST.json`, and this handoff. Supporting files are `audit_commands.txt`, `audit_import_results.json`, `audit_smoke_results.json`, `old_result_audit.json`, `protected_core_hashes_before.json`, `run_e0_a_smoke.py`, and `generate_e0_a_artifacts.py`.

## Tests

The required interpreter imported 11 ST-MGPrompt modules and all 18 requested TSLib model modules. Relevant Python files were AST-parsed. Synthetic forward smoke ran one batch only: 14 models succeeded; SegRNN, MICN, MultiPatchFormer and TimeMixer failed with recorded model/config shape errors. No full dataset, full-shape SDWPF run, official training, validation-only tuning, or benchmark aggregation was run.

## E0-B exact input and order

E0-B should receive `benchmark_implementation_matrix.csv`, `benchmark_registry_inventory.json`, `BENCHMARK_PROTOCOL_GAP_AUDIT.md`, `benchmark_dependency_report.md`, the local TSLib source and its MIT `LICENSE`, plus the formal ST-MGPrompt protocol files for read-only contract comparison. Implement, in order: (1) new independent benchmark_v2 registry and canonical names; (2) adapter interface for `(B,T,N,C)` and explicit `[B,N,H]`; (3) one SDWPF provider/mask-aware loss/metrics/checkpoint policy; (4) artifact schema and isolated smoke/formal roots; (5) migrate E1-A/E1-B sources, then E2 and E3. Do not import or modify `st_mgprompt.registry`, `run_precision_ablation.py`, `run_ablation.py`, Canonical, P0-P5 or A0-A8 from the new runtime; only read/reference them.

## Traps

  Do not treat TSLib filenames as fair implementations; do not transpose `[B,H,C]` into `[B,N,H]` without deciding node semantics; do not use its MSE/test-loss path; do not call generic HuggingFace fallback; do not add graph models by name only; do not rename STCN/STGCN before evidence; do not promote smoke runs; do not select models from test metrics; do not modify or retrain Canonical/P0/A0; do not install dependencies; do not delete files.
