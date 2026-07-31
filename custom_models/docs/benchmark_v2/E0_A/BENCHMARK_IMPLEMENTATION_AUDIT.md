# E0-A Benchmark Implementation Audit

## Executive Summary

This is an audit-only snapshot. The 28 required canonical rows are present: **18 IMPLEMENTED_PROTOCOL_RISK, 1 PARTIAL_IMPLEMENTATION, 8 MISSING, 1 NAME_CONFLICT**. No new model, registry, adapter, artifact schema, training run, dependency installation, or deletion was performed.

The 18 TSLib source models are importable and can be constructed through the generic factory, but all are protocol-risk because the path is generic `(B,T,C)`, uses generic data/target semantics, `nn.MSELoss`, no `valid_target_mask`, no official Score, and writes generic `./results`. Persistence and MovingAverage are absent. GRU is only an internal node-shared block inside ST-MGPrompt and has no standalone benchmark entry. All seven graph slots are absent locally; `STCN_STGCN` is a naming conflict caused by missing source evidence.

## Repository and environment

- Project root: `D:\PaperProject\GyxPaper2`.
- Git: `git rev-parse --show-toplevel`, branch, HEAD, and status all failed with `fatal: not a git repository`; no `.git` directory was found under the project root. This is recorded as an environment fact, not treated as a clean worktree.
- Interpreter: `D:\Apps\Miniconda3\envs\env_tslib\python.exe`; Python 3.11.15; platform `Windows-10-10.0.26200-SP0`; torch 2.7.1+cu128; CUDA 12.8; CUDA available `True`; GPU `NVIDIA GeForce GTX 1060`; numpy 2.1.2; pandas 2.3.3; scikit-learn 1.7.2.
- `rg.exe` was attempted for file enumeration but returned Windows Access Denied; UTF-8 PowerShell/Python enumeration was used instead.

## 28-model implementation matrix summary

| canonical_name | status | primary implementation | stage |
|---|---|---|---|
| Persistence | MISSING | UNKNOWN | E1-A |
| MovingAverage | MISSING | UNKNOWN | E1-A |
| GRU | PARTIAL_IMPLEMENTATION | custom_models/src/st_mgprompt/model.py | E1-A |
| DLinear | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/DLinear.py | E1-B |
| LightTS | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/LightTS.py | E1-B |
| TiDE | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/TiDE.py | E1-B |
| SegRNN | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/SegRNN.py | E1-B |
| Transformer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/Transformer.py | E2-A |
| PatchTST | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/PatchTST.py | E2-A |
| iTransformer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/iTransformer.py | E2-A |
| TimeXer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/TimeXer.py | E2-A |
| TimesNet | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/TimesNet.py | E2-B |
| MICN | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/MICN.py | E2-B |
| WPMixer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/WPMixer.py | E2-B |
| MultiPatchFormer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/MultiPatchFormer.py | E2-B |
| TimeMixer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/TimeMixer.py | E2-C |
| TSMixer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/TSMixer.py | E2-C |
| FreTS | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/FreTS.py | E2-C |
| Crossformer | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/Crossformer.py | E2-D |
| MSGNet | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/MSGNet.py | E2-D |
| TimeFilter | IMPLEMENTED_PROTOCOL_RISK | Time-Series-Library/models/TimeFilter.py | E2-D |
| GCN | MISSING | UNKNOWN | E3-B |
| STCN_STGCN | NAME_CONFLICT | UNKNOWN | E3-A |
| DCRNN | MISSING | UNKNOWN | E3-B |
| Graph WaveNet | MISSING | UNKNOWN | E3-C |
| MTGNN | MISSING | UNKNOWN | E3-C |
| AGCRN | MISSING | UNKNOWN | E3-C |
| STID | MISSING | UNKNOWN | E3-C |

Evidence for the matrix is in `benchmark_implementation_matrix.csv`; each row lists source/config paths and a `CONFIRMED`/`INFERRED` explanation in `notes`.

## Real call chains

### Legacy TSLib path

`Time-Series-Library/run.py:15-24,180-245` parses `args.model` and task. `exp/exp_basic.py:15-23,25-46` scans every `models/*.py` into a `LazyModelDict`; `exp/exp_long_term_forecasting.py:22-27` constructs `module.Model(args)`. `data_provider/data_factory.py` selects a generic dataset; `data_provider/data_loader.py:1-17,51-79` imports generic/HuggingFace datasets and fits a generic scaler. The train/validation/test path is `exp_long_term_forecasting.py:76-166`, with `nn.MSELoss` at `:37-39`; test loss is evaluated during every training epoch at `:78-79,152-155`. Test prediction and artifacts use `:168-268`, including `./test_results`, `./results`, `result_long_term_forecast.txt`, and `.npy` files. There is no official SDWPF Score, valid-target-mask path, physical clip, or H10 checkpoint rule.

### ST-MGPrompt protected path

`custom_models/src/st_mgprompt/run_st_mgprompt.py` is the CLI. `registry.py:30-196` resolves STMGPrompt names to model classes. `data.py:95-102,124-153,269-342` loads separate model-input/eval-target tables, fits train-only scalers, builds split-contained windows and loaders. `train.py` calls the model and loss; `evaluate.py:194-269,376-493` applies inverse transform, mask, clip, official Score and writes metrics artifacts. `formal_runner.py:103-145,309-429` manages P/A families and writes smoke/formal roots. This path is not a source for an external benchmark registry.

### Canonical/P0/A0 protection

Canonical is `D:\PaperProject\GyxPaper2\custom_models\results\st_mgprompt_canonical\full_fixed_dual_keep_msmgdwu_seed2026` with checkpoint SHA256 `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. `experiment_protocol.py:15-22,165-234,332-381` defines the canonical ID/root and reference resolution. P0 and A0 `reference.json` files point to this directory, are `trainable:false`, and carry the same checkpoint/config/metrics hashes. `run_st_mgprompt.py:489` and `train.py:441,589` use `load_state_dict(..., strict=True)`. No benchmark code was found coupled to P/A; the risk is namespace/result-directory collision if E0-B reuses these roots.

## Duplicate implementations, names, and old results

No duplicate source implementation was confirmed for the 18 requested TSLib models. TSLib contains many additional orphan models not in the requested 28; they are listed in `benchmark_registry_inventory.json` and must not silently enter the benchmark. The only local old result with a complete formal identity is the protected Canonical artifact. Smoke and legacy ST-MGPrompt runs exist under `custom_models/results/st_mgprompt`, which is a pollution risk because the path is under `results`, not a distinct `results_smoke`; all such runs are marked non-reusable in `old_result_audit.json`. No local SDWPF external-benchmark result was found.

## Key risks

1. TSLib auto-registration by filename is not a canonical benchmark registry.
2. The generic path uses `(B,T,C)` and `[B,H,C]`, while the target protocol requires `(B,T,N,C)` and `(B,N,10)`.
3. TSLib has no `Patv_raw`/`Patv_clean_for_input`/`valid_target_mask` contract and no official Score.
4. Default ST-MGPrompt config fields differ from the formal canonical overrides (`config.py:70-78` defaults include eval batch 64, patience 10 and min_delta 0; `experiment_protocol.py:181-196` overrides to 32/4/4, patience 6 and 0.01). E0-B must consume the formal resolved config, not raw defaults.
5. TSLib evaluates test loss during training and saves generic outputs; this is incompatible with the required validation-H10-only selection and artifact schema.
6. Smoke artifacts are present in formal-looking ST-MGPrompt result roots; benchmark_v2 must use an independent root.

## E0-B and E1-E3 recommendation

E0-B should create a new inventory-backed registry outside `st_mgprompt.registry` and TSLib, with one adapter contract, explicit shape conversion, dataset provider, mask-aware loss/metrics, validation H10 Score selection, and independent `custom_models/results/benchmark_v2` plus E0-A smoke roots. First migrate the TSLib source metadata without modifying upstream files; then implement E1-A baselines, E1-B lightweight adapters, E2 groups, and E3 graph identity/protocol. Keep the 28-model list frozen before test values are inspected.

## Static checks and unresolved questions

`static_audit` parsed 106 relevant Python files with 0 syntax errors. Placeholder/TODO hits are reported in `audit_import_results.json`; most are framework abstract methods or explicit `NotImplementedError`, not silently treated as model implementations. Import results are in `audit_import_results.json`; synthetic smoke results are in `audit_smoke_results.json`. Unresolved: graph model sources/precise papers are absent, SDWPF adapters do not exist for TSLib, and the Git metadata/worktree baseline cannot be recovered from this directory.
