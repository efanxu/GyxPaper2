# Git import audit

Generated: 2026-07-31T15:49:07.4882539+08:00

## Summary

- Files scanned: 7933
- Total size: 10563345657 bytes (9.84 GiB)
- Planned for commit: 6653 files, 27808042 bytes
- Excluded but retained locally: 1280 files, 10535537615 bytes
- >10 MiB: 189; >50 MiB: 80; >=100 MiB: 22
- Reparse points skipped: 0

## Policy

- Source, scripts, configuration, documentation, tests, schemas, location metadata, and small summaries are eligible for normal Git.
- Dataset bodies, checkpoints, arrays, logs, caches, bytecode, environments, installers, duplicate archives, and large regenerable results are excluded and remain on disk.
- No file was approved for Git LFS; no canonical checkpoint was selected.
- No local data, checkpoint, result, or history file was deleted.

## Top-level inventory

| Top-level path | Files | GiB |
|---|---:|---:|
| custom_models | 7467 | 9.41 |
| dataset | 18 | 0.42 |
| HANDOFF_FIXED_DUAL_REFACTOR.md | 1 | 0 |
| PYTHON_INTERPRETER_PATH_AUDIT.md | 1 | 0 |
| PYTHON_INTERPRETER_PATH_FIX_REPORT.md | 1 | 0 |
| Time-Series-Library | 407 | 0 |
| tsl.zip | 1 | 0 |
| scripts | 18 | 0 |
| render_word_qa.ps1 | 1 | 0 |
| RUNBOOK_FIXED_DUAL_REFACTOR.md | 1 | 0 |
| GIT_COLLABORATION.md | 1 | 0 |
| .vscode | 1 | 0 |
| build_revision.py | 1 | 0 |
| .gitignore | 1 | 0 |
| .idea | 9 | 0 |
| DELETE_MANIFEST_refactor_bytecode.json | 1 | 0 |
| DELETE_STATUS_refactor.json | 1 | 0 |
| Custom Model 协议.md | 1 | 0 |
| DELETE_MANIFEST_refactor.json | 1 | 0 |

## Large files

All files above 10 MiB are hashed in `git_import_inventory.csv`; excluded files are listed with SHA256 in `git_import_excluded_paths.txt`.

| Size MiB | Relative path | Classification | Plan | Reason |
|---:|---|---|---|---|
| 161.9 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/msgnet_e5_retry2/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 161.9 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/msgnet_e5_retry2/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 161.9 | `custom_models/results_smoke/benchmark_v2/e2_d/real_data/msgnet_cpu_default_threads_interrupted/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 161.9 | `custom_models/results_smoke/benchmark_v2/e2_d/real_data/msgnet_cpu_default_threads_interrupted/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 161.9 | `custom_models/results_smoke/benchmark_v2/e2_d/real_data/msgnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 161.9 | `custom_models/results_smoke/benchmark_v2/e2_d/real_data/msgnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 159.1 | `dataset/sdwpf_10min.parquet` | other | No | raw or processed dataset body; retained locally |
| 140.5 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 140.5 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 140.5 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 140.5 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 140.5 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 140.5 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 128.1 | `dataset/sdwpf_raw_aligned.parquet` | other | No | raw or processed dataset body; retained locally |
| 121.9 | `dataset/sdwpf_model_input_base.parquet` | other | No | raw or processed dataset body; retained locally |
| 112.5 | `custom_models/results/st_mgprompt_volatility_group_analysis/volatility_group_analysis_seed2026.zip` | archive/installer | No | duplicate archive or installer; source/config uploaded instead |
| 107.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 107.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 107.6 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 107.6 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 107.6 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 107.6 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 83.1 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 83.1 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 83.1 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 83.1 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 83.1 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 83.1 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 74.5 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real_retry1/real_data/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 74.5 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real_retry1/real_data/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 74.5 | `custom_models/results_smoke/benchmark_v2/e2_d/real_data/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 74.5 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 74.5 | `custom_models/results_smoke/benchmark_v2/e2_d/real_data/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 74.5 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 73.1 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 73.1 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 73.1 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 73.1 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 73.1 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 73.1 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/msgnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/msgnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/msgnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/msgnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2/e2_d/ordinary/msgnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/msgnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2/e2_d/ordinary/msgnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 60.6 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/msgnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2/e2_a/real_data/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2/e2_a/transformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 59.9 | `custom_models/results_smoke/benchmark_v2/e2_a/transformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e2_c/ordinary/frets_fp32_boundary_attempt/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e2_c/real_data/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e2_c/ordinary/frets_fp32_boundary_attempt/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e2_c/real_data/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e2_c/real_data/frets_fp32_boundary_attempt/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 54.8 | `custom_models/results_smoke/benchmark_v2/e2_c/real_data/frets_fp32_boundary_attempt/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real_retry2/real_data/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real_retry2/real_data/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e3_c/real_data/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e3_c/ordinary/mtgnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e3_c/real_data/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e3_c/ordinary/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 53.8 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/mtgnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 48.7 | `custom_models/results/st_mgprompt_volatility_group_analysis/volatility_group_analysis_seed2026/analysis_arrays.npz` | array-artifact | No | large or regenerable local artifact; retained locally |
| 48.5 | `custom_models/results/st_mgprompt/full_msmg_dwu_v4_2/STMGPrompt_Full_MSMGDWU/predictions.npz` | array-artifact | No | large or regenerable local artifact; retained locally |
| 47.9 | `custom_models/results/st_mgprompt/full_fairfull_v4_2/STMGPrompt_FairFull/predictions.npz` | array-artifact | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2/e2_a/timexer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 42.4 | `custom_models/results_smoke/benchmark_v2/e2_a/timexer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2/e2_d/ordinary/timefilter/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 37.9 | `custom_models/results_smoke/benchmark_v2/e2_d/ordinary/timefilter/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2/e2_a/patchtst/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 34.2 | `custom_models/results_smoke/benchmark_v2/e2_a/patchtst/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 30.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 30.6 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 30.6 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 30.6 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 30.6 | `custom_models/results_smoke/benchmark_v2/e2_b/real_data/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 30.6 | `custom_models/results_smoke/benchmark_v2/e2_b/real_data/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 26.7 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/multipatchformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 26.7 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/multipatchformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 26.7 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/multipatchformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 26.7 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/multipatchformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 26.7 | `custom_models/results_smoke/benchmark_v2/e2_b/real_data/multipatchformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 26.7 | `custom_models/results_smoke/benchmark_v2/e2_b/real_data/multipatchformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 25.3 | `custom_models/results/st_mgprompt_volatility_group_analysis/volatility_group_analysis_seed2026/logs/a0_test_stage.npz` | logs | No | large or regenerable local artifact; retained locally |
| 25.1 | `custom_models/results/st_mgprompt_volatility_group_analysis/volatility_group_analysis_seed2026/logs/a1_test_stage.npz` | logs | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2/e2_a/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2/e2_a/itransformer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 24.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/itransformer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 21.5 | `dataset/sdwpf_eval_target.parquet` | other | No | raw or processed dataset body; retained locally |
| 20.7 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/tide/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 20.7 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/tide/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 20.7 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/tide/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 20.7 | `custom_models/results_smoke/benchmark_v2/e1_b/real_data/tide/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 20.7 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/tide/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 20.7 | `custom_models/results_smoke/benchmark_v2/e1_b/real_data/tide/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2/e2_c/ordinary/frets/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.3 | `custom_models/results_smoke/benchmark_v2/e2_c/ordinary/frets/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/segrnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/segrnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/segrnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/segrnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/segrnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/segrnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2/e1_b/real_data/segrnn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 18.2 | `custom_models/results_smoke/benchmark_v2/e1_b/real_data/segrnn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results/st_mgprompt_method_full_ablation/method_full_msmgdwu_seed2026_p0_p5/P5/STMGPrompt_Full_MSMGDWU_DiffusionHistory/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results/st_mgprompt_method_full_ablation/method_full_msmgdwu_seed2026_p0_p5/P5/STMGPrompt_Full_MSMGDWU_DiffusionHistory/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/timesnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real/real_data/timesnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results_smoke/benchmark_v2/e2_b/real_data/timesnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/timesnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/timesnet/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results_smoke/benchmark_v2/e2_b/real_data/timesnet/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results/st_mgprompt_precision/precision_ablation_fair_main_seed2026_all_p0_p5/P5/STMGPrompt_FairFull_DiffusionHistory/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results/st_mgprompt_precision/precision_ablation_fair_main_seed2026_all_p0_p5/P5/STMGPrompt_FairFull_DiffusionHistory/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results/st_mgprompt_precision/precision_ablation_fixed_dual_seed2026/P5/STMGPrompt_ComponentAblation/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 14.4 | `custom_models/results/st_mgprompt_precision/precision_ablation_fixed_dual_seed2026/P5/STMGPrompt_ComponentAblation/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real_retry1/real_data/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/real_retry1/real_data/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/ordinary/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/ordinary/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e3_b/real_data/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e3_b/real_data/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/ordinary/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e3_b/ordinary/stgcn/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e3_b/ordinary/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 13.4 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/real_data/stgcn/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 11.7 | `custom_models/results/st_mgprompt_volatility_group_analysis/volatility_group_analysis_seed2026/logs/train_threshold_stage.npz` | logs | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/original/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2_uniform_bs4/e5/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2/e2_b/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2/e2_b/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2/e5_common_loss/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2/e2_b/passing_v2/wpmixer/best_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |
| 10.2 | `custom_models/results_smoke/benchmark_v2/e2_b/passing_v2/wpmixer/last_checkpoint.pt` | checkpoint | No | large or regenerable local artifact; retained locally |

## Sensitive-information review

- No filename or concrete secret-assignment pattern was detected by the non-secret scan.
