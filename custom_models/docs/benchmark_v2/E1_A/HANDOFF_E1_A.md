# HANDOFF_E1_A

E1-A is complete and stopped before formal runs or E1-B. It implements Persistence, MovingAverage, and an independent node-shared GRU inside `custom_models/src/benchmark_v2`. The sole protocol remains `protocol/benchmark_protocol_v1.json`, with unchanged hash `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`.

## Exact models

Persistence repeats each turbine's last historical `Patv_clean_for_input` for 10 horizons. MovingAverage averages the last 144 historical `Patv_clean_for_input` values in formal mode and repeats the mean for 10 horizons. Both resolve the power feature index from the frozen feature list. Their adapter converts normalized input power through the train-only input scaler to physical kW, computes the statistic in kW, then converts through the train-only `Patv_raw` target scaler into normalized target space. They have zero parameters and use `NON_TRAINABLE` artifacts.

`NodeSharedGRU` is new and does not import, subclass, copy, or wrap the ST-MGPrompt GRU. It is one unidirectional `torch.nn.GRU(16,64,num_layers=1,batch_first=True,dropout=0)` plus one `Linear(64,10)`, with 16,394 shared trainable parameters. There is no graph, node embedding, node-specific head, future input, time mark, or decoder target. It uses masked MSE, Adam lr 0.001/weight decay 0, no scheduler or gradient clipping, AMP from the protocol, and validation official Score H10 checkpoint selection.

Code paths:

- `models/{persistence,moving_average,gru}.py`
- `configs/{persistence,moving_average,gru}.py`
- `adapters/statistical_baselines.py`
- `adapters/gru.py`
- `model_runtime.py`, `model_cli.py`

## Registry and CLI

The explicit registry still has 28 canonical entries: 3 available, 25 unavailable, 1 trainable, and 2 non-trainable. The E0-A `audit_status` values remain as history. Factories use lazy explicit `module:function` paths; listing the registry does not read SDWPF or import ST-MGPrompt/TSLib. STCN/STGCN remains unresolved and all E1-B+ models remain fail-closed.

The single entry is `python -m benchmark_v2.cli` or the executable thin wrapper `custom_models/src/benchmark_v2/run_benchmark.py`. Added commands are:

- `registry-show --model <id>`
- `model-smoke --model <id>`
- `full-shape-model-smoke --model <id>`
- `real-data-model-smoke --model <id>`
- `evaluate-only --model persistence|moving_average ...`
- `train --model gru ...`

Training Persistence/MovingAverage fails with the deterministic non-trainable message and starts nothing. Training DLinear or another unavailable model reports its planned stage and starts nothing.

## Verification

Final unittest suite: 30/30 PASS, retaining the 17 E0-B tests. Protocol check PASS. Six accepted ordinary/real artifact directories validate PASS.

Ordinary smoke: all three PASS at `(2,144,4,16) -> (2,4,10)`. GRU loss `5.884469032287598`, 16,394 parameters, backward/optimizer/checkpoint/strict reload all complete.

Full-shape: all three PASS at `(32,144,134,16) -> (32,134,10)` on CUDA. Persistence and MovingAverage inverse-transform parity is true. GRU AMP/backward PASS, loss `5.408382415771484`, elapsed `0.726019199937582` seconds, peak allocated GPU memory `2305106944` bytes.

Limited real SDWPF: Persistence accepted retry2 PASS `(4,134,10)`; MovingAverage PASS `(4,134,10)`; GRU PASS `(1,134,10)` after no more than two train batches, one validation batch, one independent evaluation batch, and strict reload. The first Persistence attempt failed JSON serialization on NaN metrics; retry1 exposed NaN official Score. Both are retained as FAIL. The accepted retry followed the documented minimal framework fix.

## Framework defect and protection

`benchmark_v2.metrics.official_score` previously allowed masked raw NaNs to poison the score because `NaN * False` is NaN in NumPy. A failing regression test was run first, then `np.where(mask, error, 0.0)` was applied. This is the only intentional protected benchmark-scope difference; protocol semantics did not change.

Canonical SHA256 before/after: `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. ST-MGPrompt, Canonical directory, graph metadata, P0-P5, A0-A8, TSLib, protocol, data, losses, and schemas have matching before/after hashes. No dependency or file was deleted. `custom_models/results/benchmark_v2` did not exist before and was not created.

## Formal status and runbook

Formal Persistence evaluation: NOT_RUN. Formal MovingAverage evaluation: NOT_RUN. Formal GRU 20-epoch training: NOT_RUN. No formal result exists. Windows PyCharm, PowerShell, Linux foreground/background/unconditional-shutdown commands and exact run IDs are in `E1_A_RUNBOOK.md`.

## E1-B input

Only DLinear, LightTS, TiDE, and SegRNN belong to E1-B. Preserve the explicit Registry, node-shared B/N reshape, shared data/scaler/loss/metric/checkpoint/artifact paths, formal/smoke isolation, and validation-only selection. Wrap the audited TSLib model source behind model-specific factories/adapters; do not use the TSLib trainer or modify upstream source.

For SegRNN, reproduce the recorded `tensor dimension 0 mismatch` with a focused source-level shape trace before adapting it. Check the frozen lookback 144 and horizon 10 against `seg_len`, segment counts, reshape/permutation dimensions, and any `pred_len // seg_len` zero-length path around the recorded `encoder` return and final `y.permute(...)+seq_last`. Add a failing unit test that uses the intended node-shared `(B*N,144,16)` shape, diagnose whether configuration or source assumptions cause the zero dimension, and only then implement a local adapter/config fix. Do not alter TSLib source or change the formal horizon to hide the mismatch.

Do not start E1-B automatically.
