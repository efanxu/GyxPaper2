# E0-A Benchmark Protocol Gap Audit

## Target protocol (audit baseline)

`SDWPF`; target `Patv_raw`; input power `Patv_clean_for_input`; mask `valid_target_mask`; lookback 144; max prediction 10; H3/H6/H10; strict chronological 0.8/0.1/0.1; strides 6/3/1; batches 32/4/4; 20 epochs; patience 6; min_delta 0.01; seed 2026; AMP enabled; clip [0,1500] kW; checkpoint validation H10 official Score, lower is better; input `(B,T,N,C)`, output `(B,N,10)`.

## Confirmed current sources

- Formal ST-MGPrompt source: `custom_models/src/st_mgprompt/config.py:48-98` and `experiment_protocol.py:144-226`. The formal canonical override is protocol-aligned, including batch/early-stopping fields; raw defaults in `config.py:70-78` are not sufficient as a benchmark contract.
- Data: `data.py:95-121,124-153,156-194,269-410` loads separate input/target tables, checks alignment, fits scalers on train only, constructs split-contained windows and applies 6/3/1 strides. It explicitly records that the target mask is not model input (`:360-367`).
- Loss/metrics: `losses.py:310-348` selects the two ST-MGPrompt losses; `metrics.py:47-84,87-141,144-163` applies the mask and defines Score aliases. `Score` is the official kW→MW aligned score and `score_lower_is_better=True` (`metrics.py:112-140`).
- Evaluation: `evaluate.py:210-226,246-258` applies mask, inverse transform and clip before metrics; `evaluate.py:388-403` computes H3/H6/H10 prefix metrics. Checkpoint selection is `val_official_score_h10` min in the formal config.
- TSLib: `run.py:18-39,87-97`, `exp_long_term_forecasting.py:37-39,42-74,76-166,168-268`, and `data_loader.py:1-17,51-79` show the generic protocol and its gaps.

## Gap table

| item | required | current TSLib path | current formal ST-MGPrompt path | severity |
|---|---|---|---|---|
| dataset | SDWPF | generic ETT/custom/HuggingFace datasets; no SDWPF adapter | SDWPF parquet pair | HIGH |
| target/input | Patv_raw / Patv_clean_for_input | generic `args.target`, dataframe columns | explicit fields in config.py:59-63 | HIGH |
| mask | valid_target_mask in loss and metrics | absent | present in data/metrics/loss path | HIGH |
| features | fixed official 16 and order | `args.enc_in`, generic dataframe order | DEFAULT_16_FEATURES config.py:9-26 | HIGH |
| split/window | strict 0.8/0.1/0.1, lookback 144, contained windows | dataset-specific borders; no unified max horizon | explicit in data.py and formal protocol | HIGH |
| stride/batch | 6/3/1 and 32/4/4 | generic loader has no required stride contract; batch only train CLI | formal override has required values | HIGH |
| shape | `(B,T,N,C)` → `(B,N,10)` | `(B,T,C)` → `(B,10,C)`; 14 synthetic forwards succeed only in generic shape | model internal `[B,H,N]` converted by evaluator | HIGH |
| loss | mask-aware unified loss | MSELoss, no mask | masked_score_aligned_hybrid or method-full loss | HIGH |
| Score | official lower-is-better H10 | no official Score implementation | metrics.py:47-84 | HIGH |
| checkpoint | validation H10 Score min | validation MSE; test loss observed each epoch | formal config.py:96-98 | HIGH |
| clip | uniform [0,1500] | absent | evaluate.py:214-216 and config.py:92-95 | HIGH |
| artifacts | resolved/effective/protocol/model/checkpoints/log/metrics/metadata/status | generic checkpoint.pth, npy and text outputs | formal roots contain richer artifacts | HIGH |

## Shape and adapter findings

The TSLib synthetic helper confirms 14/18 models return `[2,10,16]` for input `[2,144,16]`; this is `[B,H,C]`, not evidence of `[B,N,H]`. `SegRNN`, `MICN`, `MultiPatchFormer`, and `TimeMixer` fail the same generic smoke due model-specific segment/decoder/scale configuration. The required adapter must decide whether each turbine is flattened into batch, treated as channels, or modeled with an explicit node dimension; this is a protocol decision for E0-B, not an automatic transpose.

## Mask, leakage, selection and artifact risks

TSLib’s `exp_long_term_forecasting.py:78-79,152-155` reads test data and reports test loss during every epoch. Even though the shown EarlyStopping call uses validation loss, test exposure is a tuning risk and cannot be accepted as a fair benchmark path. Generic `data_loader.py` also imports optional remote dataset tooling (`datasets`, `huggingface_hub`) and is unrelated to the local aligned SDWPF input/target pair. No TSLib path applies the required clip or writes H3/H6/H10 Score artifacts.

`custom_models/results/st_mgprompt` contains smoke-named runs under the formal results tree. The E0-A inventory marks them `NOT_REUSABLE`; no files were copied or promoted.

## E0-B must unify

1. One SDWPF provider and feature order.
2. One split/window/stride and batch contract.
3. One shape adapter interface with explicit node semantics.
4. One mask-aware loss/Score/MAE/RMSE/R2 implementation.
5. One validation-H10 checkpoint policy.
6. One clip policy and artifact schema.
7. Separate registry/runner/output roots from `st_mgprompt`, P0-P5, A0-A8, Canonical, `run_precision_ablation.py` and `run_ablation.py`.
