# E1-A model specifications

Protocol source: `custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json`. Protocol record: `<removed-content-record>`.

## Common contract

All three models receive only the 144-step, 16-feature historical input. They do not receive `Patv_raw`, `valid_target_mask`, future observed/calendar covariates, decoder targets, a graph, node embeddings, or node-specific parameters. `NodeSharedAdapter` performs `(B,T,N,16) -> (B*N,T,16)` and restores `(B*N,10) -> (B,N,10)`. The output semantic is future `Patv_raw` in normalized target space.

## Persistence

Definition: for each node, repeat the last visible historical `Patv_clean_for_input` value for all 10 horizons. The power feature index is resolved from the frozen `ordered_input_features`, with uniqueness, feature-count, and feature-order-record checks.

Scaler path: normalized input power -> inverse input feature scaler -> physical kW -> select final historical value -> target scaler transform -> normalized `Patv_raw`. Input and target scaler statistics are never assumed equal. A non-finite final value raises `ContractError`; the model neither fills zero nor searches backward.

Parameters: 0 total, 0 trainable. Optimizer/scheduler/epochs/early stopping: null. Registry status: `AVAILABLE_NON_TRAINABLE`.

Code: `models/persistence.py`, `configs/persistence.py`, `adapters/statistical_baselines.py`.

## MovingAverage

Definition: for each node, average the trailing 144 physical `Patv_clean_for_input` values and repeat the result for 10 horizons. Formal `ma_window=144` is fail-closed. Smoke may explicitly override the value without changing the protocol.

Scaler path: normalized input power -> inverse input feature scaler -> physical kW -> trailing mean -> target scaler transform -> normalized `Patv_raw`. `T < ma_window`, an empty window, or any NaN/Inf in the averaging window raises `ContractError`; no truncation occurs.

Parameters: 0 total, 0 trainable. Optimizer/scheduler/epochs/early stopping: null. Registry status: `AVAILABLE_NON_TRAINABLE`.

Code: `models/moving_average.py`, `configs/moving_average.py`, `adapters/statistical_baselines.py`.

## GRU

`NodeSharedGRU` is independent of ST-MGPrompt and TSLib. It contains exactly one `torch.nn.GRU` and one `torch.nn.Linear`:

- input dimension 16; hidden dimension 64; one layer;
- unidirectional; dropout 0; `batch_first=true`;
- final hidden state readout;
- `Linear(64,10)` direct multi-horizon head;
- one shared parameter set for every node and no cross-node interaction.

Parameters: 16,394 total and trainable, independent of node count. Initialization uses PyTorch defaults after `benchmark_v2.seeds` sets seed 2026.

Training: masked MSE in normalized `Patv_raw` target space; Adam, learning rate 0.001, weight decay 0, no scheduler, no gradient clipping. AMP follows the frozen protocol. Checkpoint selection is validation official Score H10, lower is better, patience 6, min delta 0.01. Test/evaluation is separate and starts only after strict best-checkpoint reload.

Registry status: `AVAILABLE_TRAINABLE`.

Code: `models/gru.py`, `configs/gru.py`, `adapters/gru.py`.
