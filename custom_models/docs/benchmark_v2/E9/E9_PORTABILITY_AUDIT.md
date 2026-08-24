# E9-A MS-MG-DWU Portability Audit

## Conclusion

`PORTABLE_WITH_TRAIN_STATE`

`FULL_MS_MG_DWU_TRANSFER_READY=true` and `PORTABLE_SUBSET_TRANSFER_READY=false`.

The exact implementation consumes prediction, target, and mask. benchmark_v2 supplies the frozen node axis, one-based horizon identity, normalized target-space identity, and protocol identity through `LossInputBundle`. The only additional dependency is loss-owned online state:

- `ema_granularity_loss`
- `initial_granularity_loss`
- `initial_granularity_fitted`
- `ema_node_loss`
- `node_weight`

These values are updated only in training mode. Validation and test reuse frozen values. They are fitted from each model's training predictions and targets, so their numeric values are intentionally not shared across models; the fitting algorithm and frozen profile are identical.

## Dependency closure

Required: prediction, target, valid-target mask, node/site axis, horizon prefixes H3/H6/H10, train-only target normalization identity, training-only model-output-dependent loss state, and the public batch/protocol identity.

Not required: Fine representation, Coarse representation, VADSP/granularity representation state, Macro Prompt, ST Prompt, Cross gate, graph embedding, hidden state, attention, model auxiliary tensor, or an ST-MGPrompt model instance.

The term “granularity” in this loss refers to public horizon prefixes, not Fine/Coarse model representations. The adapter only transposes benchmark `(B,N,H)` tensors to the original loss implementation's `(B,H,N)` order.

## Leakage and lifecycle

The loss never fits from validation/test, never reads test targets for weights, and does not update state in `eval()` mode. Checkpoints save and restore the loss state alongside model state. An all-masked batch returns `None` and skips the optimizer step.

Machine-readable details are in `E9_PORTABILITY_AUDIT.json`, `E9_LOSS_INPUT_CONTRACT.json`, and `E9_LOSS_STATE_MANIFEST.json`.
