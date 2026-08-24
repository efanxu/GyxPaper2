# E9 Design — MS-MG-DWU Transferability

## Frozen question and scope

E9 tests whether the exact public-tensor MS-MG-DWU loss transfers across six frozen external architectures. Each pair is read-only original26 Original Masked-MSE versus one newly trained E9 MS-MG-DWU run. Controls are never retrained.

The fixed models are LightTS (lightweight), TiDE (Dense/MLP encoder-decoder), PatchTST (patch Transformer), iTransformer (cross-variable Transformer), DCRNN (recurrent diffusion graph), and MTGNN (adaptive graph learning). Persistence, GRU, DLinear, Crossformer, GraphWaveNet, AGCRN, SegRNN, and MSGNet are excluded and cannot be substituted.

E5 is cancelled. No E5/common-loss source is eligible for discovery, evidence, readiness, pairing, or aggregation.

## E9-A gate

The audited implementation is `st_mgprompt.losses.MSMGDWULoss`. Its forward boundary is `pred, target, mask`; it uses public node and horizon axes and maintains online training-error EMA state. It does not consume Fine/Coarse representations, VADSP state, prompts, cross gates, hidden states, attention, graph embeddings, or model-specific auxiliary tensors.

Classification: `PORTABLE_WITH_TRAIN_STATE`. The algorithm and profile are common to all six models, while fitted state values are per-run because they depend on each model's training predictions. State updates are guarded by `self.training` and are frozen in validation/test. Full MS-MG-DWU is allowed; no portable subset is used.

## E9-B loss-only rule

The transfer effective config is derived from the frozen control config. Allowed changes are loss identity/profile/state logging, E9 run/output identity, control reference, and E9 provenance. Architecture, hyperparameters, data, batch profile, optimizer, learning rate, epochs, early stopping, seed, precision policy, checkpoint selection, metrics, clip, graph protocol, and node order must match.

Formal output root:

`custom_models/results/benchmark_v2/msmg_dwu_transfer_seed2026/`

Each formal model runs in a separate Python process. Completed identity-matching runs are safely skipped. Existing incomplete or identity-conflicting directories are not overwritten.

## Protocol source-of-truth

The base protocol file still contains historical `train_batch_size=32`. The active formal identity is the base protocol plus `uniform_train_batch4_v1`, which resolves train/val/test to 4/4/4 and forbids accumulation or fallback. All six controls record this profile.

PatchTST and iTransformer controls have the existing model-specific FP32 precision override; the other four use the profile AMP setting. E9 inherits each control's effective precision identity rather than forcing a new override.

## Evidence and readiness

Core evidence is exactly six original26 controls plus six E9 transfers. A pair is ready only when both artifacts are complete/formal, loss identities are correct, protocol and graph identity match, H3/H6/H10 metrics exist, and the config diff is loss-only.

`aggregate --require-complete` fails closed unless `CORE_E9_READY=12/12` and `LOSS_ONLY_PAIRING_READY=6/6`. It never fills missing values with zero, substitutes smoke/old runs, skips unfavorable models, or creates the formal Excel early.
