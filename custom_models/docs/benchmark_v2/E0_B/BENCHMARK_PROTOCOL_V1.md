# benchmark_v2 Protocol v1

The sole machine-readable source is `custom_models/src/benchmark_v2/protocol/benchmark_protocol_v1.json`. This document summarizes that JSON; it is not an independent default source.

- Dataset: SDWPF, 10-minute interval, 134 nodes.
- Input: `(B,T,N,C)`, `T=144`, `N=134`, `C=16`; target and mask: `(B,N,H)`, `H=10`.
- Target: `Patv_raw` in kW. Input power feature: `Patv_clean_for_input`. Mask: `valid_target_mask`.
- Feature order: `Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, T2m, Sp, RelH, Wspd_w, Wdir_w, Tp, Patv_clean_for_input`.
- Splits: strict chronological `[0.8,0.1,0.1]`; contained windows `[t-144,t) -> [t,t+10)`; strides `6/3/1`.
- Batches: `32/4/4`; 20 epochs; patience 6; min delta 0.01; seed 2026; AMP enabled.
- Evaluation: inverse target transform, mask, clip `[0,1500]` kW, then H3/H6/H10 `Score/MAE/RMSE/R2`.
- Checkpoint: validation official Score H10, lower is better.
- Normalization: standard z-score, fit train-only; default loss space is normalized target space.
- Future observed and future calendar covariates are disabled.

Protocol hash: `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`.

