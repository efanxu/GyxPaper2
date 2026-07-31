# E2-A config resolution

No validation/test search or smoke-loss tuning was performed. Common fields are frozen to `seq_len=144`, `pred_len=10`, `enc_in=16`, Adam `lr=0.001`, weight decay 0, masked MSE, AMP, seed 2026, 20 epochs, patience 6, min delta 0.01.

| model | frozen model configuration | resolution |
|---|---|---|
| Transformer | `label_len=48`, `dec_in=16`, `c_out=1`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_layers=1`, `d_ff=2048`, `factor=1`, `dropout=0.1`, `activation=gelu`, `embed=timeF`, `freq=h` | TSLib `run.py` defaults; native single-target projection |
| PatchTST | `patch_len=16`, `stride=8`, end padding 8, `patch_num=18`, `individual=false`, `decomposition=false`, `kernel_size=25`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_ff=2048`, dropout/head dropout 0.1 | exact upstream constructor defaults and formula `(144-16)/8+2=18` |
| iTransformer | `d_model=512`, `n_heads=8`, `e_layers=2`, `d_ff=2048`, `factor=1`, dropout 0.1, `activation=gelu`, `embed=timeF`, `freq=h` | TSLib defaults; time marks omitted |
| TimeXer | `features=MS`, `use_norm=true`, `patch_len=16`, `patch_num=9`, `d_model=512`, `n_heads=8`, `e_layers=2`, `d_ff=2048`, `factor=1`, dropout 0.1 | `MS` is required for last-channel endogenous power and native one-channel forecast |

All fields were frozen before smoke. No batch, node, lookback, horizon, layer, head, or hidden-width reduction occurred after OOM.
