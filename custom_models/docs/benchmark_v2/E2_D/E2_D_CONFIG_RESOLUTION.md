# E2-D config resolution

All values were frozen before the first ordinary smoke. For all three models:
`validation_search_performed=false`, `test_result_used=false`,
`smoke_result_used_for_selection=false`, and
`oom_driven_capacity_reduction=false`.

Protocol supplies `seq_len=144`, `pred_len=10`, `enc_in=16`, `c_out=16`,
batch sizes, Adam `0.001`, zero weight decay, no scheduler, seed 2026, AMP,
20 epochs, patience 6, min delta 0.01, and normalized-`Patv_raw` masked MSE.

## Crossformer

```text
d_model=32, d_ff=32, e_layers=2, factor=3
n_heads=8, dropout=0.1
seg_len=12, win_size=2 (source hard-codes)
```

The local Weather script supplies the width, depth, and factor. Local `run.py`
supplies the head count and dropout defaults. Script-only `d_layers`, `top_k`,
and `label_len` are excluded because the forecast source does not read them.

## MSGNet

```text
label_len=48, c_out=16
d_model=512, d_ff=2048, e_layers=2, n_heads=8, top_k=5
dropout=0.1, embed=timeF, freq=h
conv_channel=32, skip_channel=32, gcn_depth=2
propalpha=0.3, node_dim=10, individual=false
```

No local MSGNet task script exists. These are local `run.py` defaults plus
fields that the source actually requires. The resolver explicitly checks
`d_model >= c_out` and FFT-bin sufficiency. No width, layer, top-k, graph
depth, or branch was reduced after local OOM.

## TimeFilter

```text
c_out=16
d_model=512, d_ff=2048, e_layers=2, n_heads=8, dropout=0.1
patch_len=16, alpha=0.1, top_p=0.5, pos=1
```

No local TimeFilter task script exists. Values come from local `run.py` and
source defaults. The resolver requires exact `144 % 16 == 0` patch geometry
and `d_model % n_heads == 0`. No token, graph, gate, or layer was removed
after local OOM.
