# E2-C config resolution

All configurations were frozen from local source shape requirements and local
Weather scripts before ordinary smoke. For every model:

`validation_search_performed=false`, `test_result_used=false`,
`smoke_result_used_for_selection=false`, and
`oom_driven_capacity_reduction=false`.

The benchmark Protocol remains authoritative for `seq_len=144`, `pred_len=10`,
`enc_in=16`, batch sizes, optimizer, learning rate, epochs, early stopping,
seed, AMP, loss, and checkpoint selection. Weather-script batch, learning rate,
epoch, and forecast-horizon values were not copied.

## TimeMixer

Frozen model config:

```text
label_len=0, enc_in=16, dec_in=16, c_out=16
d_model=16, d_ff=32, e_layers=3, dropout=0.1
down_sampling_layers=3, down_sampling_window=2
down_sampling_method=avg
decomp_method=moving_avg, moving_avg=25
channel_independence=1, use_norm=1
embed=timeF, freq=h
```

The local Weather script supplies the same depth, width, downsampling count,
window, and method. The source requires the additional decomposition,
normalization, embedding, and channel-independence fields. No scale was removed
after local OOM.

## TSMixer

Model-specific effective config contains only fields actually read by source:

```text
e_layers=2
d_model=32
dropout=0.1
```

`task_name`, `seq_len`, `pred_len`, and `enc_in` come from common Protocol
resolution. `label_len`, `dec_in`, `c_out`, `d_ff`, `top_k`, and `factor` are
not claimed as TSMixer parameters because the local source does not read them.

## FreTS

```text
channel_independence="0"
embed_size=128
hidden_size=256
sparsity_threshold=0.01
scale=0.02
```

The last four values are source hard-codes and were not reduced. String `"0"`
is a type-semantic correction required by the literal source comparison and
activates the full channel-frequency identity. The raw-source CUDA AMP
diagnostic passed; final execution therefore uses native Trainer autocast with
no precision boundary override.

Smoke losses, real-data scores, test metrics, and GTX 1060 OOM outcomes were
not used to select any configuration.
