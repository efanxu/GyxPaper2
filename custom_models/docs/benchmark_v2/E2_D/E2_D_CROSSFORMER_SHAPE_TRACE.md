# E2-D Crossformer shape trace

For one flattened node item, `seq_len=144`, `pred_len=10`, `seg_len=12`,
`win_size=2`, `d_model=32`, `e_layers=2`.

| stage | shape |
|---|---|
| input | `(BN,144,16)` |
| patch/segment embedding | `(BN,16,12,32)` |
| encoder initial tokens | `(BN,16,12,32)` |
| scale block 0, no merge | `(BN,16,12,32)` |
| scale block 1, 2-to-1 merge | `(BN,16,6,32)` |
| learned decoder position tokens | `(BN,16,1,32)` |
| decoder layer 0 cross source | `(BN,16,12,32)` |
| decoder layer 1 cross source | `(BN,16,12,32)` |
| decoder layer 2 cross source | `(BN,16,6,32)` |
| each decoder layer prediction | `(BN,16,12)` |
| summed/rearranged decoder prediction | `(BN,12,16)` |
| upstream `[:, -pred_len:, :]` | `(BN,10,16)` |
| adapter-selected output | `(B,N,10)` |

`pad_in_len=144`, `pad_out_len=12`, `in_seg_num=12`, decoder segment count is
one. The adapter performs no horizon crop, shift, or reinterpretation.

Regression gradients are finite and non-zero for segment embedding; each
encoder time-attention and dimension sender/receiver/router; segment merging;
decoder self-attention/router, cross-attention, linear prediction; and learned
decoder positional embedding. Decoder tokens are learned parameters and
contain no future target or calendar data.
