# E2-B MICN shape trace

The E0-A generic configuration was reproduced unchanged with:

`x_enc=(2,144,16)`, `x_mark_enc=(2,144,4)`, `x_dec=(2,58,16)`, `x_mark_dec=(2,58,4)`, `label_len=48`, `pred_len=10`.

Traceback:

`MICN.py:210 forward -> MICN.py:167 forecast -> Embed.py:124 DataEmbedding.forward`

Error:

`RuntimeError: The size of tensor a (154) must match the size of tensor b (58) at non-singleton dimension 1`.

Source proof:

- `series_decomp_multi(x_enc)` returns seasonal/trend length 144.
- `MICN.py:165-166` creates ten zero future steps and concatenates `seasonal_init_enc[:, -seq_len:, :]`, so `seasonal_init_dec=(2,154,16)`.
- The generic TSLib decoder mark was `label_len+pred_len=48+10=58`.
- `DataEmbedding.forward` adds value/position tensors of length 154 to temporal embedding length 58, producing the exact error.
- `x_dec` values are not used for seasonal initialization; only its batch/channel shape is read when creating zeros. This is why arbitrary cropping to 58 would contradict the source.

Final Adapter shapes are:

`x_enc=(BN,144,16)`, `x_dec=(BN,154,16)`, `x_mark_enc=None`, `x_mark_dec=(BN,154,4)`.

The first 144 decoder steps equal observed history, the final 10 decoder steps are zero, and all decoder marks are zero. Target and mask never enter forward. `seq_len=144`, `pred_len=10`, and all TSLib files remain unchanged. The historical failure regression and fixed-path backward/reload tests pass.
