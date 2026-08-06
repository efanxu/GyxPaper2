# E2-D TSLib source audit

E2-D uses only `D:\PaperProject\GyxPaper2\Time-Series-Library`. The loader is
an explicit 18-ID map; it performs no scan, arbitrary import, download, dummy
fallback, TSLib Trainer use, or TSLib dataset use. The local MIT license is
`Time-Series-Library/LICENSE`, content record
`<removed-content-record>`.

## Crossformer

- Source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\Crossformer.py`
- content record: `<removed-content-record>`
- Constructor: `Model(configs)`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`.
- Dependencies: `Crossformer_EncDec.py`
  (`<removed-content-record>`),
  `SelfAttention_Family.py`
  (`<removed-content-record>`),
  and `Embed.py`
  (`<removed-content-record>`).
- Read fields: `task_name,seq_len,pred_len,enc_in,d_model,n_heads,d_ff,
  e_layers,dropout,factor`. `seg_len=12` and `win_size=2` are source
  hard-codes. Forecast does not read decoder values, marks, or mask.
- Actual input is historical `x_enc` only. There is no normalization. Native
  decoder output is `(BN,12,16)` and the upstream source itself returns its
  last ten steps `(BN,10,16)`.
- Two-stage attention first attends over segments independently per variable,
  then sends/receives through routers across the 16 within-turbine variables.
  It never sees the 134 turbines as dimensions.

## MSGNet

- Source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\MSGNet.py`
- content record: `<removed-content-record>`
- Constructor and forward signatures match Crossformer.
- Dependency: `layers/MSGBlock.py`, content record
  `<removed-content-record>`.
- Read fields: `task_name,seq_len,label_len,pred_len,enc_in,c_out,d_model,
  d_ff,e_layers,n_heads,top_k,dropout,embed,freq,conv_channel,skip_channel,
  gcn_depth,propalpha,node_dim,individual`.
- Forecast reads historical `x_enc` and optional encoder marks; E2-D passes
  marks as `None`. Decoder inputs and mask are ignored.
- Per-history-channel mean/variance normalization and matching de-normalization
  are source behavior. Native output is `(BN,10,16)`.
- `FFT_for_Period` averages amplitudes across its batch before top-k selection.
  E2-D therefore calls one shared model once per flattened prediction item.
  FFT and frequency statistics use FP32; all learned graph, attention, and
  backward paths retain AMP.
- Every graph branch owns `nodevec1 (16,10)` and `nodevec2 (10,16)`, producing
  an adaptive `(16,16)` variable adjacency. It imports no turbine graph.

## TimeFilter

- Source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\TimeFilter.py`
- content record: `<removed-content-record>`
- Constructor and forward signatures match Crossformer.
- Dependencies: `TimeFilter_layers.py`
  (`<removed-content-record>`),
  `StandardNorm.py`
  (`<removed-content-record>`),
  and `Embed.py` as above.
- Read fields: `task_name,seq_len,pred_len,c_out,enc_in,d_model,d_ff,
  patch_len,alpha,top_p,pos,n_heads,e_layers,dropout`.
- Forecast reads historical `x_enc`; decoder values/marks and encoder marks are
  unused. `Normalize(..., affine=False)` uses history-only statistics and
  de-normalizes the `(BN,10,16)` native output.
- The graph has 144 variable-time patch tokens, not 134 turbines. Training mode
  uses noisy gate logits; evaluation mode is deterministic. The backbone
  returns an auxiliary MoE loss, but upstream forecast discards it and E2-D
  does not add it to frozen masked MSE.
- E2-D keeps the upstream hard top-p forward exactly and supplies a
  straight-through gate gradient in the wrapper so the gate is trainable
  without adding the auxiliary objective.

`run.py` content record is
`<removed-content-record>`;
`README.md` content record is
`<removed-content-record>`.
No E2-D TSLib source or dependency file was modified.
