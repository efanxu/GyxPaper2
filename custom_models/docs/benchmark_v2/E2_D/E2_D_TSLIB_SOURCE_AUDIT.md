# E2-D TSLib source audit

E2-D uses only `D:\PaperProject\GyxPaper2\Time-Series-Library`. The loader is
an explicit 18-ID map; it performs no scan, arbitrary import, download, dummy
fallback, TSLib Trainer use, or TSLib dataset use. The local MIT license is
`Time-Series-Library/LICENSE`, SHA256
`8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2`.

## Crossformer

- Source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\Crossformer.py`
- SHA256: `f5892f70eb3fc320f69fa0a9b2019b68537cc5a5e5aa2ec544c15127b19e36f0`
- Constructor: `Model(configs)`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`.
- Dependencies: `Crossformer_EncDec.py`
  (`ee2cc5c8d3a44f4de67ce986e6b442488826a20fa4d8f87991941a75587936a1`),
  `SelfAttention_Family.py`
  (`6f6592aed0753e342fc04c01db70528c2eb191eeb067cd5fd4c8d348fddcb794`),
  and `Embed.py`
  (`17e7c3577324c41a0da427a199c955b782fde905aabb1f7cbc3c4e15ebd4ae35`).
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
- SHA256: `e6f8f3fbd4e0bc9d77c78e2409969c4fe4b2fc6560e7c1f5cced2a809d8e9e59`
- Constructor and forward signatures match Crossformer.
- Dependency: `layers/MSGBlock.py`, SHA256
  `01d3d446a214c33062ba098f251a1e49ed8a342dd4372e973a0a8c0dde5cd8c0`.
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
- SHA256: `6cbce92f112a9907fe6117ea23527bf1864c0ea00721a812d786e86e0a1b0724`
- Constructor and forward signatures match Crossformer.
- Dependencies: `TimeFilter_layers.py`
  (`3441713bdb54de1153afe3d5e746de4e251c07b545492180a073c4f275df6b9f`),
  `StandardNorm.py`
  (`3f690dc5fc0e395d4f1ffe6396717548d18bd04f4518258308bbe5f392703f5f`),
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

`run.py` SHA256 is
`db6dfa215520ba193fe7095380f9005949a958efa9526975410cb5d97afe3fc4`;
`README.md` SHA256 is
`2d03e4917e41917105ed647c5ea7537e1394f88b0af190e05715457b55dc5be6`.
No E2-D TSLib source or dependency file was modified.
