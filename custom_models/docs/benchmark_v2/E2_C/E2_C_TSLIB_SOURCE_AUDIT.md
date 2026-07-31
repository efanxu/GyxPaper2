# E2-C TSLib source audit

E2-C only loads the local checkout at `D:\PaperProject\GyxPaper2\Time-Series-Library`.
The explicit allowlist is exactly:

`dlinear, lightts, tide, segrnn, transformer, patchtst, itransformer, timexer, timesnet, micn, wpmixer, multipatchformer, timemixer, tsmixer, frets`.

There is no directory scan, arbitrary-path import, network download, dummy fallback,
TSLib Trainer, or TSLib dataset use. TSLib source files were not modified.
License: `Time-Series-Library/LICENSE`, MIT, SHA256
`8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2`.

## TimeMixer

- Absolute source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\TimeMixer.py`
- SHA256: `bbf378cab03d16d3e7ae907f7b3cefbac89ae1504e036270578d5d496ffbd133`
- Constructor: `Model(configs)`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`.
- Direct local imports: `layers.Autoformer_EncDec.series_decomp`
  (`48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e`),
  `layers.Embed.DataEmbedding_wo_pos`
  (`17e7c3577324c41a0da427a199c955b782fde905aabb1f7cbc3c4e15ebd4ae35`),
  and `layers.StandardNorm.Normalize`
  (`3f690dc5fc0e395d4f1ffe6396717548d18bd04f4518258308bbe5f392703f5f`).
- Required forecast fields: `task_name,seq_len,label_len,pred_len,enc_in,c_out,
  d_model,d_ff,e_layers,dropout,down_sampling_layers,down_sampling_window,
  down_sampling_method,decomp_method,moving_avg,channel_independence,use_norm,
  embed,freq`.
- Actual inputs: historical `x_enc`; optional `x_mark_enc`. E2-C passes marks as
  `None`. Decoder values, decoder marks, mask, target, and future covariates are
  not read by the forecast path.
- Native shape: `(BN,144,16) -> four scales -> (BN,10,16)`.
- Normalization: one `Normalize` module per scale uses history-only statistics;
  scale 0 de-normalizes the summed forecast.
- Interaction: no batch aggregation. With `channel_independence=1`, the source
  internally flattens the 16 historical features into its batch dimension. These
  are features inside one already-isolated turbine, never the 134 turbines.
- Multiscale identity: fixed average-pooling pyramid, moving-average
  trend/season decomposition, bottom-up season mixing, top-down trend mixing,
  and sum of four predictors.
- FFT/complex operations: only the unused `dft_decomp` alternative has FFT.
  Frozen `decomp_method=moving_avg` performs no FFT.

## TSMixer

- Absolute source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\TSMixer.py`
- SHA256: `a82942ddc22cba59f4161f3ea480eb0e474eba6f0ab2d13f2bd949f100b2ce87`
- Constructor: `Model(configs)`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`.
- Local dependencies: none beyond PyTorch.
- Fields actually read: `task_name,e_layers,pred_len,seq_len,d_model,enc_in,dropout`.
- Actual input: historical `x_enc` only. Marks, decoder, mask, target, and future
  tensors are ignored by source and rejected by the Adapter when supplied.
- Native path: `(BN,144,16) -> two ResBlocks -> Linear(144,10) ->
  (BN,10,16)`.
- Each ResBlock preserves temporal MLP, channel MLP, and two residual additions.
- The local source contains no normalization and E2-C adds none.
- There is cross-variable interaction only through each turbine's 16-feature
  channel MLP. There is no batch aggregation or cross-turbine interaction.
- FFT/complex operations: none.

## FreTS

- Absolute source: `D:\PaperProject\GyxPaper2\Time-Series-Library\models\FreTS.py`
- SHA256: `2b6c9e0cd3d4f42bc74736343147812db94f029f1a8b13b098b9841705de5c18`
- Constructor: `Model(configs)`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)`.
- Local dependencies: none; imports PyTorch and NumPy (NumPy is unused).
- Fields read: `task_name,pred_len,seq_len,enc_in,channel_independence`.
- Hard-coded identity: `embed_size=128`, `hidden_size=256`,
  `sparsity_threshold=0.01`, `scale=0.02`.
- Native shape: token embedding `(BN,16,144,128)`; optional channel rFFT over
  the 16 historical variables; temporal rFFT over 144 steps; FC input width
  `144*128=18432`; output `(BN,10,16)`.
- Complex parameter groups: channel `r1/i1/rb1/ib1`; temporal
  `r2/i2/rb2/ib2`; soft shrinkage is applied before `view_as_complex`.
- The source compares `channel_independence == '0'`. Integer `0`, integer `1`,
  and string `"1"` skip the channel learner; only string `"0"` activates it.
  E2-C therefore freezes string `"0"` explicitly.
- Raw-source CUDA AMP forward/backward passed in isolation, so native Trainer
  autocast is retained. No FP32 override is applied in the final wrapper.
- No normalization and no batch aggregation exist. Channel FFT mixes only the
  16 variables inside each turbine.

## Script and repository identity

- `run.py`: `db6dfa215520ba193fe7095380f9005949a958efa9526975410cb5d97afe3fc4`
- `README.md`: `2d03e4917e41917105ed647c5ea7537e1394f88b0af190e05715457b55dc5be6`
- Weather TimeMixer script:
  `24638b0358519d62024bc7d4e42ad76e0cb71f4b85b74482d32bd15d3816f8d2`
- Weather TSMixer script:
  `0dfcc7956d0c2ef32590623b7e4f3aa0f7b120e34be74ccb1a44f15a626baa55`

None of the three imports VADSP, an ST-MGPrompt graph, Macro Prompt, Reverse
Cross, Cross Fusion, ST Prompt, or MS-MG-DWU.
