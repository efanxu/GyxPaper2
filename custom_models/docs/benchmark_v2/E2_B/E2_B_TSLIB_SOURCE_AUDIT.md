# E2-B TSLib source audit

E2-B uses only the local `THUML/Time-Series-Library` checkout through the explicit lazy loader. The final allowlist is exactly:

`dlinear, lightts, tide, segrnn, transformer, patchtst, itransformer, timexer, timesnet, micn, wpmixer, multipatchformer`.

There is no directory scan, network access, TSLib Trainer, TSLib dataset, arbitrary-string import, or source modification. The license is local MIT text at `Time-Series-Library/LICENSE`, content record `<removed-content-record>`.

## TimesNet

- Source: `Time-Series-Library/models/TimesNet.py`
- content record: `<removed-content-record>`
- Imports: torch, `layers.Embed.DataEmbedding`, `layers.Conv_Blocks.Inception_Block_V1`.
- Constructor: `Model(configs)`; forecast fields are `task_name`, `seq_len`, `label_len`, `pred_len`, `enc_in`, `c_out`, `d_model`, `d_ff`, `e_layers`, `top_k`, `num_kernels`, `dropout`, `embed`, `freq`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`. Long/short forecast uses only `x_enc` and optional `x_mark_enc`; decoder values and decoder marks are ignored.
- Shape: `(BN,144,16)` is embedded, linearly extended to 154 steps, processed by two TimesBlocks, projected to 16 channels, and cropped to `(BN,10,16)`.
- Normalization: per-sample/per-input-channel mean and variance from historical `x_enc`; de-normalization uses the same history statistics. Therefore `c_out` is frozen to 16; `c_out=1` is not a clean native single-target path because source de-normalization broadcasts against 16 history channels.
- Multiscale mechanism: each TimesBlock runs rFFT over its 154-step representation derived only from historical `x_enc`, dynamically selects `top_k=5` non-DC frequencies, converts them to periods, pads each branch to a period multiple, applies six 2-D inception kernels (`1,3,5,7,9,11`), crops to 154, and softmax-fuses period branches.
- Constraints: `top_k` must not exceed the available non-DC rFFT bins; periods must be positive. Padding is internal only and final forecast remains 10.
- AMP compatibility: Windows CUDA FP16 cuFFT rejects non-power-of-two length 154. The local wrapper performs only FFT in FP32; convolution and backward remain under benchmark AMP. TSLib files are unchanged.
- Node isolation: upstream frequency selection averages its batch. The Adapter calls the same shared model once per node group `(B,144,16)`, so no turbine can affect another turbine's selected periods.

## MICN

- Source: `Time-Series-Library/models/MICN.py`
- content record: `<removed-content-record>`
- Imports: torch, `DataEmbedding`, `series_decomp`, `series_decomp_multi`.
- Constructor: `Model(configs, conv_kernel=[12,16])`; forecast reads `task_name`, `seq_len`, `pred_len`, `enc_in`, `c_out`, `d_model`, `n_heads`, `d_layers`, `dropout`, `embed`, `freq`.
- Forward: `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`. Forecast uses `x_enc`, the batch/channel shape of `x_dec`, and `x_mark_dec`; `x_mark_enc` is ignored.
- Shape: decomposition keeps `(BN,144,16)`. Trend is linearly mapped `144→10`. Seasonal decoder is historical seasonal 144 plus zero future 10, hence length 154. Output is `(BN,10,16)`.
- Multiscale mechanism: conv kernels `[12,16]`; even kernels induce odd decomposition kernels `[13,17]`; source formulas induce isometric kernels `[13,10]`. Both branches decompose, downsample, isometric-convolve, transpose-convolve, and fuse with a Conv2d across both scales.
- Constraints: decomposition kernels must be odd; input length must support the derived convolution/isometric kernels. No internal normalization exists.
- Time marks: `DataEmbedding` requires the supplied mark length to match 154. E2-B supplies all-zero `(BN,154,4)` marks and no calendar data.
- Output: frozen `c_out=16`, then dynamic `Patv_clean_for_input` channel selection. A nominal `c_out=1` is not native single-target because the 16-channel trend addition broadcasts it back to 16 channels.

## WPMixer

- Source: `Time-Series-Library/models/WPMixer.py`
- content record: `<removed-content-record>`
- Direct dependency: `layers.DWT_Decomposition.Decomposition`; that local file embeds differentiable DWT code and imports `pywt`. Installed versions: PyWavelets 1.9.0, torch 2.7.1+cu128.
- Constructor: `Model(args, tfactor=5, dfactor=5, wavelet='db2', level=1, stride=8, no_decomposition=False)`. It reads `task_name`, `seq_len`, `pred_len`, `batch_size`, `c_out`, `d_model`, `dropout`, `patch_len`, `device`, `use_amp`.
- Forward: standard five-argument signature; forecast uses historical `x_enc` only. Marks and decoder are ignored.
- Shape/geometry: db2 level-1 DWT maps input length 144 to approximation/detail lengths `[73,73]`; prediction length 10 maps to `[6,6]`. Each band uses patch length 16, stride 8, repeated-last-value end padding 8, and 9 patches. Each ResolutionBranch contains two Mixer blocks and predicts six coefficients; inverse DWT fuses both bands and crops reconstruction to 10.
- WP means wavelet-patch mixing in the actual source: fixed wavelet frequency bands plus patch/embedding mixers. It is not a replacement wavelet implementation.
- Normalization: history mean/std per channel, with matching de-normalization.
- Output: source requires `c_out=16` as its channel count and emits `(BN,10,16)`; Adapter dynamically selects the frozen power-associated channel.
- AMP compatibility: local wrapper keeps DWT/IDWT in FP32 because the custom autograd filters are FP32; both learned ResolutionBranches remain under the unified Trainer AMP. No scale or parameter was removed.

## MultiPatchFormer

- Source: `Time-Series-Library/models/MultiPatchFormer.py`
- content record: `<removed-content-record>`
- Imports: torch, math, einops 0.8.1, local `AttentionLayer` and `FullAttention`.
- Constructor: `Model(configs)`; reads `task_name`, `seq_len`, `pred_len`, `enc_in`, `e_layers`, `d_model`, `d_ff`, `n_heads`, `dropout`.
- Forward: standard five-argument signature; forecast uses only `x_enc`. Marks and decoder are ignored.
- Normalization: history mean/std per channel, with matching de-normalization.
- Four fixed branches retain patch lengths `[8,16,24,32]`. Final E2-B geometry is `(stride,padding,count)` of `(8,0,18)`, `(8,8,18)`, `(7,0,18)`, `(7,8,18)`. Per-branch embeddings are `(BN*16,18,64)` and concatenate on features to `(BN*16,18,256)`.
- Fusion: positional encoding, shared temporal attention, channel-wise attention, then eight semi-autoregressive forecast chunks. For `pred_len=10`, chunks are `1,1,1,1,1,1,1,3`.
- Compatibility: source defaults produce counts `18,18,19,20`. E2-B changes only branch-3 padding `7→0`, branch-4 stride `6→7`, branch-4 padding `6→8`, and reconstructs forecast heads 5–8 with the exact cumulative chunk widths required by horizon 10. No patch is cropped, no branch is disabled, and all four embedding weights receive gradients.
- Output: always `(BN,10,16)`; dynamic frozen power-channel selection is used.

## Shared imported sources

`layers/Embed.py` content record `<removed-content-record>`; `layers/Conv_Blocks.py` `<removed-content-record>`; `layers/Autoformer_EncDec.py` `<removed-content-record>`; `layers/DWT_Decomposition.py` `<removed-content-record>`; `layers/SelfAttention_Family.py` `<removed-content-record>`; `run.py` `<removed-content-record>`; `exp_basic.py` `<removed-content-record>`; `exp_long_term_forecasting.py` `<removed-content-record>`.

TSLib `exp_long_term_forecasting.py` and its dataset path are audited but never invoked.
