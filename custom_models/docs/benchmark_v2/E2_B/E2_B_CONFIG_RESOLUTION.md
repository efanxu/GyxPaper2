# E2-B config resolution

No validation search, test use, smoke-loss selection, or OOM-driven capacity reduction was performed:

`validation_search_performed=false`, `test_result_used=false`, `oom_driven_capacity_reduction=false`.

Common frozen fields remain `seq_len=144`, `pred_len=10`, `enc_in=16`, Adam `lr=0.001`, weight decay 0, no scheduler, masked MSE in normalized `Patv_raw` space, AMP, seed 2026, 20 epochs, patience 6, min delta 0.01.

| model | upstream/local evidence | effective engineering config | required changes |
|---|---|---|---|
| TimesNet | `run.py`: top_k 5, kernels 6; local official Weather long-forecast: `d_model=32,d_ff=32,e_layers=2,top_k=5` | `c_out=16,label_len=48,d_model=32,d_ff=32,e_layers=2,top_k=5,num_kernels=6,dropout=.1,embed=timeF,freq=h` | FFT only FP32 for CUDA length 154; node-grouped calls prevent upstream batch-mean FFT from coupling turbines. No capacity change. |
| MICN | official Weather long-forecast width 32; source constructor kernels `[12,16]` | `label_len=144,c_out=16,d_model=32,n_heads=8,d_layers=1,dropout=.1,conv=[12,16],decomp=[13,17],isometric=[13,10]` | decoder/mark length 154 is shape-required; future 10 values and all 154 marks are zero. |
| WPMixer | source defaults `db2,level=1,tfactor=5,dfactor=5,stride=8`; official patch 16 and Weather first-horizon width 256 | `c_out=16,d_model=256,dropout=.1,patch_len=16,db2,level=1,tfactor=5,dfactor=5,stride=8` | custom DWT/IDWT fixed FP32 while learned mixers use Trainer AMP; shape/capacity unchanged. |
| MultiPatchFormer | official Weather first-horizon `d_model=256,d_ff=512,e_layers=1,n_heads=8`; source patch lengths `[8,16,24,32]` | same capacity; strides `[8,8,7,7]`, paddings `[0,8,0,8]`, counts `[18,18,18,18]` | shape-required branch padding/stride compatibility and exact horizon-10 semi-autoregressive input widths. No crop/branch removal. |

All four emit 16 channels because their source normalization/trend/channel structure does not provide a clean native one-channel forecast. The Adapter resolves the frozen feature order and uniquely selects `Patv_clean_for_input` to supervise normalized `Patv_raw`; index 15 is verified dynamically, not silently assumed.
