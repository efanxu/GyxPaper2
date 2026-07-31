# E5 loss identity audit

- Formal definition: `custom_models/src/st_mgprompt/losses.py::masked_score_aligned_hybrid_loss`
- Source SHA256: `f90b99df342de7b1b1ee6ea39bbf30cc84eceb11663b38dfc7070d3905afc32b`
- Read-only adapter: benchmark `(B,N,H)` is transposed to A8 `(B,H,N)`.
- Formula: `0.5 * masked MAE + 0.5 * sqrt(masked MSE + 1e-6)` per valid `(B,N)` row; valid rows are then averaged.
- All-masked behavior: `None`, exactly as A8.
- Space: normalized `Patv_raw`; no inverse transform and no physical clipping in training loss.
- Dependencies: prediction, target, mask, frozen `eps=1e-6` only.
- AMP: canonical A8 implementation is called directly from the source file without importing `st_mgprompt.model`.
