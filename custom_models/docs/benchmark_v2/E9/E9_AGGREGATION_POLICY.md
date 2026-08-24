# E9 Aggregation Policy

Formal aggregation consumes only `ORIGINAL26_CONTROL` and `E9_TRANSFER`. `OPTIONAL_STMG_REFERENCE` is related background only and is excluded from core counts and averages.

For Score, MAE, and RMSE:

`Improvement(%) = (Original Masked-MSE - MS-MG-DWU) / Original Masked-MSE × 100`

Positive values favor MS-MG-DWU. Absolute error delta is also retained. R2 uses only `R2_gain = R2_transfer - R2_control`.

Every model reports H3, H6, and H10. Summary outputs retain positive and negative results and separate engineering portability, loss-only protocol validity, optimization stability, empirical positive transfer, and cross-family coverage.

The strict aggregate requires six ready controls, six ready transfers, and six valid loss-only pairs. Missing, non-finite, incomplete, smoke, ambiguous, protocol-invalid, wrong-loss, or config-diff-invalid evidence causes a nonzero exit. Missing values are never converted to zero and the formal workbook is not created while incomplete.
