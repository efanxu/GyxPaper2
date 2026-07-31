# E2-C TimeMixer scale trace

Let `Q=B*N` be the outer node-shared batch.

| stage | scale 0 | scale 1 | scale 2 | scale 3 |
|---|---|---|---|---|
| history | `(Q,144,16)` | `(Q,72,16)` | `(Q,36,16)` | `(Q,18,16)` |
| Normalize input/output | same | same | same | same |
| CI reshape | `(Q*16,144,1)` | `(Q*16,72,1)` | `(Q*16,36,1)` | `(Q*16,18,1)` |
| embedding | `(Q*16,144,16)` | `(Q*16,72,16)` | `(Q*16,36,16)` | `(Q*16,18,16)` |
| season/trend | two tensors of embedding shape | same | same | same |
| predictor pre-projection | `(Q*16,10,16)` | `(Q*16,10,16)` | `(Q*16,10,16)` | `(Q*16,10,16)` |
| projected/reshaped | `(Q,10,16)` | `(Q,10,16)` | `(Q,10,16)` | `(Q,10,16)` |

Season mixing runs bottom-up `144 -> 72 -> 36 -> 18`; trend mixing runs
top-down `18 -> 36 -> 72 -> 144`. Four `(Q,10,16)` forecasts are stacked and
summed, then scale-0 history statistics de-normalize the result.

Regression evidence (`test_e2_c_models`):

- all four predictors received non-zero gradients;
- every one of three PDM blocks received gradients;
- every season down-mixing and trend up-mixing layer received non-zero gradients;
- no scale was disabled or cropped;
- changing one turbine input left every other turbine output unchanged.

The internal `Q*16` dimension represents 16 features for each already-isolated
turbine. It never represents 134 turbines and does not create separate
parameter sets.
