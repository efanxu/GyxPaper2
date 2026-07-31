# E2-D model specifications

All three use one parameter-shared model:

`(B,144,N,16) -> (B,N,144,16) -> (B*N,144,16) -> upstream
-> (B*N,10,16) -> validated Patv_clean_for_input channel -> (B,N,10)`.

The selected output channel is resolved uniquely from the frozen feature order.
Supervision remains normalized `Patv_raw`. Target, mask, test target, future
observations/weather/calendar, turbine location, graph provider, node
embedding, and node-specific heads do not enter forward.

| model | parameters | native output | final output | local exact | Registry |
|---|---:|---|---|---|---|
| Crossformer | 117,668 | `(BN,10,16)` after upstream 12-to-10 slice | `(B,N,10)` | PASS | `AVAILABLE_TRAINABLE` |
| MSGNet | 13,294,292 | `(BN,10,16)` | `(B,N,10)` | `FAIL_OOM` in forward | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` |
| TimeFilter | 4,801,994 | `(BN,10,16)` | `(B,N,10)` | `FAIL_OOM` in forward | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` |

Parameter count does not depend on `N`. MSGNet uses per-item calls because its
FFT top-k statistic otherwise aggregates the batch. TimeFilter and MSGNet
internal graphs describe 16 variables or 144 variable-time tokens,
respectively; neither is an E3 134-turbine graph.
