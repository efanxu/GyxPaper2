# E2-C model specifications

All three models use one node-shared parameter set:

`(B,144,N,16) -> (B,N,144,16) -> (B*N,144,16) -> upstream ->
(B*N,10,16) -> dynamic Patv_clean_for_input channel -> (B,N,10)`.

`cross_node_interaction=false`, `node_embedding=false`,
`node_specific_head=false`, `uses_graph=false`. Parameter count is independent
of `N`; no 134-model replication or node chunking exists.

| model | parameters | native output | final output | local exact status | Registry |
|---|---:|---|---|---|---|
| TimeMixer | 192,285 | `(BN,10,16)` | `(B,N,10)` | `FAIL_OOM`, forward | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` |
| TSMixer | 22,378 | `(BN,10,16)` | `(B,N,10)` | `PASS`, backward complete | `AVAILABLE_TRAINABLE` |
| FreTS | 4,787,594 | `(BN,10,16)` | `(B,N,10)` | `FAIL_OOM`, channel FFT forward | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` |

The selected channel is resolved from the frozen ordered feature names and
validated to occur exactly once. It is not averaged, assumed to be channel 0,
or used as the supervision target: supervision remains normalized
`Patv_raw`.

Target, mask, test target, future observed/weather/exogenous tensors, and real
calendar marks are excluded from forward. Non-zero future tensors and supplied
marks fail closed. Target and mask enter loss/evaluation only.
