# E2-B model specifications

All models use one node-shared parameter set:

`(B,144,N,16) -> (B*N,144,16) -> upstream -> (B*N,10,16) -> frozen power channel -> (B,N,10)`.

`cross_node_interaction=false`, `node_embedding=false`, `node_specific_head=false`, and parameter counts do not depend on `N`. TimesNet is grouped by node during forward because upstream FFT averages across its batch; the same parameters are still shared across all nodes.

| model | parameters | ordinary | exact local full-shape | limited real | Registry |
|---|---:|---|---|---|---|
| TimesNet | 1,196,810 | PASS | PASS, 5,414,120,960 B peak | PASS | `AVAILABLE_TRAINABLE` |
| MICN | 95,354 | PASS | PASS, 1,956,011,520 B peak | PASS | `AVAILABLE_TRAINABLE` |
| WPMixer | 2,667,788 | PASS | `FAIL_OOM`, 12,459,016,192 B peak allocated record | PASS | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` |
| MultiPatchFormer | 2,344,900 | PASS | `FAIL_OOM`, 12,657,478,144 B peak allocated record | PASS | `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED` |

The final Registry has 28 entries: 7 locally full-shape trainable, 6 trainable with hardware preflight, 2 non-trainable available, and 13 true blocked/unavailable. Formal-runnable trainable total is 13.

All forward paths exclude target, mask, future observations, future weather, future calendar marks, and test targets. MICN alone requires a mark placeholder; it is all zero and length 154.
