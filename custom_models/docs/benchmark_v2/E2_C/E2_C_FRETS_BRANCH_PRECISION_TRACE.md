# E2-C FreTS branch and precision trace

Machine-readable evidence:
`custom_models/results_smoke/benchmark_v2/e2_c/diagnostics/frets_branch_amp_diagnostics.json`.

| supplied value | Python type | channel learner | channel gradients | output |
|---|---|---|---|---|
| `0` | int | skipped | `r1/i1/rb1/ib1 = null` | `(1,10,16)` |
| `1` | int | skipped | null | `(1,10,16)` |
| `"0"` | str | executed | all non-zero | `(1,10,16)` |
| `"1"` | str | skipped | null | `(1,10,16)` |

Temporal `r2/i2/rb2/ib2` gradients were non-zero in all four cases. Final
frozen value is string `"0"`; both channel and temporal learners are active.
Token embedding and FC gradients are also finite and non-zero.

Raw-source CUDA AMP with string `"0"` completed forward and backward at the
isolated diagnostic shape, returning FP16. Therefore the final wrapper does
not override autocast: Trainer AMP is enabled and PyTorch uses its native
FFT/complex autocast behavior.

A preventive full-forward FP32 implementation attempt had already produced
PASS ordinary/real artifacts and an OOM full-shape artifact. Those artifacts
are retained under `frets_fp32_boundary_attempt` but are explicitly superseded
and not used for final selection or routing. Final native-AMP runs were
repeated: ordinary and real-data PASS; exact full-shape failed only by CUDA OOM
at `FreTS.py:65` channel `rfft`. There was no dtype, complex, NaN, or shape
failure.
