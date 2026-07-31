# E3-B STGCN architecture trace

Frozen `Kt=3`, valid temporal convolution, and two ST-Conv blocks produce:

| stage | shape for B=1 |
|---|---|
| input | `(1,144,134,16)` |
| block 1 temporal 1 | `(1,142,134,64)` |
| block 1 Chebyshev graph bottleneck | `(1,142,134,16)` |
| block 1 temporal 2 | `(1,140,134,64)` |
| block 2 temporal 1 | `(1,138,134,64)` |
| block 2 Chebyshev graph bottleneck | `(1,138,134,16)` |
| block 2 temporal 2 | `(1,136,134,64)` |
| collapse conv | `(1,128,1,134)` |
| prediction | `(1,134,10)` |

Chebyshev support is `(3,134,134)` with `T0=I`, `T1=L_tilde`, and
`T2=2*L_tilde^2-I`. Focused tests verify recurrence and aggregation by hand.
Both GLU filter/gate halves, both graph convolutions and all three Chebyshev
parameter slices, both block LayerNorms, output collapse, output LayerNorm, and
horizon head receive finite non-zero gradients.
