# MTGNN temporal and graph trace

## Temporal geometry

Each dilated inception layer has kernels 2, 3, 6, and 7, dilation exponent 1,
and aligns all branches to the shortest result:

| Layer | Input | k2 | k3 | k6 | k7 | Aligned output |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 144 | 143 | 142 | 139 | 138 | 138 |
| 2 | 138 | 137 | 136 | 133 | 132 | 132 |
| 3 | 132 | 131 | 130 | 127 | 126 | 126 |

The receptive field is `1 + 3×(7-1) = 19`. `skip0` uses kernel 144; the three
layer skip kernels are 138, 132, and 126; `skipE` uses 126. Every skip path
therefore ends at width 1 before the horizon head.

## Graph geometry

The constructor is directed and antisymmetric before top-k. It removes
self-loops and retains exactly 20 outgoing edges per row. Ordinary smoke:
2,680 edges, density `0.14925373134328357`, outgoing degree exactly 20,
incoming degree 0–59, maximum asymmetry 1.0, hash
`7a07d0e5a7e156dcffbcabeff7bf47a18716bc43bc2783e4dfa34f774a37d1b1`.
Forward and reverse MixProp both use depth 2. Physical supports are context-only.
Node splitting/sampling is disabled.
