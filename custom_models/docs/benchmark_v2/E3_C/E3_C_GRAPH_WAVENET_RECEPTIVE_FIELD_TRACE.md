# Graph WaveNet receptive-field trace

For kernel size 2, each block has dilations 1 and 2 and the dilation schedule
resets for each of four blocks:

`[1,2,1,2,1,2,1,2]`.

The receptive field is:

`1 + 4 × ((2-1)×1 + (2-1)×2) = 13`.

Starting from temporal width 144, the eight valid causal convolutions produce:

`144 → 143 → 141 → 140 → 138 → 137 → 135 → 134 → 132`.

Residual tensors are right-aligned to the valid causal tail. Skip tensors are
also right-aligned before addition. The end projection preserves temporal width
132 and only index `-1` is used as the forecast anchor. Thus every horizon is
derived from the last valid causal state, whose actual temporal receptive field
is the final 13 observed steps; no arbitrary middle crop or future value is
used.
