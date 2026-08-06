# Graph WaveNet support trace

Support order is stable and explicit:

1. `P_forward`, record `<removed-content-record>`
2. `P_reverse`, record `<removed-content-record>`
3. adaptive `softmax(ReLU(E1 @ E2))`

For each support, diffusion bases are `A·X` and `A²·X`; the residual input is
included once outside those support-specific terms. Both physical matrices are
runtime float32, non-persistent, non-trainable copies of the frozen float64
bundle matrices. Adaptive parameters are trainable and checkpointed.

Focused tests verified all support branches and both diffusion orders receive
gradients. Ordinary smoke adaptive record was
`<removed-content-record>`
and was reproduced after strict reload.
