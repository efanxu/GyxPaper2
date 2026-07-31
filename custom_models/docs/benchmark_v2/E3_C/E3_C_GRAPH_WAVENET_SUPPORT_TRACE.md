# Graph WaveNet support trace

Support order is stable and explicit:

1. `P_forward`, hash `59b39f001b3118557b55b88041b27b4ce7a743a85c0bc7948210945f7e3b8c03`
2. `P_reverse`, hash `89f09d41a8cae126c5668d4fe55934819c0a7489797b8e09fc0def0cc05da4fe`
3. adaptive `softmax(ReLU(E1 @ E2))`

For each support, diffusion bases are `A·X` and `A²·X`; the residual input is
included once outside those support-specific terms. Both physical matrices are
runtime float32, non-persistent, non-trainable copies of the frozen float64
bundle matrices. Adaptive parameters are trainable and checkpointed.

Focused tests verified all support branches and both diffusion orders receive
gradients. Ordinary smoke adaptive hash was
`0c2dc80e419305ec2f803f9edef8baf780bad99625b2c644b7f75e0a8fa89c78`
and was reproduced after strict reload.
