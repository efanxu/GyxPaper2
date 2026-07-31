# Batch=4 Memory Feasibility

Historical local-machine status: `BLOCKED_MACHINE_OOM`.

Global engineering status: `READY_FOR_TARGET_MACHINE_VALIDATION`.

Exact shape was `B=4,T=144,N=134,C=16,H=10,AMP=true` on an NVIDIA GeForce
GTX 1060 (6 GiB). known-OOM6 counts: PASS=4,
FAIL_OOM=2, FAIL_NON_OOM=0.

SegRNN failed during backward. MSGNet failed during forward. Transformer,
PatchTST, FreTS, and TimeFilter passed forward/backward with finite gradients.
No fallback batch, gradient accumulation, checkpointing, offload, or model
capacity change was attempted.

Because this is not the intended approximately 48 GiB formal GPU, these results
block this machine but do not substitute for a rerun on the intended device.
They do not prevent any target machine from producing its own machine-bound
exact preflight artifacts.
