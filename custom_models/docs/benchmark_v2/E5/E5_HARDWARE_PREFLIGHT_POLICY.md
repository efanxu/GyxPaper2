# E5 hardware preflight policy

A masked-MSE PASS is not reusable for E5. The E5 identity additionally binds
the experiment profile, E5 protocol record, loss id/source/profile records,
base config/source listing, benchmark protocol, exact B/T/N/C/H, precision,
seed, and graph/node/timestamp identities. E5 benchmark models use a uniform
FP32 overlay; Original and other profiles retain their own precision policy.
PASS requires a completed forward, common-loss backward, and Adam optimizer
update with finite output, loss, gradients, parameters, and optimizer state.
PatchTST uses E5-only encoder activation checkpointing to keep the uniform
FP32 policy within the Batch4 memory envelope without changing parameters.
OOM routes only the E5 variant to
`E5_AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`; non-OOM failures block.
The parent launcher remains CPU-only and starts a fresh formal worker only
after the preflight worker exits successfully. No target-machine preflight was
run in E5-A.
