# E5 hardware preflight policy

A masked-MSE PASS is not reusable for E5. The E5 identity additionally binds
the experiment profile, E5 protocol hash, loss id/source/profile hashes,
base config/source closure, benchmark protocol, exact B/T/N/C/H, AMP, seed,
and graph/node/timestamp identities. PASS requires completed forward and
backward. OOM routes only the E5 variant to
`E5_AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`; non-OOM failures block.
The parent launcher remains CPU-only and starts a fresh formal worker only
after the preflight worker exits successfully. No target-machine preflight was
run in E5-A.
