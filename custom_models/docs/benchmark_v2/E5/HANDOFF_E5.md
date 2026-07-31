# HANDOFF E5-A

E5-A engineering preparation is complete and stops before formal execution.
E4 is deferred. The matrix is 26 trainable + 2 evaluate-only + 1 formal A8
reference. Loss profile `0fc1fca238d7161d3a0257d639a11df8cc92ad2a1c4d61eab234338fa24b59fa` and E5 protocol
`a7c5b5a5b15b5ab70d08bf926b9de351f05740940f16409c801eed419287141a` are frozen. A8 is VALID and was
neither copied nor retrained.

Ordinary smoke: 28/28 PASS. Exact full-shape: 16 PASS, 10 pure OOM routes, zero
non-OOM failures. Limited real SDWPF: 26/26 PASS. Use `E5_RUNBOOK.md` only
after all 28 original Full runs finish and are frozen.

Formal E5 train/evaluate, target-machine preflight, final aggregation, E4 and
E6 are NOT_RUN/NOT_STARTED.
