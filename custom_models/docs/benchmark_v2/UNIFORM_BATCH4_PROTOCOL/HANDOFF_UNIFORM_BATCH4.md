# Handoff: Uniform Batch=4

Global engineering status is `READY_FOR_TARGET_MACHINE_VALIDATION`.

The local `GTX1060_WINDOWS` status is `BLOCKED_MACHINE_OOM`: SegRNN failed in
backward and MSGNet failed in forward. Those immutable artifacts are historical
evidence for that machine only.

On every target, run platform PRECHECK and the exact suite-specific preflight.
Original, E5, Full, and A8 have independent machine-bound gates. A target PASS
automatically releases only its matching formal script; no JSON is edited by
hand.

Original 28 and E5 core may run in parallel on two machines with matching
frozen identities and disjoint output roots. E5 final reference, 29/29
readiness, and require-complete aggregation remain blocked until both the new
batch4 A8 and frozen Original 28 are complete.

See `UNIFORM_BATCH4_RUNBOOK.md` for exact Windows and Linux commands.
