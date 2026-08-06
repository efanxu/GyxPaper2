# HANDOFF E2-D

E2-D is complete and stops before formal runs and E3.

- Crossformer: 117,668 parameters; ordinary/real/exact local PASS;
  `AVAILABLE_TRAINABLE`.
- MSGNet: 13,294,292 parameters; ordinary/limited-real PASS; exact local
  `FAIL_OOM`; requires matching target-machine preflight.
- TimeFilter: 4,801,994 parameters; ordinary/limited-real PASS; exact local
  `FAIL_OOM`; requires matching target-machine preflight.

All use one shared model and dynamic validated `Patv_clean_for_input` output
selection with normalized `Patv_raw` supervision. MSGNet is isolated per
prediction item because upstream FFT averages batch. MSGNet's `(16,16)` graph
and TimeFilter's 144-token graph are within-turbine structures, not E3 turbine
graphs.

Allowlist: 18 explicit IDs. Registry: 28 total, 19 trainable, 9 locally exact
PASS, 10 preflight-required, 2 non-trainable, 7 blocked/unavailable.

Use `E2_D_RUNBOOK.md`. Target high-memory preflight, all three formal Full
runs, formal evaluation, and E3 are NOT_RUN/NOT_STARTED. The next permitted
stage is E3-A only: freeze graph protocol, resolve STCN/STGCN naming, and record
the 134-turbine node order and graph.
