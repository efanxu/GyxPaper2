# HANDOFF E2-C

E2-C is complete and stops before formal runs and E2-D.

- TimeMixer: source content record `<removed-content-record>`;
  192,285 parameters; ordinary/real PASS; exact local `FAIL_OOM`;
  `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.
- TSMixer: source content record `<removed-content-record>`;
  22,378 parameters; ordinary/real/exact local PASS;
  `AVAILABLE_TRAINABLE`.
- FreTS: source content record `<removed-content-record>`;
  4,787,594 parameters; string `"0"` enables both FFT branches; native AMP;
  ordinary/real PASS; exact local `FAIL_OOM`;
  `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.

The explicit allowlist has exactly 15 IDs. All three use one node-shared model,
dynamic validated `Patv_clean_for_input` output-channel selection, and
normalized `Patv_raw` supervision. No target, mask, future observation,
weather, calendar, graph, node embedding, or node-specific head enters forward.

Registry: 28 total; 16 formal-runnable trainable; 8 local exact PASS; 8
preflight-required; 2 non-trainable available; 10 blocked/unavailable.

Tests: 84/84 full suite, 11/11 E2-C focused, 9/9 hardware flow. Ordinary and
real smoke are 3/3 PASS. TimeMixer/FreTS local exact OOM artifacts and all
FreTS diagnostic/superseded artifacts are retained.

Protocol record remains
`<removed-content-record>`;
Canonical checkpoint remains
`<removed-content-record>`.
Use `E2_C_RUNBOOK.md`. Target high-memory preflight and all formal runs are
NOT_RUN. E2-D is NOT_STARTED; its only proposed inputs are Crossformer, MSGNet,
and TimeFilter.
