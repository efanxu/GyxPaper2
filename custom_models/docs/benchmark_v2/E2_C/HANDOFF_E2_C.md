# HANDOFF E2-C

E2-C is complete and stops before formal runs and E2-D.

- TimeMixer: source SHA256 `bbf378cab03d16d3e7ae907f7b3cefbac89ae1504e036270578d5d496ffbd133`;
  192,285 parameters; ordinary/real PASS; exact local `FAIL_OOM`;
  `AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED`.
- TSMixer: source SHA256 `a82942ddc22cba59f4161f3ea480eb0e474eba6f0ab2d13f2bd949f100b2ce87`;
  22,378 parameters; ordinary/real/exact local PASS;
  `AVAILABLE_TRAINABLE`.
- FreTS: source SHA256 `2b6c9e0cd3d4f42bc74736343147812db94f029f1a8b13b098b9841705de5c18`;
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

Protocol hash remains
`0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`;
Canonical checkpoint remains
`f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`.
Use `E2_C_RUNBOOK.md`. Target high-memory preflight and all formal runs are
NOT_RUN. E2-D is NOT_STARTED; its only proposed inputs are Crossformer, MSGNet,
and TimeFilter.
