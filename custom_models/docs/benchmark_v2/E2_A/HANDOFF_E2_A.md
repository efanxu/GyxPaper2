# HANDOFF E2-A

E2-A is complete and stops before formal runs and E2-B.

Final runtime classes:

- locally verified trainable: GRU, DLinear, LightTS, iTransformer, TimeXer;
- trainable with exact hardware preflight: TiDE, SegRNN, Transformer, PatchTST;
- available non-trainable: Persistence, MovingAverage;
- true blocked/unavailable: remaining 17 entries.

Transformer and PatchTST ordinary/real engineering checks pass, but exact local GTX 1060 full-shape attempts are `FAIL_OOM`; they require RTX 4060 preflight. iTransformer and TimeXer exact local AMP backward passed and are `AVAILABLE_TRAINABLE`.

Use `E2_A_RUNBOOK.md` and the appended RTX 4060 chapter in `E1_B_RUNBOOK.md`. A single `train` command performs preflight then training automatically when required. Do not edit Registry manually or reuse a mismatched preflight.

Protection: protocol hash remains `0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b`; Canonical checkpoint remains `f08c822f512384aaf7700b9f5e6049a940d63f385829a43f4223920b583bba7a`. TSLib, ST-MGPrompt, Canonical, P0–P5/A0–A8, E1-A, and TiDE/SegRNN model/config/adapter behavior are protected by matching before/after snapshots.

RTX 4060 preflight: NOT_RUN. E1-B formal Full: NOT_RUN. E2-A formal Full: NOT_RUN. E2-B: NOT_STARTED.
