# HANDOFF E9

## Current state

E9-A and E9-B engineering are implemented and locally validated. No formal long training was started.

Portability is `PORTABLE_WITH_TRAIN_STATE`. The exact MS-MG-DWU implementation uses only prediction/target/mask plus public node/horizon/protocol identity and training-only online EMA state. It has no Fine/Coarse, VADSP, Prompt, Cross-gate, hidden-state, attention, graph-embedding, or model-auxiliary dependency. Full transfer is allowed; no subset or renaming is needed.

The original26 workbook is `C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx`. All six frozen controls reconcile to formal artifacts and are genuine `masked_mse`, batch4, seed2026 controls. Proposed loss-only config diff passes 6/6.

Current readiness is:

- `PORTABILITY_AUDIT_READY=PASS`
- `ORIGINAL_MASKED_MSE_REFERENCE_READY=6/6`
- `MSMG_DWU_TRANSFER_READY=0/6`
- `LOSS_ONLY_PAIRING_READY=0/6`
- `CORE_E9_READY=6/12`
- `FULL_MS_MG_DWU_TRANSFER_READY=true`
- `PORTABLE_SUBSET_TRANSFER_READY=false`

## Resume sequence

On Linux, run `E9_PRECHECK_LINUX.sh`, then `E9_PREFLIGHT_ALL_6_LINUX.sh`. Only after all required preflights pass, run `E9_RUN_ALL_6_LINUX.sh` or the autoshutdown wrapper. Do not change batch/shape/model settings after a failure and do not overwrite failed directories.

The run-all script always writes `E9_TRANSFER_READINESS.json`, which can reach 6/6 without the desktop workbook but explicitly does not imply core E9 readiness. If `ORIGINAL26_XLSX` is exported, the script additionally performs the full workbook-backed readiness audit.

After syncing the six transfer run directories back to the fixed output root, run readiness with explicit `--original26-xlsx`. Run `aggregate --require-complete` only after core 12/12 and pairing 6/6.

## Integrity constraints

Controls remain read-only and are not retrained. Historical formal artifacts, A0/A8, and cancelled common-loss material were not modified or adopted. Smoke/preflight outputs cannot enter evidence. Exact initial-state equality is currently `NOT_VERIFIED`; claim only same seed/source/config/protocol unless future artifacts prove matching initial-state identity.

No git commit or push was performed. Stop after E9; do not start E10 automatically.
