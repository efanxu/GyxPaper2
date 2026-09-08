# HANDOFF E8

## Current state

E8 engineering, audit, read-only diagnostic hooks, statistics, fail-closed aggregation, report generation path, tests, and Runbook are implemented. No training was started.

Current readiness is `INTERNAL_PROMPT_FUSION_READY=0/5`, `ORIGINAL26_EXTERNAL_READY=6/6`, and `CORE_E8_READY=6/11`. Strict formal aggregation was exercised and exited 2; no formal Excel/result CSV was created.

## Evidence facts

The external workbook is `C:\Users\12811\Desktop\实验结果\original26_filtered20.xlsx`. Its exact six E8 model IDs have unique COMPLETED/PASS runs, complete H3/H6/H10 metrics, and exact artifact agreement.

The available A0/A4/A5/A6/A7 fixed-dual artifacts record train batch32, lack the current batch4 profile and `resolved_config.json`, and do not have the current formal run-status schema. They cannot be current E8 causal evidence and cannot be used for formal diagnostics.

Static config/source audit passes. A5 disables Fine-to-Coarse while retaining Coarse/Macro-to-Fine; A6 disables both directions while retaining both branches. The direction decomposition predicate is statically valid, but the result table remains blocked until authentic current-batch4 internal evidence exists.

Macro Prompt is latent only; no physical trend sign result is allowed. A0 uses attention temporal pooling, while current A4 uses equal-weight mean pooling with the same four-token capacity. A0 ST Prompt is node + future-step + granularity; current A7 removes only node identity and remains future-step + granularity, with the same ST Prompt decoder. Cross Fusion is real attention + gate + residual + LayerNorm. Fine and Coarse both preserve length 144.

Grouped analysis is `BLOCKED_DEFINITION_MISSING`, and paired-window analysis is blocked by missing current-protocol common prediction/window exports.

Compileall/import passed. All 13 E8 contract tests passed, including artifact-tool inspection/render verification of the required 16-sheet Excel schema and the diagnostic prediction/parameter/buffer/checkpoint/mode/RNG side-effect contract.

## Resume rule

Resume only when authentic current-batch4 A0/A4/A5/A6/A7 formal artifacts are available. Re-run `audit`, `diagnostics`, then `aggregate --require-complete`. Do not retrain as part of E8, repair source metadata, restore E5, add A8, relabel batch32 artifacts, use smoke, substitute excluded models, modify protocol, or start E9.
