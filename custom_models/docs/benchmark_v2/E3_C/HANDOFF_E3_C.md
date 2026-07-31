# HANDOFF E3-C

E3-C engineering integration is complete.

Graph WaveNet, MTGNN, AGCRN, and STID are native 134-turbine models with
frozen configurations and explicit physical/adaptive/identity policies. All
four passed ordinary smoke, limited real SDWPF smoke, exact local full-shape
AMP forward/loss/backward, strict reload, artifact validation, leakage,
identity, isolation, gradient, and source-closure checks.

Final status for every model: `AVAILABLE_TRAINABLE`, locally exact PASS,
mandatory formal hardware preflight not required.

Registry: 28 total, 26 trainable, 16 locally exact PASS, 10 earlier
preflight-required, 2 available non-trainable, 0 blocked. TSLib allowlist: 18.

Use `E3_C_RUNBOOK.md` for prepared commands. Formal Full, formal evaluation,
target-machine preflight, and E4 are `NOT_RUN`. The formal root
`custom_models/results/benchmark_v2/e3_c_seed2026` does not exist.

Next-phase input is the frozen E3-C implementation plus its source-closure,
graph/identity policies, smoke summaries, tests, and protection snapshots.
Do not treat smoke losses or learned-graph diagnostics as model-selection or
paper-result evidence.
