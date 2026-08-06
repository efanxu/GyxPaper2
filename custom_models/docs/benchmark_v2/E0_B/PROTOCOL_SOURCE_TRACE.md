# Protocol source trace

| Field group | Source and decision |
|---|---|
| Dataset, target, mask, lookback, horizon, split, stride, clip, checkpoint | E0-A audit and formal `experiment_protocol.py`/canonical overrides; these fields are frozen by the task protocol. |
| 16 features and order | `st_mgprompt/config.py:9-26`; copied exactly and recorded. `Patv_raw` and `valid_target_mask` are excluded from input. |
| Timestamp/node keys, alignment, window boundary, train-only scaler | `st_mgprompt/data.py:95-194, 269-410`; actual semantics preserved in the independent provider. |
| Target loss space | `st_mgprompt/losses.py:27-84` documents normalized target-space losses; benchmark_v2 defaults to `masked_mse` there. |
| Metric order and official Score | `st_mgprompt/metrics.py:47-141` and `evaluate.py:210-225, 388-403`; runtime reimplements the formula without importing ST-MGPrompt. |
| Raw `config.py` defaults | Not used to override formal values: its patience/min-delta/eval batch defaults differ from the frozen formal contract. |
| TSLib generic protocol | Read-only comparison only; `(B,T,C)->(B,H,C)`, generic MSE and test-in-epoch path are explicitly rejected. |
| Missing `PLAN.md`/`HANDOFF.md` | Not present in the task workspace at E0-B start; recorded as a pre-existing material gap. No unresolved formal field was guessed. |

