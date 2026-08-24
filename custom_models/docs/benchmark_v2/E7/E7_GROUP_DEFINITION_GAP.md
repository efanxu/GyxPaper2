# E7 Group Definition Gap

Status: `BLOCKED_DEFINITION_MISSING`

Training-fitted volatility and Ramp threshold artifacts exist, but E7 cannot locate all of the following frozen public definitions:

- `E10_GROUP_DEFINITION_MANIFEST.json`
- `test_window_identity_manifest.json`
- `shared_difficult_top10_window_ids.json`

Therefore E7 does not invent thresholds, does not construct model-specific groups, and does not publish grouped metrics. This does not change core 10-evidence readiness.
