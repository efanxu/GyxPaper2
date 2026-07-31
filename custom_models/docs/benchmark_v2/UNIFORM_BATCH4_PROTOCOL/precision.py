from __future__ import annotations

from typing import Any


TRANSFORMER_PRECISION_REASON = (
    "AMP-only forward overflow reproduced in upstream Transformer attention "
    "on benchmark_v2 uniform batch4; keep model-specific FP32 forward policy."
)
TRANSFORMER_BATCH_PROFILE_ID = "uniform_train_batch4_v1"


def apply_model_precision_policy(runtime: Any):
    """Apply only audited model-specific precision overrides.

    The frozen batch profile remains AMP-enabled.  Transformer is the only
    model with a proven forward overflow, so its effective training precision
    is recorded separately and all other models retain the profile behavior.
    """

    if getattr(runtime, "model_id", None) != "transformer":
        return runtime
    if (
        runtime.effective_config.get("training_batch_profile_id")
        != TRANSFORMER_BATCH_PROFILE_ID
    ):
        return runtime
    requested_amp_enabled = bool(
        runtime.effective_config.get("amp_enabled", False)
    )
    runtime.effective_config["amp_enabled"] = False
    runtime.effective_config["precision_policy"] = "fp32"
    runtime.effective_config["precision_resolution"] = {
        "requested_amp_enabled": requested_amp_enabled,
        "effective_amp_enabled": False,
        "reason": TRANSFORMER_PRECISION_REASON,
    }
    return runtime
