from __future__ import annotations

from typing import Any


TRANSFORMER_PRECISION_REASON = (
    "AMP-only forward overflow reproduced in upstream Transformer attention "
    "on benchmark_v2 uniform batch4; keep model-specific FP32 forward policy."
)
E9_FP32_PRECISION_REASON = (
    "AMP non-finite gradients/predictions reproduced during formal "
    "benchmark_v2 uniform batch4 training; keep model-specific FP32 policy."
)
TRANSFORMER_BATCH_PROFILE_ID = "uniform_train_batch4_v1"
FP32_MODEL_IDS = frozenset({"transformer", "patchtst", "itransformer"})
E9_FP32_MODEL_IDS = frozenset({"patchtst", "itransformer"})


def _fp32_precision_reason(model_id: str) -> str:
    if model_id in E9_FP32_MODEL_IDS:
        return E9_FP32_PRECISION_REASON
    return TRANSFORMER_PRECISION_REASON


def expected_model_precision_identity(
    model_id: str, training_batch_profile_id: str | None
) -> dict[str, Any]:
    """Return the frozen precision identity for a model/profile pair."""

    if training_batch_profile_id != TRANSFORMER_BATCH_PROFILE_ID:
        return {
            "amp_enabled": True,
            "precision_policy": "profile_default",
        }
    if model_id in FP32_MODEL_IDS:
        return {
            "amp_enabled": False,
            "precision_policy": "fp32",
            "precision_resolution": {
                "requested_amp_enabled": True,
                "effective_amp_enabled": False,
                "reason": _fp32_precision_reason(model_id),
            },
        }
    return {
        "amp_enabled": True,
        "precision_policy": "profile_default",
        "precision_resolution": {
            "requested_amp_enabled": True,
            "effective_amp_enabled": True,
            "reason": "uniform_train_batch4_v1 profile default; no model-specific override.",
        },
    }


def apply_model_precision_policy(runtime: Any):
    """Apply only audited model-specific precision overrides.

    The frozen batch profile remains AMP-enabled by default.  The original
    Transformer entry and the E9 PatchTST/iTransformer controls have audited
    FP32 exceptions; all other models retain the profile behavior.
    """

    if (
        runtime.effective_config.get("training_batch_profile_id")
        != TRANSFORMER_BATCH_PROFILE_ID
    ):
        return runtime
    model_id = getattr(runtime, "model_id", None)
    if model_id not in FP32_MODEL_IDS:
        runtime.effective_config.setdefault("precision_policy", "profile_default")
        runtime.effective_config.setdefault(
            "precision_resolution",
            {
                "requested_amp_enabled": True,
                "effective_amp_enabled": True,
                "reason": (
                    "uniform_train_batch4_v1 profile default; "
                    "no model-specific override."
                ),
            },
        )
        return runtime
    requested_amp_enabled = bool(
        runtime.effective_config.get("amp_enabled", False)
    )
    runtime.effective_config["amp_enabled"] = False
    runtime.effective_config["precision_policy"] = "fp32"
    runtime.effective_config["precision_resolution"] = {
        "requested_amp_enabled": requested_amp_enabled,
        "effective_amp_enabled": False,
        "reason": _fp32_precision_reason(model_id),
    }
    return runtime
