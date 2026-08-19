from __future__ import annotations

from typing import Any


TRANSFORMER_PRECISION_REASON = (
    "AMP-only forward overflow reproduced in upstream Transformer attention "
    "on benchmark_v2 uniform batch4; keep model-specific FP32 forward policy."
)
TRANSFORMER_BATCH_PROFILE_ID = "uniform_train_batch4_v1"
E5_PROFILE_IDS = frozenset(
    {"e5_common_loss_v1", "e5_common_loss_architecture_v1"}
)
E5_FP32_REASON = (
    "E5 uses one uniform FP32 policy for every benchmark model so its common-loss "
    "comparison cannot be interrupted or biased by model-specific AMP overflow."
)


def expected_model_precision_identity(
    model_id: str,
    training_batch_profile_id: str | None,
    *,
    experiment_profile_id: str | None = None,
) -> dict[str, Any]:
    """Return the frozen precision identity for a model/profile pair."""

    if training_batch_profile_id != TRANSFORMER_BATCH_PROFILE_ID:
        return {
            "amp_enabled": True,
            "precision_policy": "profile_default",
        }
    if experiment_profile_id in E5_PROFILE_IDS:
        return {
            "amp_enabled": False,
            "precision_policy": "fp32",
            "precision_resolution": {
                "requested_amp_enabled": True,
                "effective_amp_enabled": False,
                "reason": E5_FP32_REASON,
            },
        }
    if model_id == "transformer":
        return {
            "amp_enabled": False,
            "precision_policy": "fp32",
            "precision_resolution": {
                "requested_amp_enabled": True,
                "effective_amp_enabled": False,
                "reason": TRANSFORMER_PRECISION_REASON,
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

    E5 is uniformly FP32. Other experiments keep the frozen AMP-enabled batch
    profile, except for the existing Transformer overflow override.
    """

    if (
        runtime.effective_config.get("training_batch_profile_id")
        != TRANSFORMER_BATCH_PROFILE_ID
    ):
        return runtime
    experiment_profile_id = runtime.effective_config.get("experiment_profile_id")
    e5_fp32 = experiment_profile_id in E5_PROFILE_IDS
    if not e5_fp32 and getattr(runtime, "model_id", None) != "transformer":
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
        "reason": E5_FP32_REASON if e5_fp32 else TRANSFORMER_PRECISION_REASON,
    }
    if e5_fp32 and getattr(runtime, "model_id", None) == "patchtst":
        from .models.patchtst import enable_e5_gradient_checkpointing

        enable_e5_gradient_checkpointing(runtime.model)
        runtime.effective_config["activation_checkpointing"] = (
            "patchtst_encoder_layers"
        )
    return runtime
