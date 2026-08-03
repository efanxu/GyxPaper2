from __future__ import annotations

from .loss_profile import CLI_PROFILE_ID
from ...original_scope26 import (
    CURRENT_SCOPE26_ID,
    CURRENT_TRAINING_PROFILE_ID,
    is_current_scope26_request,
    validate_current_scope26_request,
)


E5_SCOPE27_ID = "e5_batch4_scope27_seed2026"
BENCHMARK_SCOPE26_ID = "benchmark_v2_batch4_scope26_seed2026"
TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
PREFLIGHT_POLICY_ID = "e5_scope27_no_preflight_artifact_required_v1"

E5_SCOPE27_TRAINABLE_MODELS = (
    "gru",
    "dlinear",
    "lightts",
    "tide",
    "transformer",
    "patchtst",
    "itransformer",
    "timexer",
    "timesnet",
    "micn",
    "wpmixer",
    "multipatchformer",
    "timemixer",
    "tsmixer",
    "frets",
    "crossformer",
    "timefilter",
    "gcn",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "mtgnn",
    "agcrn",
    "stid",
)
E5_SCOPE27_EVALUATE_ONLY_MODELS = ("persistence", "moving_average")
E5_SCOPE27_REFERENCE_MODELS = ("st_mgprompt_a8",)
E5_SCOPE27_EXCLUDED_MODELS = ("segrnn", "msgnet")


def is_scope27_train_request(
    *,
    model_id: str,
    formal_scope_id: str | None,
    experiment_profile: str | None,
    training_profile: str | None,
) -> bool:
    """Authorize the narrowly scoped E5 batch4 path that has no artifact preflight gate."""
    return (
        formal_scope_id == E5_SCOPE27_ID
        and experiment_profile == CLI_PROFILE_ID
        and training_profile == TRAINING_PROFILE_ID
        and model_id in E5_SCOPE27_TRAINABLE_MODELS
    )


def is_current_scope26_train_request(
    *,
    model_id: str,
    formal_scope_id: str | None,
    experiment_profile: str | None,
    training_profile: str | None,
) -> bool:
    """Return whether a request is bound to the active Original scope26."""

    return is_current_scope26_request(
        model_id=model_id,
        formal_scope_id=formal_scope_id,
        experiment_profile=experiment_profile,
        training_profile=training_profile,
        trainable=True,
    )


def validate_scope27_request(
    *,
    model_id: str,
    formal_scope_id: str | None,
    experiment_profile: str | None,
    training_profile: str | None,
    trainable: bool,
) -> None:
    if formal_scope_id is None:
        return
    if formal_scope_id == CURRENT_SCOPE26_ID:
        validate_current_scope26_request(
            model_id=model_id,
            formal_scope_id=formal_scope_id,
            experiment_profile=experiment_profile,
            training_profile=training_profile,
            trainable=trainable,
        )
        return
    allowed = (
        E5_SCOPE27_TRAINABLE_MODELS
        if trainable
        else E5_SCOPE27_EVALUATE_ONLY_MODELS
    )
    if (
        formal_scope_id != E5_SCOPE27_ID
        or experiment_profile != CLI_PROFILE_ID
        or training_profile != TRAINING_PROFILE_ID
        or model_id not in allowed
    ):
        raise ValueError(
            "Invalid E5 scope27 request binding: "
            f"scope={formal_scope_id!r}, model={model_id!r}, "
            f"experiment_profile={experiment_profile!r}, "
            f"training_profile={training_profile!r}"
        )
