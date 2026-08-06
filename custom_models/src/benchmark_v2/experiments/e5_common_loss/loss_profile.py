from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


CLI_PROFILE_ID = "e5_common_loss_v1"
PROFILE_ID = "e5_common_loss_architecture_v1"
DEFAULT_PROFILE_ID = "default_benchmark_v1"
LOSS_ID = "masked_score_aligned_hybrid"
ALLOWLIST = (DEFAULT_PROFILE_ID, CLI_PROFILE_ID)
SOURCE_FUNCTION = "st_mgprompt.losses.masked_score_aligned_hybrid_loss"
PROTOCOL_PATH = Path(__file__).resolve().with_name("e5_common_loss_protocol_v1.json")


def loss_profile_payload() -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "cli_profile_id": CLI_PROFILE_ID,
        "loss_id": LOSS_ID,
        "source_function": SOURCE_FUNCTION,
        "aggregation": "masked mean over valid targets",
        "horizons": [3, 6, 10],
    }


def get_profile_metadata(profile: str | None) -> dict[str, Any]:
    selected = profile or DEFAULT_PROFILE_ID
    if selected not in ALLOWLIST:
        raise ValueError(f"Unknown experiment profile: {selected}")
    if selected == DEFAULT_PROFILE_ID:
        return {"profile_id": DEFAULT_PROFILE_ID, "loss_id": "masked_mse", "source_function": "benchmark_v2.losses.masked_mse"}
    return loss_profile_payload()


def e5_loss_adapter(prediction, target, mask, eps: float = 1e-6):
    from st_mgprompt.losses import masked_score_aligned_hybrid_loss
    return masked_score_aligned_hybrid_loss(prediction, target, mask, eps=eps)


def loss_for_profile(profile: str | None) -> Callable:
    if (profile or DEFAULT_PROFILE_ID) == CLI_PROFILE_ID:
        return e5_loss_adapter
    from ...losses import get_loss
    return get_loss("masked_mse")


def load_e5_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E5 protocol must be an object.")
    return payload


def check_loss_profile(profile: str | None) -> dict[str, Any]:
    metadata = get_profile_metadata(profile)
    return {"status": "PASS", **metadata}
