from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


CLI_PROFILE_ID = "e5_common_loss_v1"
PROFILE_ID = "e5_common_loss_architecture_v1"
DEFAULT_PROFILE_ID = "default_benchmark_v1"
LOSS_ID = "masked_score_aligned_hybrid"
ALLOWLIST = (DEFAULT_PROFILE_ID, CLI_PROFILE_ID)
BENCHMARK_PROTOCOL_HASH = (
    "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b"
)
SOURCE_FUNCTION = "st_mgprompt.losses.masked_score_aligned_hybrid_loss"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
LOSS_SOURCE_PATH = PROJECT_ROOT / "custom_models/src/st_mgprompt/losses.py"
PROTOCOL_PATH = Path(__file__).resolve().with_name("e5_common_loss_protocol_v1.json")


def stable_hash(payload: Mapping[str, Any]) -> str:
    material = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loss_profile_payload() -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "cli_profile_id": CLI_PROFILE_ID,
        "loss_id": LOSS_ID,
        "loss_source_path": LOSS_SOURCE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "loss_source_function": SOURCE_FUNCTION,
        "loss_source_hash": file_sha256(LOSS_SOURCE_PATH),
        "adapter": "benchmark_BNH_to_a8_BHN_transpose_v1",
        "loss_space": "normalized Patv_raw",
        "physical_clip_in_training_loss": False,
        "inverse_transform_in_training_loss": False,
        "formula": "0.5*per_node_masked_MAE_horizon + 0.5*per_node_masked_RMSE_horizon",
        "epsilon": 1e-6,
        "mae_weight": 0.5,
        "rmse_weight": 0.5,
        "horizon_aggregation": "masked mean per B,N across H",
        "site_aggregation": "mean across valid B,N rows",
        "mask_normalization": "per B,N valid horizon count, clamp_min(1)",
        "all_masked_behavior": "None",
        "dependencies": ["prediction", "target", "mask", "epsilon=1e-6"],
    }


def get_profile_metadata(profile: str | None) -> dict[str, Any]:
    selected = DEFAULT_PROFILE_ID if profile in (None, "", DEFAULT_PROFILE_ID) else profile
    if selected not in ALLOWLIST:
        raise ValueError(
            f"Unknown experiment profile {selected!r}; allowed={list(ALLOWLIST)}"
        )
    if selected == DEFAULT_PROFILE_ID:
        return {
            "cli_profile_id": DEFAULT_PROFILE_ID,
            "profile_id": DEFAULT_PROFILE_ID,
            "loss_id": "masked_mse",
        }
    payload = loss_profile_payload()
    payload["loss_profile_hash"] = stable_hash(payload)
    protocol = load_e5_protocol()
    payload["e5_common_loss_protocol_hash"] = protocol[
        "e5_common_loss_protocol_hash"
    ]
    return payload


def e5_loss_adapter(prediction, target, mask, eps: float = 1e-6):
    """Call the formal A8 loss source after the explicit BNH->BHN axis adapter."""
    module_name = "_benchmark_v2_e5_readonly_a8_losses"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, LOSS_SOURCE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load formal A8 loss source: {LOSS_SOURCE_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    masked_score_aligned_hybrid_loss = module.masked_score_aligned_hybrid_loss

    return masked_score_aligned_hybrid_loss(
        prediction.transpose(1, 2),
        target.transpose(1, 2),
        mask.transpose(1, 2),
        eps=eps,
    )


def loss_for_profile(profile: str | None) -> Callable:
    selected = DEFAULT_PROFILE_ID if profile in (None, "", DEFAULT_PROFILE_ID) else profile
    if selected == DEFAULT_PROFILE_ID:
        from ...losses import get_loss

        return get_loss("masked_mse")
    if selected != CLI_PROFILE_ID:
        raise ValueError(
            f"Unknown experiment profile {selected!r}; allowed={list(ALLOWLIST)}"
        )
    return e5_loss_adapter


def load_e5_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    logical = {
        key: value
        for key, value in payload.items()
        if key != "e5_common_loss_protocol_hash"
    }
    expected = stable_hash(logical)
    if payload.get("e5_common_loss_protocol_hash") != expected:
        raise ValueError("E5 common-loss protocol logical hash mismatch")
    profile = loss_profile_payload()
    if payload.get("loss_source_hash") != profile["loss_source_hash"]:
        raise ValueError("E5 loss source hash mismatch")
    if payload.get("loss_profile_hash") != stable_hash(profile):
        raise ValueError("E5 loss profile hash mismatch")
    if payload.get("base_benchmark_protocol_hash") != BENCHMARK_PROTOCOL_HASH:
        raise ValueError("E5 base benchmark protocol hash mismatch")
    return payload


def check_loss_profile(profile: str | None) -> dict[str, Any]:
    metadata = get_profile_metadata(profile)
    if metadata["cli_profile_id"] != CLI_PROFILE_ID:
        raise ValueError("loss-profile-check requires e5_common_loss_v1")
    protocol = load_e5_protocol()
    return {
        "status": "PASS",
        "allowlist": list(ALLOWLIST),
        "profile": metadata,
        "protocol": protocol,
        "canonical_source_exists": LOSS_SOURCE_PATH.is_file(),
        "imports_st_mgprompt_model": False,
        "creates_gpu_model": False,
    }
