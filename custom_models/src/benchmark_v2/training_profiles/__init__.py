from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ProtocolError


PROFILE_SCHEMA_VERSION = "uniform_training_batch_profile_v1"
UNIFORM_BATCH4_PROFILE_ID = "uniform_train_batch4_v1"
PROFILE_ALLOWLIST = (UNIFORM_BATCH4_PROFILE_ID,)
PROFILE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingBatchProfile:
    payload: dict[str, Any]

    @property
    def profile_id(self) -> str:
        return str(self.payload["profile_id"])

    @property
    def train_batch_size(self) -> int:
        return int(self.payload["train_batch_size"])

    @property
    def val_batch_size(self) -> int:
        return int(self.payload["val_batch_size"])

    @property
    def test_batch_size(self) -> int:
        return int(self.payload["test_batch_size"])

    def identity(self) -> dict[str, Any]:
        return {
            "training_batch_profile_id": self.profile_id,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "test_batch_size": self.test_batch_size,
            "gradient_accumulation_steps": int(self.payload["gradient_accumulation_steps"]),
            "effective_train_batch_size": int(self.payload["effective_train_batch_size"]),
            "amp_enabled": bool(self.payload["amp_enabled"]),
        }


def _validate(payload: dict[str, Any]) -> None:
    required = {
        "schema_version", "profile_id", "train_batch_size", "val_batch_size",
        "test_batch_size", "gradient_accumulation_steps",
        "effective_train_batch_size", "amp_enabled", "created_for",
        "selection_reason", "compatibility_policy",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ProtocolError(f"Training batch profile is missing keys: {missing}")
    if payload["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ProtocolError("Unsupported training batch profile schema.")
    if payload["profile_id"] not in PROFILE_ALLOWLIST:
        raise ProtocolError(f"Training batch profile is not allowlisted: {payload['profile_id']}")
    expected = {
        "train_batch_size": 4, "val_batch_size": 4, "test_batch_size": 4,
        "gradient_accumulation_steps": 1, "effective_train_batch_size": 4,
        "amp_enabled": True,
    }
    mismatches = {key: {"actual": payload.get(key), "expected": value} for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise ProtocolError(f"Invalid uniform batch4 profile: {mismatches}")


def load_training_profile(profile_id: str | None) -> TrainingBatchProfile | None:
    if profile_id in (None, ""):
        return None
    if profile_id not in PROFILE_ALLOWLIST:
        raise ProtocolError(f"Unknown training batch profile {profile_id!r}; allowed={PROFILE_ALLOWLIST}")
    payload = json.loads((PROFILE_ROOT / f"{profile_id}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("Training batch profile JSON root must be an object.")
    _validate(payload)
    return TrainingBatchProfile(payload=payload)


def batch_identity(profile_id: str | None) -> dict[str, Any]:
    profile = load_training_profile(profile_id)
    return {} if profile is None else profile.identity()


def apply_training_profile(runtime, profile_id: str | None):
    profile = load_training_profile(profile_id)
    if profile is not None:
        runtime.effective_config.update(profile.identity())
    return runtime


def resolved_batch_sizes(protocol: Mapping[str, Any], profile_id: str | None) -> dict[str, int]:
    profile = load_training_profile(profile_id)
    if profile is None:
        return {"train": int(protocol["train_batch_size"]), "val": int(protocol["val_batch_size"]), "test": int(protocol["test_batch_size"])}
    return {"train": profile.train_batch_size, "val": profile.val_batch_size, "test": profile.test_batch_size}
