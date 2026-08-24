from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ArtifactError


EXPLICIT_CONFIG_KEYS = (
    "scope_id", "model_id", "run_id", "entry_type", "seed", "loss_id",
    "precision_policy", "train_batch_size", "val_batch_size", "test_batch_size",
    "gradient_accumulation_steps", "formal_training", "artifact_profile",
)


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


class CheckpointManager:
    def __init__(self, run_dir: str | Path, *, protocol_id: str, model_id: str, resolved_config: dict[str, Any], effective_config: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_id = protocol_id
        self.model_id = model_id
        self.resolved_config = dict(resolved_config)
        self.run_metadata = {key: effective_config[key] for key in EXPLICIT_CONFIG_KEYS if key in effective_config}

    def save(self, name: str, *, epoch: int, global_step: int, monitor_value: float, model, loss_fn=None, optimizer=None, scheduler=None, amp_scaler=None, seed_state: dict[str, Any] | None = None) -> Path:
        import torch
        payload = {
            "schema_version": "checkpoint_schema_v2",
            "run_id": self.run_dir.name,
            "model_id": self.model_id,
            "protocol_id": self.protocol_id,
            "resolved_config": self.resolved_config,
            "run_metadata": self.run_metadata,
            "epoch": int(epoch), "global_step": int(global_step),
            "monitor_name": "validation_official_score_h10",
            "monitor_value": float(monitor_value),
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss_fn.state_dict() if hasattr(loss_fn, "state_dict") else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "amp_scaler_state_dict": amp_scaler.state_dict() if amp_scaler is not None else None,
            "seed_state": seed_state or _rng_state(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.run_dir / name
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return path

    def load(self, path: str | Path, model, loss_fn=None, optimizer=None, scheduler=None, amp_scaler=None) -> dict[str, Any]:
        import torch
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("model_id") not in (None, self.model_id):
            raise ArtifactError("Checkpoint model_id conflicts with the requested model.")
        if payload.get("run_id") not in (None, self.run_dir.name):
            raise ArtifactError("Checkpoint run_id conflicts with the requested run.")
        stored = payload.get("run_metadata")
        if isinstance(stored, dict):
            conflicts = {key: (stored.get(key), value) for key, value in self.run_metadata.items() if key in stored and stored.get(key) != value}
            if conflicts:
                raise ArtifactError(f"Checkpoint explicit configuration conflict: {conflicts}")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if loss_fn is not None and payload.get("loss_state_dict") is not None:
            loss_fn.load_state_dict(payload["loss_state_dict"], strict=True)
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if amp_scaler is not None and payload.get("amp_scaler_state_dict") is not None:
            amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
        return payload
