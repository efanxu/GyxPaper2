from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ArtifactError


GRAPH_IDENTITY_KEYS = (
    "graph_id",
    "graph_protocol_hash",
    "node_order_hash",
    "graph_bundle_hash",
    "location_source_hash",
    "selected_k",
    "graph_support_names",
    "graph_support_hashes",
    "graph_runtime_dtype",
    "graph_context_id",
    "uses_physical_support",
    "physical_support_names",
    "physical_support_hashes",
    "adaptive_graph_policy",
    "node_identity_policy",
    "temporal_identity_policy",
    "adaptive_initialization_policy",
)
BATCH_IDENTITY_KEYS = (
    "base_benchmark_protocol_hash",
    "training_batch_profile_id",
    "training_batch_profile_hash",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "gradient_accumulation_steps",
    "effective_train_batch_size",
)


def config_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rng_state() -> dict[str, Any]:
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


class CheckpointManager:
    def __init__(self, run_dir: str | Path, *, protocol_hash: str, model_id: str, resolved_config: dict[str, Any], effective_config: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_hash = protocol_hash
        self.model_id = model_id
        self.resolved_hash = config_hash(resolved_config)
        self.effective_hash = config_hash(effective_config)
        self.graph_identity = {
            key: effective_config[key]
            for key in GRAPH_IDENTITY_KEYS
            if key in effective_config
        }
        self.batch_identity = {
            key: effective_config[key]
            for key in BATCH_IDENTITY_KEYS
            if key in effective_config
        }

    def save(self, name: str, *, epoch: int, global_step: int, monitor_value: float, model, optimizer=None, scheduler=None, amp_scaler=None, seed_state: dict[str, Any] | None = None) -> Path:
        import torch
        payload = {
            "schema_version": "checkpoint_schema_v1", "run_id": self.run_dir.name, "model_id": self.model_id,
            "protocol_hash": self.protocol_hash, "resolved_config_hash": self.resolved_hash, "effective_config_hash": self.effective_hash,
            "graph_identity": dict(self.graph_identity),
            "batch_identity": dict(self.batch_identity),
            "epoch": int(epoch), "global_step": int(global_step), "monitor_name": "validation_official_score_h10", "monitor_value": float(monitor_value),
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None, "amp_scaler_state_dict": amp_scaler.state_dict() if amp_scaler is not None else None,
            "seed_state": seed_state or _rng_state(), "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.run_dir / name
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load(self, path: str | Path, model, optimizer=None, scheduler=None, amp_scaler=None) -> dict[str, Any]:
        import torch
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        expected = {"model_id": self.model_id, "protocol_hash": self.protocol_hash, "resolved_config_hash": self.resolved_hash, "effective_config_hash": self.effective_hash, "graph_identity": self.graph_identity}
        if self.batch_identity:
            expected["batch_identity"] = self.batch_identity
        mismatches = {k: (payload.get(k), v) for k, v in expected.items() if payload.get(k) != v}
        if mismatches:
            raise ArtifactError(f"Checkpoint resume hash/model mismatch: {mismatches}")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if amp_scaler is not None and payload.get("amp_scaler_state_dict") is not None:
            amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
        return payload
