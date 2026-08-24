from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .constants import E9_SCOPE_ID, LOSS_ID, LOSS_PROFILE, MODEL_IDS, TRAINING_PROFILE_ID, TRANSFER_ROOT
from .variants import canonical_transfer_run_id

ALLOWED_EXACT = {
    "loss", "loss_id", "loss_profile", "loss_protocol", "loss_space",
    "run_id", "output_root", "scope_id", "entry_id", "current_scope_entry_id",
    "active_scope_id", "current_scope_manifest_hash", "formal_worker_pid",
    "control_run_reference", "e9_provenance", "loss_state_policy",
}
ALLOWED_PREFIXES = ("loss_profile.", "e9_provenance.", "loss_state_policy.")
PLATFORM_PATH_FIELDS = {"upstream_source_path", "upstream_license_path"}


def build_transfer_config(control_effective: dict[str, Any], model_id: str, output_root: str | Path = TRANSFER_ROOT) -> dict[str, Any]:
    if model_id not in MODEL_IDS:
        raise ValueError(f"Model is outside the frozen E9 scope: {model_id}")
    result = deepcopy(control_effective)
    result.update({
        "loss": LOSS_ID,
        "loss_id": LOSS_ID,
        "loss_profile": deepcopy(LOSS_PROFILE),
        "loss_protocol": "e9_loss_only_transfer",
        "run_id": canonical_transfer_run_id(model_id),
        "output_root": str(Path(output_root)),
        "scope_id": E9_SCOPE_ID,
        "entry_id": f"e9_transfer_{model_id}",
        "formal_training": True,
        "training_batch_profile_id": TRAINING_PROFILE_ID,
        "control_run_reference": {
            "run_id": control_effective.get("run_id"),
            "loss_id": control_effective.get("loss"),
            "read_only": True,
        },
        "loss_state_policy": {
            "fit_split": "train", "validation_test_frozen": True,
            "model_output_dependent": True, "algorithm_shared": True, "values_shared": False,
        },
        "e9_provenance": {
            "experiment": "E9-B", "comparison": "loss-only",
            "e5_consumed": False, "control_retrained": False,
        },
    })
    result.pop("active_scope_id", None)
    result.pop("current_scope_entry_id", None)
    result.pop("current_scope_manifest_hash", None)
    result.pop("formal_worker_pid", None)
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def _allowed(path: str) -> bool:
    top = path.split(".", 1)[0]
    return top in ALLOWED_EXACT or path in ALLOWED_EXACT or path.startswith(ALLOWED_PREFIXES)


def compare_configs(control: dict[str, Any], transfer: dict[str, Any], *, graph_model: bool = False) -> dict[str, Any]:
    left, right = _flatten(control), _flatten(transfer)
    differences = []
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            differences.append({"path": path, "control": left.get(path), "transfer": right.get(path), "allowed": _allowed(path)})
    unexpected = [row for row in differences if not row["allowed"]]
    loss_changed = control.get("loss") == "masked_mse" and transfer.get("loss") == LOSS_ID
    model_source_match = all(
        control.get(key) == transfer.get(key)
        for key in ("model_id", "source_closure_hash", "model_config_hash")
        if control.get(key) is not None or transfer.get(key) is not None
    )
    architecture_keys = ("model_id", "wrapper_path", "adapter_path", "implementation_class")
    architecture_match = all(control.get(key) == transfer.get(key) for key in architecture_keys if key in control or key in transfer)
    data_keys = ("target_column", "selected_power_feature", "target_col", "input_patv_col", "seq_len", "lookback", "pred_len", "horizon", "enc_in", "input_dim")
    data_match = all(control.get(key) == transfer.get(key) for key in data_keys if key in control or key in transfer)
    training_keys = (
        "optimizer", "learning_rate", "weight_decay", "scheduler", "epochs", "patience", "min_delta", "seed",
        "amp_enabled", "train_batch_size", "val_batch_size", "test_batch_size", "gradient_accumulation_steps",
        "checkpoint_metric", "checkpoint_direction", "training_batch_profile_id",
    )
    training_match = all(control.get(key) == transfer.get(key) for key in training_keys if key in control or key in transfer)
    graph_keys = ("graph_id", "graph_context_id", "ordered_node_ids", "selected_k", "graph_support_names", "graph_support_shapes", "adaptive_graph_policy")
    graph_match = (not graph_model) or all(control.get(key) == transfer.get(key) for key in graph_keys if key in control or key in transfer)
    valid = not unexpected and loss_changed and model_source_match and architecture_match and data_match and training_match and graph_match
    return {
        "allowed_differences": [row for row in differences if row["allowed"]],
        "unexpected_differences": unexpected,
        "model_source_match": model_source_match,
        "model_architecture_match": architecture_match,
        "model_hyperparameter_match": not any(
            not row["allowed"] and any(token in row["path"].lower() for token in ("hidden", "layer", "patch", "dropout", "graph", "subgraph", "kernel", "d_model"))
            for row in differences
        ),
        "data_protocol_match": data_match,
        "training_protocol_match": training_match,
        "graph_protocol_match": graph_match,
        "loss_identity_changed": loss_changed,
        "loss_only_diff_valid": valid,
    }


def build_config_diff_audit(control_audit: dict[str, Any], output_root: str | Path = TRANSFER_ROOT) -> dict[str, Any]:
    controls = {row["model_id"]: row for row in control_audit["controls"]}
    pairs = []
    for model_id in MODEL_IDS:
        control = controls[model_id]
        transfer = build_transfer_config(control.get("effective_config", {}), model_id, output_root)
        diff = compare_configs(control.get("effective_config", {}), transfer, graph_model=model_id in {"dcrnn", "mtgnn"})
        pairs.append({
            "model_id": model_id, "control_run_id": control.get("run_id"),
            "transfer_run_id": canonical_transfer_run_id(model_id), **diff,
            "pair_status": "PASS" if diff["loss_only_diff_valid"] else "INVALID_CONFIG_DIFF",
            "proposed_effective_config": transfer,
        })
    ready = sum(row["loss_only_diff_valid"] for row in pairs)
    return {
        "schema_version": "e9_loss_only_config_diff_v1",
        "ready_count": ready, "LOSS_ONLY_CONFIG_READY": f"{ready}/6", "pairs": pairs,
    }
