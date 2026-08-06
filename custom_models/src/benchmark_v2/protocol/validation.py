from __future__ import annotations

from typing import Any, Mapping

from ..errors import ProtocolError


REQUIRED_FIELDS = (
    "protocol_id", "protocol_version", "dataset_id", "sample_interval_minutes", "expected_node_count",
    "target_column", "input_power_column", "mask_column", "target_unit", "ordered_input_features",
    "feature_count", "input_tensor_shape", "prediction_tensor_shape",
    "target_tensor_shape", "mask_tensor_shape", "lookback", "max_pred_len", "eval_horizons",
    "split_ratio", "split_mode", "window_boundary_policy", "train_sample_stride", "val_sample_stride",
    "test_sample_stride", "train_batch_size", "val_batch_size", "test_batch_size", "epochs",
    "early_stopping_patience", "early_stopping_min_delta", "default_seed", "amp_enabled",
    "physical_power_min_kw", "physical_power_max_kw", "checkpoint_metric", "checkpoint_horizon",
    "checkpoint_direction", "normalization_method", "normalization_fit_scope", "target_loss_space",
    "future_observed_covariates", "future_calendar_covariates", "non_graph_node_semantics",
    "formal_output_root", "smoke_output_root", "score_definition_version", "metrics_contract_version",
    "artifact_schema_version",
)


def check_protocol(protocol: Mapping[str, Any], mode: str = "formal") -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in protocol]
    if missing:
        raise ProtocolError(f"Missing protocol fields: {missing}")
    features = list(protocol["ordered_input_features"])
    if len(features) != 16 or protocol["feature_count"] != 16:
        raise ProtocolError("Formal benchmark protocol requires exactly 16 ordered input features.")
    if protocol["target_column"] != "Patv_raw" or protocol["input_power_column"] != "Patv_clean_for_input":
        raise ProtocolError("Target/input power columns do not match the frozen SDWPF contract.")
    if protocol["mask_column"] != "valid_target_mask":
        raise ProtocolError("Mask column does not match the frozen SDWPF contract.")
    if list(protocol["eval_horizons"]) != [3, 6, 10]:
        raise ProtocolError("Formal horizons must be [3, 6, 10].")
    if list(protocol["split_ratio"]) != [0.8, 0.1, 0.1]:
        raise ProtocolError("Formal split ratio must be [0.8, 0.1, 0.1].")
    if protocol["split_mode"] != "strict_chronological":
        raise ProtocolError("Formal split mode must be strict_chronological.")
    if protocol["future_observed_covariates"] != "disabled" or protocol["future_calendar_covariates"] != "disabled":
        raise ProtocolError("Future covariates must remain disabled in E0-B.")
    if protocol["checkpoint_horizon"] != 10 or protocol["checkpoint_direction"] != "lower_is_better":
        raise ProtocolError("Checkpoint policy must monitor validation official Score H10, lower is better.")
    if mode not in {"formal", "smoke"}:
        raise ProtocolError(f"Unknown protocol mode: {mode}")
    return {
        "status": "PASS",
        "mode": mode,
        "fixed_fields_checked": list(REQUIRED_FIELDS),
        "formal_shape": mode == "formal",
    }
