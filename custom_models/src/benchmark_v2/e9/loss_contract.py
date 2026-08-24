from __future__ import annotations

from typing import Any


def loss_input_contract() -> dict[str, Any]:
    fields = [
        {
            "field_name": "prediction", "shape": ["B", "N", "H"], "dtype": "floating",
            "semantic": "model forecast in normalized target space", "source": "BenchmarkOutput.prediction",
            "gradient_required": True, "training_fitted": False, "model_specific": False,
            "normalization_space": "normalized_target_space", "mask_semantics": "weighted only where mask=true",
            "validation_rule": "exactly matches target and mask; H=10",
        },
        {
            "field_name": "target", "shape": ["B", "N", "H"], "dtype": "floating",
            "semantic": "Patv_raw transformed by train-only target scaler", "source": "BenchmarkBatch.target",
            "gradient_required": False, "training_fitted": False, "model_specific": False,
            "normalization_space": "normalized_target_space", "mask_semantics": "invalid target is excluded",
            "validation_rule": "exactly matches prediction and mask",
        },
        {
            "field_name": "mask", "shape": ["B", "N", "H"], "dtype": "bool or explicit 0/1",
            "semantic": "valid_target_mask", "source": "BenchmarkBatch.mask", "gradient_required": False,
            "training_fitted": False, "model_specific": False, "normalization_space": "not_applicable",
            "mask_semantics": "true/1 participates; all-masked returns None and skips optimizer step",
            "validation_rule": "only bool/0/1 values; exact tensor alignment",
        },
        {
            "field_name": "site_id", "shape": ["N"], "dtype": "canonical identifier",
            "semantic": "frozen node axis used by site EMA weights", "source": "BenchmarkBatch.node_ids",
            "gradient_required": False, "training_fitted": False, "model_specific": False,
            "normalization_space": "not_applicable", "mask_semantics": "site valid iff any valid element exists",
            "validation_rule": "length equals N; graph models preserve ordered node ids 1..134",
        },
        {
            "field_name": "horizon_index", "shape": ["H"], "dtype": "int",
            "semantic": "one-based forecast offsets with prefixes H3/H6/H10", "source": "public protocol",
            "gradient_required": False, "training_fitted": False, "model_specific": False,
            "normalization_space": "not_applicable", "mask_semantics": "prefix loss uses matching mask prefix",
            "validation_rule": "exactly 1..10; eval_horizons exactly [3,6,10]",
        },
        {
            "field_name": "normalization_identity", "shape": [], "dtype": "string",
            "semantic": "identity of train-only standard z-score target space", "source": "formal protocol",
            "gradient_required": False, "training_fitted": True, "model_specific": False,
            "normalization_space": "train_only standard_zscore", "mask_semantics": "fit excludes invalid targets",
            "validation_rule": "target_loss_space=normalized_target_space and normalization_fit_scope=train_only",
        },
        {
            "field_name": "protocol_identity", "shape": [], "dtype": "string",
            "semantic": "sdwpf_benchmark_v2 plus uniform_train_batch4_v1", "source": "formal protocol/profile",
            "gradient_required": False, "training_fitted": False, "model_specific": False,
            "normalization_space": "not_applicable", "mask_semantics": "not_applicable",
            "validation_rule": "protocol and batch profile must match control",
        },
    ]
    return {
        "schema_version": "e9_loss_input_contract_v1",
        "contract_type": "LossInputBundle",
        "tensor_order": "(B,N,H)",
        "fields": fields,
        "forbidden_fields": [
            "hidden_states", "attention", "graph_embedding", "Fine representation", "Coarse representation",
            "Prompt representation", "Cross gate", "model-specific auxiliary tensor",
        ],
        "contract_hash": "NOT_GENERATED_REPOSITORY_POLICY",
        "validation_status": "PASS",
    }


def loss_profile_manifest(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e9_loss_profile_v1",
        "profile": dict(profile),
        "source_hash": "NOT_GENERATED_REPOSITORY_POLICY",
        "profile_hash": "NOT_GENERATED_REPOSITORY_POLICY",
        "contract_hash": "NOT_GENERATED_REPOSITORY_POLICY",
        "identity_note": "Repository instructions prohibit generating new hashes; path/symbol/profile values are recorded directly.",
    }


def loss_state_manifest() -> dict[str, Any]:
    return {
        "schema_version": "e9_loss_state_manifest_v1",
        "fit_scope": "training batches only",
        "fit_range": "formal SDWPF train split (strict chronological first 0.8)",
        "uses_validation": False,
        "uses_test": False,
        "uses_test_target": False,
        "updates_in_validation": False,
        "updates_in_test": False,
        "model_output_dependent_state": True,
        "state_shared_across_models": False,
        "algorithm_shared_across_models": True,
        "state_fields": [
            {"name": "ema_granularity_loss", "shape": [3], "initial": 1.0},
            {"name": "initial_granularity_loss", "shape": [3], "initial": 1.0},
            {"name": "initial_granularity_fitted", "shape": [3], "initial": False},
            {"name": "ema_node_loss", "shape": [134], "initial": 1.0},
            {"name": "node_weight", "shape": [134], "initial": 1.0},
        ],
        "checkpointed": True,
        "state_hash": "AVAILABLE_AFTER_FORMAL_RUN_NOT_GENERATED_BY_E9",
        "fitting_source_hash": "NOT_GENERATED_REPOSITORY_POLICY",
        "leakage_audit": "PASS",
    }
