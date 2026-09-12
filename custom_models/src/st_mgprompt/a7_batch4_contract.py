from __future__ import annotations

import json
from pathlib import Path
from typing import Any


A7_SCOPE_ID = "st_mgprompt_a7_batch4_seed2026"
A7_MODEL_ID = "st_mgprompt_a7"
A7_VARIANT = "A7"
A7_DEFINITION = "w/o MS-MG-DWU"
A7_RUN_ID = "component_ablation_a7_bs4_seed2026"
A7_OUTPUT_ROOT = "custom_models/results/st_mgprompt_uniform_bs4"
A7_RUN_RELATIVE_PATH = f"{A7_OUTPUT_ROOT}/{A7_RUN_ID}/STMGPrompt_ComponentAblation"
A7_REFERENCE_ID = "STMGPrompt_A7_loss_msa_hybrid_bs4_seed2026"
A7_REFERENCE_RELATIVE_PATH = "custom_models/logs/uniform_bs4/audit/a7/A7_BATCH4_REFERENCE.json"
TRAINING_ROLE = "ST_MGPROMPT_COMPONENT_ABLATION"
TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
LOSS_ID = "masked_score_aligned_hybrid"
LOSS_PROTOCOL = "fair_main"
PRECISION_POLICY = "fp32"
DATA_SIGNATURE_SCHEMA_VERSION = "st_mgprompt_a7_data_signature_v2"
DATASET_ID = "SDWPF"
INPUT_RELATIVE_PATH = "dataset/sdwpf_model_input_base.parquet"
TARGET_RELATIVE_PATH = "dataset/sdwpf_eval_target.parquet"
TARGET_COL = "Patv_raw"
INPUT_PATV_COL = "Patv_clean_for_input"
TARGET_MASK_COL = "valid_target_mask"
FEATURE_ORDER = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp", "Patv_clean_for_input"]
DATA_SPLIT_RATIOS = [0.8, 0.1, 0.1]
DATA_STRIDES = {"train": 6, "val": 3, "test": 1}

EXPECTED_CONFIG = {
    "scope_id": A7_SCOPE_ID, "model_id": A7_MODEL_ID, "run_id": A7_RUN_ID,
    "component_ablation": A7_VARIANT, "variant": A7_VARIANT,
    "definition": A7_DEFINITION,
    "training_batch_profile_id": TRAINING_PROFILE_ID, "train_batch_size": 4,
    "val_batch_size": 4, "test_batch_size": 4,
    "gradient_accumulation_steps": 1, "seed": 2026, "lookback": 144,
    "max_pred_len": 10, "loss_function": LOSS_ID,
    "precision_policy": PRECISION_POLICY, "amp_enabled": False,
    "formal_training": True,
}


class A7ContractError(ValueError):
    pass


def _repo_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root or Path(__file__).resolve().parents[3]).resolve()


def graph_identity(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(project_root)
    protocol = json.loads((root / "custom_models/src/benchmark_v2/protocol/graph_protocol_v1.json").read_text(encoding="utf-8"))
    order = json.loads((root / "custom_models/src/benchmark_v2/protocol/graph_v1/node_order_v1.json").read_text(encoding="utf-8"))
    matrix_files = protocol.get("matrix_files", {})
    return {
        "graph_id": protocol.get("graph_id"), "node_count": protocol.get("node_count"),
        "selected_k": protocol.get("selected_k"),
        "ordered_node_ids": order.get("ordered_node_ids"),
        "matrix_names": sorted(matrix_files),
        "matrix_shapes": {name: value.get("shape") for name, value in matrix_files.items()},
    }


def variant_contract() -> dict[str, Any]:
    return {
        "scope_id": A7_SCOPE_ID, "model_id": A7_MODEL_ID, "run_id": A7_RUN_ID,
        "variant": A7_VARIANT, "definition": A7_DEFINITION,
        "training_role": TRAINING_ROLE, "training_profile_id": TRAINING_PROFILE_ID,
        "batch_size": 4, "lookback": 144, "node_count": 134,
        "feature_count": len(FEATURE_ORDER), "horizon": 10,
        "loss_id": LOSS_ID, "precision": PRECISION_POLICY,
    }


def loss_identity(project_root: str | Path | None = None) -> dict[str, Any]:
    del project_root
    return {"loss_id": LOSS_ID, "loss_protocol": LOSS_PROTOCOL, "source_function": "st_mgprompt.losses.masked_score_aligned_hybrid_loss"}


def precision_identity() -> dict[str, Any]:
    return {"precision_policy": PRECISION_POLICY, "amp_enabled": False}


__all__ = [name for name in globals() if name.isupper()] + ["A7ContractError", "graph_identity", "loss_identity", "precision_identity", "variant_contract"]
