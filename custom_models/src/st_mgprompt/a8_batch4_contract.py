"""The current, formal ST-MGPrompt A8 Batch4 contract.

This module is deliberately independent from the historical A8 result and
from the E5 benchmark manifest.  It is used by the A8 gate and by the formal
runner to calculate the identities that a future E5 run is allowed to read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


A8_SCOPE_ID = "st_mgprompt_a8_batch4_prerequisite_seed2026"
A8_MODEL_ID = "st_mgprompt_a8"
A8_VARIANT = "A8"
A8_DEFINITION = "w/o MS-MG-DWU"
A8_RUN_ID = "component_ablation_a8_bs4_seed2026"
A8_OUTPUT_ROOT = "custom_models/results/st_mgprompt_uniform_bs4"
A8_RUN_RELATIVE_PATH = (
    f"{A8_OUTPUT_ROOT}/{A8_RUN_ID}/STMGPrompt_ComponentAblation"
)
A8_REFERENCE_ID = "STMGPrompt_A8_loss_msa_hybrid_bs4_seed2026_reference"
A8_REFERENCE_RELATIVE_PATH = (
    "custom_models/logs/uniform_bs4/audit/e5_scope27/"
    "E5_A8_BATCH4_REFERENCE.json"
)
TRAINING_ROLE = "E5_BATCH4_PREREQUISITE"
TRAINING_PROFILE_ID = "uniform_train_batch4_v1"
LOSS_ID = "masked_score_aligned_hybrid"
LOSS_PROTOCOL = "fair_main"
PRECISION_POLICY = "fp32"

EXPECTED_CONFIG: dict[str, Any] = {
    "model_id": A8_MODEL_ID,
    "model_name": "STMGPrompt_ComponentAblation",
    "variant": A8_VARIANT,
    "component_ablation": A8_VARIANT,
    "definition": A8_DEFINITION,
    "use_msmg_dwu": False,
    "loss_function": LOSS_ID,
    "loss_protocol": LOSS_PROTOCOL,
    "seed": 2026,
    "lookback": 144,
    "max_pred_len": 10,
    "eval_horizons": [3, 6, 10],
    "split_ratios": [0.8, 0.1, 0.1],
    "train_sample_stride": 6,
    "val_sample_stride": 3,
    "test_sample_stride": 1,
    "train_batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "effective_train_batch_size": 4,
    "epochs": 20,
    "patience": 6,
    "early_stopping_min_delta": 0.01,
    "checkpoint_selection_metric": "val_official_score_h10",
    "physical_power_min_kw": 0.0,
    "physical_power_max_kw": 1500.0,
    "amp_enabled": False,
    "precision_policy": PRECISION_POLICY,
    "training_role": TRAINING_ROLE,
    "trained_for_e5_scope27": True,
    "consumed_read_only_by_e5": True,
    "checkpoint_copied": False,
    "metrics_copied": False,
    "warm_started_from_historical_a8": False,
}

# The closure is explicit.  Keeping the list here makes a new dynamic import
# an actionable contract change instead of silently changing the experiment.
A8_SOURCE_CLOSURE_RELATIVE_PATHS: tuple[str, ...] = (
    "custom_models/src/st_mgprompt/__init__.py",
    "custom_models/src/st_mgprompt/artifact_status.py",
    "custom_models/src/st_mgprompt/canonical_artifact.py",
    "custom_models/src/st_mgprompt/check_protocol.py",
    "custom_models/src/st_mgprompt/config.py",
    "custom_models/src/st_mgprompt/coupling_block.py",
    "custom_models/src/st_mgprompt/cross_fusion.py",
    "custom_models/src/st_mgprompt/data.py",
    "custom_models/src/st_mgprompt/decoder.py",
    "custom_models/src/st_mgprompt/diagnostics.py",
    "custom_models/src/st_mgprompt/evaluate.py",
    "custom_models/src/st_mgprompt/experiment_protocol.py",
    "custom_models/src/st_mgprompt/graph_layers.py",
    "custom_models/src/st_mgprompt/graph_prior.py",
    "custom_models/src/st_mgprompt/losses.py",
    "custom_models/src/st_mgprompt/metrics.py",
    "custom_models/src/st_mgprompt/model.py",
    "custom_models/src/st_mgprompt/prompt_alignment.py",
    "custom_models/src/st_mgprompt/registry.py",
    "custom_models/src/st_mgprompt/run_st_mgprompt.py",
    "custom_models/src/st_mgprompt/temporal_layers.py",
    "custom_models/src/st_mgprompt/train.py",
    "custom_models/src/st_mgprompt/volatility_patching.py",
    "custom_models/src/benchmark_v2/training_profiles/__init__.py",
    "custom_models/src/benchmark_v2/training_profiles/uniform_train_batch4_v1.json",
)

GRAPH_PROTOCOL_RELATIVE_PATH = "custom_models/src/benchmark_v2/protocol/graph_protocol_v1.json"
GRAPH_BUNDLE_RELATIVE_PATH = (
    "custom_models/src/benchmark_v2/protocol/graph_v1/graph_bundle_manifest_v1.json"
)
GRAPH_NODE_ORDER_RELATIVE_PATH = (
    "custom_models/src/benchmark_v2/protocol/graph_v1/node_order_v1.json"
)
GRAPH_NODE_METADATA_RELATIVE_PATH = (
    "custom_models/src/benchmark_v2/protocol/graph_v1/node_metadata_v1.json"
)
LOCATION_RELATIVE_PATH = "dataset/sdwpf_turb_location_elevation.csv"
TREND_RESOURCE_RELATIVE_PATH = "dataset/sdwpf_model_input_base.parquet"
DELTA_RESOURCE_RELATIVE_PATH = "dataset/sdwpf_model_input_base.parquet"

GRAPH_PARAMETERS: dict[str, Any] = {
    "node_count": 134,
    "coordinate_columns": ["x", "y"],
    "elevation_used_in_edge_distance": False,
    "macro_graph_source": "causal_trend_similarity",
    "micro_graph_source": "distance_delta_wspd",
    "macro_top_k": 10,
    "micro_top_k": 5,
    "macro_alpha": 0.7,
    "micro_alpha": 0.3,
    "macro_similarity_metric": "pearson",
    "micro_similarity_metric": "cosine",
    "trend_method": "causal_moving_average",
    "trend_window": 36,
    "delta_definition": "Wspd[t] - Wspd[t-1], train split only",
    "fit_split": "train_only",
}


class A8ContractError(ValueError):
    """Raised when the current A8 contract cannot be proven."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_relative(root: Path, relative_path: str) -> Path:
    raw = relative_path.replace("\\", "/")
    if not raw or raw.startswith("/") or ":" in raw.split("/", 1)[0]:
        raise A8ContractError(f"A8 contract path is not repository-relative: {relative_path}")
    candidate = (root / Path(raw)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise A8ContractError(f"A8 contract path escapes repository: {relative_path}") from exc
    return candidate


def source_closure(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(project_root)
    records: list[dict[str, str]] = []
    for relative_path in A8_SOURCE_CLOSURE_RELATIVE_PATHS:
        path = _resolve_relative(root, relative_path)
        if not path.is_file() or path.is_symlink():
            raise A8ContractError(f"A8 source closure file is missing or unsafe: {relative_path}")
        records.append(
            {"path": relative_path.replace("\\", "/"), "sha256": sha256_file(path)}
        )
    records.sort(key=lambda item: item["path"])
    return {
        "schema_version": "st_mgprompt_a8_source_closure_v1",
        "dynamic_imports": [],
        "files": records,
        "canonical_combined_hash": canonical_hash(records),
    }


def _resource_record(root: Path, relative_path: str) -> dict[str, str]:
    path = _resolve_relative(root, relative_path)
    if not path.is_file() or path.is_symlink():
        raise A8ContractError(f"A8 graph resource is missing or unsafe: {relative_path}")
    return {"path": relative_path.replace("\\", "/"), "sha256": sha256_file(path)}


def graph_identity(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(project_root)
    protocol_path = _resolve_relative(root, GRAPH_PROTOCOL_RELATIVE_PATH)
    bundle_path = _resolve_relative(root, GRAPH_BUNDLE_RELATIVE_PATH)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise A8ContractError(f"A8 graph protocol cannot be read: {exc}") from exc
    if not isinstance(protocol, dict) or not isinstance(bundle, dict):
        raise A8ContractError("A8 graph protocol and bundle must be JSON objects.")
    source = source_closure(root)
    resource_paths = (
        GRAPH_PROTOCOL_RELATIVE_PATH,
        GRAPH_BUNDLE_RELATIVE_PATH,
        GRAPH_NODE_ORDER_RELATIVE_PATH,
        GRAPH_NODE_METADATA_RELATIVE_PATH,
        LOCATION_RELATIVE_PATH,
        TREND_RESOURCE_RELATIVE_PATH,
        DELTA_RESOURCE_RELATIVE_PATH,
    )
    resources = [_resource_record(root, value) for value in resource_paths]
    generation_protocol = {
        "schema_version": "st_mgprompt_a8_graph_generation_protocol_v1",
        "graph_prior_source_hash": next(
            record["sha256"]
            for record in source["files"]
            if record["path"] == "custom_models/src/st_mgprompt/graph_prior.py"
        ),
        "parameters": GRAPH_PARAMETERS,
        "resource_paths": [record["path"] for record in resources],
    }
    generation_protocol_hash = canonical_hash(generation_protocol)
    resource_hashes = {record["path"]: record["sha256"] for record in resources}
    macro_material = {
        "source": GRAPH_PARAMETERS["macro_graph_source"],
        "generation_protocol_hash": generation_protocol_hash,
        "trend_resource_hash": resource_hashes[TREND_RESOURCE_RELATIVE_PATH],
        "location_elevation_resource_hash": resource_hashes[LOCATION_RELATIVE_PATH],
        "parameters": {
            key: GRAPH_PARAMETERS[key]
            for key in (
                "macro_graph_source",
                "macro_top_k",
                "macro_alpha",
                "macro_similarity_metric",
                "trend_method",
                "trend_window",
            )
        },
    }
    micro_material = {
        "source": GRAPH_PARAMETERS["micro_graph_source"],
        "generation_protocol_hash": generation_protocol_hash,
        "delta_resource_hash": resource_hashes[DELTA_RESOURCE_RELATIVE_PATH],
        "location_elevation_resource_hash": resource_hashes[LOCATION_RELATIVE_PATH],
        "parameters": {
            key: GRAPH_PARAMETERS[key]
            for key in (
                "micro_graph_source",
                "micro_top_k",
                "micro_alpha",
                "micro_similarity_metric",
                "delta_definition",
            )
        },
    }
    graph_protocol_hash = protocol.get("graph_protocol_hash")
    node_order_hash = bundle.get("node_order_hash") or protocol.get("node_order_hash")
    graph_bundle_hash = bundle.get("graph_bundle_hash")
    if not all(isinstance(value, str) and value for value in (graph_protocol_hash, node_order_hash, graph_bundle_hash)):
        raise A8ContractError("A8 benchmark graph identity is incomplete.")
    return {
        "schema_version": "st_mgprompt_a8_macro_micro_graph_identity_v1",
        "graph_contract_id": "st_mgprompt_macro_micro_graph_v1",
        "macro_graph_source": GRAPH_PARAMETERS["macro_graph_source"],
        "micro_graph_source": GRAPH_PARAMETERS["micro_graph_source"],
        "macro_graph_hash": canonical_hash(macro_material),
        "micro_graph_hash": canonical_hash(micro_material),
        "node_count": int(bundle.get("node_count", GRAPH_PARAMETERS["node_count"])),
        "node_order_hash": node_order_hash,
        "location_elevation_resource_hash": resource_hashes[LOCATION_RELATIVE_PATH],
        "trend_resource_hash": resource_hashes[TREND_RESOURCE_RELATIVE_PATH],
        "delta_resource_hash": resource_hashes[DELTA_RESOURCE_RELATIVE_PATH],
        "graph_parameters": dict(GRAPH_PARAMETERS),
        "graph_generation_protocol_hash": generation_protocol_hash,
        "graph_protocol_hash": graph_protocol_hash,
        "benchmark_graph_bundle_hash": graph_bundle_hash,
        "resource_paths": resources,
    }


def variant_contract() -> dict[str, Any]:
    return {
        "schema_version": "st_mgprompt_a8_batch4_variant_contract_v1",
        "scope_id": A8_SCOPE_ID,
        "model_id": A8_MODEL_ID,
        "variant": A8_VARIANT,
        "definition": A8_DEFINITION,
        "component_ablation": A8_VARIANT,
        "use_msmg_dwu": False,
        "loss_function": LOSS_ID,
        "loss_protocol": LOSS_PROTOCOL,
        "training_profile_id": TRAINING_PROFILE_ID,
        "precision_policy": PRECISION_POLICY,
        "amp_enabled": False,
        "warm_start_prohibited": True,
        "checkpoint_copy_prohibited": True,
        "metrics_copy_prohibited": True,
    }


def variant_contract_hash() -> str:
    return canonical_hash(variant_contract())


def loss_identity(project_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(project_root)
    loss_path = _resolve_relative(root, "custom_models/src/st_mgprompt/losses.py")
    return {
        "loss_id": LOSS_ID,
        "loss_protocol": LOSS_PROTOCOL,
        "loss_source_path": "custom_models/src/st_mgprompt/losses.py",
        "loss_source_hash": sha256_file(loss_path),
        "loss_identity_hash": canonical_hash(
            {
                "loss_id": LOSS_ID,
                "loss_protocol": LOSS_PROTOCOL,
                "loss_source_hash": sha256_file(loss_path),
            }
        ),
    }


def precision_identity() -> dict[str, Any]:
    return {
        "precision_policy": PRECISION_POLICY,
        "amp_enabled": False,
        "amp_dtype": "float16",
    }


__all__ = [
    "A8ContractError",
    "A8_SCOPE_ID",
    "A8_MODEL_ID",
    "A8_VARIANT",
    "A8_DEFINITION",
    "A8_RUN_ID",
    "A8_OUTPUT_ROOT",
    "A8_RUN_RELATIVE_PATH",
    "A8_REFERENCE_ID",
    "A8_REFERENCE_RELATIVE_PATH",
    "TRAINING_ROLE",
    "TRAINING_PROFILE_ID",
    "LOSS_ID",
    "LOSS_PROTOCOL",
    "PRECISION_POLICY",
    "EXPECTED_CONFIG",
    "A8_SOURCE_CLOSURE_RELATIVE_PATHS",
    "GRAPH_PARAMETERS",
    "canonical_hash",
    "sha256_file",
    "source_closure",
    "graph_identity",
    "variant_contract",
    "variant_contract_hash",
    "loss_identity",
    "precision_identity",
]
