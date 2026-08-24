from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ANALYSIS_ROOT, EXPECTED_GRAPH, INTERNAL_ROOT, PROJECT_ROOT
from .graph_statistics import distance_weight_analysis, load_locations, matched_edge_overlap, matrix_statistics, row_top_k
from .io_utils import read_json, write_csv, write_json


def _save_matrix(root: Path, model_id: str, name: str, matrix: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    target = root / model_id
    target.mkdir(parents=True, exist_ok=True)
    npz_path = target / f"{name}.npz"
    np.savez_compressed(npz_path, matrix=np.asarray(matrix, dtype=np.float64))
    edge_rows = []
    for source, target_index in np.argwhere(np.asarray(matrix) != 0):
        edge_rows.append({"source_TurbID": int(source + 1), "target_TurbID": int(target_index + 1), "weight": float(matrix[source, target_index])})
    csv_path = target / f"{name}_edges.csv"
    write_csv(csv_path, edge_rows, ("source_TurbID", "target_TurbID", "weight"))
    metadata_path = target / f"{name}_metadata.json"
    write_json(metadata_path, {**metadata, "matrix_path": str(npz_path), "edge_list_path": str(csv_path), "content_hash": "NOT_COMPUTED_BY_REPOSITORY_POLICY"})
    return {"model_id": model_id, "matrix_name": name, "matrix": matrix, "matrix_path": str(npz_path), "metadata_path": str(metadata_path)}


def _load_benchmark_bundle():
    from ..graph import load_graph_bundle

    return load_graph_bundle()


def _load_mtgnn(checkpoint: Path, config: dict[str, Any], bundle):
    import torch
    from ..models.graph_models.mtgnn import create_model

    model = create_model(config=config, graph_bundle=bundle)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        learned = model.learned_adjacency().detach().cpu().numpy().astype(np.float64)
    return model, learned


def _state_unchanged(before: dict, after: dict) -> bool:
    import torch

    return before.keys() == after.keys() and all(torch.equal(before[key], after[key]) for key in before)


def _snapshot(model) -> dict:
    snapshot = {f"parameter:{key}": value.detach().cpu().clone() for key, value in model.named_parameters()}
    snapshot.update({f"buffer:{key}": value.detach().cpu().clone() for key, value in model.named_buffers()})
    return snapshot


def _mtgnn_raw_scores(model) -> np.ndarray:
    import torch

    constructor = model.graph_constructor
    with torch.no_grad():
        left = torch.tanh(constructor.tanh_alpha * constructor.lin1(constructor.emb1(constructor.node_index)))
        right = torch.tanh(constructor.tanh_alpha * constructor.lin2(constructor.emb2(constructor.node_index)))
        relation = left @ right.T - right @ left.T
        row_scale = relation.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sigmoid(constructor.tanh_alpha * relation / row_scale).cpu().numpy().astype(np.float64)


def side_effect_audit(model, extraction, *, checkpoint: Path, input_shape=(1, 144, 134, 16), tolerance=1e-7) -> dict[str, Any]:
    import torch

    before_stat = checkpoint.stat()
    was_training = bool(model.training)
    model.eval()
    fixed = torch.zeros(*input_shape, dtype=torch.float32)
    before_state = _snapshot(model)
    with torch.no_grad():
        prediction_a = model(fixed)
        extraction()
        prediction_b = model(fixed)
    after_state = _snapshot(model)
    model.train(was_training)
    after_stat = checkpoint.stat()
    delta = float((prediction_a - prediction_b).abs().max().cpu())
    passed = delta <= tolerance and _state_unchanged(before_state, after_state) and before_stat.st_size == after_stat.st_size and before_stat.st_mtime_ns == after_stat.st_mtime_ns and bool(model.training) == was_training
    return {
        "status": "PASS" if passed else "FAIL",
        "prediction_max_abs_delta": delta,
        "determinism_tolerance": tolerance,
        "parameters_and_buffers_unchanged": _state_unchanged(before_state, after_state),
        "checkpoint_file_unchanged": before_stat.st_size == after_stat.st_size and before_stat.st_mtime_ns == after_stat.st_mtime_ns,
        "checkpoint_content_hash_check": "NOT_COMPUTED_BY_REPOSITORY_POLICY",
        "mode_restored": bool(model.training) == was_training,
    }


def run_graph_diagnostics(output_root: str | Path = ANALYSIS_ROOT) -> dict[str, Any]:
    root = Path(output_root).resolve()
    diagnostics_root = root / "graph_structure_diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    bundle = _load_benchmark_bundle()
    node_ids = list(bundle.ordered_node_ids)
    if node_ids != list(range(1, 135)):
        raise ValueError("GraphBundle node order is not TurbID 1..134.")
    metadata = {
        "graph_id": bundle.spec.graph_id,
        "node_count": bundle.spec.node_count,
        "selected_k": bundle.spec.selected_k,
        "ordered_node_ids": node_ids,
        "read_only": True,
        "training_executed": False,
    }
    extracted = []
    fixed = {
        "graph_protocol": ("prior_A_directed", bundle.matrices["A_directed"]),
        "gcn": ("effective_A_gcn", bundle.matrices["A_gcn"]),
        "stgcn": ("effective_L_tilde", bundle.matrices["L_tilde"]),
        "dcrnn_forward": ("effective_P_forward", bundle.matrices["P_forward"]),
        "dcrnn_reverse": ("effective_P_reverse", bundle.matrices["P_reverse"]),
    }
    for model_key, (name, matrix) in fixed.items():
        model_id = model_key.split("_")[0]
        extracted.append(_save_matrix(diagnostics_root, model_id, name, matrix, {**metadata, "extraction_method": "frozen GraphBundle support"}))
    mtgnn_dir = PROJECT_ROOT / "custom_models" / "results" / "benchmark_v2_uniform_bs4" / "e3_c_seed2026" / "MTGNN_native_adaptive_k20_gdep2_l3_bs4_seed2026"
    mtgnn_config = read_json(mtgnn_dir / "effective_config.json")
    mtgnn_model, mtgnn = _load_mtgnn(mtgnn_dir / "best_checkpoint.pt", mtgnn_config, bundle)
    mtgnn_raw = _mtgnn_raw_scores(mtgnn_model)
    extracted.append(_save_matrix(diagnostics_root, "mtgnn", "raw_learned_scores", mtgnn_raw, {**metadata, "extraction_method": "best checkpoint graph-constructor scores before native top-k masking"}))
    extracted.append(_save_matrix(diagnostics_root, "mtgnn", "learned_effective_native_top20", mtgnn, {**metadata, "extraction_method": "best checkpoint model.learned_adjacency", "native_k": 20}))
    extracted.append(_save_matrix(diagnostics_root, "mtgnn", "learned_matched_top4", row_top_k(mtgnn, 4), {**metadata, "extraction_method": "diagnostic row-wise top-k projection", "matched_k": 4}))
    side_effect = {
        "schema_version": "e7_diagnostic_side_effect_audit_v1",
        "mtgnn": side_effect_audit(mtgnn_model, lambda: mtgnn_model.learned_adjacency(), checkpoint=mtgnn_dir / "best_checkpoint.pt"),
        "internal_st_mgprompt": {
            "status": "NOT_RUN_PROTOCOL_MISMATCH",
            "reason": "A0/A1/A2/A3 formal artifacts have train_batch_size=32, not current batch4.",
        },
    }
    stats = [matrix_statistics(item["matrix"], model_id=item["model_id"], matrix_name=item["matrix_name"]) for item in extracted]
    prior = bundle.matrices["A_directed"]
    overlaps = [
        {"model_id": "mtgnn", "prior_matrix": "A_directed", "learned_matrix": "learned_effective_native_top20", **matched_edge_overlap(prior, mtgnn, k=4)}
    ]
    coordinates = load_locations(PROJECT_ROOT / "dataset" / "sdwpf_turb_location_elevation.csv", node_ids)
    distance_summaries, distance_bins = [], []
    for item in extracted:
        summary, bins = distance_weight_analysis(item["matrix"], coordinates, model_id=item["model_id"], matrix_name=item["matrix_name"])
        distance_summaries.append(summary)
        distance_bins.extend(bins)
    flat_stats = []
    for row in stats:
        flat_stats.append({key: value for key, value in row.items() if key not in {"top_influence_nodes", "top_weighted_edges", "weight_quantiles"}} | {f"weight_{key}": value for key, value in row["weight_quantiles"].items()})
    write_csv(root / "e7_graph_structure_statistics.csv", flat_stats)
    write_csv(root / "e7_prior_learned_overlap.csv", overlaps)
    write_csv(root / "e7_distance_weight_analysis.csv", distance_bins)
    write_json(root / "E7_DIAGNOSTIC_SIDE_EFFECT_AUDIT.json", side_effect)
    write_json(diagnostics_root / "diagnostics_manifest.json", {
        "schema_version": "e7_graph_diagnostics_manifest_v1",
        "status": "PARTIAL_EXTERNAL_ONLY",
        "graph_protocol": metadata,
        "exports": [{key: value for key, value in item.items() if key != "matrix"} for item in extracted],
        "statistics": stats,
        "overlaps": overlaps,
        "distance_summaries": distance_summaries,
        "graph_free_controls": {"stid": "NOT_APPLICABLE", "tsmixer": "NOT_APPLICABLE"},
        "internal_diagnostics": "BLOCKED_CURRENT_BATCH4_EVIDENCE_MISSING",
    })
    return {
        "status": "PARTIAL_EXTERNAL_ONLY",
        "export_count": len(extracted),
        "side_effect_status": side_effect["mtgnn"]["status"],
        "diagnostics_root": str(diagnostics_root),
    }
