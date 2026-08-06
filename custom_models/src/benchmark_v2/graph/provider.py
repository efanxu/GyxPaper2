from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .builder import MATRIX_FILENAMES, NODE_COUNT
from .contracts import GraphBundle, GraphSpec
from .validation import (
    GraphProtocolError,
    canonical_node_ids,
    validate_matrices,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "protocol" / "graph_protocol_v1.json"
)
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "protocol" / "graph_v1"
DEFAULT_LOCATION_PATH = PROJECT_ROOT / "dataset" / "sdwpf_turb_location_elevation.csv"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphProtocolError(f"{path} must contain a JSON object.")
    return payload


def _validate_location(path: Path, ordered_node_ids: tuple[Any, ...]) -> None:
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"TurbID", "x", "y", "Ele"}
    missing = required - set(frame.columns)
    if missing:
        raise GraphProtocolError(f"Location columns missing: {sorted(missing)}")
    if len(frame) != NODE_COUNT:
        raise GraphProtocolError(
            f"Location row count must be {NODE_COUNT}, got {len(frame)}."
        )
    ids = canonical_node_ids(frame["TurbID"].astype(int).tolist())
    if ids != ordered_node_ids:
        raise GraphProtocolError("Location node order does not match graph node order.")


def load_graph_bundle(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    location_path: str | Path = DEFAULT_LOCATION_PATH,
    validate_location_source: bool = True,
) -> GraphBundle:
    protocol = _read_json(Path(protocol_path))
    bundle_source = Path(bundle_dir)
    if protocol.get("schema_version") != "graph_protocol_v1":
        raise GraphProtocolError("Unsupported graph protocol schema.")
    graph_id = protocol.get("graph_id")
    if not isinstance(graph_id, str) or not graph_id:
        raise GraphProtocolError("Graph protocol requires graph_id.")
    if int(protocol.get("node_count", -1)) != NODE_COUNT:
        raise GraphProtocolError("Graph protocol node count is not 134.")
    selected_k = int(protocol.get("selected_k", -1))
    if selected_k <= 0:
        raise GraphProtocolError("Graph protocol selected_k must be positive.")

    node_order_payload = _read_json(bundle_source / "node_order_v1.json")
    metadata_payload = _read_json(bundle_source / "node_metadata_v1.json")
    manifest = _read_json(bundle_source / "graph_bundle_manifest_v1.json")
    ordered_node_ids = canonical_node_ids(
        node_order_payload.get("ordered_node_ids", [])
    )
    if len(ordered_node_ids) != NODE_COUNT or len(set(ordered_node_ids)) != NODE_COUNT:
        raise GraphProtocolError("Graph node IDs must contain 134 unique values.")
    protocol_ids = protocol.get("ordered_node_ids")
    if protocol_ids is not None and tuple(protocol_ids) != ordered_node_ids:
        raise GraphProtocolError("Protocol and bundle node orders differ.")
    records = metadata_payload.get("records")
    if not isinstance(records, list) or len(records) != NODE_COUNT:
        raise GraphProtocolError("Node metadata must contain 134 records.")
    if [record.get("node_id") for record in records] != list(ordered_node_ids):
        raise GraphProtocolError("Node metadata order differs from graph node order.")
    if validate_location_source:
        _validate_location(Path(location_path), ordered_node_ids)

    declared_files = protocol.get("matrix_files")
    if not isinstance(declared_files, dict) or set(declared_files) != set(MATRIX_FILENAMES):
        raise GraphProtocolError("Graph protocol matrix names are incomplete.")
    manifest_files = manifest.get("matrix_files")
    if not isinstance(manifest_files, dict) or set(manifest_files) != set(MATRIX_FILENAMES):
        raise GraphProtocolError("Graph bundle manifest matrix names are incomplete.")
    matrices: dict[str, np.ndarray] = {}
    for name, filename in MATRIX_FILENAMES.items():
        declaration = declared_files[name]
        if not isinstance(declaration, dict):
            raise GraphProtocolError(f"Matrix declaration for {name} must be an object.")
        if declaration.get("filename") != filename:
            raise GraphProtocolError(f"Unexpected matrix filename for {name}.")
        if declaration.get("shape") != [NODE_COUNT, NODE_COUNT]:
            raise GraphProtocolError(f"Unexpected matrix shape declaration for {name}.")
        matrix = np.load(bundle_source / filename, allow_pickle=False)
        matrices[name] = np.asarray(matrix, dtype=np.float64)
    validate_matrices(matrices, node_count=NODE_COUNT)

    spec = GraphSpec(
        graph_id=graph_id,
        schema_version=protocol["schema_version"],
        node_count=NODE_COUNT,
        coordinate_system=protocol["coordinate_system"],
        distance_metric=protocol["distance_metric"],
        elevation_used=bool(protocol["elevation_policy"]["used_in_edge_distance"]),
        k_selection_rule=protocol["k_selection_rule"],
        selected_k=selected_k,
        directedness=json.dumps(protocol["directedness"], ensure_ascii=False, sort_keys=True),
        self_loop_policy=protocol["self_loop_policy"],
        weight_formula=protocol["weight_formula"],
        normalization_formulas=protocol["normalization_formulas"],
    )
    return GraphBundle(
        spec=spec,
        ordered_node_ids=ordered_node_ids,
        **matrices,
    )


def graph_protocol_check() -> dict[str, Any]:
    bundle = load_graph_bundle()
    diagnostics = validate_matrices(bundle.matrices, node_count=bundle.spec.node_count)
    return {
        "status": "PASS",
        "schema_version": bundle.spec.schema_version,
        "graph_id": bundle.spec.graph_id,
        "node_count": bundle.spec.node_count,
        "selected_k": bundle.spec.selected_k,
        "ordered_node_ids": list(bundle.ordered_node_ids),
        "matrix_names": list(bundle.matrices),
        "matrix_validation": diagnostics,
        "cuda_context_created": False,
        "model_constructed": False,
        "formal_result_directory_created": False,
    }
