from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .builder import MATRIX_FILENAMES, NODE_COUNT
from .contracts import GraphBundle, GraphSpec
from .hashing import (
    file_sha256,
    graph_bundle_hash,
    graph_protocol_hash,
    matrix_hash,
    node_order_hash,
    stable_hash,
)
from .validation import GraphProtocolError, validate_matrices


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "protocol" / "graph_protocol_v1.json"
)
DEFAULT_BUNDLE_DIR = (
    Path(__file__).resolve().parents[1] / "protocol" / "graph_v1"
)
DEFAULT_LOCATION_PATH = (
    PROJECT_ROOT / "dataset" / "sdwpf_turb_location_elevation.csv"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphProtocolError(f"{path} must contain a JSON object.")
    return payload


def load_graph_bundle(
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    location_path: str | Path = DEFAULT_LOCATION_PATH,
    validate_location_source: bool = True,
) -> GraphBundle:
    protocol_source = Path(protocol_path)
    bundle_source = Path(bundle_dir)
    protocol = _read_json(protocol_source)
    declared_protocol_hash = protocol.get("graph_protocol_hash")
    calculated_protocol_hash = graph_protocol_hash(protocol)
    if declared_protocol_hash != calculated_protocol_hash:
        raise GraphProtocolError(
            "Frozen graph protocol hash mismatch: "
            f"{declared_protocol_hash} != {calculated_protocol_hash}"
        )
    if protocol.get("schema_version") != "graph_protocol_v1":
        raise GraphProtocolError("Unsupported graph protocol schema.")
    if int(protocol.get("node_count", -1)) != NODE_COUNT:
        raise GraphProtocolError("Frozen graph protocol node count is not 134.")
    node_order_payload = _read_json(bundle_source / "node_order_v1.json")
    metadata_payload = _read_json(bundle_source / "node_metadata_v1.json")
    manifest = _read_json(bundle_source / "graph_bundle_manifest_v1.json")
    ordered_node_ids = tuple(node_order_payload.get("ordered_node_ids", []))
    order_hash = node_order_hash(ordered_node_ids)
    if order_hash != protocol["node_order_hash"]:
        raise GraphProtocolError("Frozen node-order hash mismatch.")
    semantic_metadata = dict(metadata_payload)
    declared_metadata_hash = semantic_metadata.pop("node_metadata_hash", None)
    for audit_only_key in (
        "location_source_sha256",
        "source_path",
        "created_by",
    ):
        semantic_metadata.pop(audit_only_key, None)
    calculated_metadata_hash = stable_hash(semantic_metadata)
    if (
        declared_metadata_hash != calculated_metadata_hash
        or declared_metadata_hash != protocol["node_metadata_hash"]
    ):
        raise GraphProtocolError("Frozen node-metadata hash mismatch.")
    if validate_location_source:
        current_location_hash = file_sha256(location_path)
        if current_location_hash != protocol["location_source_sha256"]:
            raise GraphProtocolError(
                "Current location source hash does not match frozen graph identity."
            )
    matrices: dict[str, np.ndarray] = {}
    matrix_hashes: dict[str, str] = {}
    for name, filename in MATRIX_FILENAMES.items():
        path = bundle_source / filename
        matrix = np.load(path, allow_pickle=False)
        digest = matrix_hash(matrix)
        expected = protocol["matrix_files"][name]["canonical_matrix_hash"]
        if digest != expected:
            raise GraphProtocolError(
                f"Frozen matrix hash mismatch for {name}: {digest} != {expected}"
            )
        matrices[name] = np.asarray(matrix, dtype=np.float64)
        matrix_hashes[name] = digest
    numerical = validate_matrices(matrices, node_count=NODE_COUNT)
    calculated_bundle_hash = graph_bundle_hash(
        graph_id=protocol["graph_id"],
        node_count=NODE_COUNT,
        node_order_digest=order_hash,
        matrix_hashes=matrix_hashes,
    )
    if (
        calculated_bundle_hash != protocol["graph_bundle_hash"]
        or calculated_bundle_hash != manifest.get("graph_bundle_hash")
    ):
        raise GraphProtocolError("Frozen graph-bundle hash mismatch.")
    if manifest.get("graph_protocol_hash") != calculated_protocol_hash:
        raise GraphProtocolError("Bundle manifest graph-protocol hash mismatch.")
    spec = GraphSpec(
        graph_id=protocol["graph_id"],
        schema_version=protocol["schema_version"],
        node_count=NODE_COUNT,
        coordinate_system=protocol["coordinate_system"],
        distance_metric=protocol["distance_metric"],
        elevation_used=bool(
            protocol["elevation_policy"]["used_in_edge_distance"]
        ),
        k_selection_rule=protocol["k_selection_rule"],
        selected_k=int(protocol["selected_k"]),
        directedness=json.dumps(
            protocol["directedness"], ensure_ascii=False, sort_keys=True
        ),
        self_loop_policy=protocol["self_loop_policy"],
        weight_formula=protocol["weight_formula"],
        normalization_formulas=protocol["normalization_formulas"],
        graph_protocol_hash=calculated_protocol_hash,
    )
    return GraphBundle(
        spec=spec,
        ordered_node_ids=ordered_node_ids,
        node_order_hash=order_hash,
        node_metadata_hash=declared_metadata_hash,
        location_source_hash=protocol["location_source_sha256"],
        matrix_hashes=matrix_hashes,
        graph_bundle_hash=calculated_bundle_hash,
        graph_protocol_hash=calculated_protocol_hash,
        **matrices,
    )


def graph_protocol_check() -> dict[str, Any]:
    bundle = load_graph_bundle()
    diagnostics = validate_matrices(
        bundle.matrices, node_count=bundle.spec.node_count
    )
    return {
        "status": "PASS",
        "schema_version": bundle.spec.schema_version,
        "graph_id": bundle.spec.graph_id,
        "node_count": bundle.spec.node_count,
        "selected_k": bundle.spec.selected_k,
        "node_order_hash": bundle.node_order_hash,
        "node_metadata_hash": bundle.node_metadata_hash,
        "location_source_hash": bundle.location_source_hash,
        "matrix_hashes": dict(bundle.matrix_hashes),
        "graph_bundle_hash": bundle.graph_bundle_hash,
        "graph_protocol_hash": bundle.graph_protocol_hash,
        "matrix_validation": diagnostics,
        "cuda_context_created": False,
        "model_constructed": False,
        "formal_result_directory_created": False,
    }
