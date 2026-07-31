from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark_v2.graph.builder import (  # noqa: E402
    MATRIX_FILENAMES,
    build_physical_graph,
    make_protocol_payload,
    node_schema_source_hash,
)
from benchmark_v2.graph.hashing import file_sha256  # noqa: E402


INPUT_PATH = PROJECT_ROOT / "dataset" / "sdwpf_model_input_base.parquet"
TARGET_PATH = PROJECT_ROOT / "dataset" / "sdwpf_eval_target.parquet"
LOCATION_PATH = (
    PROJECT_ROOT / "dataset" / "sdwpf_turb_location_elevation.csv"
)
PROTOCOL_PATH = (
    SOURCE_ROOT / "benchmark_v2" / "protocol" / "graph_protocol_v1.json"
)
BUNDLE_DIR = SOURCE_ROOT / "benchmark_v2" / "protocol" / "graph_v1"
DOC_DIR = Path(__file__).resolve().parent


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_node_sources() -> dict[str, Any]:
    node_column = "TurbID"
    timestamp_column = "Tmstamp"
    model_keys = pd.read_parquet(
        INPUT_PATH, columns=[timestamp_column, node_column]
    )
    target_keys = pd.read_parquet(
        TARGET_PATH, columns=[timestamp_column, node_column]
    )
    if model_keys.duplicated([timestamp_column, node_column]).any():
        raise ValueError("Input contains duplicate (Tmstamp,TurbID) keys.")
    if target_keys.duplicated([timestamp_column, node_column]).any():
        raise ValueError("Target contains duplicate (Tmstamp,TurbID) keys.")
    if not model_keys.reset_index(drop=True).equals(
        target_keys.reset_index(drop=True)
    ):
        raise ValueError("Input and target key order differs.")
    if model_keys[node_column].isna().any() or target_keys[node_column].isna().any():
        raise ValueError("Input or target contains null TurbID.")
    input_ids = sorted(
        model_keys[node_column].astype(int).unique().tolist()
    )
    target_ids = sorted(
        target_keys[node_column].astype(int).unique().tolist()
    )
    location = pd.read_csv(LOCATION_PATH)
    location_ids = sorted(location[node_column].astype(int).tolist())
    if input_ids != target_ids or input_ids != location_ids:
        raise ValueError("Input/target/location node sets differ.")
    timestamp_count = int(model_keys[timestamp_column].nunique())
    input_source_hash = node_schema_source_hash(
        logical_path="dataset/sdwpf_model_input_base.parquet",
        node_id_column=node_column,
        node_id_type="integer",
        ordered_node_ids=input_ids,
        row_count=len(model_keys),
        timestamp_count=timestamp_count,
    )
    target_source_hash = node_schema_source_hash(
        logical_path="dataset/sdwpf_eval_target.parquet",
        node_id_column=node_column,
        node_id_type="integer",
        ordered_node_ids=target_ids,
        row_count=len(target_keys),
        timestamp_count=int(target_keys[timestamp_column].nunique()),
    )
    return {
        "node_id_field": node_column,
        "node_id_type": "integer",
        "canonical_order_source": (
            "SDWPFDataProvider.from_files: sorted("
            "model_df['TurbID'].astype(int).unique())"
        ),
        "ordered_node_ids": input_ids,
        "node_count": len(input_ids),
        "input_node_ids": input_ids,
        "target_node_ids": target_ids,
        "location_node_ids": location_ids,
        "input_target_key_order_aligned": True,
        "input_target_location_sets_equal": True,
        "input_duplicate_key_count": 0,
        "target_duplicate_key_count": 0,
        "input_null_node_id_count": 0,
        "target_null_node_id_count": 0,
        "location_null_node_id_count": int(
            location[node_column].isna().sum()
        ),
        "location_duplicate_node_id_count": int(
            location[node_column].duplicated().sum()
        ),
        "location_duplicate_coordinate_count": int(
            location.duplicated(["x", "y"]).sum()
        ),
        "location_coordinate_null_count": int(
            location[["x", "y"]].isna().sum().sum()
        ),
        "row_count": len(model_keys),
        "timestamp_count": timestamp_count,
        "input_source_hash": input_source_hash,
        "target_source_hash": target_source_hash,
        "input_file_sha256_audit_only": file_sha256(INPUT_PATH),
        "target_file_sha256_audit_only": file_sha256(TARGET_PATH),
        "location_source_sha256": file_sha256(LOCATION_PATH),
        "input_source_hash_semantics": (
            "node schema only; feature values are excluded"
        ),
        "target_source_hash_semantics": (
            "node schema only; target and mask values are excluded"
        ),
    }


def build_all() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    audit = audit_node_sources()
    location = pd.read_csv(LOCATION_PATH)
    build = build_physical_graph(audit["ordered_node_ids"], location)
    protocol = make_protocol_payload(
        build,
        location_source_hash=audit["location_source_sha256"],
        input_source_hash=audit["input_source_hash"],
        target_source_hash=audit["target_source_hash"],
    )
    return audit, build, protocol


def identity_summary(build: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_order_hash": build.node_order_hash,
        "node_metadata_hash": build.node_metadata_hash,
        "matrix_hashes": build.matrix_hashes,
        "graph_bundle_hash": build.graph_bundle_hash,
        "graph_protocol_hash": protocol["graph_protocol_hash"],
    }


def freeze(
    audit: dict[str, Any], build: Any, protocol: dict[str, Any]
) -> None:
    if PROTOCOL_PATH.exists() or BUNDLE_DIR.exists():
        raise FileExistsError(
            "Frozen graph_v1 already exists; overwrite is forbidden. "
            "Use --output-new-version in a future task for a new version."
        )
    BUNDLE_DIR.mkdir(parents=True, exist_ok=False)
    for name, filename in MATRIX_FILENAMES.items():
        np.save(BUNDLE_DIR / filename, build.matrices[name], allow_pickle=False)
    node_order_payload = {
        "schema_version": "node_order_v1",
        "node_count": len(build.ordered_node_ids),
        "ordered_node_ids": list(build.ordered_node_ids),
        "node_id_type": "integer",
        "node_id_source": (
            "benchmark_v2 SDWPFDataProvider.from_files N-axis order"
        ),
        "input_source_hash": audit["input_source_hash"],
        "target_source_hash": audit["target_source_hash"],
        "location_source_hash": audit["location_source_sha256"],
        "input_file_sha256_audit_only": audit[
            "input_file_sha256_audit_only"
        ],
        "target_file_sha256_audit_only": audit[
            "target_file_sha256_audit_only"
        ],
        "node_order_hash": build.node_order_hash,
        "created_by": "E3-A build_graph_protocol_v1.py",
    }
    metadata_payload = {
        **build.node_metadata,
        "source_path": "dataset/sdwpf_turb_location_elevation.csv",
        "location_source_sha256": audit["location_source_sha256"],
        "created_by": "E3-A build_graph_protocol_v1.py",
    }
    manifest = {
        "schema_version": "graph_bundle_manifest_v1",
        "graph_id": protocol["graph_id"],
        "node_count": protocol["node_count"],
        "node_order_hash": build.node_order_hash,
        "node_metadata_hash": build.node_metadata_hash,
        "location_source_hash": audit["location_source_sha256"],
        "matrix_files": protocol["matrix_files"],
        "matrix_hashes": build.matrix_hashes,
        "graph_bundle_hash": build.graph_bundle_hash,
        "graph_protocol_hash": protocol["graph_protocol_hash"],
        "identity_note": (
            "Canonical matrix content hashes, not .npy container bytes, "
            "define matrix identity."
        ),
    }
    write_json(BUNDLE_DIR / "node_order_v1.json", node_order_payload)
    write_json(BUNDLE_DIR / "node_metadata_v1.json", metadata_payload)
    write_json(BUNDLE_DIR / "graph_bundle_manifest_v1.json", manifest)
    write_json(PROTOCOL_PATH, protocol)
    audit = {**audit, "node_order_hash": build.node_order_hash}
    write_json(DOC_DIR / "node_order_audit.json", audit)
    write_json(DOC_DIR / "graph_build_diagnostics.json", build.diagnostics)
    write_json(
        DOC_DIR / "graph_hash_manifest.json",
        {
            **identity_summary(build, protocol),
            "location_source_sha256": audit["location_source_sha256"],
            "input_source_hash": audit["input_source_hash"],
            "target_source_hash": audit["target_source_hash"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-freeze",
        action="store_true",
        help="Create graph_v1 only when it does not already exist.",
    )
    parser.add_argument(
        "--verify-rebuild",
        action="store_true",
        help="Build twice in memory and require identical canonical identity.",
    )
    parser.add_argument(
        "--output-new-version",
        help=(
            "Reserved explicit version output. E3-A refuses this option so "
            "graph_v2 cannot be created in this task."
        ),
    )
    args = parser.parse_args()
    if args.output_new_version:
        raise ValueError(
            "--output-new-version is reserved for a future approved task; "
            "E3-A must not create graph_v2."
        )
    if PROTOCOL_PATH.exists() or BUNDLE_DIR.exists():
        raise FileExistsError(
            "Frozen graph_v1 already exists; overwrite is forbidden. "
            "Use --output-new-version in a future approved task."
        )
    audit, build, protocol = build_all()
    verification: dict[str, Any] | None = None
    if args.verify_rebuild:
        audit_second, build_second, protocol_second = build_all()
        first = identity_summary(build, protocol)
        second = identity_summary(build_second, protocol_second)
        if first != second or audit["ordered_node_ids"] != audit_second[
            "ordered_node_ids"
        ]:
            raise RuntimeError("Two consecutive graph rebuild identities differ.")
        verification = {
            "status": "PASS",
            "first": first,
            "second": second,
            "all_canonical_hashes_identical": True,
        }
        write_json(DOC_DIR / "double_rebuild_hash_verification.json", verification)
    if args.initial_freeze:
        freeze(audit, build, protocol)
    print(
        json.dumps(
            {
                "status": "PASS",
                "initial_freeze": bool(args.initial_freeze),
                "verify_rebuild": verification,
                **identity_summary(build, protocol),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
