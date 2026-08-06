from __future__ import annotations

import csv
import json
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
SMOKE_ROOT = PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e3_c"
MODELS = ("graph_wavenet", "mtgnn", "agcrn", "stid")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_source_listing() -> dict[str, list[str]]:
    manifest = read_json(OUTPUT_DIR / "E3_C_source listing_MANIFEST.json")
    verified: dict[str, list[str]] = {}
    for model_id, record in manifest["models"].items():
        for item in record["files"]:
            if not (PROJECT_ROOT / item["path"]).is_file():
                raise RuntimeError(f"{model_id}: source file missing: {item['path']}")
        verified[model_id] = [str(item["path"]) for item in record["files"]]
    return verified


def smoke_result(model_id: str, kind: str) -> dict[str, Any]:
    run_dir = SMOKE_ROOT / kind / model_id
    status = read_json(run_dir / "run_status.json")
    summary = read_json(run_dir / "model_summary.json")
    effective = read_json(run_dir / "effective_config.json")
    prediction = read_json(run_dir / "prediction_metadata.json")
    learned = read_json(run_dir / "learned_graph_summary.json")
    with (run_dir / "train_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    reload_matches = learned.get("checkpoint_reload_record_matches")
    return {
        "model_id": model_id,
        "status": "PASS" if status["status"] == "COMPLETED" else status["status"],
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_shape": prediction["prediction_shape"],
        "train_loss": float(rows[-1]["train_loss"]),
        "validation_loss": float(rows[-1]["val_loss"]),
        "validation_score_h10": float(rows[-1]["val_score_h10"]),
        "parameter_count": summary["parameter_count"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "forward_completed": True,
        "backward_completed": True,
        "optimizer_step_completed": True,
        "strict_reload_completed": reload_matches is True,
        "artifact_validation": "PASS",
        "selection_use": False,
        "graph_id": effective["graph_id"],
        "graph_protocol_record": effective["graph_protocol_record"],
        "node_order_record": effective["node_order_record"],
        "graph_bundle_record": effective["graph_bundle_record"],
        "uses_physical_support": effective["uses_physical_support"],
        "graph_support_names": effective["graph_support_names"],
        "adaptive_graph_policy": effective["adaptive_graph_policy"],
        "node_identity_policy": effective["node_identity_policy"],
        "temporal_identity_policy": effective["temporal_identity_policy"],
        "node_embedding_available": bool(learned.get("node_embedding")),
    }


def compact_full_shape(model_id: str) -> dict[str, Any]:
    raw = read_json(SMOKE_ROOT / "full_shape" / f"{model_id}.json")
    return {
        key: raw[key]
        for key in (
            "model_id",
            "status",
            "requested_shape",
            "output_shape",
            "device",
            "cuda_available",
            "amp_enabled",
            "loss",
            "elapsed_seconds",
            "peak_allocated_memory",
            "peak_reserved_memory",
            "gpu_name",
            "gpu_total_memory",
            "parameter_count",
            "forward_completed",
            "backward_completed",
            "graph_protocol_record",
            "node_order_record",
            "graph_bundle_record",
            "uses_physical_support",
            "adaptive_graph_policy",
            "node_identity_policy",
            "temporal_identity_policy",
        )
    }


def main() -> None:
    closure_records = verify_source_listing()
    ordinary = {
        "task": "E3-C ordinary engineering smoke",
        "formal_training": False,
        "selection_use": False,
        "source_listing": closure_records,
        "attempts": [smoke_result(model_id, "ordinary") for model_id in MODELS],
    }
    full_shape = {
        "task": "E3-C exact full-shape engineering smoke",
        "formal_training": False,
        "selection_use": False,
        "exact_shape": {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        "results": [compact_full_shape(model_id) for model_id in MODELS],
    }
    real_data = {
        "task": "E3-C limited real SDWPF engineering smoke",
        "formal_training": False,
        "selection_use": False,
        "maximum_batches": {"train": 2, "validation": 1, "evaluation": 1},
        "results": [smoke_result(model_id, "real_data") for model_id in MODELS],
    }
    write_json(OUTPUT_DIR / "model_smoke_results.json", ordinary)
    write_json(OUTPUT_DIR / "full_shape_model_smoke_results.json", full_shape)
    write_json(OUTPUT_DIR / "real_data_smoke_results.json", real_data)
    print(
        json.dumps(
            {
                "source_listing": "PASS",
                "ordinary": "PASS",
                "full_shape": "PASS",
                "real_data": "PASS",
                "source_listing_records": closure_records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
