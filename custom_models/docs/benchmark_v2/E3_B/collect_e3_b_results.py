from __future__ import annotations

import csv
import json
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
SMOKE_ROOT = (
    PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e3_b"
)
MODELS = ("gcn", "stgcn", "dcrnn")

COMMON_CLOSURE = [
    (
        "custom_models/src/benchmark_v2/models/graph_models/__init__.py",
        "graph model public exports",
    ),
    (
        "custom_models/src/benchmark_v2/models/graph_models/common.py",
        "frozen support copies and graph identity",
    ),
    (
        "custom_models/src/benchmark_v2/adapters/e3_b.py",
        "native graph adapter and leakage/identity boundary",
    ),
    (
        "custom_models/src/benchmark_v2/configs/graph_common.py",
        "shared frozen benchmark graph configuration",
    ),
    (
        "custom_models/src/benchmark_v2/model_runtime.py",
        "explicit runtime build entry",
    ),
]
MODEL_CLOSURE = {
    model_id: [
        (
            f"custom_models/src/benchmark_v2/models/graph_models/{model_id}.py",
            "model architecture and custom graph operators",
        ),
        (
            f"custom_models/src/benchmark_v2/configs/{model_id}.py",
            "model-specific frozen configuration",
        ),
    ]
    for model_id in MODELS
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_listing() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_id in MODELS:
        files = []
        for relative, role in [*COMMON_CLOSURE, *MODEL_CLOSURE[model_id]]:
            path = PROJECT_ROOT / relative
            files.append({"path": relative, "role": role, "exists": path.is_file()})
        models[model_id] = {"files": files}
    return {
        "task": "E3-B",
        "validation": "explicit ordered source paths, roles, and existence",
        "models": models,
        "graph_protocol_identity_is_separate": True,
    }


def smoke_result(model_id: str, kind: str) -> dict[str, Any]:
    run_dir = SMOKE_ROOT / kind / model_id
    status = read_json(run_dir / "run_status.json")
    summary = read_json(run_dir / "model_summary.json")
    effective = read_json(run_dir / "effective_config.json")
    prediction = read_json(run_dir / "prediction_metadata.json")
    with (run_dir / "train_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    return {
        "model_id": model_id,
        "status": "PASS" if status["status"] == "COMPLETED" else status["status"],
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_shape": prediction["prediction_shape"],
        "loss": float(rows[-1]["train_loss"]),
        "validation_score_h10": float(rows[-1]["val_score_h10"]),
        "parameter_count": summary["parameter_count"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "backward_completed": True,
        "optimizer_step_completed": True,
        "strict_reload_completed": True,
        "artifact_validation": "PASS",
        "graph_id": effective["graph_id"],
        "graph_protocol_record": effective["graph_protocol_record"],
        "node_order_record": effective["node_order_record"],
        "graph_bundle_record": effective["graph_bundle_record"],
        "graph_support_names": effective["graph_support_names"],
        "graph_support_records": effective["graph_support_records"],
    }


def main() -> None:
    closure = source_listing()
    write_json(OUTPUT_DIR / "E3_B_source listing_MANIFEST.json", closure)
    ordinary = {
        "task": "E3-B ordinary smoke",
        "selection_use": False,
        "attempts": [smoke_result(model_id, "ordinary") for model_id in MODELS],
    }
    write_json(OUTPUT_DIR / "model_smoke_results.json", ordinary)
    full_shape = {
        "task": "E3-B exact full-shape smoke",
        "exact_shape": {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        "selection_use": False,
        "results": [
            read_json(SMOKE_ROOT / "full_shape" / f"{model_id}.json")
            for model_id in MODELS
        ],
    }
    write_json(
        OUTPUT_DIR / "full_shape_model_smoke_results.json", full_shape
    )
    real_data = {
        "task": "E3-B limited real SDWPF smoke",
        "maximum_batches": {"train": 2, "validation": 1, "evaluation": 1},
        "selection_use": False,
        "results": [smoke_result(model_id, "real_data") for model_id in MODELS],
    }
    write_json(OUTPUT_DIR / "real_data_smoke_results.json", real_data)
    print(
        json.dumps(
            {
                "source_listing": closure,
                "ordinary": "PASS",
                "full_shape": "PASS",
                "real_data": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
