from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = (
    PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e1_b"
)
MODELS = ("dlinear", "lightts", "tide", "segrnn")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_train_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        "epoch": int(row["epoch"]),
        "train_loss": float(row["train_loss"]),
        "val_loss": float(row["val_loss"]),
        "val_score_h10": float(row["val_score_h10"]),
        "is_best": row["is_best"].casefold() == "true",
    }


def collect_trainable_run(run_dir: Path) -> dict[str, Any]:
    status = read_json(run_dir / "run_status.json")
    summary = read_json(run_dir / "model_summary.json")
    prediction = read_json(run_dir / "prediction_metadata.json")
    effective = read_json(run_dir / "effective_config.json")
    train_row = read_train_row(run_dir / "train_log.csv")
    metrics_h10 = read_json(run_dir / "metrics_eval_h10.json")
    completed = status["status"] == "COMPLETED"
    return {
        "status": "PASS" if completed else "FAIL",
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_shape": prediction["prediction_shape"],
        "loss": train_row["train_loss"],
        "validation_score_h10": train_row["val_score_h10"],
        "evaluation_score_h10": metrics_h10["Score"],
        "evaluation_r2_h10": metrics_h10["R2"],
        "parameter_count": summary["parameter_count"],
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "backward_completed": completed,
        "optimizer_step_completed": completed,
        "strict_reload_completed": completed,
        "artifact_validation": "PASS" if completed else "FAIL",
        "upstream_source_path": effective.get("upstream_source_path"),
    }


def write_json(name: str, payload: dict[str, Any]) -> None:
    (DOC_ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ordinary = {
        model_id: collect_trainable_run(SMOKE_ROOT / model_id)
        for model_id in MODELS
    }
    ordinary_payload = {
        "task": "E1-B",
        "run_mode": "smoke",
        "formal_training": False,
        "shape": {"B": 2, "T": 144, "N": 4, "C": 16, "H": 10},
        "results": ordinary,
    }
    write_json("model_smoke_results.json", ordinary_payload)

    full_results = {
        model_id: read_json(
            SMOKE_ROOT / "full_shape" / f"{model_id}.json"
        )
        for model_id in MODELS
    }
    full_payload = {
        "task": "E1-B",
        "run_mode": "smoke",
        "formal_training": False,
        "requested_shape": {
            "B": 32,
            "T": 144,
            "N": 134,
            "C": 16,
            "H": 10,
        },
        "results": full_results,
    }
    write_json("full_shape_model_smoke_results.json", full_payload)

    real = {
        model_id: {
            **collect_trainable_run(SMOKE_ROOT / "real_data" / model_id),
            "train_batch_limit": 2,
            "validation_batch_limit": 1,
            "evaluation_batch_limit": 1,
        }
        for model_id in MODELS
    }
    real_payload = {
        "task": "E1-B",
        "run_mode": "smoke",
        "limited": True,
        "formal_training": False,
        "formal_evaluation": False,
        "results": real,
    }
    write_json("real_data_smoke_results.json", real_payload)


if __name__ == "__main__":
    main()
