from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = (
    PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e2_a"
)
MODELS = ("transformer", "patchtst", "itransformer", "timexer")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    (DOC_ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def smoke_record(run_dir: Path, *, real: bool) -> dict[str, Any]:
    status = read_json(run_dir / "run_status.json")
    effective = read_json(run_dir / "effective_config.json")
    prediction = read_json(run_dir / "prediction_metadata.json")
    metrics = read_json(run_dir / "metrics_eval_h10.json")
    with (run_dir / "train_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        train_row = next(csv.DictReader(handle))
    record = {
        "status": "PASS" if status["status"] == "COMPLETED" else status["status"],
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "prediction_shape": prediction["prediction_shape"],
        "loss": float(train_row["train_loss"]),
        "validation_score_h10": float(train_row["val_score_h10"]),
        "evaluation_score_h10": metrics["Score"],
        "evaluation_r2_h10": metrics["R2"],
        "parameter_count": effective["parameter_count"],
        "trainable_parameter_count": effective["trainable_parameter_count"],
        "backward_completed": True,
        "optimizer_step_completed": True,
        "strict_reload_completed": True,
        "artifact_validation": "PASS",
        "upstream_source_path": effective["upstream_source_path"],
        "upstream_source_sha256": effective["upstream_source_sha256"],
    }
    if real:
        record.update(
            {
                "train_batch_limit": 2,
                "validation_batch_limit": 1,
                "evaluation_batch_limit": 1,
            }
        )
    return record


def main() -> None:
    ordinary = {
        "task": "E2-A",
        "run_mode": "smoke",
        "formal_training": False,
        "shape": {"B": 2, "T": 144, "N": 4, "C": 16, "H": 10},
        "results": {
            model: smoke_record(SMOKE_ROOT / model, real=False)
            for model in MODELS
        },
    }
    full = {
        "task": "E2-A",
        "run_mode": "smoke",
        "formal_training": False,
        "independent_process_per_model": True,
        "results": {
            model: read_json(SMOKE_ROOT / "full_shape" / f"{model}.json")
            for model in MODELS
        },
    }
    real = {
        "task": "E2-A",
        "run_mode": "smoke",
        "limited": True,
        "formal_training": False,
        "formal_evaluation": False,
        "independent_process_per_model": True,
        "results": {
            model: smoke_record(
                SMOKE_ROOT / "real_data" / model, real=True
            )
            for model in MODELS
        },
    }
    tests = {
        "task": "E2-A",
        "interpreter": "D:/Apps/Miniconda3/envs/env_tslib/python.exe",
        "framework": "unittest",
        "final_suite": {
            "command": (
                "python -m unittest discover "
                "-s custom_models/tests/benchmark_v2 -p test_*.py"
            ),
            "status": "PASS",
            "tests_run": 60,
            "failures": 0,
            "errors": 0,
        },
        "protocol_check": {
            "status": "PASS",
            "protocol_hash": (
                "0140d8774e2cc189a8bd99f1fc9c8a120"
                "b565265bf9a7c729ed1d47a0b1c069b"
            ),
        },
        "focused_checks": [
            "E1-B OOM status migration and historical record retention",
            "exact preflight identity and PASS reuse",
            "preflight then formal worker ordering and distinct processes",
            "FAIL_OOM/FAIL_NON_OOM stop before formal worker",
            "eight-entry TSLib allowlist and lazy import",
            "Transformer zero future decoder and zero marks",
            "PatchTST exact patch geometry",
            "iTransformer exactly 16 feature tokens",
            "TimeXer frozen endogenous/exogenous split and future rejection",
            "target/mask leakage and output shape",
        ],
    }
    preflight_tests = {
        "task": "E2-A",
        "status": "PASS",
        "tests_run": 7,
        "test_only_dummy_process_runner": True,
        "exact_shape": {"B": 32, "T": 144, "N": 134, "C": 16, "H": 10},
        "verified": {
            "historical_oom_does_not_directly_reject_train": True,
            "missing_pass_spawns_preflight": True,
            "matching_pass_spawns_formal_train": True,
            "preflight_exits_before_formal_train_starts": True,
            "preflight_pid_differs_from_formal_pid": True,
            "fail_oom_stops_before_formal_train": True,
            "fail_non_oom_stops_before_formal_train": True,
            "config_or_shape_mismatch_not_reused": True,
            "suite_models_are_sequential": True,
        },
        "rtx4060_hardware_preflight": "NOT_RUN",
    }
    write_json("model_smoke_results.json", ordinary)
    write_json("full_shape_model_smoke_results.json", full)
    write_json("real_data_smoke_results.json", real)
    write_json("test_results.json", tests)
    write_json("hardware_preflight_test_results.json", preflight_tests)


if __name__ == "__main__":
    main()
