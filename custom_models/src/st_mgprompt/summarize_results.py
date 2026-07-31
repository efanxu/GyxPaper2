from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.config import resolve_project_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metric_rows(results_root: str | Path = "custom_models/results/st_mgprompt") -> list[dict[str, Any]]:
    root = resolve_project_path(results_root)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for metrics_path in root.glob("*/*/metrics.csv"):
        run_dir = metrics_path.parent
        config = _read_json(run_dir / "config.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        if config.get("smoke") and config.get("smoke_result_mode", "isolated") == "isolated":
            continue
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out = dict(row)
                out.update(
                    {
                        "run_id": config.get("run_id", run_dir.parent.name),
                        "model_name": config.get("model_name", run_dir.name),
                        "metrics_path": str(metrics_path),
                        "checkpoint": manifest.get("best_checkpoint", str(run_dir / "best_checkpoint.pt")),
                        "eligible_for_fair_main_table": config.get("eligible_for_fair_main_table"),
                        "loss_function": config.get("loss_function"),
                        "loss_protocol": config.get("loss_protocol"),
                        "smoke": bool(config.get("smoke", False)),
                    }
                )
                for key in ["horizon", "MAE", "RMSE", "R2", "SMAPE", "MAPE", "official_align_score", "score", "Score"]:
                    if key in out and out[key] not in {None, ""}:
                        try:
                            out[key] = float(out[key])
                        except ValueError:
                            pass
                rows.append(out)
    rows.sort(key=lambda r: (float(r.get("horizon", 0)), float(r.get("score", r.get("Score", float("inf"))))))
    return rows


def write_summary(
    results_root: str | Path = "custom_models/results/st_mgprompt",
    output_csv: str | Path = "custom_models/results/metrics_all_custom.csv",
    output_json: str | Path = "custom_models/results/metrics_all_custom.json",
) -> dict[str, Any]:
    rows = collect_metric_rows(results_root)
    csv_path = resolve_project_path(output_csv)
    json_path = resolve_project_path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "horizon",
        "model_name",
        "run_id",
        "score",
        "Score",
        "MAE",
        "RMSE",
        "R2",
        "SMAPE",
        "MAPE",
        "official_align_score",
        "num_valid_points",
        "eligible_for_fair_main_table",
        "loss_function",
        "loss_protocol",
        "checkpoint",
        "metrics_path",
        "smoke",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    payload = {
        "score_lower_is_better": True,
        "sorted_by": ["horizon", "score"],
        "num_rows": len(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "num_rows": len(rows)}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_seed_aggregate(run_root: str | Path, model_name: str) -> dict[str, Any]:
    root = resolve_project_path(run_root)
    seed_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"run_root": str(root), "model_name": model_name, "seeds": []}
    metric_paths = sorted({path.resolve() for path in root.glob(f"seed_*/{model_name}/metrics.csv")})
    for metrics_path in metric_paths:
        run_dir = metrics_path.parent
        seed_text = run_dir.parent.name.replace("seed_", "")
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        cfg = _read_json(run_dir / "config.json")
        checkpoint_meta = {}
        try:
            import torch

            checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            checkpoint_meta = {
                "best_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch")),
                "best_val_score": checkpoint.get("best_metric_value"),
            }
        except Exception as exc:
            checkpoint_meta = {"checkpoint_read_error": str(exc)}
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seed_rows.append(
                    {
                        "seed": seed,
                        "model": cfg.get("model_name", model_name),
                        "horizon": int(float(row["horizon"])),
                        "MAE": _to_float(row.get("MAE")),
                        "RMSE": _to_float(row.get("RMSE")),
                        "R2": _to_float(row.get("R2")),
                        "Score": _to_float(row.get("Score", row.get("score"))),
                        "best_epoch": checkpoint_meta.get("best_epoch"),
                        "best_val_score": checkpoint_meta.get("best_val_score"),
                    }
                )
        manifest["seeds"].append({"seed": seed, "run_dir": str(run_dir), **checkpoint_meta})
    seed_rows.sort(key=lambda r: (r["horizon"], r["seed"]))
    seedwise_path = root / "metrics_seedwise.csv"
    mean_std_path = root / "metrics_mean_std.csv"
    manifest_path = root / "seed_run_manifest.json"
    if seed_rows:
        with seedwise_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["seed", "model", "horizon", "MAE", "RMSE", "R2", "Score", "best_epoch", "best_val_score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seed_rows)
        aggregate_rows = []
        horizons = sorted({row["horizon"] for row in seed_rows})
        for horizon in horizons:
            group = [row for row in seed_rows if row["horizon"] == horizon]
            out: dict[str, Any] = {"model": model_name, "horizon": horizon, "num_seeds": len(group)}
            for metric in ["MAE", "RMSE", "R2", "Score"]:
                values = [float(row[metric]) for row in group]
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate_rows.append(out)
        with mean_std_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "model",
                "horizon",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "RMSE_std",
                "R2_mean",
                "R2_std",
                "Score_mean",
                "Score_std",
                "num_seeds",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_rows)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "metrics_seedwise": str(seedwise_path),
        "metrics_mean_std": str(mean_std_path),
        "seed_run_manifest": str(manifest_path),
        "num_seed_rows": len(seed_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ST-MGPrompt result metrics.")
    parser.add_argument("--results-root", default="custom_models/results/st_mgprompt")
    parser.add_argument("--output-csv", default="custom_models/results/metrics_all_custom.csv")
    parser.add_argument("--output-json", default="custom_models/results/metrics_all_custom.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(write_summary(args.results_root, args.output_csv, args.output_json), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
