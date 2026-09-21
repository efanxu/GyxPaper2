from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from st_mgprompt.config import resolve_project_path
from st_mgprompt.empirical_protocol import EMPIRICAL_PROTOCOL_ID, EMPIRICAL_RESULT_ROOT


HORIZONS = (3, 6, 10)
METRICS = ("Score", "MAE", "RMSE", "R2")
REQUIRED_RUN_ARTIFACTS = (
    "active_config.json",
    "config.json",
    "run_status.json",
    "train_log.csv",
    "best_checkpoint.pt",
    "metrics.csv",
    "metrics_eval_h3.json",
    "metrics_eval_h6.json",
    "metrics_eval_h10.json",
    "model_summary.json",
    "evaluation_complete.json",
)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_status(run_dir: Path) -> str:
    if all((run_dir / name).is_file() for name in REQUIRED_RUN_ARTIFACTS):
        return "completed"
    if (run_dir / "failure.json").is_file():
        return "failed"
    status = _json(run_dir / "run_status.json")
    if status.get("current_stage") == "PROCESS_EXCEPTION":
        return "failed"
    return "partial" if any(run_dir.iterdir()) else "pending"


def _metric_rows(
    metrics_dir: Path,
    *,
    family: str,
    variant_id: str,
    display_name: str,
    run_id: str,
    seed: int | None,
    model_name: str,
    status: str,
    reference_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        path = metrics_dir / f"metrics_eval_h{horizon}.json"
        metric = _json(path)
        if not metric:
            continue
        row: dict[str, Any] = {
            "protocol_id": EMPIRICAL_PROTOCOL_ID,
            "family": family,
            "variant_id": variant_id,
            "display_name": display_name,
            "run_id": run_id,
            "seed": seed,
            "model_name": model_name,
            "horizon": horizon,
            "status": status,
            "reference_only": reference_only,
            "metrics_source": str(path.resolve()),
        }
        for name in METRICS:
            value = metric.get(name, metric.get(name.lower()))
            row[name] = _finite_float(value)
        row["valid_target_count"] = metric.get("valid_target_count", metric.get("num_valid_points"))
        rows.append(row)
    return rows


def collect_empirical_rows(results_root: str | Path = EMPIRICAL_RESULT_ROOT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = resolve_project_path(results_root)
    metric_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return metric_rows, status_rows
    for family_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "summary"):
        for run_root in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            dry = _json(run_root / "dry_run_report.json")
            variant_id = str(dry.get("variant_id") or run_root.name.split("_")[0]).upper()
            display_name = str(dry.get("display_name") or variant_id)
            reference = _json(run_root / "reference.json")
            if reference:
                source = Path(str(reference.get("source_run_dir", "")))
                source_config = _json(source / "active_config.json") or _json(source / "config.json")
                seed_value = source_config.get("seed")
                seed = int(seed_value) if isinstance(seed_value, (int, float)) else None
                source_exists = source.is_dir()
                status = "reference-only" if source_exists else "partial"
                status_rows.append(
                    {
                        "family": family_dir.name,
                        "variant_id": variant_id,
                        "display_name": reference.get("display_name", display_name),
                        "run_id": run_root.name,
                        "seed": seed,
                        "model_name": source_config.get("model_name"),
                        "status": status,
                        "reference_only": True,
                        "run_dir": str(run_root.resolve()),
                        "metrics_source_dir": str(source.resolve()),
                    }
                )
                if source_exists:
                    metric_rows.extend(
                        _metric_rows(
                            source,
                            family=family_dir.name,
                            variant_id=variant_id,
                            display_name=str(reference.get("display_name", display_name)),
                            run_id=run_root.name,
                            seed=seed,
                            model_name=str(source_config.get("model_name") or source.name),
                            status=status,
                            reference_only=True,
                        )
                    )
                continue
            seed_dirs = sorted(path for path in run_root.glob("seed_*") if path.is_dir())
            if not seed_dirs and dry.get("implementation_status") == "pending":
                status_rows.append(
                    {
                        "family": family_dir.name,
                        "variant_id": variant_id,
                        "display_name": display_name,
                        "run_id": run_root.name,
                        "seed": None,
                        "model_name": None,
                        "status": "pending",
                        "reference_only": False,
                        "run_dir": str(run_root.resolve()),
                        "metrics_source_dir": None,
                    }
                )
            for seed_dir in seed_dirs:
                try:
                    seed = int(seed_dir.name.removeprefix("seed_"))
                except ValueError:
                    continue
                model_dirs = sorted(path for path in seed_dir.iterdir() if path.is_dir())
                for model_dir in model_dirs:
                    identity = _json(model_dir / "empirical_identity.json")
                    variant_meta = identity.get("variant") if isinstance(identity.get("variant"), dict) else {}
                    current_variant = str(variant_meta.get("variant_id") or variant_id)
                    current_display = str(variant_meta.get("display_name") or display_name)
                    status = _run_status(model_dir)
                    status_rows.append(
                        {
                            "family": family_dir.name,
                            "variant_id": current_variant,
                            "display_name": current_display,
                            "run_id": run_root.name,
                            "seed": seed,
                            "model_name": model_dir.name,
                            "status": status,
                            "reference_only": False,
                            "run_dir": str(model_dir.resolve()),
                            "metrics_source_dir": str(model_dir.resolve()),
                        }
                    )
                    metric_rows.extend(
                        _metric_rows(
                            model_dir,
                            family=family_dir.name,
                            variant_id=current_variant,
                            display_name=current_display,
                            run_id=run_root.name,
                            seed=seed,
                            model_name=model_dir.name,
                            status=status,
                            reference_only=False,
                        )
                    )
    metric_rows.sort(key=lambda row: (row["family"], row["variant_id"], row["horizon"], row["seed"] or -1))
    status_rows.sort(key=lambda row: (row["family"], row["variant_id"], row["seed"] or -1))
    return metric_rows, status_rows


def _wide_rows(metric_rows: list[dict[str, Any]], status_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for status in status_rows:
        key = (status["family"], status["variant_id"], status["run_id"], status["seed"], status["model_name"])
        grouped[key] = dict(status)
    for row in metric_rows:
        key = (row["family"], row["variant_id"], row["run_id"], row["seed"], row["model_name"])
        wide = grouped.setdefault(key, {name: row.get(name) for name in ("family", "variant_id", "display_name", "run_id", "seed", "model_name", "status", "reference_only")})
        for metric in METRICS:
            wide[f"{metric}_H{row['horizon']}"] = row.get(metric)
    return list(grouped.values())


def _aggregate_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[
            (
                row["family"],
                row["variant_id"],
                row["display_name"],
                row["run_id"],
                int(row["horizon"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (family, variant_id, display_name, run_id, horizon), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "family": family,
            "variant_id": variant_id,
            "display_name": display_name,
            "run_id": run_id,
            "horizon": horizon,
            "status": "completed" if all(item["status"] in {"completed", "reference-only"} for item in group) else "partial",
        }
        seeds = sorted({item["seed"] for item in group if item["seed"] is not None})
        row["seed_count"] = len(seeds)
        row["seeds"] = json.dumps(seeds, ensure_ascii=False)
        for metric in METRICS:
            values_by_seed = {
                str(item["seed"]): item[metric]
                for item in group
                if item.get("seed") is not None and item.get(metric) is not None
            }
            values = list(values_by_seed.values())
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if len(values) == 1 else None)
            row[f"{metric}_seed_count"] = len(values)
            row[f"{metric}_single_seed_values"] = json.dumps(values_by_seed, ensure_ascii=False, sort_keys=True)
        output.append(row)
    return output


def summarize_empirical(results_root: str | Path = EMPIRICAL_RESULT_ROOT) -> dict[str, Any]:
    root = resolve_project_path(results_root)
    summary_root = root / "summary"
    metric_rows, status_rows = collect_empirical_rows(root)
    wide_rows = _wide_rows(metric_rows, status_rows)
    aggregate_rows = _aggregate_rows(metric_rows)
    long_path = summary_root / "empirical_metrics_long.csv"
    wide_path = summary_root / "empirical_metrics_wide.csv"
    seed_path = summary_root / "seed_summary.csv"
    status_path = summary_root / "run_status_summary.csv"
    _write_csv(long_path, metric_rows)
    _write_csv(wide_path, wide_rows)
    _write_csv(seed_path, aggregate_rows)
    _write_csv(status_path, status_rows)
    manifest = {
        "protocol_id": EMPIRICAL_PROTOCOL_ID,
        "results_root": str(root.resolve()),
        "long_row_count": len(metric_rows),
        "wide_row_count": len(wide_rows),
        "aggregate_row_count": len(aggregate_rows),
        "status_counts": {
            status: sum(row["status"] == status for row in status_rows)
            for status in sorted({row["status"] for row in status_rows})
        },
        "missing_values_are_not_zero_filled": True,
        "outputs": [str(path.resolve()) for path in (long_path, wide_path, seed_path, status_path)],
    }
    summary_root.mkdir(parents=True, exist_ok=True)
    (summary_root / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize empirical_analysis_v1 results.")
    parser.add_argument("--results-root", default=EMPIRICAL_RESULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(summarize_empirical(args.results_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
