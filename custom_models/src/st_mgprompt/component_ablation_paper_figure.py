from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RESULT_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "st_mgprompt_component_ablation"
    / "component_ablation_fixed_dual_seed2026"
)
CANONICAL_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "st_mgprompt_canonical"
    / "full_fixed_dual_keep_msmgdwu_seed2026"
)
OUTPUT_ROOT = RESULT_ROOT / "paper_figure"
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
FIGURE_DATA_ROOT = OUTPUT_ROOT / "figure_data"
FIGURE_ROOT = OUTPUT_ROOT / "figures"

HORIZONS = (3, 6, 10)
EXPECTED_SHAPE = (5104, 134, 10)
EXPECTED_COUNTS = {3: 1_256_027, 6: 2_512_061, 10: 4_186_848}
VARIANT_NAMES = {
    "A0": "Canonical Full",
    "A1": "w/o Spatial Graph",
    "A2": "w/o Adaptive Graph",
    "A3": "w/o Diffusion",
    "A4": "Mean-Pooling Macro Prompt",
    "A5": "Short-context Reverse Cross",
    "A6": "Early-History Macro Cross Fusion",
    "A7": "Shared-Projection Cross Fusion",
    "A8": "w/o MS-MG-DWU",
}
COLORS = {
    "A0": "#1976D2",
    "A1": "#F5A623",
    "A2": "#26B6B2",
    "A3": "#8C7BD6",
    "A4": "#F06464",
    "A5": "#62A8E5",
    "A6": "#E6B85C",
    "A7": "#57B9A8",
    "A8": "#B07AA1",
}
METRIC_COLORS = {**COLORS, "A0": "#4F98D3"}
LINE_STYLES = {
    "A0": "-",
    "A1": "--",
    "A2": "-.",
    "A3": ":",
    "A4": (0, (5, 2)),
    "A5": (0, (3, 1, 1, 1)),
    "A6": (0, (7, 2)),
    "A7": (0, (2, 2)),
    "A8": (0, (4, 1, 1, 1)),
}
HORIZON_COLORS = {3: "#1976D2", 6: "#F5A623", 10: "#EF5350"}
METRIC_KEYS = ("MAE", "RMSE", "R2", "Score")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def variant_run_dir(variant: str) -> Path:
    if variant == "A0":
        return CANONICAL_ROOT
    return RESULT_ROOT / variant / "STMGPrompt_ComponentAblation"


def official_metrics(variant: str) -> dict[int, dict[str, float]]:
    run_dir = variant_run_dir(variant)
    result: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        row = read_json(run_dir / f"metrics_eval_h{horizon}.json")
        result[horizon] = {key: float(row[key]) for key in METRIC_KEYS}
    return result


def load_graph_data(config: Any) -> dict[str, Any]:
    graph_dir = config.resolve_path(config.graph_output_root) / config.graph_tag
    return {
        "A_macro_trend": np.load(
            graph_dir / "macro_trend_adjacency.npy", allow_pickle=False
        ),
        "A_micro_local": np.load(
            graph_dir / "micro_local_adjacency.npy", allow_pickle=False
        ),
        "metadata": read_json(graph_dir / "metadata.json"),
    }


def protocol_audit() -> dict[str, Any]:
    import torch

    from st_mgprompt.config import STMGPromptConfig
    from st_mgprompt.experiment_protocol import config_diff, semantic_config

    canonical_config = STMGPromptConfig.from_json(CANONICAL_ROOT / "config.json")
    canonical_semantic = semantic_config(canonical_config)
    records = []
    for variant in VARIANT_NAMES:
        run_dir = variant_run_dir(variant)
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        config = STMGPromptConfig.from_json(config_path)
        config.validate()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_config = checkpoint.get("config", {})
        file_semantic = semantic_config(config)
        checkpoint_semantic = semantic_config(checkpoint_config)
        semantic_mismatches = sorted(
            key
            for key in file_semantic
            if file_semantic.get(key) != checkpoint_semantic.get(key)
        )
        if variant == "A0":
            diff = {
                "passed": file_semantic == canonical_semantic,
                "actual_diff_fields": [],
                "unexpected_diff_fields": [],
                "missing_expected_diff_fields": [],
            }
        else:
            diff = config_diff(config, variant, family="component_ablation")
        protocol_path = run_dir / "protocol_check.json"
        protocol_passed = bool(
            protocol_path.exists() and read_json(protocol_path).get("passed")
        )
        metrics_present = all(
            (run_dir / f"metrics_eval_h{horizon}.json").is_file()
            for horizon in HORIZONS
        )
        accepted_checkpoint_mismatches = (
            {"site_weight_mode", "vadsp_gate_mode"} if variant == "A0" else set()
        )
        checkpoint_semantic_passed = set(semantic_mismatches).issubset(
            accepted_checkpoint_mismatches
        )
        passed = bool(
            checkpoint_path.is_file()
            and metrics_present
            and protocol_passed
            and diff["passed"]
            and checkpoint_semantic_passed
        )
        note = None
        if variant == "A1":
            note = (
                "The formal checkpoint and config.json are a valid A1 run. "
                "A later incomplete launch overwrote active_config.json and "
                "requested_config.json with A8 values and left an orphan "
                "TRAIN_STARTED event; those mutable files are not used here."
            )
        records.append(
            {
                "variant": variant,
                "display_name": VARIANT_NAMES[variant],
                "run_dir": str(run_dir.resolve()),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_best_epoch": checkpoint.get("best_epoch"),
                "protocol_check_passed": protocol_passed,
                "metrics_present": metrics_present,
                "formal_config_diff": diff,
                "checkpoint_config_semantic_mismatches": semantic_mismatches,
                "accepted_checkpoint_semantic_mismatches": sorted(
                    accepted_checkpoint_mismatches
                ),
                "passed": passed,
                "note": note,
            }
        )
    all_passed = all(row["passed"] for row in records)
    audit = {
        "canonical_source": str(CANONICAL_ROOT.resolve()),
        "component_result_root": str(RESULT_ROOT.resolve()),
        "source_artifacts_modified": False,
        "variants": records,
        "all_variants_protocol_comparable": all_passed,
    }
    write_json(OUTPUT_ROOT / "protocol_audit.json", audit)
    if not all_passed:
        failed = [row["variant"] for row in records if not row["passed"]]
        raise RuntimeError(f"Protocol audit failed for: {failed}")
    return audit


def infer_variant(
    variant: str,
    data: Any,
    graph_data: dict[str, Any],
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    from st_mgprompt.config import STMGPromptConfig
    from st_mgprompt.evaluate import (
        _finalize_stream_accumulator,
        _new_stream_accumulator,
        _update_stream_accumulator,
        apply_physical_clip,
    )
    from st_mgprompt.registry import build_model
    from st_mgprompt.train import _autocast_context, set_seed

    run_dir = variant_run_dir(variant)
    config = STMGPromptConfig.from_json(run_dir / "config.json")
    config.device = device_name
    set_seed(int(config.seed))
    device = torch.device(device_name)
    model = build_model(config, data.input_dim, graph_data=graph_data).to(device)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    validation_batch = next(iter(data.train_loader))
    with torch.inference_mode():
        model(validation_batch["x"].to(device).float())

    pred_parts: list[np.ndarray] = []
    accumulators = {h: _new_stream_accumulator() for h in HORIZONS}
    total_batches = len(data.test_loader)
    with torch.inference_mode():
        for batch_number, batch in enumerate(data.test_loader, start=1):
            x = batch["x"].to(device).float()
            with _autocast_context(device, bool(config.amp_enabled)):
                output = model(x)
            pred_norm = output["pred"].detach().float().cpu().numpy()
            pred_raw = data.scalers["target"].inverse_transform(pred_norm).astype(
                np.float32
            )
            pred_eval = apply_physical_clip(pred_raw, config).astype(np.float32)
            pred_parts.append(pred_eval)
            y_true = batch["y_raw"].numpy().astype(np.float32)
            mask = batch["valid_target_mask"].numpy().astype(bool)
            for horizon in HORIZONS:
                _update_stream_accumulator(
                    accumulators[horizon],
                    pred_eval[:, :horizon, :],
                    y_true[:, :horizon, :],
                    mask[:, :horizon, :],
                    num_nodes=config.num_nodes,
                )
            if batch_number == 1 or batch_number % 256 == 0 or batch_number == total_batches:
                print(
                    f"{variant}: inference batch {batch_number}/{total_batches}",
                    flush=True,
                )

    pred_bhn = np.concatenate(pred_parts, axis=0)
    pred_snh = np.transpose(pred_bhn, (0, 2, 1)).astype(np.float32)
    recomputed = {
        horizon: _finalize_stream_accumulator(
            accumulators[horizon],
            physical_clip_applied=bool(config.enable_physical_clip_eval),
        )
        for horizon in HORIZONS
    }
    return pred_snh, {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "recomputed_metrics": recomputed,
    }


def common_test_arrays(data: Any) -> dict[str, np.ndarray]:
    target_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    sample_indices: list[int] = []
    for batch in data.test_loader:
        target_parts.append(batch["y_raw"].numpy().astype(np.float32))
        mask_parts.append(batch["valid_target_mask"].numpy().astype(bool))
        sample_indices.extend(
            int(value)
            for value in batch["prediction_start_index"].numpy().tolist()
        )
    y_true = np.transpose(np.concatenate(target_parts), (0, 2, 1))
    mask = np.transpose(np.concatenate(mask_parts), (0, 2, 1))
    sample_index = np.asarray(sample_indices, dtype=np.int64)
    dataset = data.test_loader.dataset
    timestamps = np.asarray(
        [str(dataset.timestamps[index]) for index in sample_index], dtype="U32"
    )
    node_ids = np.asarray(data.turbine_ids, dtype=np.int64)
    return {
        "y_true_snh": y_true.astype(np.float32),
        "valid_target_mask_snh": mask.astype(bool),
        "sample_index": sample_index,
        "prediction_start_timestamp": timestamps,
        "node_id": node_ids,
    }


def validate_common_arrays(common: dict[str, np.ndarray]) -> dict[int, int]:
    if common["y_true_snh"].shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected target shape: {common['y_true_snh'].shape}"
        )
    if common["valid_target_mask_snh"].shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected mask shape: {common['valid_target_mask_snh'].shape}"
        )
    counts = {
        horizon: int(
            common["valid_target_mask_snh"][:, :, :horizon].sum()
        )
        for horizon in HORIZONS
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected valid counts: {counts}")
    return counts


def export_predictions(device: str) -> dict[str, Any]:
    import torch

    from st_mgprompt.config import STMGPromptConfig
    from st_mgprompt.data import make_dataloaders

    protocol_audit()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    base_config = STMGPromptConfig.from_json(CANONICAL_ROOT / "config.json")
    base_config.device = device
    data = make_dataloaders(base_config)["bundle"]
    graph_data = load_graph_data(base_config)
    common = common_test_arrays(data)
    valid_counts = validate_common_arrays(common)
    atomic_savez(PREDICTION_ROOT / "common_test_targets.npz", **common)

    records = []
    for variant in VARIANT_NAMES:
        print(f"START {variant}: {VARIANT_NAMES[variant]}", flush=True)
        pred_snh, inference = infer_variant(
            variant, data, graph_data, device
        )
        if pred_snh.shape != EXPECTED_SHAPE:
            raise RuntimeError(f"{variant} prediction shape is {pred_snh.shape}")
        atomic_savez(
            PREDICTION_ROOT / f"{variant}_predictions.npz",
            y_pred_eval_snh=pred_snh,
        )
        official = official_metrics(variant)
        comparisons = []
        for horizon in HORIZONS:
            recomputed = inference["recomputed_metrics"][horizon]
            comparisons.append(
                {
                    "horizon": horizon,
                    "official": official[horizon],
                    "recomputed": {
                        key: float(recomputed[key]) for key in METRIC_KEYS
                    },
                    "absolute_difference": {
                        key: abs(
                            official[horizon][key] - float(recomputed[key])
                        )
                        for key in METRIC_KEYS
                    },
                    "mae_same_at_one_decimal": round(
                        official[horizon]["MAE"], 1
                    )
                    == round(float(recomputed["MAE"]), 1),
                }
            )
        record = {
            "variant": variant,
            "display_name": VARIANT_NAMES[variant],
            "run_dir": str(variant_run_dir(variant).resolve()),
            "prediction_file": str(
                (PREDICTION_ROOT / f"{variant}_predictions.npz").resolve()
            ),
            "prediction_shape": list(pred_snh.shape),
            "checkpoint_epoch": inference["checkpoint_epoch"],
            "checkpoint_best_epoch": inference["checkpoint_best_epoch"],
            "metric_comparison": comparisons,
            "mean_annotation_gate_pass": all(
                row["mae_same_at_one_decimal"] for row in comparisons
            ),
        }
        records.append(record)
        print(
            f"DONE {variant}: mean/MAE display gate "
            f"{'PASS' if record['mean_annotation_gate_pass'] else 'FAIL'}",
            flush=True,
        )
        del pred_snh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = {
        "task": "ST-MGPrompt A0-A8 evaluation-only prediction export",
        "training_performed": False,
        "source_artifacts_modified": False,
        "shape_protocol": "[sample,node,horizon]",
        "target": "Patv_raw",
        "mask": "valid_target_mask",
        "physical_clipping_applied": True,
        "expected_shape": list(EXPECTED_SHAPE),
        "valid_counts": {str(key): value for key, value in valid_counts.items()},
        "common_target_file": str(
            (PREDICTION_ROOT / "common_test_targets.npz").resolve()
        ),
        "variants": records,
        "all_mean_annotation_gates_pass": all(
            row["mean_annotation_gate_pass"] for row in records
        ),
    }
    write_json(PREDICTION_ROOT / "prediction_export_manifest.json", manifest)
    if not manifest["all_mean_annotation_gates_pass"]:
        raise RuntimeError("At least one recomputed MAE does not round to the formal MAE")
    return manifest


def load_common() -> dict[str, np.ndarray]:
    path = PREDICTION_ROOT / "common_test_targets.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_prediction(variant: str) -> np.ndarray:
    path = PREDICTION_ROOT / f"{variant}_predictions.npz"
    with np.load(path, allow_pickle=False) as data:
        return data["y_pred_eval_snh"].copy()


def write_metric_table() -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANT_NAMES:
        metrics = official_metrics(variant)
        for horizon in HORIZONS:
            rows.append(
                {
                    "variant": variant,
                    "display_name": VARIANT_NAMES[variant],
                    "horizon": horizon,
                    **metrics[horizon],
                }
            )
    FIGURE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DATA_ROOT / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_tracking_data(
    common: dict[str, np.ndarray], variants: list[str]
) -> list[dict[str, Any]]:
    y_true = common["y_true_snh"]
    mask = common["valid_target_mask_snh"]
    sample_index = common["sample_index"]
    first_common_target_index = int(sample_index[0]) + max(HORIZONS) - 1
    rows: list[dict[str, Any]] = []
    predictions = {variant: load_prediction(variant) for variant in variants}
    common_points: list[tuple[int, dict[int, int]]] = []
    target_index = first_common_target_index
    last_target_index = int(sample_index[-1]) + min(HORIZONS) - 1
    while target_index <= last_target_index and len(common_points) < 400:
        rows_by_horizon = {
            horizon: target_index - (horizon - 1) - int(sample_index[0])
            for horizon in HORIZONS
        }
        if all(
            0 <= sample_row < len(sample_index)
            and bool(mask[sample_row, :, horizon - 1].any())
            for horizon, sample_row in rows_by_horizon.items()
        ):
            common_points.append((target_index, rows_by_horizon))
        target_index += 1
    if len(common_points) != 400:
        raise RuntimeError(
            f"Only {len(common_points)} common valid tracking timestamps were found"
        )
    for horizon in HORIZONS:
        for point, (target_index, rows_by_horizon) in enumerate(common_points):
            sample_row = rows_by_horizon[horizon]
            horizon_index = horizon - 1
            valid = mask[sample_row, :, horizon_index]
            row: dict[str, Any] = {
                "horizon": horizon,
                "point": point + 1,
                "target_index": target_index,
                "timestamp": str(
                    common["prediction_start_timestamp"][
                        target_index - int(sample_index[0])
                    ]
                ),
                "Actual": float(y_true[sample_row, valid, horizon_index].mean()),
            }
            for variant, prediction in predictions.items():
                row[variant] = float(
                    prediction[sample_row, valid, horizon_index].mean()
                )
            rows.append(row)
    path = FIGURE_DATA_ROOT / "tracking.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_error_distribution_data(
    common: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    y_true = common["y_true_snh"]
    mask = common["valid_target_mask_snh"]
    edges = np.linspace(0.0, 1500.0, 301, dtype=np.float64)
    counts = np.zeros((len(HORIZONS), len(VARIANT_NAMES), len(edges) - 1), dtype=np.int64)
    stats = np.zeros((len(HORIZONS), len(VARIANT_NAMES), 7), dtype=np.float64)
    for variant_index, variant in enumerate(VARIANT_NAMES):
        prediction = load_prediction(variant)
        for horizon_index, horizon in enumerate(HORIZONS):
            valid = mask[:, :, :horizon]
            errors = np.abs(
                prediction[:, :, :horizon] - y_true[:, :, :horizon]
            )[valid].astype(np.float64)
            counts[horizon_index, variant_index], _ = np.histogram(
                errors, bins=edges
            )
            stats[horizon_index, variant_index] = np.asarray(
                [
                    float(errors.mean()),
                    float(np.quantile(errors, 0.05)),
                    float(np.quantile(errors, 0.25)),
                    float(np.quantile(errors, 0.50)),
                    float(np.quantile(errors, 0.75)),
                    float(np.quantile(errors, 0.95)),
                    float(errors.size),
                ]
            )
        del prediction
        gc.collect()
    atomic_savez(
        FIGURE_DATA_ROOT / "error_distribution.npz",
        bin_edges_kw=edges,
        histogram_counts=counts,
        stats=stats,
        horizons=np.asarray(HORIZONS, dtype=np.int64),
        variants=np.asarray(list(VARIANT_NAMES), dtype="U2"),
        stat_names=np.asarray(
            ["mean", "q05", "q25", "median", "q75", "q95", "count"],
            dtype="U8",
        ),
    )
    return {
        "bin_edges_kw": edges,
        "histogram_counts": counts,
        "stats": stats,
    }


def rounded_patch(fig: Any, bounds: tuple[float, float, float, float], **kwargs: Any) -> Any:
    from matplotlib.patches import FancyBboxPatch

    x, y, width, height = bounds
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.004,rounding_size=0.009",
        transform=fig.transFigure,
        clip_on=False,
        **kwargs,
    )
    fig.add_artist(patch)
    return patch


def panel_header(
    fig: Any,
    bounds: tuple[float, float, float, float],
    label: str,
    title: str,
) -> None:
    from matplotlib.patches import Polygon

    x, y, width, height = bounds
    rounded_patch(
        fig,
        bounds,
        facecolor="white",
        edgecolor="#7DD3FC",
        linewidth=0.85,
        linestyle=(0, (3.0, 2.2)),
        zorder=-10,
    )
    header_y = y + height - 0.043
    label_width = 0.057 if width > 0.5 else 0.060
    if label:
        polygon = Polygon(
            [
                [x + 0.006, header_y],
                [x + 0.006 + label_width - 0.012, header_y],
                [x + 0.006 + label_width, header_y + 0.0185],
                [x + 0.006 + label_width - 0.012, header_y + 0.037],
                [x + 0.006, header_y + 0.037],
            ],
            transform=fig.transFigure,
            facecolor="#2E7FB5",
            edgecolor="none",
            zorder=10,
        )
        fig.add_artist(polygon)
        fig.text(
            x + 0.006 + (label_width - 0.010) / 2,
            header_y + 0.0185,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
            zorder=20,
        )
        title_x = x + 0.006 + label_width + 0.008
    else:
        title_x = x + 0.016
    fig.text(
        title_x,
        header_y + 0.025,
        title,
        ha="left",
        va="center",
        color="#082B68",
        fontsize=11.4 if width < 0.5 else 13.2,
        fontweight="bold",
        zorder=20,
    )


def style_axis(ax: Any) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#52606D")
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5, width=0.6, colors="#1F2937")
    ax.grid(axis="y", color="#E6EEF5", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)


def draw_tracking(
    fig: Any,
    tracking_rows: list[dict[str, Any]],
    tracking_variants: list[str],
) -> None:
    panel = (0.018, 0.405, 0.575, 0.578)
    panel_header(
        fig,
        panel,
        "",
        "Forecast tracking",
    )
    row_bounds = [
        (0.120, 0.797, 0.455, 0.130),
        (0.120, 0.614, 0.455, 0.130),
        (0.120, 0.431, 0.455, 0.130),
    ]
    methods = tracking_variants
    horizon_labels = {
        3: "Three-step",
        6: "Six-step",
        10: "Ten-step",
    }
    for horizon, bounds in zip(HORIZONS, row_bounds):
        side_color = HORIZON_COLORS[horizon]
        rounded_patch(
            fig,
            (0.024, bounds[1] - 0.026, 0.036, bounds[3] + 0.052),
            facecolor=side_color + "24",
            edgecolor="none",
            zorder=-2,
        )
        fig.text(
            0.042,
            bounds[1] + bounds[3] / 2,
            horizon_labels[horizon],
            ha="center",
            va="center",
            rotation=90,
            color=side_color,
            fontsize=12.5,
            fontweight="bold",
        )
        rounded_patch(
            fig,
            (0.066, bounds[1] - 0.026, 0.516, bounds[3] + 0.052),
            facecolor="white",
            edgecolor="none",
            zorder=-1,
        )
        ax = fig.add_axes(bounds)
        data = [row for row in tracking_rows if int(row["horizon"]) == horizon]
        x = np.arange(1, 401)
        ax.plot(
            x,
            [row["Actual"] for row in data],
            color="#E60012",
            lw=0.62,
            label="Actual",
            alpha=0.92,
            zorder=12,
        )
        for method in methods:
            is_full_model = method == "A0"
            ax.plot(
                x,
                [row[method] for row in data],
                color=COLORS[method],
                lw=0.76 if is_full_model else 0.43,
                linestyle=LINE_STYLES[method],
                label=method,
                alpha=0.96 if is_full_model else 0.70,
                zorder=11 if is_full_model else 4,
            )
        style_axis(ax)
        ax.set_xlim(0, 400)
        ax.set_xticks(np.arange(0, 401, 50))
        ax.set_ylim(0, 2000)
        ax.set_yticks(np.arange(0, 2001, 400))
        fig.text(
            0.078,
            bounds[1] + bounds[3] / 2,
            "Power output (kW)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=7.2,
            fontweight="bold",
            color="#111827",
        )
        ax.set_xlabel("Testing set", fontsize=7.2, fontweight="bold", labelpad=1)
        handles, labels = ax.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        legend_order = [
            "Actual",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
        ]
        ax.legend(
            [handle_by_label[label] for label in legend_order],
            legend_order,
            ncol=10,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.012),
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.86,
            fontsize=8.5,
            handlelength=0.55,
            handletextpad=0.05,
            columnspacing=0.10,
            labelspacing=0.0,
            borderaxespad=0.0,
            borderpad=0.12,
        )


def draw_metrics(fig: Any, metric_rows: list[dict[str, Any]]) -> None:
    panel = (0.603, 0.405, 0.379, 0.578)
    panel_header(
        fig,
        panel,
        "",
        "Metric comparison",
    )
    chart_bounds = [
        (0.635, 0.814, 0.329, 0.092),
        (0.635, 0.700, 0.329, 0.092),
        (0.635, 0.586, 0.329, 0.092),
        (0.635, 0.472, 0.329, 0.092),
    ]
    labels = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R2", "Score": "Score"}
    variants = list(VARIANT_NAMES)
    lookup = {
        (row["variant"], int(row["horizon"])): row for row in metric_rows
    }
    for metric, bounds in zip(METRIC_KEYS, chart_bounds):
        ax = fig.add_axes(bounds)
        x = np.arange(len(HORIZONS), dtype=float) * 0.64
        width = 0.050
        for variant_index, variant in enumerate(variants):
            offset = (variant_index - (len(variants) - 1) / 2) * width
            values = [lookup[(variant, horizon)][metric] for horizon in HORIZONS]
            ax.bar(
                x + offset,
                values,
                width=width * 0.86,
                color=METRIC_COLORS[variant],
                edgecolor="white" if variant != "A0" else "#2C6FA3",
                linewidth=0.18 if variant != "A0" else 0.48,
                alpha=0.90,
                zorder=3,
            )
        style_axis(ax)
        ax.set_title(
            labels[metric],
            fontsize=7.7,
            fontweight="bold",
            y=0.91,
            pad=0.0,
            color="#111827",
        )
        ax.set_xticks(x, ["H=3", "H=6", "H=10"])
        if metric == "R2":
            values = [float(row[metric]) for row in metric_rows]
            lower = max(0.0, math.floor((min(values) - 0.02) * 20) / 20)
            ax.set_ylim(lower, 1.0)
        else:
            maximum = max(float(row[metric]) for row in metric_rows)
            ax.set_ylim(0, maximum * 1.26)
        ax.set_xlim(-0.28, x[-1] + 0.28)
        ax.tick_params(labelsize=5.35, length=1.8, width=0.45, pad=0.8)
        ax.yaxis.set_major_locator(__import__("matplotlib.ticker", fromlist=["MaxNLocator"]).MaxNLocator(3))

    legend_x = [0.650, 0.757, 0.864]
    legend_y = [0.449, 0.430, 0.411]
    for index, variant in enumerate(variants):
        col = index % 3
        row = index // 3
        x = legend_x[col]
        y = legend_y[row]
        rounded_patch(
            fig,
            (x, y, 0.008, 0.008),
            facecolor=METRIC_COLORS[variant],
            edgecolor="none",
            zorder=2,
        )
        fig.text(
            x + 0.011,
            y + 0.004,
            variant,
            ha="left",
            va="center",
            fontsize=5.8,
            fontweight="bold" if variant == "A0" else "normal",
            color="#172554",
        )


def draw_violin_from_hist(
    ax: Any,
    x: float,
    edges: np.ndarray,
    counts: np.ndarray,
    color: str,
    width: float = 0.52,
) -> None:
    centers = (edges[:-1] + edges[1:]) / 2
    kernel_x = np.arange(-24, 25, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 7.0) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts.astype(float), kernel, mode="same")
    cumulative = np.cumsum(counts, dtype=float)
    if cumulative[-1] > 0:
        body_index = min(
            int(np.searchsorted(cumulative, cumulative[-1] * 0.997, side="left")),
            len(centers) - 1,
        )
        body_top = centers[body_index]
        rare_mask = (centers > body_top) & (counts > 0)
        smooth = np.where(centers <= body_top, smooth, 0.0)
    else:
        rare_mask = np.zeros_like(counts, dtype=bool)
    if smooth.max() > 0:
        half_width = width * np.power(smooth / smooth.max(), 0.50)
    else:
        half_width = np.zeros_like(smooth)
    ax.fill_betweenx(
        centers,
        x - half_width,
        x + half_width,
        facecolor=color,
        edgecolor=color,
        linewidth=0.45,
        alpha=0.68,
        zorder=2,
    )
    if np.any(rare_mask):
        rare_centers = centers[rare_mask]
        rare_counts = counts[rare_mask].astype(float)
        keep = np.unique(
            np.linspace(
                0,
                rare_centers.size - 1,
                min(8, rare_centers.size),
                dtype=int,
            )
        )
        rare_centers = rare_centers[keep]
        rare_counts = rare_counts[keep]
        rare_index = np.arange(rare_centers.size, dtype=float)
        rare_x = x + 0.16 * np.sin(rare_index * 2.35 + x * 0.70)
        ax.scatter(
            rare_x,
            rare_centers,
            s=0.08 + 0.04 * np.log1p(rare_counts),
            facecolor=color,
            edgecolor="none",
            alpha=0.11,
            zorder=1,
        )


def draw_error_distributions(
    fig: Any,
    distribution: dict[str, np.ndarray],
) -> None:
    panel = (0.018, 0.080, 0.964, 0.305)
    panel_header(
        fig,
        panel,
        "",
        "Error distribution analysis",
    )
    bounds_list = [
        (0.060, 0.105, 0.282, 0.198),
        (0.366, 0.105, 0.282, 0.198),
        (0.672, 0.105, 0.282, 0.198),
    ]
    edges = distribution["bin_edges_kw"]
    counts = distribution["histogram_counts"]
    stats = distribution["stats"]
    y_max = float(edges[-1])
    variants = list(VARIANT_NAMES)
    horizon_titles = {
        3: "Three-step",
        6: "Six-step",
        10: "Ten-step",
    }
    for horizon_index, (horizon, bounds) in enumerate(zip(HORIZONS, bounds_list)):
        rounded_patch(
            fig,
            (bounds[0] - 0.004, bounds[1] + bounds[3] + 0.001, bounds[2] + 0.008, 0.027),
            facecolor=HORIZON_COLORS[horizon] + "20",
            edgecolor="none",
            zorder=0,
        )
        fig.text(
            bounds[0] + bounds[2] / 2,
            bounds[1] + bounds[3] + 0.0145,
            horizon_titles[horizon],
            ha="center",
            va="center",
            color=HORIZON_COLORS[horizon],
            fontsize=8.9,
            fontweight="bold",
        )
        ax = fig.add_axes(bounds)
        for variant_index, variant in enumerate(variants):
            x = variant_index
            draw_violin_from_hist(
                ax,
                x,
                edges,
                counts[horizon_index, variant_index],
                COLORS[variant],
            )
            mean, q05, q25, median, q75, q95, _ = stats[horizon_index, variant_index]
            whisker_low = q25 - 0.45 * (q25 - q05)
            whisker_high = q75 + 0.45 * (q95 - q75)
            ax.vlines(x, whisker_low, whisker_high, color="#273444", lw=0.62, zorder=4)
            ax.hlines(
                [whisker_low, whisker_high],
                x - 0.085,
                x + 0.085,
                color="#273444",
                lw=0.52,
                zorder=4,
            )
            ax.add_patch(
                __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                    (x - 0.17, q25),
                    0.34,
                    q75 - q25,
                    facecolor=COLORS[variant],
                    edgecolor="#273444",
                    linewidth=0.6,
                    alpha=0.85,
                    zorder=5,
                )
            )
            ax.hlines(median, x - 0.17, x + 0.17, color="white", lw=1.0, zorder=6)
            ax.scatter([x], [mean], s=8, facecolor="white", edgecolor="#273444", linewidth=0.45, zorder=7)
            label_y = y_max * 0.945
            ax.text(
                x,
                label_y,
                f"Mean\n{mean:.1f}",
                ha="center",
                va="top",
                fontsize=5.6,
                color=COLORS[variant],
                fontweight="bold" if variant == "A0" else "normal",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 0.12,
                },
                zorder=8,
            )
        style_axis(ax)
        ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.set_ylim(0, y_max)
        ax.set_xticks(np.arange(len(variants)), variants)
        ax.set_xlabel("Ablation variant", fontsize=7.0, fontweight="bold", labelpad=2)
        ax.tick_params(axis="x", labelsize=6.0, pad=2)
        if horizon_index == 0:
            ax.set_ylabel(
                "Absolute error (kW)",
                fontsize=6.2,
                fontweight="bold",
                labelpad=-0.8,
            )


def render_figure(
    metric_rows: list[dict[str, Any]],
    tracking_rows: list[dict[str, Any]],
    tracking_variants: list[str],
    distribution: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(8.0, 10.0), facecolor="white")
    draw_tracking(fig, tracking_rows, tracking_variants)
    draw_metrics(fig, metric_rows)
    draw_error_distributions(fig, distribution)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_ROOT / "st_mgprompt_component_ablation_A0_A8.pdf"
    svg_path = FIGURE_ROOT / "st_mgprompt_component_ablation_A0_A8.svg"
    png_path = FIGURE_ROOT / "st_mgprompt_component_ablation_A0_A8_600dpi.png"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)


def build_figure_data_and_plot() -> dict[str, Any]:
    protocol_audit()
    manifest_path = PREDICTION_ROOT / "prediction_export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Prediction exports are missing. Run the export command first."
        )
    prediction_manifest = read_json(manifest_path)
    if not prediction_manifest.get("all_mean_annotation_gates_pass"):
        raise RuntimeError("Prediction export metric gate did not pass")
    common = load_common()
    validate_common_arrays(common)
    metric_rows = write_metric_table()
    tracking_variants = list(VARIANT_NAMES)
    tracking_rows = build_tracking_data(common, tracking_variants)
    distribution = build_error_distribution_data(common)

    mean_checks = []
    official_lookup = {
        (row["variant"], int(row["horizon"])): float(row["MAE"])
        for row in metric_rows
    }
    for horizon_index, horizon in enumerate(HORIZONS):
        for variant_index, variant in enumerate(VARIANT_NAMES):
            mean = float(distribution["stats"][horizon_index, variant_index, 0])
            official = official_lookup[(variant, horizon)]
            mean_checks.append(
                {
                    "variant": variant,
                    "horizon": horizon,
                    "distribution_mean": mean,
                    "official_mae": official,
                    "absolute_difference": abs(mean - official),
                    "same_at_one_decimal": round(mean, 1) == round(official, 1),
                }
            )
    if not all(row["same_at_one_decimal"] for row in mean_checks):
        raise RuntimeError("Distribution means fail the MAE display-rounding gate")
    write_json(FIGURE_DATA_ROOT / "mean_mae_audit.json", mean_checks)
    write_json(
        FIGURE_DATA_ROOT / "tracking_selection.json",
        {
            "selection_rule": "Show the canonical model and every A1-A8 ablation",
            "selected_variants": tracking_variants,
            "selected_names": [
                VARIANT_NAMES[variant] for variant in tracking_variants
            ],
            "window": "first 400 common target timestamps aligned across H3/H6/H10",
            "aggregation": "mean Patv over valid_target_mask turbines",
        },
    )
    render_figure(metric_rows, tracking_rows, tracking_variants, distribution)
    result = {
        "tracking_variants": tracking_variants,
        "metric_rows": len(metric_rows),
        "tracking_rows": len(tracking_rows),
        "mean_mae_gate_pass": True,
        "outputs": {
            "pdf": str(
                (FIGURE_ROOT / "st_mgprompt_component_ablation_A0_A8.pdf").resolve()
            ),
            "svg": str(
                (FIGURE_ROOT / "st_mgprompt_component_ablation_A0_A8.svg").resolve()
            ),
            "png_600dpi": str(
                (FIGURE_ROOT / "st_mgprompt_component_ablation_A0_A8_600dpi.png").resolve()
            ),
        },
    }
    write_json(OUTPUT_ROOT / "figure_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and plot the ST-MGPrompt A0-A8 paper ablation figure."
    )
    parser.add_argument(
        "command", choices=["audit", "export", "plot", "all"], nargs="?", default="all"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "audit":
        print(json.dumps(protocol_audit(), indent=2, ensure_ascii=False))
        return 0
    if args.command in {"export", "all"}:
        export_predictions(args.device)
    if args.command in {"plot", "all"}:
        result = build_figure_data_and_plot()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
