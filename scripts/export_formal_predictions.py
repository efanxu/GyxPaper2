from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RESULT_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "benchmark_v2_uniform_bs4"
)
EXPORT_ROOT = RESULT_ROOT / "prediction_exports"
CANDIDATE_ROOT = EXPORT_ROOT / "candidates"
CANONICAL_ROOT = (
    PROJECT_ROOT
    / "custom_models"
    / "results"
    / "st_mgprompt_canonical"
    / "full_fixed_dual_keep_msmgdwu_seed2026"
)
INPUT_PATH = PROJECT_ROOT / "dataset" / "sdwpf_model_input_base.parquet"
TARGET_PATH = PROJECT_ROOT / "dataset" / "sdwpf_eval_target.parquet"

EXPECTED_SHAPE = (5104, 134, 10)
EXPECTED_COUNTS = {3: 1_256_027, 6: 2_512_061, 10: 4_186_848}
METRIC_NAMES = ("Score", "MAE", "RMSE", "R2")
METRIC_TOLERANCE = 1e-5

MODEL_SPECS = {
    "proposed_model": {
        "display_name": "Proposed model",
        "model_id": "st_mgprompt_canonical",
        "run_dir": CANONICAL_ROOT,
        "export_name": "proposed_model_predictions.npz",
    },
    "lightts": {
        "display_name": "LightTS",
        "model_id": "lightts",
        "run_dir": RESULT_ROOT
        / "basic_lightweight_seed2026"
        / "LightTS_node_shared_chunk8_bs4_seed2026",
        "export_name": "lightts_predictions.npz",
    },
    "movingaverage": {
        "display_name": "MovingAverage",
        "model_id": "moving_average",
        "run_dir": RESULT_ROOT
        / "basic_lightweight_seed2026"
        / "MovingAverage_w144_bs4_seed2026",
        "export_name": "movingaverage_predictions.npz",
    },
    "tide": {
        "display_name": "TiDE",
        "model_id": "tide",
        "run_dir": RESULT_ROOT
        / "basic_lightweight_seed2026"
        / "TiDE_node_shared_d512_bs4_seed2026",
        "export_name": "tide_predictions.npz",
    },
    "transformer": {
        "display_name": "Transformer",
        "model_id": "transformer",
        "run_dir": RESULT_ROOT
        / "e2_a_seed2026"
        / "Transformer_node_shared_d512_bs4_seed2026",
        "export_name": "transformer_predictions.npz",
    },
    "micn": {
        "display_name": "MICN",
        "model_id": "micn",
        "run_dir": RESULT_ROOT
        / "e2_b_seed2026"
        / "MICN_node_shared_k12_16_d32_bs4_seed2026",
        "export_name": "micn_predictions.npz",
    },
    "timesnet": {
        "display_name": "TimesNet",
        "model_id": "timesnet",
        "run_dir": RESULT_ROOT
        / "e2_b_seed2026"
        / "TimesNet_node_shared_d32_k5_bs4_seed2026",
        "export_name": "timesnet_predictions.npz",
    },
    "wpmixer": {
        "display_name": "WPMixer",
        "model_id": "wpmixer",
        "run_dir": RESULT_ROOT
        / "e2_b_seed2026"
        / "WPMixer_node_shared_db2_l1_p16_s8_d256_bs4_seed2026",
        "export_name": "wpmixer_predictions.npz",
    },
    "frets": {
        "display_name": "FreTS",
        "model_id": "frets",
        "run_dir": RESULT_ROOT
        / "e2_c_seed2026"
        / "FreTS_node_shared_tcfft_e128_h256_bs4_seed2026",
        "export_name": "frets_predictions.npz",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def official_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for horizon in (3, 6, 10):
        row = read_json(run_dir / f"metrics_eval_h{horizon}.json")
        result[str(horizon)] = {
            name: float(row[name]) for name in METRIC_NAMES
        }
    return result


def select_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        str(int(row["horizon"])): {
            name: float(row[name]) for name in METRIC_NAMES
        }
        for row in rows
    }


def compare_metrics(
    display_name: str,
    official: dict[str, dict[str, float]],
    recomputed: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], float, bool]:
    comparisons = []
    max_diff = 0.0
    passed = True
    print("Model | Horizon | Metric | Official | Recomputed | AbsDiff | RelDiff | Status")
    for horizon in (3, 6, 10):
        for metric in METRIC_NAMES:
            left = official[str(horizon)][metric]
            right = recomputed[str(horizon)][metric]
            diff = abs(left - right)
            rel_diff = diff / max(abs(left), 1e-12)
            ok = bool(np.isfinite(diff) and diff <= METRIC_TOLERANCE)
            max_diff = max(max_diff, diff)
            passed = passed and ok
            row = {
                "model": display_name,
                "horizon": horizon,
                "metric": metric,
                "official": left,
                "recomputed": right,
                "absolute_difference": diff,
                "relative_difference": rel_diff,
                "pass": ok,
                "status": "EXACT_PASS" if ok else "OUTSIDE_EXACT_TOLERANCE",
            }
            comparisons.append(row)
            print(
                f"{display_name} | H{horizon} | {metric} | {left:.12g} | "
                f"{right:.12g} | {diff:.6g} | {rel_diff:.6g} | "
                f"{'EXACT_PASS' if ok else 'OUTSIDE_EXACT_TOLERANCE'}",
                flush=True,
            )
    return comparisons, max_diff, passed


def validate_arrays(
    y_true: np.ndarray,
    y_pred_eval: np.ndarray,
    mask: np.ndarray,
    sample_index: np.ndarray,
) -> dict[str, int]:
    if y_true.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"y_true shape {y_true.shape} != {EXPECTED_SHAPE}")
    if y_pred_eval.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"y_pred_eval shape {y_pred_eval.shape} != {EXPECTED_SHAPE}")
    if mask.shape != EXPECTED_SHAPE:
        raise RuntimeError(f"mask shape {mask.shape} != {EXPECTED_SHAPE}")
    if sample_index.shape != (EXPECTED_SHAPE[0],):
        raise RuntimeError(f"sample_index shape {sample_index.shape} is invalid")
    counts = {
        horizon: int(mask[:, :, :horizon].sum()) for horizon in (3, 6, 10)
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"valid counts {counts} != {EXPECTED_COUNTS}")
    return counts


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def benchmark_dataset_signature(provider) -> dict[str, Any]:
    return {
        "dataset_id": "SDWPF",
        "timeline_shape": list(provider.x_raw.shape),
        "timestamp_count": len(provider.timestamps),
        "first_timestamp": str(provider.timestamps[0]),
        "last_timestamp": str(provider.timestamps[-1]),
        "node_count": len(provider.node_ids),
        "node_ids": list(provider.node_ids),
        "feature_names": list(provider.feature_names),
        "split_ratio": list(provider.protocol["split_ratio"]),
        "lookback": int(provider.protocol["lookback"]),
        "max_pred_len": int(provider.protocol["max_pred_len"]),
        "test_sample_stride": int(provider.protocol["test_sample_stride"]),
    }


def export_benchmark_model(
    spec: dict[str, Any], device: str, run_label: str
) -> dict[str, Any]:
    import torch

    from benchmark_v2.adapters import adapter_forward
    from benchmark_v2.data import SDWPFDataProvider
    from benchmark_v2.metrics import evaluate_horizons
    from benchmark_v2.model_cli import _move_batch
    from benchmark_v2.model_runtime import build_model_runtime
    from benchmark_v2.precision import apply_model_precision_policy
    from benchmark_v2.protocol import load_protocol
    from benchmark_v2.runtime import ProviderBatchIterable
    from benchmark_v2.seeds import seed_everything
    from benchmark_v2.training_profiles import apply_training_profile

    run_dir = Path(spec["run_dir"])
    model_id = str(spec["model_id"])
    effective = read_json(run_dir / "effective_config.json")
    protocol = load_protocol()
    seed_everything(int(effective.get("seed", protocol["default_seed"])))
    provider = SDWPFDataProvider.from_files(
        INPUT_PATH, TARGET_PATH, protocol=protocol
    )
    if int(effective["test_batch_size"]) != 4:
        raise RuntimeError(f"{model_id}: formal test_batch_size is not 4")
    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="formal",
        input_scaler=provider.scalers["input"],
        target_scaler=provider.scalers["target"],
    )
    apply_training_profile(runtime, effective.get("training_batch_profile_id"))
    apply_model_precision_policy(runtime)

    checkpoint_path: Path | None = None
    checkpoint_identity: dict[str, Any] = {
        "checkpoint_path": None,
        "checkpoint_epoch": None,
        "checkpoint_file_size": None,
        "checkpoint_last_write_time": None,
    }
    if model_id != "moving_average":
        checkpoint_path = run_dir / "best_checkpoint.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("model_id") not in (None, model_id):
            raise RuntimeError(
                f"checkpoint model_id {payload.get('model_id')} != {model_id}"
            )
        if payload.get("run_id") not in (None, run_dir.name):
            raise RuntimeError(
                f"checkpoint run_id {payload.get('run_id')} != {run_dir.name}"
            )
        runtime.model.load_state_dict(payload["model_state_dict"], strict=True)
        checkpoint_stat = checkpoint_path.stat()
        checkpoint_identity = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_global_step": payload.get("global_step"),
            "checkpoint_monitor_name": payload.get("monitor_name"),
            "checkpoint_monitor_value": payload.get("monitor_value"),
            "checkpoint_file_size": int(checkpoint_stat.st_size),
            "checkpoint_last_write_time": checkpoint_stat.st_mtime,
            "checkpoint_strict_load": True,
        }

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    runtime.model.to(target_device).eval()
    pred_parts = []
    target_parts = []
    mask_parts = []
    sample_indices: list[int] = []
    batches = ProviderBatchIterable(provider, "test", 4)
    total_batches = (len(provider.starts["test"]) + 3) // 4
    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            moved = _move_batch(batch, target_device)
            if model_id == "moving_average":
                output = runtime.adapter(
                    runtime.model,
                    moved,
                    expected_horizon=int(protocol["max_pred_len"]),
                    expected_features=int(protocol["feature_count"]),
                )
            else:
                output = adapter_forward(
                    runtime.adapter,
                    runtime.model,
                    moved,
                    expected_horizon=int(protocol["max_pred_len"]),
                    expected_features=int(protocol["feature_count"]),
                )
            pred_parts.append(
                runtime.inverse_target(output.prediction).detach().cpu()
            )
            target = (
                moved.target_raw_or_inverse_transform
                if moved.target_raw_or_inverse_transform is not None
                else runtime.inverse_target(moved.target)
            )
            target_parts.append(target.detach().cpu())
            mask_parts.append(moved.mask.detach().cpu())
            sample_indices.extend(int(v) for v in batch.window_end_indices)
            if batch_number == 1 or batch_number % 128 == 0 or batch_number == total_batches:
                print(
                    f"{spec['display_name']}: inference batch "
                    f"{batch_number}/{total_batches}",
                    flush=True,
                )

    pred_raw = torch.cat(pred_parts).numpy().astype(np.float32, copy=False)
    y_true = torch.cat(target_parts).numpy().astype(np.float32, copy=False)
    mask = torch.cat(mask_parts).numpy().astype(bool, copy=False)
    sample_index = np.asarray(sample_indices, dtype=np.int64)
    pred_eval = np.clip(
        pred_raw,
        float(protocol["physical_power_min_kw"]),
        float(protocol["physical_power_max_kw"]),
    ).astype(np.float32)
    rows = evaluate_horizons(
        pred_raw,
        y_true,
        mask,
        protocol["eval_horizons"],
        num_nodes=int(pred_raw.shape[1]),
        physical_clip=(
            float(protocol["physical_power_min_kw"]),
            float(protocol["physical_power_max_kw"]),
        ),
    )
    timestamps = np.asarray(
        [str(provider.timestamps[int(index)]) for index in sample_index], dtype="U32"
    )
    node_ids = np.asarray(provider.node_ids, dtype=np.int64)
    scalers = {
        name: scaler.to_dict() for name, scaler in provider.scalers.items()
    }
    record = {
        "display_name": spec["display_name"],
        "formal_run_directory": run_dir,
        "checkpoint_path": checkpoint_path,
        "config_path": run_dir / "effective_config.json",
        "prediction_export_path": EXPORT_ROOT / spec["export_name"],
        "prediction_shape": list(pred_eval.shape),
        "dtype": {
            "y_true_snh": str(y_true.dtype),
            "y_pred_eval_snh": str(pred_eval.dtype),
            "valid_target_mask_snh": str(mask.dtype),
        },
        "mask_shape": list(mask.shape),
        "sample_count": int(pred_eval.shape[0]),
        "node_count": int(pred_eval.shape[1]),
        "horizon_count": int(pred_eval.shape[2]),
        "physical_clipping_applied": True,
        "physical_clip_range_kw": [
            float(protocol["physical_power_min_kw"]),
            float(protocol["physical_power_max_kw"]),
        ],
        "scaler": scalers,
        # The frozen benchmark-v2 effective configs encode the formal seed in
        # the run/scope identifiers rather than as a top-level field.
        "seed": 2026,
        "batch_size": int(effective["test_batch_size"]),
        "dataset_signature": benchmark_dataset_signature(provider),
        "checkpoint_identity": checkpoint_identity,
        "formal_metric_source": [
            str((run_dir / f"metrics_eval_h{horizon}.json").resolve())
            for horizon in (3, 6, 10)
        ],
        "run_label": run_label.upper(),
    }
    return finish_export(
        spec,
        record,
        y_true,
        pred_raw,
        pred_eval,
        mask,
        sample_index,
        timestamps,
        node_ids,
        rows,
        run_label,
    )


def load_canonical_graph_read_only(config) -> dict[str, Any]:
    graph_dir = config.resolve_path(config.graph_output_root) / config.graph_tag
    macro_path = graph_dir / "macro_trend_adjacency.npy"
    micro_path = graph_dir / "micro_local_adjacency.npy"
    metadata_path = graph_dir / "metadata.json"
    return {
        "A_macro_trend": np.load(macro_path, allow_pickle=False),
        "A_micro_local": np.load(micro_path, allow_pickle=False),
        "metadata": read_json(metadata_path) if metadata_path.exists() else {},
    }


def canonical_dataset_signature(data, config) -> dict[str, Any]:
    dataset = data.test_loader.dataset
    return {
        "dataset_id": "SDWPF",
        "timeline_shape": list(dataset.x.shape),
        "timestamp_count": len(dataset.timestamps),
        "first_timestamp": str(dataset.timestamps[0]),
        "last_timestamp": str(dataset.timestamps[-1]),
        "node_count": len(data.turbine_ids),
        "node_ids": list(data.turbine_ids),
        "feature_names": list(data.feature_cols),
        "split_ratio": list(config.split_ratios),
        "lookback": int(config.lookback),
        "max_pred_len": int(config.max_pred_len),
        "test_sample_stride": int(config.test_sample_stride),
    }


def export_proposed_model(
    spec: dict[str, Any], device: str, run_label: str
) -> dict[str, Any]:
    import torch

    from st_mgprompt.config import STMGPromptConfig
    from st_mgprompt.data import make_dataloaders
    from st_mgprompt.evaluate import apply_physical_clip
    from st_mgprompt.registry import build_model
    from st_mgprompt.train import _autocast_context, set_seed
    from st_mgprompt.evaluate import (
        _finalize_stream_accumulator,
        _new_stream_accumulator,
        _update_stream_accumulator,
    )

    run_dir = Path(spec["run_dir"])
    config_path = run_dir / "config.json"
    config = STMGPromptConfig.from_json(config_path)
    config.device = device
    set_seed(int(config.seed))
    if int(config.test_batch_size or config.eval_batch_size) != 4:
        raise RuntimeError("Proposed model formal test_batch_size is not 4")
    data = make_dataloaders(config)["bundle"]
    graph_data = load_canonical_graph_read_only(config)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = build_model(config, data.input_dim, graph_data=graph_data).to(target_device)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    # Reproduce the formal runner's final shape-validation forward before the
    # test pass.  The run performed this one batch in FP32 (outside autocast).
    validation_batch = next(iter(data.train_loader))
    with torch.inference_mode():
        model(validation_batch["x"].to(target_device).float())

    pred_raw_parts = []
    pred_eval_parts = []
    target_parts = []
    mask_parts = []
    sample_indices: list[int] = []
    total_batches = len(data.test_loader)
    stream_accum = {
        int(horizon): _new_stream_accumulator()
        for horizon in config.eval_horizons
    }
    with torch.inference_mode():
        for batch_number, batch in enumerate(data.test_loader, start=1):
            x = batch["x"].to(target_device).float()
            with _autocast_context(target_device, bool(config.amp_enabled)):
                output = model(x)
            pred_norm = output["pred"].detach().float().cpu().numpy()
            pred_raw = data.scalers["target"].inverse_transform(pred_norm).astype(
                np.float32
            )
            pred_eval = apply_physical_clip(pred_raw, config).astype(np.float32)
            pred_raw_parts.append(pred_raw)
            pred_eval_parts.append(pred_eval)
            target_parts.append(batch["y_raw"].numpy().astype(np.float32))
            mask_parts.append(batch["valid_target_mask"].numpy().astype(bool))
            for horizon in config.eval_horizons:
                _update_stream_accumulator(
                    stream_accum[int(horizon)],
                    pred_eval[:, : int(horizon), :],
                    batch["y_raw"].numpy().astype(np.float32)[:, : int(horizon), :],
                    batch["valid_target_mask"].numpy().astype(bool)[:, : int(horizon), :],
                    num_nodes=config.num_nodes,
                )
            sample_indices.extend(
                int(v) for v in batch["prediction_start_index"].numpy().tolist()
            )
            if batch_number == 1 or batch_number % 128 == 0 or batch_number == total_batches:
                print(
                    f"Proposed model: inference batch {batch_number}/{total_batches}",
                    flush=True,
                )

    pred_raw_bhn = np.concatenate(pred_raw_parts, axis=0)
    pred_eval_bhn = np.concatenate(pred_eval_parts, axis=0)
    y_true_bhn = np.concatenate(target_parts, axis=0)
    mask_bhn = np.concatenate(mask_parts, axis=0)
    sample_index = np.asarray(sample_indices, dtype=np.int64)
    rows = [
        {
            "horizon": int(horizon),
            **_finalize_stream_accumulator(
                stream_accum[int(horizon)],
                physical_clip_applied=bool(config.enable_physical_clip_eval),
            ),
        }
        for horizon in config.eval_horizons
    ]
    y_true = np.transpose(y_true_bhn, (0, 2, 1)).astype(np.float32)
    pred_raw = np.transpose(pred_raw_bhn, (0, 2, 1)).astype(np.float32)
    pred_eval = np.transpose(pred_eval_bhn, (0, 2, 1)).astype(np.float32)
    mask = np.transpose(mask_bhn, (0, 2, 1)).astype(bool)
    dataset = data.test_loader.dataset
    timestamps = np.asarray(
        [str(dataset.timestamps[int(index)]) for index in sample_index], dtype="U32"
    )
    node_ids = np.asarray(data.turbine_ids, dtype=np.int64)
    record = {
        "display_name": spec["display_name"],
        "formal_run_directory": run_dir,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "prediction_export_path": EXPORT_ROOT / spec["export_name"],
        "prediction_shape": list(pred_eval.shape),
        "dtype": {
            "y_true_snh": str(y_true.dtype),
            "y_pred_eval_snh": str(pred_eval.dtype),
            "valid_target_mask_snh": str(mask.dtype),
        },
        "mask_shape": list(mask.shape),
        "sample_count": int(pred_eval.shape[0]),
        "node_count": int(pred_eval.shape[1]),
        "horizon_count": int(pred_eval.shape[2]),
        "physical_clipping_applied": bool(config.enable_physical_clip_eval),
        "physical_clip_range_kw": [
            float(config.physical_power_min_kw),
            float(config.physical_power_max_kw),
        ],
        "scaler": {
            name: scaler.to_dict() for name, scaler in data.scalers.items()
        },
        "seed": int(config.seed),
        "batch_size": int(config.test_batch_size or config.eval_batch_size),
        "dataset_signature": canonical_dataset_signature(data, config),
        "canonical_source": read_json(run_dir / "canonical_manifest.json")[
            "source_directory"
        ],
        "graph_resources": {
            "macro": str(
                (config.resolve_path(config.graph_output_root) / config.graph_tag / "macro_trend_adjacency.npy").resolve()
            ),
            "micro": str(
                (config.resolve_path(config.graph_output_root) / config.graph_tag / "micro_local_adjacency.npy").resolve()
            ),
            "read_only": True,
        },
        "checkpoint_identity": {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_best_epoch": payload.get("best_epoch"),
            "checkpoint_best_metric_name": payload.get("best_metric_name"),
            "checkpoint_best_metric_value": payload.get("best_metric_value"),
            "checkpoint_file_size": int(checkpoint_path.stat().st_size),
            "checkpoint_last_write_time": checkpoint_path.stat().st_mtime,
            "checkpoint_strict_load": True,
        },
        "formal_metric_source": [
            str((run_dir / f"metrics_eval_h{horizon}.json").resolve())
            for horizon in (3, 6, 10)
        ],
        "run_label": run_label.upper(),
    }
    return finish_export(
        spec,
        record,
        y_true,
        pred_raw,
        pred_eval,
        mask,
        sample_index,
        timestamps,
        node_ids,
        rows,
        run_label,
    )


def finish_export(
    spec: dict[str, Any],
    record: dict[str, Any],
    y_true: np.ndarray,
    pred_raw: np.ndarray,
    pred_eval: np.ndarray,
    mask: np.ndarray,
    sample_index: np.ndarray,
    timestamps: np.ndarray,
    node_ids: np.ndarray,
    rows: list[dict[str, Any]],
    run_label: str,
) -> dict[str, Any]:
    import platform
    import sys

    import torch

    record["runtime"] = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    counts = validate_arrays(y_true, pred_eval, mask, sample_index)
    official = official_metrics(Path(spec["run_dir"]))
    recomputed = select_metrics(rows)
    comparisons, max_diff, metrics_pass = compare_metrics(
        str(spec["display_name"]), official, recomputed
    )
    record.update(
        {
            "valid_count_h3": counts[3],
            "valid_count_h6": counts[6],
            "valid_count_h10": counts[10],
            "official_metrics_h3": official["3"],
            "official_metrics_h6": official["6"],
            "official_metrics_h10": official["10"],
            "recomputed_metrics_h3": recomputed["3"],
            "recomputed_metrics_h6": recomputed["6"],
            "recomputed_metrics_h10": recomputed["10"],
            "metric_comparisons": comparisons,
            "metric_tolerance": METRIC_TOLERANCE,
            "max_metric_abs_diff": max_diff,
            "metrics_reproduced": metrics_pass,
            "exact_reproduction_pass": metrics_pass,
            "shape_mask_count_gate_pass": counts == EXPECTED_COUNTS,
            "validation_pass": metrics_pass and counts == EXPECTED_COUNTS,
        }
    )
    if run_label == "a" and metrics_pass:
        export_path = EXPORT_ROOT / str(spec["export_name"])
        record["prediction_role"] = "exact_formal_reproduction"
        record["classification"] = "EXACT_PASS"
    else:
        base_stem = Path(str(spec["export_name"])).stem
        suffix = "_candidate.npz" if run_label == "a" else "_candidate_run_b.npz"
        export_path = CANDIDATE_ROOT / f"{base_stem}{suffix}"
        record["prediction_role"] = "candidate_visualization_prediction"
        record["classification"] = "PENDING_TWO_RUN_DIAGNOSIS"
    record["prediction_export_path"] = export_path
    metadata_json = json.dumps(json_safe(record), ensure_ascii=False)
    atomic_save_npz(
        export_path,
        {
            "y_true_snh": y_true.astype(np.float32, copy=False),
            "y_pred_eval_snh": pred_eval.astype(np.float32, copy=False),
            "valid_target_mask_snh": mask.astype(bool, copy=False),
            "sample_index": sample_index.astype(np.int64, copy=False),
            "prediction_start_index": sample_index.astype(np.int64, copy=False),
            "timestamp": timestamps,
            "node_id": node_ids,
            "y_pred_raw_snh": pred_raw.astype(np.float32, copy=False),
            "export_metadata_json": np.asarray(metadata_json),
        },
    )
    print(
        f"EXPORTED {'EXACT' if metrics_pass and run_label == 'a' else 'CANDIDATE'}: "
        f"{export_path.resolve()}",
        flush=True,
    )
    return json_safe(record)


def export_one(name: str, device: str, run_label: str) -> dict[str, Any]:
    spec = MODEL_SPECS[name]
    print(f"START EXPORT: {spec['display_name']}", flush=True)
    if name == "proposed_model":
        record = export_proposed_model(spec, device, run_label)
    else:
        record = export_benchmark_model(spec, device, run_label)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return record


def load_export(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = MODEL_SPECS[name]
    path = EXPORT_ROOT / str(spec["export_name"])
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["export_metadata_json"].item()))
        arrays = {
            "y_true_snh": data["y_true_snh"].copy(),
            "valid_target_mask_snh": data["valid_target_mask_snh"].copy(),
            "sample_index": data["sample_index"].copy(),
        }
    return metadata, arrays


def finalize_manifest() -> dict[str, Any]:
    reference_meta, reference = load_export("proposed_model")
    del reference_meta
    records = []
    alignment_rows = []
    all_pass = True
    for name, spec in MODEL_SPECS.items():
        metadata, arrays = load_export(name)
        mask_aligned = np.array_equal(
            arrays["valid_target_mask_snh"],
            reference["valid_target_mask_snh"],
        )
        sample_aligned = np.array_equal(
            arrays["sample_index"], reference["sample_index"]
        )
        common_valid = (
            arrays["valid_target_mask_snh"]
            & reference["valid_target_mask_snh"]
        )
        y_true_aligned = bool(
            mask_aligned
            and np.allclose(
                arrays["y_true_snh"][common_valid],
                reference["y_true_snh"][common_valid],
                rtol=1e-6,
                atol=1e-5,
                equal_nan=True,
            )
        )
        alignment = {
            "y_true_aligned_on_formal_valid_mask": y_true_aligned,
            "mask_aligned": mask_aligned,
            "sample_aligned": sample_aligned,
            "comparison_rtol": 1e-6,
            "comparison_atol": 1e-5,
        }
        metadata["alignment_to_proposed_model"] = alignment
        metadata["validation_pass"] = bool(
            metadata["validation_pass"]
            and y_true_aligned
            and mask_aligned
            and sample_aligned
        )
        all_pass = all_pass and bool(metadata["validation_pass"])
        records.append(metadata)
        alignment_rows.append(
            {
                "model": spec["display_name"],
                **alignment,
            }
        )
    manifest = {
        "task": "formal checkpoint inference-only prediction backfill",
        "training_performed": False,
        "figures_generated": False,
        "shape_protocol": "[sample,node,horizon]",
        "horizon_evaluation_protocol": {
            "H3": "[:, :, :3]",
            "H6": "[:, :, :6]",
            "H10": "[:, :, :10]",
        },
        "expected_valid_counts": {str(k): v for k, v in EXPECTED_COUNTS.items()},
        "metric_tolerance": METRIC_TOLERANCE,
        "alignment_reference": "Proposed model",
        "alignment_scope": "formal valid_target_mask positions for y_true",
        "models": records,
        "alignment": alignment_rows,
        "all_models_validation_pass": all_pass,
        "ready_for_joint_distribution_figures": all_pass,
    }
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = EXPORT_ROOT / "prediction_export_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(
        "Model | Prediction exported | shape | H3 count | H6 count | "
        "H10 count | Metrics reproduced | y_true aligned"
    )
    for record in records:
        aligned = record["alignment_to_proposed_model"]
        print(
            f"{record['display_name']} | PASS | {record['prediction_shape']} | "
            f"{record['valid_count_h3']} | {record['valid_count_h6']} | "
            f"{record['valid_count_h10']} | "
            f"{'PASS' if record['metrics_reproduced'] else 'FAIL'} | "
            f"{'PASS' if aligned['y_true_aligned_on_formal_valid_mask'] else 'FAIL'}"
        )
    print(f"MANIFEST: {manifest_path.resolve()}")
    print(f"ALL MODELS: {'PASS' if all_pass else 'FAIL'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only export of formal benchmark predictions."
    )
    parser.add_argument(
        "--model",
        choices=[*MODEL_SPECS.keys(), "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-label", choices=["a", "b"], default="a")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.finalize_only:
        manifest = finalize_manifest()
        return 0 if manifest["all_models_validation_pass"] else 1
    names = list(MODEL_SPECS) if args.model == "all" else [args.model]
    for name in names:
        export_one(name, args.device, args.run_label)
    if args.model == "all":
        manifest = finalize_manifest()
        return 0 if manifest["all_models_validation_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
