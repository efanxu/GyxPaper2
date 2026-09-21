"""Create the E9 Original/MS-MG-DWU prediction joint-grid figures.

The script deliberately keeps data preparation on the benchmark-v2 path.  It
does not reconstruct observations from aggregate spreadsheets: it first looks
for a real prediction export next to each formal run and, when none exists,
performs test-only inference from ``best_checkpoint.pt`` using the formal
SDWPF provider, adapter, inverse transform, mask, and physical clipping.

The full ten-run inference is intentionally a user-run operation.  Importing
this module and asking for ``--help`` do not import torch or start inference.

The final entry point renders two figures in one invocation.  Each figure has
five model rows and H3/H6/H10 columns; the same ten formal runs are loaded
once and reused for both figures and all three horizons.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

HORIZONS = (3, 6, 10)
METRIC_NAMES = ("Score", "MAE", "RMSE", "R2")
METRIC_TOLERANCE = 1e-5
DEFAULT_MAX_SCATTER_POINTS = 40_000
DEFAULT_SEED = 2026
DEFAULT_DENSITY_BINS = 160
DEFAULT_HISTOGRAM_BINS = 40


@dataclass(frozen=True)
class PanelSpec:
    model_id: str
    display_name: str
    original_run_id: str
    transfer_run_id: str
    column: int


PANEL_SPECS: tuple[PanelSpec, ...] = (
    PanelSpec(
        "lightts",
        "LightTS",
        "LightTS_node_shared_chunk8_bs4_seed2026",
        "LightTS_node_shared_chunk8_bs4_msmg_dwu_seed2026",
        0,
    ),
    PanelSpec(
        "patchtst",
        "PatchTST",
        "PatchTST_node_shared_p16_s8_bs4_seed2026",
        "PatchTST_node_shared_p16_s8_bs4_msmg_dwu_seed2026",
        1,
    ),
    PanelSpec(
        "itransformer",
        "iTransformer",
        "iTransformer_node_shared_d512_bs4_seed2026",
        "iTransformer_node_shared_d512_bs4_msmg_dwu_seed2026",
        2,
    ),
    PanelSpec(
        "dcrnn",
        "DCRNN",
        "DCRNN_native_dualrw_k2_h64_l2_ar_bs4_seed2026",
        "DCRNN_native_dualrw_k2_h64_l2_ar_bs4_msmg_dwu_seed2026",
        3,
    ),
    PanelSpec(
        "mtgnn",
        "MTGNN",
        "MTGNN_native_adaptive_k20_gdep2_l3_bs4_seed2026",
        "MTGNN_native_adaptive_k20_gdep2_l3_bs4_msmg_dwu_seed2026",
        4,
    ),
)


class E9FigureError(RuntimeError):
    """Raised when the formal data contract cannot be established."""


@dataclass
class PanelData:
    spec: PanelSpec
    condition: str
    run_dir: Path
    run_id: str
    model_id: str
    actual_snh: np.ndarray
    prediction_snh: np.ndarray
    mask_snh: np.ndarray
    sample_index: np.ndarray | None
    checkpoint_path: Path | None
    prediction_artifact: Path | None
    prediction_source: str
    inference_executed: bool
    artifact_metadata: dict[str, Any]
    effective_config: dict[str, Any]
    official_metrics: dict[int, dict[str, float]]
    recomputed_metrics: dict[int, dict[str, float]]
    metric_comparisons: dict[int, dict[str, dict[str, Any]]]

    @property
    def condition_label(self) -> str:
        return "Original" if self.condition == "original" else "E9 MS-MG-DWU"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise E9FigureError(f"Required JSON artifact is missing: {path}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E9FigureError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ignored_run_path(path: Path) -> bool:
    return any(part.lower() == ".original_scope26_quarantine" for part in path.parts)


def resolve_run_directory(root: Path, run_id: str) -> Path:
    """Resolve one exact formal run and exclude quarantined copies."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise E9FigureError(f"Run root does not exist or is not a directory: {root}")
    direct = root / run_id
    if direct.is_dir() and not _ignored_run_path(direct):
        return direct
    matches = sorted(
        path
        for path in root.rglob(run_id)
        if path.is_dir() and path.name == run_id and not _ignored_run_path(path)
    )
    if not matches:
        raise E9FigureError(f"Formal run {run_id!r} was not found below {root}")
    if len(matches) != 1:
        joined = "\n".join(str(path) for path in matches)
        raise E9FigureError(f"Formal run {run_id!r} is ambiguous:\n{joined}")
    return matches[0]


def validate_formal_run_roots(original_root: Path, transfer_root: Path) -> None:
    """Validate all ten formal run locations before starting any inference."""

    roots = (
        ("Original", original_root),
        ("E9 MS-MG-DWU", transfer_root),
    )
    missing_roots = [
        f"{condition}: {root}"
        for condition, root in roots
        if not root.is_dir()
    ]
    if missing_roots:
        raise E9FigureError(
            "Input run-root preflight failed before inference; missing directory(s):\n"
            + "\n".join(missing_roots)
        )

    missing_runs: list[str] = []
    for condition, root in roots:
        for spec in PANEL_SPECS:
            run_id = (
                spec.original_run_id
                if condition == "Original"
                else spec.transfer_run_id
            )
            try:
                resolve_run_directory(root, run_id)
            except E9FigureError as exc:
                missing_runs.append(f"{condition}/{spec.display_name}: {exc}")
    if missing_runs:
        raise E9FigureError(
            "Formal run preflight failed before inference; unresolved run(s):\n"
            + "\n".join(missing_runs)
        )


def _validate_run_identity(run_dir: Path, expected_model_id: str, expected_run_id: str) -> dict[str, Any]:
    effective = _read_json(run_dir / "effective_config.json")
    actual_model_id = effective.get("model_id")
    if actual_model_id != expected_model_id:
        raise E9FigureError(
            f"{expected_run_id}: effective_config model_id={actual_model_id!r}, "
            f"expected {expected_model_id!r}"
        )
    configured_run_id = effective.get("run_id")
    if configured_run_id not in (None, expected_run_id):
        raise E9FigureError(
            f"{expected_run_id}: effective_config run_id={configured_run_id!r}"
        )
    prediction_metadata = _read_json(
        run_dir / "prediction_metadata.json", required=False
    )
    for key, expected in (("model_id", expected_model_id), ("run_id", expected_run_id)):
        value = prediction_metadata.get(key)
        if value not in (None, expected):
            raise E9FigureError(
                f"{expected_run_id}: prediction_metadata {key}={value!r}, expected {expected!r}"
            )
    return effective


def _metric_value(row: Mapping[str, Any], name: str, path: Path) -> float:
    aliases = (name, "R²") if name == "R2" else (name,)
    for alias in aliases:
        if alias in row:
            value = float(row[alias])
            if math.isfinite(value):
                return value
            break
    raise E9FigureError(f"Missing or non-finite {name} in {path}")


def read_official_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        path = run_dir / f"metrics_eval_h{horizon}.json"
        row = _read_json(path)
        result[horizon] = {
            name: _metric_value(row, name, path) for name in METRIC_NAMES
        }
    return result


def _candidate_prediction_paths(
    run_dir: Path,
    *,
    project_root: Path,
    search_roots: Sequence[Path],
) -> list[Path]:
    candidates: list[Path] = []
    for name in (
        "predictions.npz",
        "prediction.npz",
        "predictions_eval.npz",
        "prediction_export.npz",
        f"{run_dir.name}_predictions.npz",
    ):
        candidates.append(run_dir / name)
    candidates.extend(
        path
        for path in sorted(run_dir.glob("*.npz"))
        if "pred" in path.name.lower()
    )

    metadata = _read_json(run_dir / "prediction_metadata.json", required=False)
    for key in (
        "prediction_path",
        "prediction_export_path",
        "predictions_path",
        "artifact_path",
    ):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw:
            path = Path(raw)
            candidates.extend(
                [path, project_root / path, run_dir / path]
                if not path.is_absolute()
                else [path]
            )

    # Existing inference exporters keep a manifest outside the run directory.
    # Read only manifests below the two user-supplied result roots; this avoids
    # treating an unrelated prediction file as a formal run artifact.
    manifest_paths: set[Path] = set()
    for root in search_roots:
        if root.is_dir():
            manifest_paths.update(root.rglob("prediction_export_manifest.json"))
    for manifest_path in sorted(manifest_paths):
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError, E9FigureError):
            continue
        records = manifest.get("models", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            formal_run = record.get("formal_run_directory")
            export_path = record.get("prediction_export_path")
            if not isinstance(formal_run, str) or not isinstance(export_path, str):
                continue
            try:
                same_run = Path(formal_run).resolve() == run_dir.resolve()
            except OSError:
                same_run = False
            if not same_run:
                continue
            path = Path(export_path)
            candidates.extend(
                [path, manifest_path.parent / path, project_root / path]
                if not path.is_absolute()
                else [path]
            )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = path.expanduser()
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _pick_npz_array(data: Mapping[str, Any], names: Sequence[str]) -> tuple[np.ndarray | None, str | None]:
    for name in names:
        if name in data:
            return np.asarray(data[name]), name
    return None, None


def _normalise_artifact_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    physical_clip: tuple[float, float] | None,
    source: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    actual, actual_key = _pick_npz_array(
        arrays,
        ("y_true_snh", "actual_snh", "y_true", "actual", "target_snh", "target", "y_kw_bhn"),
    )
    prediction, prediction_key = _pick_npz_array(
        arrays,
        (
            "y_pred_eval_snh",
            "prediction_snh",
            "prediction",
            "y_pred_eval",
            "pred_kw_bhn",
            "pred_kw_eval_bhn",
            "y_pred_raw_snh",
            "pred_raw_snh",
        ),
    )
    mask, mask_key = _pick_npz_array(
        arrays,
        ("valid_target_mask_snh", "mask_snh", "mask", "valid_target_mask", "mask_bhn"),
    )
    if actual is None or prediction is None or mask is None:
        raise E9FigureError(
            f"Prediction artifact {source} must contain actual, prediction, and mask arrays; "
            f"keys={sorted(arrays)}"
        )

    bhn_keys = {"y_kw_bhn", "pred_kw_bhn", "pred_kw_eval_bhn", "mask_bhn"}
    if actual_key in bhn_keys or prediction_key in bhn_keys or mask_key in bhn_keys:
        actual = actual.transpose(0, 2, 1) if actual_key in bhn_keys else actual
        prediction = prediction.transpose(0, 2, 1) if prediction_key in bhn_keys else prediction
        mask = mask.transpose(0, 2, 1) if mask_key in bhn_keys else mask

    actual = np.asarray(actual, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    mask = np.asarray(mask).astype(bool, copy=False)
    if actual.ndim != 3 or actual.shape != prediction.shape or actual.shape != mask.shape:
        raise E9FigureError(
            f"Prediction artifact {source} arrays must align as [sample,node,horizon], "
            f"got actual={actual.shape}, prediction={prediction.shape}, mask={mask.shape}"
        )
    if actual.shape[-1] < max(HORIZONS):
        raise E9FigureError(
            f"Prediction artifact {source} has horizon={actual.shape[-1]}, "
            f"but H{max(HORIZONS)} is required"
        )
    raw_prediction = prediction_key in {"y_pred_raw_snh", "pred_raw_snh"}
    if raw_prediction and physical_clip is not None:
        prediction = np.clip(prediction, physical_clip[0], physical_clip[1]).astype(
            np.float32, copy=False
        )

    sample_index, sample_index_key = _pick_npz_array(
        arrays, ("sample_index", "prediction_start_index", "window_end_index")
    )
    if sample_index is not None:
        sample_index = np.asarray(sample_index, dtype=np.int64).reshape(-1)
        if sample_index.shape != (actual.shape[0],):
            raise E9FigureError(
                f"Prediction artifact {source} sample index shape {sample_index.shape} "
                f"does not match sample count {actual.shape[0]}"
            )
    metadata: dict[str, Any] = {
        "source_path": source,
        "array_keys": sorted(str(key) for key in arrays),
        "actual_key": actual_key,
        "prediction_key": prediction_key,
        "mask_key": mask_key,
        "sample_index_key": sample_index_key,
        "raw_prediction_clipped_for_evaluation": raw_prediction and physical_clip is not None,
    }
    export_metadata = arrays.get("export_metadata_json")
    if export_metadata is not None:
        try:
            raw_metadata = export_metadata.item() if np.asarray(export_metadata).ndim == 0 else export_metadata
            if isinstance(raw_metadata, bytes):
                raw_metadata = raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, str):
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    metadata["embedded_export_metadata"] = parsed
        except (ValueError, TypeError, json.JSONDecodeError):
            metadata["embedded_export_metadata_parse_error"] = True
    return actual, prediction, mask, sample_index, metadata


def load_prediction_artifact(
    candidates: Iterable[Path], *, physical_clip: tuple[float, float] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, Path, dict[str, Any]] | None:
    invalid: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with np.load(candidate, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            actual, prediction, mask, sample_index, metadata = _normalise_artifact_arrays(
                arrays, physical_clip=physical_clip, source=candidate
            )
            return actual, prediction, mask, sample_index, candidate, metadata
        except (OSError, ValueError, KeyError, E9FigureError) as exc:
            invalid.append(f"{candidate}: {type(exc).__name__}: {exc}")
    if invalid:
        raise E9FigureError(
            "Prediction artifact candidates were found but invalid:\n" + "\n".join(invalid)
        )
    return None


def _resolve_dataset_paths(project_root: Path) -> tuple[Path, Path]:
    input_path = project_root / "dataset" / "sdwpf_model_input_base.parquet"
    target_path = project_root / "dataset" / "sdwpf_eval_target.parquet"
    missing = [str(path) for path in (input_path, target_path) if not path.is_file()]
    if missing:
        raise E9FigureError("Formal SDWPF data files are missing:\n" + "\n".join(missing))
    return input_path, target_path


def _resolve_physical_clip(protocol: Mapping[str, Any]) -> tuple[float, float] | None:
    lower = protocol.get("physical_power_min_kw")
    upper = protocol.get("physical_power_max_kw")
    if lower is None or upper is None:
        return None
    lower, upper = float(lower), float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise E9FigureError(f"Invalid physical clip range in benchmark protocol: {lower}, {upper}")
    return lower, upper


def run_formal_test_inference(
    *,
    project_root: Path,
    run_dir: Path,
    model_id: str,
    protocol: Mapping[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, dict[str, Any]]:
    """Run the benchmark-v2 test path once, without training or artifact writes."""

    import torch

    from benchmark_v2.adapters import adapter_forward
    from benchmark_v2.data import SDWPFDataProvider
    from benchmark_v2.model_cli import _move_batch
    from benchmark_v2.model_runtime import build_model_runtime
    from benchmark_v2.precision import apply_model_precision_policy
    from benchmark_v2.protocol import load_protocol
    from benchmark_v2.runtime import ProviderBatchIterable
    from benchmark_v2.seeds import seed_everything
    from benchmark_v2.training_profiles import apply_training_profile

    input_path, target_path = _resolve_dataset_paths(project_root)
    effective = _read_json(run_dir / "effective_config.json")
    seed = int(effective.get("seed", protocol.get("default_seed", DEFAULT_SEED)))
    seed_everything(seed)
    provider = SDWPFDataProvider.from_files(
        input_path, target_path, protocol=load_protocol()
    )
    batch_size = int(effective.get("test_batch_size", protocol.get("test_batch_size", 4)))
    if batch_size <= 0:
        raise E9FigureError(f"{run_dir.name}: invalid formal test batch size {batch_size}")

    runtime = build_model_runtime(
        model_id,
        protocol,
        run_mode="formal",
        input_scaler=provider.scalers["input"],
        target_scaler=provider.scalers["target"],
    )
    apply_training_profile(runtime, effective.get("training_batch_profile_id"))
    apply_model_precision_policy(runtime)

    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not checkpoint_path.is_file():
        raise E9FigureError(f"{run_dir.name}: formal best checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise E9FigureError(f"{run_dir.name}: checkpoint has no model_state_dict")
    if payload.get("model_id") not in (None, model_id):
        raise E9FigureError(
            f"{run_dir.name}: checkpoint model_id={payload.get('model_id')!r}, expected {model_id!r}"
        )
    if payload.get("run_id") not in (None, run_dir.name):
        raise E9FigureError(
            f"{run_dir.name}: checkpoint run_id={payload.get('run_id')!r}"
        )
    runtime.model.load_state_dict(payload["model_state_dict"], strict=True)

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise E9FigureError("CUDA was requested but is not available")
    runtime.model.to(target_device).eval()

    predictions: list[Any] = []
    targets: list[Any] = []
    masks: list[Any] = []
    sample_indices: list[int] = []
    batches = ProviderBatchIterable(provider, "test", batch_size)
    total_batches = (len(provider.starts["test"]) + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            moved = _move_batch(batch, target_device)
            output = adapter_forward(
                runtime.adapter,
                runtime.model,
                moved,
                expected_horizon=int(protocol["max_pred_len"]),
                expected_features=int(protocol["feature_count"]),
            )
            predictions.append(runtime.inverse_target(output.prediction).detach().cpu())
            target = (
                moved.target_raw_or_inverse_transform
                if moved.target_raw_or_inverse_transform is not None
                else runtime.inverse_target(moved.target)
            )
            targets.append(target.detach().cpu())
            masks.append(moved.mask.detach().cpu())
            sample_indices.extend(int(value) for value in batch.window_end_indices)
            if batch_number == 1 or batch_number % 128 == 0 or batch_number == total_batches:
                print(
                    f"INFERENCE {run_dir.name}: batch {batch_number}/{total_batches}",
                    flush=True,
                )

    if not predictions:
        raise E9FigureError(f"{run_dir.name}: formal test loader produced no batches")
    actual = torch.cat(targets).numpy().astype(np.float32, copy=False)
    prediction_raw = torch.cat(predictions).numpy().astype(np.float32, copy=False)
    mask = torch.cat(masks).numpy().astype(bool, copy=False)
    physical_clip = _resolve_physical_clip(protocol)
    prediction = (
        np.clip(prediction_raw, physical_clip[0], physical_clip[1]).astype(np.float32, copy=False)
        if physical_clip is not None
        else prediction_raw
    )
    sample_index = np.asarray(sample_indices, dtype=np.int64)
    metadata = {
        "inference_device": str(target_device),
        "formal_test_batch_size": batch_size,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "checkpoint_monitor_name": payload.get("monitor_name"),
        "checkpoint_monitor_value": payload.get("monitor_value"),
        "target_source": "BenchmarkBatch.target_raw_or_inverse_transform",
        "prediction_source": "best_checkpoint.pt via benchmark_v2 test adapter",
        "physical_clip_applied": physical_clip is not None,
        "physical_clip_range_kw": physical_clip,
    }
    del runtime, provider, payload
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return actual, prediction, mask, sample_index, checkpoint_path, metadata


def _load_or_infer_panel(
    *,
    project_root: Path,
    root: Path,
    spec: PanelSpec,
    condition: str,
    protocol: Mapping[str, Any],
    device: str,
) -> PanelData:
    run_id = spec.original_run_id if condition == "original" else spec.transfer_run_id
    run_dir = resolve_run_directory(root, run_id)
    effective = _validate_run_identity(run_dir, spec.model_id, run_id)
    physical_clip = _resolve_physical_clip(protocol)
    artifact = load_prediction_artifact(
        _candidate_prediction_paths(
            run_dir,
            project_root=project_root,
            search_roots=(root, project_root / "custom_models" / "results"),
        ),
        physical_clip=physical_clip,
    )
    checkpoint_path: Path | None = None
    if artifact is None:
        print(f"PREDICTION SOURCE {run_id}: formal checkpoint inference", flush=True)
        actual, prediction, mask, sample_index, checkpoint_path, source_metadata = run_formal_test_inference(
            project_root=project_root,
            run_dir=run_dir,
            model_id=spec.model_id,
            protocol=protocol,
            device=device,
        )
        prediction_source = "formal_test_checkpoint_inference"
        inference_executed = True
        prediction_artifact = None
        artifact_metadata = source_metadata
    else:
        actual, prediction, mask, sample_index, prediction_artifact, artifact_metadata = artifact
        prediction_source = "existing_prediction_artifact"
        inference_executed = False

    expected_horizon = int(protocol.get("max_pred_len", max(HORIZONS)))
    if actual.ndim != 3 or actual.shape != prediction.shape or actual.shape != mask.shape:
        raise E9FigureError(
            f"{run_id}: arrays do not align as [sample,node,horizon]: "
            f"actual={actual.shape}, prediction={prediction.shape}, mask={mask.shape}"
        )
    if actual.shape[-1] < max(HORIZONS) or actual.shape[-1] != expected_horizon:
        raise E9FigureError(
            f"{run_id}: prediction horizon {actual.shape[-1]} does not match formal horizon {expected_horizon}"
        )
    if actual.shape[1] != int(protocol.get("expected_node_count", actual.shape[1])):
        raise E9FigureError(
            f"{run_id}: node count {actual.shape[1]} does not match benchmark protocol"
        )

    official = read_official_metrics(run_dir)
    from benchmark_v2.metrics import evaluate_horizons

    recomputed_rows = evaluate_horizons(
        prediction,
        actual,
        mask,
        HORIZONS,
        num_nodes=int(actual.shape[1]),
        physical_clip=physical_clip,
    )
    recomputed = {
        int(row["horizon"]): {name: float(row[name]) for name in METRIC_NAMES}
        for row in recomputed_rows
    }
    comparisons: dict[int, dict[str, dict[str, Any]]] = {}
    for horizon in HORIZONS:
        comparisons[horizon] = {}
        for name in METRIC_NAMES:
            official_value = float(official[horizon][name])
            recomputed_value = float(recomputed[horizon][name])
            difference = abs(official_value - recomputed_value)
            comparisons[horizon][name] = {
                "official": official_value,
                "recomputed_from_prediction": recomputed_value,
                "absolute_difference": difference,
                "within_tolerance": bool(difference <= METRIC_TOLERANCE),
                "rounded_to_3dp_matches": round(official_value, 3) == round(recomputed_value, 3),
            }
            if difference > METRIC_TOLERANCE:
                print(
                    f"WARNING METRIC DIFFERENCE run_id={run_id} H{horizon} {name}: "
                    f"official={official_value:.12g}, recomputed={recomputed_value:.12g}, "
                    f"abs_diff={difference:.6g}",
                    file=sys.stderr,
                    flush=True,
                )

    return PanelData(
        spec=spec,
        condition=condition,
        run_dir=run_dir,
        run_id=run_id,
        model_id=spec.model_id,
        actual_snh=actual,
        prediction_snh=prediction,
        mask_snh=mask,
        sample_index=sample_index,
        checkpoint_path=checkpoint_path,
        prediction_artifact=prediction_artifact,
        prediction_source=prediction_source,
        inference_executed=inference_executed,
        artifact_metadata=artifact_metadata,
        effective_config=effective,
        official_metrics=official,
        recomputed_metrics=recomputed,
        metric_comparisons=comparisons,
    )


def validate_panel_alignment(panels: Sequence[PanelData]) -> dict[str, Any]:
    if len(panels) != 10:
        raise E9FigureError(f"Expected 10 panels, got {len(panels)}")
    reference = panels[0]
    shape = reference.actual_snh.shape
    shape_matches = all(panel.actual_snh.shape == shape for panel in panels)
    if not shape_matches:
        raise E9FigureError("All ten formal runs must have the same [sample,node,horizon] shape")
    mask_matches = [
        bool(np.array_equal(reference.mask_snh, panel.mask_snh)) for panel in panels[1:]
    ]
    if not all(mask_matches):
        raise E9FigureError(
            "The ten formal runs do not share an identical valid_target_mask; "
            "refusing to make a visually incomparable joint grid"
        )
    actual_matches = [
        bool(
            np.allclose(
                reference.actual_snh[reference.mask_snh],
                panel.actual_snh[reference.mask_snh],
                rtol=1e-6,
                atol=1e-5,
                equal_nan=True,
            )
        )
        for panel in panels[1:]
    ]
    if not all(actual_matches):
        raise E9FigureError(
            "The ten formal runs do not share the same Patv_raw target values; "
            "refusing to make a mixed-target figure"
        )
    sample_index_matches = [
        reference.sample_index is not None
        and panel.sample_index is not None
        and bool(np.array_equal(reference.sample_index, panel.sample_index))
        for panel in panels[1:]
    ]
    return {
        "reference_run_id": reference.run_id,
        "shape": list(shape),
        "shape_identical": shape_matches,
        "valid_target_mask_identical": all(mask_matches),
        "valid_target_mask_matches_by_panel": mask_matches,
        "Patv_raw_target_aligned": all(actual_matches),
        "Patv_raw_target_matches_by_panel": actual_matches,
        "sample_index_available_for_all": all(panel.sample_index is not None for panel in panels),
        "sample_index_identical_when_available": all(sample_index_matches)
        if all(panel.sample_index is not None for panel in panels)
        else None,
    }


def select_shared_sample_positions(
    panels: Sequence[PanelData], *, horizon: int, max_points: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    population = np.ones(panels[0].actual_snh.shape[:2] + (horizon,), dtype=bool)
    for panel in panels:
        population &= panel.mask_snh[:, :, :horizon]
        population &= np.isfinite(panel.actual_snh[:, :, :horizon])
        population &= np.isfinite(panel.prediction_snh[:, :, :horizon])
    valid_positions = np.flatnonzero(population.ravel())
    rng = np.random.default_rng(int(seed))
    sample_count = min(int(max_points), int(valid_positions.size))
    if sample_count == 0:
        raise E9FigureError("No common finite valid test positions are available for plotting")
    if sample_count == valid_positions.size:
        selected = valid_positions.copy()
    else:
        selected = np.sort(rng.choice(valid_positions, size=sample_count, replace=False))
    return selected, {
        "seed": int(seed),
        "horizon": int(horizon),
        "max_scatter_points_per_panel": int(max_points),
        "shared_population_count": int(valid_positions.size),
        "sampled_count_per_panel": int(sample_count),
        "same_flattened_positions_for_all_panels": True,
        "population_definition": "intersection of valid_target_mask and finite Actual/Prediction across all ten panels",
        "flattening_order": "sample,node,horizon_prefix C-order",
    }


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(axis / sigma))
    return kernel / kernel.sum()


def _smooth_2d(values: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, values)
    return np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="same"), 0, horizontal)


def compute_point_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    axis_limits: tuple[float, float],
    bins: int = DEFAULT_DENSITY_BINS,
) -> np.ndarray:
    """Estimate local 2-D density using a fixed, smoothed histogram grid."""

    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    lower, upper = axis_limits
    x_edges = np.linspace(lower, upper, int(bins) + 1)
    y_edges = np.linspace(lower, upper, int(bins) + 1)
    histogram, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    smoothed = _smooth_2d(histogram)
    x_index = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    y_index = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    density = smoothed[x_index, y_index]
    return np.maximum(density.astype(np.float64, copy=False), np.finfo(np.float64).tiny)


def _smooth_histogram_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    total = float(counts.sum())
    density = counts.astype(np.float64) / max(total, 1.0) / widths
    if density.size >= 5:
        density = np.convolve(density, _gaussian_kernel(1.1), mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, density


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid].astype(np.float64, copy=False), y[valid].astype(np.float64, copy=False)
    if x.size < 2:
        return None
    x_mean, y_mean = float(x.mean()), float(y.mean())
    denominator = float(np.square(x - x_mean).sum())
    if denominator <= np.finfo(np.float64).eps:
        return None
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    intercept = float(y_mean - slope * x_mean)
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        return None
    return slope, intercept


def resolve_axis_limits(
    panels: Sequence[PanelData], *, physical_clip: tuple[float, float] | None, horizon: int
) -> tuple[float, float, dict[str, Any]]:
    if physical_clip is not None:
        lower, upper = physical_clip
        return lower, upper, {
            "source": "benchmark_protocol_physical_power_clip",
            "physical_clip_applied_to_prediction": True,
            "limits_kw": [lower, upper],
        }
    values: list[np.ndarray] = []
    for panel in panels:
        valid = panel.mask_snh[:, :, :horizon]
        values.extend(
            [
                panel.actual_snh[:, :, :horizon][valid],
                panel.prediction_snh[:, :, :horizon][valid],
            ]
        )
    combined = np.concatenate(values).astype(np.float64, copy=False)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        raise E9FigureError("Cannot derive shared axis limits from empty formal arrays")
    lower, upper = float(combined.min()), float(combined.max())
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.02, 1.0)
    else:
        padding = (upper - lower) * 0.02
    return lower - padding, upper + padding, {
        "source": "all_ten_formal_evaluated_arrays",
        "physical_clip_applied_to_prediction": False,
        "limits_kw": [lower - padding, upper + padding],
    }


def _main_ticks(lower: float, upper: float) -> np.ndarray:
    if math.isclose(lower, 0.0) and math.isclose(upper, 1500.0):
        return np.arange(0.0, 1500.1, 500.0)
    return np.linspace(lower, upper, 4)


def _format_metric(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.3f}"


def _plot_values(
    panel: PanelData,
    *,
    horizon: int,
    sample_positions: np.ndarray,
    axis_limits: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    actual = panel.actual_snh[:, :, :horizon].ravel()[sample_positions]
    prediction = panel.prediction_snh[:, :, :horizon].ravel()[sample_positions]
    lower, upper = axis_limits
    visible = (
        np.isfinite(actual)
        & np.isfinite(prediction)
        & (actual >= lower)
        & (actual <= upper)
        & (prediction >= lower)
        & (prediction <= upper)
    )
    return actual[visible], prediction[visible]


def compute_density_normalization(
    panels: Sequence[PanelData],
    *,
    horizons: Sequence[int],
    sample_positions_by_horizon: Mapping[int, np.ndarray],
    axis_limits: tuple[float, float],
    density_bins: int = DEFAULT_DENSITY_BINS,
) -> tuple[float, float]:
    """Get one density scale shared by both figures and all 30 panels."""

    all_densities: list[np.ndarray] = []
    for panel in panels:
        for horizon in horizons:
            x, y = _plot_values(
                panel,
                horizon=horizon,
                sample_positions=sample_positions_by_horizon[horizon],
                axis_limits=axis_limits,
            )
            density = compute_point_density(
                x, y, axis_limits=axis_limits, bins=density_bins
            )
            if density.size:
                all_densities.append(density)
    if not all_densities:
        raise E9FigureError("No finite sampled points are available for density normalization")
    combined = np.concatenate(all_densities)
    lower, upper = float(combined.min()), float(combined.max())
    if math.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def render_joint_grid(
    panels: Sequence[PanelData],
    *,
    horizons: Sequence[int] = HORIZONS,
    sample_positions_by_horizon: Mapping[int, np.ndarray],
    axis_limits: tuple[float, float],
    output_stem: Path,
    figure_title: str,
    density_normalization: tuple[float, float] | None = None,
    density_bins: int = DEFAULT_DENSITY_BINS,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> dict[str, Any]:
    """Render one five-model, three-horizon figure."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    horizons = tuple(int(horizon) for horizon in horizons)
    if horizons != HORIZONS:
        raise ValueError(f"The final E9 figures require horizons {HORIZONS}")
    if len(panels) != 5:
        raise ValueError(f"Each split figure requires five model panels, got {len(panels)}")
    for horizon in horizons:
        if horizon not in sample_positions_by_horizon:
            raise ValueError(f"Missing shared sample positions for H{horizon}")

    lower, upper = axis_limits
    ticks = _main_ticks(lower, upper)
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    # The point arrays are stored in row-major model/horizon order so the
    # panel audit can use the same labels as the visual grid.
    panel_records: list[tuple[PanelData, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for row, panel in enumerate(panels):
        for column, horizon in enumerate(horizons):
            actual, prediction = _plot_values(
                panel,
                horizon=horizon,
                sample_positions=sample_positions_by_horizon[horizon],
                axis_limits=axis_limits,
            )
            density = compute_point_density(
                actual,
                prediction,
                axis_limits=axis_limits,
                bins=density_bins,
            )
            panel_records.append((panel, horizon, actual, prediction, density))
            x_values.append(actual)
            y_values.append(prediction)
            densities.append(density)

    if density_normalization is None:
        finite_densities = np.concatenate([value for value in densities if value.size])
        density_min = float(finite_densities.min())
        density_max = float(finite_densities.max())
        if math.isclose(density_min, density_max):
            density_max = density_min + 1.0
    else:
        density_min, density_max = map(float, density_normalization)
        if not density_min < density_max:
            raise ValueError("density_normalization must be an increasing pair")
    norm = Normalize(vmin=density_min, vmax=density_max)
    cmap = plt.get_cmap("GnBu")
    histogram_edges = np.linspace(lower, upper, histogram_bins + 1)

    # Final paper layout: prioritize the scatter area, keep the marginal
    # distributions visibly subordinate, and use the inter-row space for the
    # two-line panel heading instead of carrying redundant outer padding.
    figure = plt.figure(figsize=(15.2, 17.4), constrained_layout=False)
    outer = GridSpec(
        5,
        3,
        figure=figure,
        left=0.052,
        right=0.918,
        bottom=0.040,
        top=0.914,
        wspace=0.105,
        hspace=0.315,
    )
    panel_audits: list[dict[str, Any]] = []
    figure.suptitle(figure_title, fontsize=22.0, fontweight="bold", y=0.982)
    for index, (panel, horizon, x, y, density) in enumerate(panel_records):
        row, column = divmod(index, 3)
        inner = outer[row, column].subgridspec(
            2,
            2,
            height_ratios=(0.31, 1.0),
            width_ratios=(1.0, 0.27),
            hspace=0.025,
            wspace=0.030,
        )
        top = figure.add_subplot(inner[0, 0])
        main = figure.add_subplot(inner[1, 0])
        right = figure.add_subplot(inner[1, 1], sharey=main)
        empty = figure.add_subplot(inner[0, 1])
        empty.axis("off")

        order = np.argsort(density, kind="stable")
        main.scatter(
            x[order],
            y[order],
            c=density[order],
            cmap=cmap,
            norm=norm,
            s=3.0,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )
        main.plot([lower, upper], [lower, upper], color="black", linewidth=1.05, zorder=3)
        fit = _linear_fit(
            panel.actual_snh[:, :, :horizon][panel.mask_snh[:, :, :horizon]],
            panel.prediction_snh[:, :, :horizon][panel.mask_snh[:, :, :horizon]],
        )
        if fit is not None:
            slope, intercept = fit
            main.plot(
                [lower, upper],
                [slope * lower + intercept, slope * upper + intercept],
                color="#d62728",
                linestyle="--",
                linewidth=1.1,
                zorder=4,
            )

        metric = panel.recomputed_metrics[horizon]
        main.text(
            0.975,
            0.045,
            "\n".join(
                (
                    f"Score = {_format_metric(metric['Score'])}",
                    f"R² = {_format_metric(metric['R2'])}",
                    f"RMSE = {_format_metric(metric['RMSE'])}",
                    f"MAE = {_format_metric(metric['MAE'])}",
                )
            ),
            transform=main.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.2,
            linespacing=1.16,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.8},
            zorder=5,
        )
        main.text(
            0.025,
            0.965,
            f"({chr(ord('a') + index)})",
            transform=main.transAxes,
            fontsize=10.8,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.8},
            zorder=6,
        )
        main.set_xlim(lower, upper)
        main.set_ylim(lower, upper)
        main.set_xticks(ticks)
        main.set_yticks(ticks)
        main.grid(color="#d9d9d9", linewidth=0.45, alpha=0.55)
        main.set_axisbelow(True)
        main.tick_params(labelsize=9.5, length=3.2, width=0.8, pad=2.0)
        main.set_xlabel("Actual Value", fontsize=11.0, labelpad=3.0)
        main.set_ylabel("Predicted Value", fontsize=11.0, labelpad=3.0)
        main.spines["left"].set_linewidth(0.85)
        main.spines["bottom"].set_linewidth(0.85)
        main.spines["top"].set_visible(False)
        main.spines["right"].set_visible(False)

        top.hist(
            x,
            bins=histogram_edges,
            color="#4c956c",
            alpha=0.62,
            edgecolor="white",
            linewidth=0.3,
            density=True,
        )
        centers, density_line = _smooth_histogram_density(x, histogram_edges)
        top.plot(centers, density_line, color="#1b6e43", linewidth=1.15)
        top.set_xlim(lower, upper)
        top.set_xticks(ticks)
        top.tick_params(axis="x", labelbottom=False, bottom=False)
        top.tick_params(axis="y", labelleft=False, left=False)
        top.set_ylabel("")
        top.text(
            0.5,
            1.30,
            panel.spec.display_name,
            transform=top.transAxes,
            ha="center",
            va="bottom",
            fontsize=14.0,
            fontweight="bold",
            clip_on=False,
        )
        top.text(
            0.5,
            1.045,
            f"{horizon}-Step",
            transform=top.transAxes,
            ha="center",
            va="bottom",
            fontsize=11.4,
            clip_on=False,
        )
        top.spines["bottom"].set_linewidth(0.6)
        top.spines["top"].set_visible(False)
        top.spines["left"].set_visible(False)
        top.spines["right"].set_visible(False)

        right.hist(
            y,
            bins=histogram_edges,
            orientation="horizontal",
            color="#377eb8",
            alpha=0.58,
            edgecolor="white",
            linewidth=0.3,
            density=True,
        )
        centers, density_line = _smooth_histogram_density(y, histogram_edges)
        right.plot(density_line, centers, color="#185a9d", linewidth=1.15)
        right.set_ylim(lower, upper)
        right.set_yticks(ticks)
        right.tick_params(axis="y", labelleft=False, left=False)
        right.tick_params(axis="x", labelbottom=False, bottom=False)
        right.set_xlabel("")
        right.spines["left"].set_linewidth(0.6)
        right.spines["top"].set_visible(False)
        right.spines["right"].set_visible(False)
        right.spines["bottom"].set_visible(False)

        panel_audits.append(
            {
                "panel": f"({chr(ord('a') + index)})",
                "run_id": panel.run_id,
                "model_name": panel.spec.display_name,
                "horizon": int(horizon),
                "density_min": float(density.min()) if density.size else None,
                "density_max": float(density.max()) if density.size else None,
                "visible_scatter_points": int(x.size),
                "linear_fit": None
                if fit is None
                else {"slope": float(fit[0]), "intercept": float(fit[1])},
            }
        )

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(np.asarray([density_min, density_max]))
    colorbar_axis = figure.add_axes([0.930, 0.175, 0.0125, 0.650])
    colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label("Point Density", fontsize=11.0, labelpad=4.0)
    colorbar.ax.tick_params(labelsize=9.0, length=3.0, pad=2.0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    for path, kwargs in (
        (png_path, {"dpi": 600}),
        (pdf_path, {}),
        (svg_path, {}),
    ):
        figure.savefig(path, bbox_inches="tight", pad_inches=0.025, **kwargs)
    plt.close(figure)
    return {
        "files": {"png": png_path, "pdf": pdf_path, "svg": svg_path},
        "figure_title": figure_title,
        "panel_count": len(panel_audits),
        "horizons": list(horizons),
        "row_models": [panel.spec.display_name for panel in panels],
        "png_dpi": 600,
        "axis_limits_kw": [lower, upper],
        "axis_ticks_kw": ticks,
        "histogram_bins": int(histogram_bins),
        "density_bins": int(density_bins),
        "density_normalization": {"vmin": density_min, "vmax": density_max, "colormap": "GnBu"},
        "common_colorbar": "Point Density",
        "scatter_points_rasterized_in_vector_containers": True,
        "panel_audits": panel_audits,
    }


def build_audit(
    *,
    panels: Sequence[PanelData],
    alignment: Mapping[str, Any],
    sampling_by_horizon: Mapping[str, Any],
    axis: Mapping[str, Any],
    figures: Mapping[str, Any],
    project_root: Path,
    original_root: Path,
    transfer_root: Path,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if len(panels) != 10:
        raise E9FigureError(f"Expected ten formal runs for audit, got {len(panels)}")

    runs: list[dict[str, Any]] = []
    figure_panels: list[dict[str, Any]] = []
    for panel in panels:
        figure_key = "original" if panel.condition == "original" else "e9_msmg_dwu"
        model_row = panel.spec.column
        runs.append(
            {
                "model_id": panel.model_id,
                "model_name": panel.spec.display_name,
                "condition": panel.condition,
                "condition_label": panel.condition_label,
                "run_id": panel.run_id,
                "run_directory": panel.run_dir,
                "checkpoint": panel.checkpoint_path,
                "prediction_artifact": panel.prediction_artifact,
                "prediction_source": panel.prediction_source,
                "inference_executed": panel.inference_executed,
                "shape": list(panel.actual_snh.shape),
                "mask_dtype": str(panel.mask_snh.dtype),
                "actual_dtype": str(panel.actual_snh.dtype),
                "prediction_dtype": str(panel.prediction_snh.dtype),
                "valid_target_count": {
                    f"H{h}": int(panel.mask_snh[:, :, :h].sum()) for h in HORIZONS
                },
                "finite_valid_count": {
                    f"H{h}": int(
                        (
                            panel.mask_snh[:, :, :h]
                            & np.isfinite(panel.actual_snh[:, :, :h])
                            & np.isfinite(panel.prediction_snh[:, :, :h])
                        ).sum()
                    )
                    for h in HORIZONS
                },
                "official_metrics": {
                    f"H{h}": panel.official_metrics[h] for h in HORIZONS
                },
                "recomputed_metrics_from_full_formal_arrays": {
                    f"H{h}": panel.recomputed_metrics[h] for h in HORIZONS
                },
                "metric_comparison": {
                    f"H{h}": panel.metric_comparisons[h] for h in HORIZONS
                },
                "artifact_metadata": panel.artifact_metadata,
            }
        )
        for column, horizon in enumerate(HORIZONS):
            panel_letter = chr(ord("a") + model_row * len(HORIZONS) + column)
            figure_panels.append(
                {
                    "figure": figure_key,
                    "panel": f"({panel_letter})",
                    "row": int(model_row + 1),
                    "column": int(column + 1),
                    "model_id": panel.model_id,
                    "model_name": panel.spec.display_name,
                    "condition": panel.condition,
                    "horizon": int(horizon),
                    "run_id": panel.run_id,
                    "run_directory": panel.run_dir,
                    "checkpoint": panel.checkpoint_path,
                    "prediction_artifact": panel.prediction_artifact,
                    "prediction_source": panel.prediction_source,
                    "inference_executed": panel.inference_executed,
                    "shape": list(panel.actual_snh.shape),
                    "valid_target_count": int(panel.mask_snh[:, :, :horizon].sum()),
                    "finite_valid_count": int(
                        (
                            panel.mask_snh[:, :, :horizon]
                            & np.isfinite(panel.actual_snh[:, :, :horizon])
                            & np.isfinite(panel.prediction_snh[:, :, :horizon])
                        ).sum()
                    ),
                    "sampled_scatter_count": sampling_by_horizon[f"H{horizon}"][
                        "sampled_count_per_panel"
                    ],
                    "official_metrics": panel.official_metrics[horizon],
                    "recomputed_metrics_from_full_formal_arrays": panel.recomputed_metrics[
                        horizon
                    ],
                    "displayed_metrics": panel.recomputed_metrics[horizon],
                    "metric_comparison": panel.metric_comparisons[horizon],
                }
            )
    return {
        "schema_version": "e9_prediction_jointgrid_audit_v2",
        "task": "E9 Original/MS-MG-DWU prediction-vs-actual two-figure joint grid",
        "project_root": project_root,
        "original_root": original_root,
        "transfer_root": transfer_root,
        "training_performed": False,
        "figure_count": 2,
        "panels_per_figure": 15,
        "panel_count": len(figure_panels),
        "horizons": list(HORIZONS),
        "seed": int(seed),
        "device": device,
        "target_column": "Patv_raw",
        "mask_column": "valid_target_mask",
        "shape_protocol": "[sample,node,horizon]",
        "horizon_evaluation_protocol": "prefix [:, :, :horizon], matching benchmark_v2.evaluate_horizons",
        "metric_order_in_each_panel": ["Score", "R2", "RMSE", "MAE"],
        "metric_policy": "Displayed Score/R2/RMSE/MAE are recomputed from all finite formal test positions; official JSON values are retained for audit comparison.",
        "metric_tolerance": METRIC_TOLERANCE,
        "panel_layout": "two figures, each 5 rows x 3 columns; rows are models and columns are H3/H6/H10",
        "figure_titles": {
            "original": "Original Masked-MSE",
            "e9_msmg_dwu": "E9 MS-MG-DWU",
        },
        "alignment": alignment,
        "sampling_by_horizon": sampling_by_horizon,
        "axis": axis,
        "figures": figures,
        "runs": runs,
        "panels": figure_panels,
    }


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two E9 prediction joint-grid figures: Original and E9 "
            "MS-MG-DWU, each with 5 model rows x H3/H6/H10 columns."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="GyxPaper2 project root containing dataset/ and custom_models/.",
    )
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=DEFAULT_MAX_SCATTER_POINTS,
        help="Maximum deterministic scatter points per panel.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto selects CUDA when available.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_scatter_points <= 0:
        raise SystemExit("--max-scatter-points must be positive")
    project_root = args.project_root.expanduser().resolve()
    original_root = args.original_root.expanduser().resolve()
    transfer_root = args.transfer_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_formal_run_roots(original_root, transfer_root)
    device = _resolve_device(args.device)

    from benchmark_v2.protocol import load_protocol

    protocol = load_protocol()
    physical_clip = _resolve_physical_clip(protocol)
    panels: list[PanelData] = []
    for spec in PANEL_SPECS:
        panels.append(
            _load_or_infer_panel(
                project_root=project_root,
                root=original_root,
                spec=spec,
                condition="original",
                protocol=protocol,
                device=device,
            )
        )
    for spec in PANEL_SPECS:
        panels.append(
            _load_or_infer_panel(
                project_root=project_root,
                root=transfer_root,
                spec=spec,
                condition="e9",
                protocol=protocol,
                device=device,
            )
        )

    alignment = validate_panel_alignment(panels)
    sample_positions_by_horizon: dict[int, np.ndarray] = {}
    sampling_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        sample_positions, sampling = select_shared_sample_positions(
            panels,
            horizon=horizon,
            max_points=args.max_scatter_points,
            seed=args.seed,
        )
        sample_positions_by_horizon[horizon] = sample_positions
        sampling_by_horizon[f"H{horizon}"] = sampling
    axis_limits = resolve_axis_limits(
        panels,
        physical_clip=physical_clip,
        horizon=max(HORIZONS),
    )
    axis = axis_limits[2]
    density_normalization = compute_density_normalization(
        panels,
        horizons=HORIZONS,
        sample_positions_by_horizon=sample_positions_by_horizon,
        axis_limits=axis_limits[:2],
    )
    original_panels = panels[: len(PANEL_SPECS)]
    e9_panels = panels[len(PANEL_SPECS) :]
    figure_specs = (
        (
            "original",
            original_panels,
            "Original Masked-MSE",
            output_dir / "E9_Original_MaskedMSE_prediction_jointgrid_H3_H6_H10_final",
        ),
        (
            "e9_msmg_dwu",
            e9_panels,
            "E9 MS-MG-DWU",
            output_dir / "E9_MS_MG_DWU_prediction_jointgrid_H3_H6_H10_final",
        ),
    )
    figures: dict[str, Any] = {}
    for figure_key, figure_panels, figure_title, output_stem in figure_specs:
        figures[figure_key] = render_joint_grid(
            figure_panels,
            horizons=HORIZONS,
            sample_positions_by_horizon=sample_positions_by_horizon,
            axis_limits=axis_limits[:2],
            output_stem=output_stem,
            figure_title=figure_title,
            density_normalization=density_normalization,
        )
    audit = build_audit(
        panels=panels,
        alignment=alignment,
        sampling_by_horizon=sampling_by_horizon,
        axis=axis,
        figures=figures,
        project_root=project_root,
        original_root=original_root,
        transfer_root=transfer_root,
        seed=args.seed,
        device=device,
    )
    audit_path = output_dir / "E9_Original_MS_MG_DWU_prediction_jointgrid_final_audit.json"
    _write_json(audit_path, audit)
    print("E9 Original/E9 prediction joint grids completed.", flush=True)
    for figure_key in ("original", "e9_msmg_dwu"):
        files = figures[figure_key]["files"]
        print(f"{figure_key} PNG: {files['png']}", flush=True)
        print(f"{figure_key} PDF: {files['pdf']}", flush=True)
        print(f"{figure_key} SVG: {files['svg']}", flush=True)
    print(f"AUDIT: {audit_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
