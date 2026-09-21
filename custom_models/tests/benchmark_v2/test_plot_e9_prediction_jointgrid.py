from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

loaded = sys.modules.get("benchmark_v2")
if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == Path(__file__).parent / "__init__.py":
    del sys.modules["benchmark_v2"]

from benchmark_v2.plot_e9_prediction_jointgrid import (  # noqa: E402
    HORIZONS,
    PANEL_SPECS,
    PanelData,
    build_audit,
    compute_point_density,
    compute_density_normalization,
    load_prediction_artifact,
    render_joint_grid,
    select_shared_sample_positions,
    validate_panel_alignment,
)


def _fixture_panels() -> list[PanelData]:
    rng = np.random.default_rng(2026)
    actual = rng.uniform(0.0, 1500.0, size=(4, 3, 10)).astype(np.float32)
    mask = np.ones_like(actual, dtype=bool)
    panels: list[PanelData] = []
    for index in range(10):
        row = index // 5
        model_index = index % 5
        prediction = np.clip(
            actual + rng.normal(0.0, 30.0 + model_index, size=actual.shape),
            0.0,
            1500.0,
        ).astype(np.float32)
        spec = PANEL_SPECS[model_index]
        metrics = {
            horizon: {"Score": 1.0, "MAE": 2.0, "RMSE": 3.0, "R2": 0.9}
            for horizon in (3, 6, 10)
        }
        panels.append(
            PanelData(
                spec=spec,
                condition="original" if row == 0 else "e9",
                run_dir=Path("fixture") / spec.original_run_id,
                run_id=f"fixture-{index}",
                model_id=spec.model_id,
                actual_snh=actual,
                prediction_snh=prediction,
                mask_snh=mask,
                sample_index=np.arange(actual.shape[0], dtype=np.int64),
                checkpoint_path=None,
                prediction_artifact=None,
                prediction_source="fixture",
                inference_executed=False,
                artifact_metadata={},
                effective_config={},
                official_metrics=metrics,
                recomputed_metrics=metrics,
                metric_comparisons={horizon: {} for horizon in HORIZONS},
            )
        )
    return panels


def test_alignment_and_shared_sampling_are_deterministic():
    panels = _fixture_panels()
    alignment = validate_panel_alignment(panels)
    assert alignment["valid_target_mask_identical"]
    first, audit = select_shared_sample_positions(
        panels, horizon=10, max_points=7, seed=2026
    )
    second, _ = select_shared_sample_positions(
        panels, horizon=10, max_points=7, seed=2026
    )
    assert np.array_equal(first, second)
    assert len(first) == 7
    assert audit["same_flattened_positions_for_all_panels"]


def test_prediction_artifact_loader_accepts_formal_export_keys(tmp_path: Path):
    actual = np.ones((2, 3, 10), dtype=np.float32)
    prediction = np.full_like(actual, 2.0)
    mask = np.ones_like(actual, dtype=bool)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        y_true_snh=actual,
        y_pred_eval_snh=prediction,
        valid_target_mask_snh=mask,
        sample_index=np.array([10, 11], dtype=np.int64),
    )
    loaded = load_prediction_artifact([path], physical_clip=(0.0, 1500.0))
    assert loaded is not None
    loaded_actual, loaded_prediction, loaded_mask, sample_index, source, _ = loaded
    assert source == path
    assert np.array_equal(loaded_actual, actual)
    assert np.array_equal(loaded_prediction, prediction)
    assert np.array_equal(loaded_mask, mask)
    assert np.array_equal(sample_index, np.array([10, 11]))


def test_density_and_lightweight_render(tmp_path: Path):
    x = np.array([0.0, 10.0, 10.0, 100.0], dtype=np.float64)
    y = np.array([0.0, 10.0, 20.0, 100.0], dtype=np.float64)
    density = compute_point_density(x, y, axis_limits=(0.0, 1500.0), bins=32)
    assert density.shape == x.shape
    assert np.isfinite(density).all()

    panels = _fixture_panels()
    positions_by_horizon = {}
    for horizon in HORIZONS:
        positions_by_horizon[horizon], _ = select_shared_sample_positions(
            panels, horizon=horizon, max_points=8, seed=2026
        )
    density_normalization = compute_density_normalization(
        panels,
        horizons=HORIZONS,
        sample_positions_by_horizon=positions_by_horizon,
        axis_limits=(0.0, 1500.0),
        density_bins=32,
    )
    stem = tmp_path / "fixture_jointgrid_original_H3_H6_H10"
    result = render_joint_grid(
        panels[:5],
        horizons=HORIZONS,
        sample_positions_by_horizon=positions_by_horizon,
        axis_limits=(0.0, 1500.0),
        output_stem=stem,
        figure_title="Original Masked-MSE",
        density_normalization=density_normalization,
        density_bins=32,
    )
    assert result["png_dpi"] == 600
    assert result["panel_count"] == 15
    assert result["horizons"] == [3, 6, 10]
    assert all(path.is_file() for path in result["files"].values())

    sampling = {
        f"H{horizon}": {
            "sampled_count_per_panel": len(positions_by_horizon[horizon]),
        }
        for horizon in HORIZONS
    }
    audit = build_audit(
        panels=panels,
        alignment=validate_panel_alignment(panels),
        sampling_by_horizon=sampling,
        axis={"limits_kw": [0.0, 1500.0]},
        figures={"original": result, "e9_msmg_dwu": result},
        project_root=PROJECT_ROOT,
        original_root=PROJECT_ROOT / "original",
        transfer_root=PROJECT_ROOT / "transfer",
        seed=2026,
        device="cpu",
    )
    assert audit["figure_count"] == 2
    assert audit["panel_count"] == 30
    assert len(audit["panels"]) == 30
    assert audit["metric_order_in_each_panel"] == ["Score", "R2", "RMSE", "MAE"]
