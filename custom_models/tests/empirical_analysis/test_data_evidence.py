from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.empirical_analysis.build_data_evidence import (
    CAUSAL_WINDOWS,
    build_evidence,
    causal_downsample_upsample,
    causal_rolling_mean,
)


def test_causal_rolling_mean_has_no_future_access() -> None:
    values = np.arange(80, dtype=np.float64).reshape(40, 2)
    baseline = causal_rolling_mean(values, 6)
    changed = values.copy()
    changed[21:] += 10_000
    perturbed = causal_rolling_mean(changed, 6)
    np.testing.assert_allclose(baseline[:21], perturbed[:21])
    expected = values[15:21].mean(axis=0)
    np.testing.assert_allclose(baseline[20], expected)


def test_downsample_upsample_is_causal_and_records_provenance() -> None:
    values = np.arange(96, dtype=np.float64).reshape(48, 2)
    baseline, source_start, source_end = causal_downsample_upsample(values, 6)
    assert np.all(source_end <= np.arange(values.shape[0]))
    assert np.all(source_start <= source_end)
    changed = values.copy()
    changed[25:] -= 50_000
    perturbed, _, _ = causal_downsample_upsample(changed, 6)
    np.testing.assert_allclose(baseline[:25], perturbed[:25])
    np.testing.assert_allclose(baseline[12], values[6:12].mean(axis=0))


def test_smoke_build_writes_required_auditable_outputs(tmp_path: Path) -> None:
    result = build_evidence(output_dir=tmp_path, smoke=True, make_plots=False)
    required = {
        "data_overview.csv",
        "turbine_summary.csv",
        "temporal_statistics.csv",
        "horizon_feature_association.csv",
        "alignment_event_metrics.csv",
        "graph_semantic_fit.csv",
        "graph_neighbor_overlap.csv",
        "baseline_horizon_difficulty.csv",
        "baseline_turbine_difficulty.csv",
        "analysis_metadata.json",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    metadata = json.loads((tmp_path / "analysis_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "smoke"
    assert metadata["temporal_protocol"]["coarse_windows"] == list(CAUSAL_WINDOWS)
    assert metadata["train_only_protocol"]["validation_or_test_targets_used_for_training_statistics"] is False
    assert metadata["alignment_audit"]["resampled_all_sources_causal"] is True
    assert metadata["difficulty_audit"]["test_targets_used_for_threshold"] is False
    assert result["num_nodes"] == 8

    overview = pd.read_csv(tmp_path / "data_overview.csv")
    assert overview["split"].tolist() == ["train", "val", "test"]
    turbine = pd.read_csv(tmp_path / "turbine_summary.csv")
    assert set(turbine["signal"]) == {"Patv_raw", "Wspd", "Patv_clean_for_input"}
    assert turbine["TurbID"].nunique() == 8
    associations = pd.read_csv(tmp_path / "horizon_feature_association.csv")
    assert set(associations["future_horizon"]) == {3, 6, 10}
    assert (associations["target_start_offset"] == 1).all()
    graph_fit = pd.read_csv(tmp_path / "graph_semantic_fit.csv")
    assert "semantic_fit_lift_vs_distance" in graph_fit


def test_smoke_build_can_render_all_six_figures(tmp_path: Path) -> None:
    result = build_evidence(output_dir=tmp_path, smoke=True, make_plots=True)
    assert result["figure_count"] == 12
    pngs = sorted((tmp_path / "figures").glob("*.png"))
    svgs = sorted((tmp_path / "figures").glob("*.svg"))
    assert len(pngs) == 6
    assert len(svgs) == 6
    assert all(path.stat().st_size > 1_000 for path in [*pngs, *svgs])
