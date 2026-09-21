from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from st_mgprompt.empirical_protocol import FROZEN_PROTOCOL_FIELDS, apply_empirical_variant, assert_empirical_expected_diff
from st_mgprompt.experiment_protocol import CANONICAL_ID, canonical_config
from st_mgprompt.graph_layers import AdaptiveGraphBuilder, FixedPriorGraphBuilder
from st_mgprompt.graph_prior import (
    EPS,
    build_distance_prior_graph,
    build_graph_artifacts,
    build_macro_trend_graph,
    build_micro_local_graph,
    degree_preserving_random_rewire,
)
from st_mgprompt.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _distinct_graphs(num_nodes: int = 4) -> dict:
    micro = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    macro = np.zeros_like(micro)
    for idx in range(num_nodes):
        micro[idx, (idx + 1) % num_nodes] = 1.0
        macro[idx, (idx - 1) % num_nodes] = 1.0
    return {
        "A_macro_trend": macro,
        "A_micro_local": micro,
        "metadata": {"graph_uses_train_only_statistics": True, "fit_split": "train_only"},
    }


def _small_model(variant: str, num_nodes: int = 4):
    config = apply_empirical_variant(None, variant, "G")
    config.hidden_dim = 8
    config.num_nodes = num_nodes
    config.dropout = 0.0
    config.use_train_robust_volatility = False
    config.diagnostics_level = "minimal"
    model = build_model(config, input_dim=len(config.feature_cols), graph_data=_distinct_graphs(num_nodes))
    return config, model


def _locations(num_nodes: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TurbID": np.arange(1, num_nodes + 1),
            "x": np.arange(num_nodes, dtype=np.float32),
            "y": np.square(np.arange(num_nodes, dtype=np.float32)),
        }
    )


def test_g0_matches_canonical_config_and_checkpoint_graph_identity() -> None:
    config = apply_empirical_variant(None, "G0", "G")
    canonical = canonical_config()
    for field in (
        "branch_graph_assignment",
        "graph_prior_component",
        "adaptive_support_mode",
        "graph_rewire_mode",
        "graph_operator",
    ):
        assert getattr(config, field) == getattr(canonical, field)
    checkpoint = PROJECT_ROOT / "custom_models/results/st_mgprompt_canonical" / CANONICAL_ID / "best_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    assert state["A_macro_prior"].shape == (134, 134)
    assert state["A_micro_prior"].shape == (134, 134)


@pytest.mark.parametrize(
    "variant,fine_id,coarse_id",
    [
        ("G0", "micro", "macro"),
        ("G1", "micro", "micro"),
        ("G2", "macro", "macro"),
        ("G3", "macro", "micro"),
    ],
)
def test_branch_assignment_changes_actual_graph_inputs(variant: str, fine_id: str, coarse_id: str) -> None:
    _, model = _small_model(variant)
    fine_prior, coarse_prior, actual_fine_id, actual_coarse_id = model.assigned_graph_priors()
    assert (actual_fine_id, actual_coarse_id) == (fine_id, coarse_id)
    expected = {"micro": model.A_micro_prior, "macro": model.A_macro_prior}
    assert fine_prior.data_ptr() == expected[fine_id].data_ptr()
    assert coarse_prior.data_ptr() == expected[coarse_id].data_ptr()
    output = model(torch.randn(2, 12, 4, 16))
    assert output["aux"]["fine_graph_id"] == fine_id
    assert output["aux"]["coarse_graph_id"] == coarse_id


def test_distance_only_and_statistics_only_have_exclusive_sources() -> None:
    rng = np.random.default_rng(7)
    series = rng.normal(size=(48, 5)).astype(np.float32)
    mask = np.ones_like(series, dtype=bool)
    locations = _locations(5)
    g4 = apply_empirical_variant(None, "G4", "G")
    g4.num_nodes = 5
    setattr(g4, "_graph_num_nodes", 5)
    setattr(g4, "_graph_turbine_ids", list(range(1, 6)))
    distance = build_distance_prior_graph(locations, g4)
    assert np.allclose(build_macro_trend_graph(series, mask, locations, g4), distance)
    assert np.allclose(build_micro_local_graph(series, mask, locations, g4), distance)

    g5 = apply_empirical_variant(None, "G5", "G")
    g5.num_nodes = 5
    setattr(g5, "_graph_num_nodes", 5)
    setattr(g5, "_graph_turbine_ids", list(range(1, 6)))
    macro = build_macro_trend_graph(series, mask, locations, g5)
    micro = build_micro_local_graph(series, mask, locations, g5)
    assert np.isfinite(macro).all() and np.isfinite(micro).all()
    assert np.allclose(macro.sum(axis=1), 1.0)
    assert np.allclose(micro.sum(axis=1), 1.0)
    assert not np.allclose(macro, distance)


def test_g6_allows_candidates_outside_prior_and_g7_has_no_adaptive_parameters() -> None:
    _, free_model = _small_model("G6")
    assert isinstance(free_model.micro_graph_builder, AdaptiveGraphBuilder)
    fine_graph, coarse_graph, trace = free_model.effective_graphs()
    assert trace["fine_edges_outside_assigned_prior"] > 0
    assert trace["coarse_edges_outside_assigned_prior"] > 0
    assert torch.all(fine_graph > 0) and torch.all(coarse_graph > 0)

    _, fixed_model = _small_model("G7")
    assert isinstance(fixed_model.micro_graph_builder, FixedPriorGraphBuilder)
    assert isinstance(fixed_model.macro_graph_builder, FixedPriorGraphBuilder)
    assert not any(name.endswith(".E1") or name.endswith(".E2") for name, _ in fixed_model.named_parameters())
    fixed_model(torch.randn(2, 12, 4, 16))["pred"].sum().backward()
    assert not any("graph_builder.E" in name for name, _ in fixed_model.named_parameters())


def test_g8_preserves_each_row_degree_self_loop_state_and_shape() -> None:
    adjacency = np.array(
        [
            [0.0, 0.6, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.5],
            [0.4, 0.0, 0.2, 0.0, 0.4],
            [0.3, 0.0, 0.0, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    rewired = degree_preserving_random_rewire(adjacency, seed=2026, keep_self=False)
    assert rewired.shape == adjacency.shape
    assert np.array_equal(np.count_nonzero(adjacency > EPS, axis=1), np.count_nonzero(rewired > EPS, axis=1))
    assert np.count_nonzero(np.diag(rewired)) == 0
    assert np.isfinite(rewired).all()
    assert np.allclose(rewired.sum(axis=1), 1.0)


def _fake_bundle(tmp_path: Path, future_shift: float):
    rng = np.random.default_rng(11)
    time_count, train_end, nodes = 24, 12, 4
    x = rng.normal(size=(time_count, nodes, 16)).astype(np.float32)
    y = rng.normal(size=(time_count, nodes)).astype(np.float32)
    x[train_end:] += future_shift
    y[train_end:] += future_shift
    dataset = SimpleNamespace(x=x, y_raw=y, mask=np.ones_like(y, dtype=np.float32))
    config = apply_empirical_variant(None, "G5", "G")
    config.num_nodes = nodes
    config.output_root = str(tmp_path)
    config.run_id = f"future_{int(future_shift)}"
    location_path = tmp_path / f"locations_{int(future_shift)}.csv"
    _locations(nodes).to_csv(location_path, index=False)
    config.location_path = str(location_path)
    bundle = SimpleNamespace(
        train_loader=SimpleNamespace(dataset=dataset),
        split_indices={"train": (0, train_end)},
        feature_cols=list(config.feature_cols),
        num_nodes=nodes,
        turbine_ids=list(range(1, nodes + 1)),
    )
    return bundle, config


def test_validation_or_test_changes_do_not_change_train_fitted_graphs(tmp_path: Path) -> None:
    bundle_a, config_a = _fake_bundle(tmp_path, 0.0)
    bundle_b, config_b = _fake_bundle(tmp_path, 10000.0)
    graph_a = build_graph_artifacts(bundle_a, config_a)
    graph_b = build_graph_artifacts(bundle_b, config_b)
    assert np.array_equal(graph_a["A_macro_trend"], graph_b["A_macro_trend"])
    assert np.array_equal(graph_a["A_micro_local"], graph_b["A_micro_local"])
    assert graph_a["metadata"]["trend_statistics_fit_split"] == "train_only"


@pytest.mark.parametrize("variant", [f"G{i}" for i in range(1, 9)])
def test_g_variants_preserve_shape_finiteness_and_frozen_protocol(variant: str) -> None:
    config, model = _small_model(variant)
    report = assert_empirical_expected_diff(apply_empirical_variant(None, variant, "G"), variant, "G")
    assert report["passed"]
    assert not report["frozen_protocol_changes"]
    canonical = canonical_config().to_dict()
    formal = apply_empirical_variant(None, variant, "G").to_dict()
    for field in FROZEN_PROTOCOL_FIELDS:
        assert formal[field] == canonical[field], (variant, field)
    output = model(torch.randn(2, 12, 4, len(config.feature_cols)))["pred"]
    assert output.shape == (2, 10, 4)
    assert torch.isfinite(output).all()


def test_manifest_fields_are_serializable_and_match_contract() -> None:
    rows = []
    for variant in [f"G{i}" for i in range(9)]:
        config = apply_empirical_variant(None, variant, "G")
        rows.append(
            {
                "variant": variant,
                "branch_graph_assignment": config.branch_graph_assignment,
                "graph_prior_component": config.graph_prior_component,
                "adaptive_support_mode": config.adaptive_support_mode,
                "graph_rewire_mode": config.graph_rewire_mode,
            }
        )
    assert len(json.dumps(rows)) > 0
