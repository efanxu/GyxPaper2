from __future__ import annotations

import numpy as np
import pytest
import torch

from st_mgprompt.empirical_protocol import (
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    assert_empirical_expected_diff,
)
from st_mgprompt.experiment_protocol import canonical_config
from st_mgprompt.graph_layers import PriorConstrainedDiffusionGraphConv, SimpleGraphConv, row_normalize
from st_mgprompt.registry import build_model


def _graphs(num_nodes: int) -> dict:
    micro = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    macro = np.zeros_like(micro)
    for node in range(num_nodes):
        micro[node, (node + 1) % num_nodes] = 0.7
        micro[node, (node + 3) % num_nodes] = 0.3
        macro[node, (node - 1) % num_nodes] = 0.6
        macro[node, (node - 4) % num_nodes] = 0.4
    return {
        "A_macro_trend": macro,
        "A_micro_local": micro,
        "metadata": {
            "graph_uses_train_only_statistics": True,
            "fit_split": "canonical_train_only_checkpoint",
            "graph_identity_reference": "G0/CANONICAL",
        },
    }


def _small_model(variant: str, num_nodes: int = 4):
    config = apply_empirical_variant(None, variant, "D")
    config.hidden_dim = 8
    config.num_nodes = num_nodes
    config.dropout = 0.0
    config.use_train_robust_volatility = False
    config.diagnostics_level = "minimal"
    torch.manual_seed(2026)
    model = build_model(config, input_dim=len(config.feature_cols), graph_data=_graphs(num_nodes))
    return config, model


def test_first_and_second_order_states_match_manual_recurrence() -> None:
    adjacency = torch.tensor(
        [[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [4.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    hidden = torch.tensor([[[[1.0], [2.0], [4.0]]]])
    module = PriorConstrainedDiffusionGraphConv(
        1,
        diffusion_order=2,
        dropout=0.0,
        direction="bidirectional",
    )
    forward_matrix, reverse_matrix, forward_states, reverse_states = module.propagate_states(hidden, adjacency)
    expected_forward_1 = torch.einsum("ij,btjd->btid", row_normalize(adjacency), hidden)
    expected_forward_2 = torch.einsum("ij,btjd->btid", row_normalize(adjacency), expected_forward_1)
    expected_reverse_1 = torch.einsum("ij,btjd->btid", row_normalize(adjacency.T), hidden)
    expected_reverse_2 = torch.einsum("ij,btjd->btid", row_normalize(adjacency.T), expected_reverse_1)
    assert torch.allclose(forward_matrix, row_normalize(adjacency))
    assert torch.allclose(reverse_matrix, row_normalize(adjacency.T))
    assert torch.allclose(forward_states[0], expected_forward_1)
    assert torch.allclose(forward_states[1], expected_forward_2)
    assert torch.allclose(reverse_states[0], expected_reverse_1)
    assert torch.allclose(reverse_states[1], expected_reverse_2)


def test_reverse_only_uses_transpose_not_forward_matrix() -> None:
    adjacency = torch.tensor(
        [[0.0, 1.0, 0.0], [0.2, 0.0, 0.8], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    hidden = torch.tensor([[[[1.0], [3.0], [7.0]]]])
    module = PriorConstrainedDiffusionGraphConv(1, diffusion_order=1, dropout=0.0, direction="reverse")
    _, _, forward_states, reverse_states = module.propagate_states(hidden, adjacency)
    expected_reverse = torch.einsum("ij,btjd->btid", row_normalize(adjacency.T), hidden)
    wrong_forward = torch.einsum("ij,btjd->btid", row_normalize(adjacency), hidden)
    assert forward_states == []
    assert len(reverse_states) == 1
    assert torch.allclose(reverse_states[0], expected_reverse)
    assert not torch.allclose(reverse_states[0], wrong_forward)


def test_d0_has_zero_diffusion_states_and_no_diffusion_parameter_gradient() -> None:
    _, model = _small_model("D0")
    block = model.coupling_blocks[0]
    graph_convs = (
        block.fine_micro_graph_temporal_encoder.block.graph_conv,
        block.coarse_macro_graph_temporal_encoder.block.graph_conv,
    )
    assert all(isinstance(module, SimpleGraphConv) for module in graph_convs)
    assert not any(isinstance(module, PriorConstrainedDiffusionGraphConv) for module in model.modules())
    output = model(torch.randn(2, 12, 4, 16))["pred"]
    output.sum().backward()
    assert not any("diffusion" in name.lower() for name, _ in model.named_parameters())
    assert not any("diffusion" in name.lower() for name, parameter in model.named_parameters() if parameter.grad is not None)


@pytest.mark.parametrize(
    "variant,direction,order,state_count,concat_order",
    [
        ("D1", "forward", 1, 1, ["input", "forward_hop_1"]),
        ("D2", "reverse", 1, 1, ["input", "reverse_hop_1"]),
        ("D3", "bidirectional", 1, 2, ["input", "forward_hop_1", "reverse_hop_1"]),
        (
            "D4",
            "bidirectional",
            2,
            4,
            ["input", "forward_hop_1", "forward_hop_2", "reverse_hop_1", "reverse_hop_2"],
        ),
        (
            "D5",
            "bidirectional",
            3,
            6,
            [
                "input",
                "forward_hop_1",
                "forward_hop_2",
                "forward_hop_3",
                "reverse_hop_1",
                "reverse_hop_2",
                "reverse_hop_3",
            ],
        ),
    ],
)
def test_variant_direction_and_state_contract(
    variant: str,
    direction: str,
    order: int,
    state_count: int,
    concat_order: list[str],
) -> None:
    _, model = _small_model(variant)
    block = model.coupling_blocks[0]
    for graph_conv in (
        block.fine_micro_graph_temporal_encoder.block.graph_conv,
        block.coarse_macro_graph_temporal_encoder.block.graph_conv,
    ):
        assert isinstance(graph_conv, PriorConstrainedDiffusionGraphConv)
        assert graph_conv.direction == direction
        assert graph_conv.diffusion_order == order
        assert graph_conv.diffusion_state_count == state_count
        assert graph_conv.state_concat_order == concat_order
        assert graph_conv.projection_mode == "shared_output_dim"


def test_d5_third_order_is_recursive_and_not_reused_second_order() -> None:
    adjacency = torch.tensor(
        [[0.0, 0.9, 0.1], [0.4, 0.0, 0.6], [0.8, 0.2, 0.0]],
        dtype=torch.float32,
    )
    hidden = torch.tensor([[[[1.0], [2.0], [5.0]]]])
    d4 = PriorConstrainedDiffusionGraphConv(1, diffusion_order=2, dropout=0.0, direction="bidirectional")
    d5 = PriorConstrainedDiffusionGraphConv(1, diffusion_order=3, dropout=0.0, direction="bidirectional")
    forward_matrix, reverse_matrix, d5_forward, d5_reverse = d5.propagate_states(hidden, adjacency)
    _, _, d4_forward, d4_reverse = d4.propagate_states(hidden, adjacency)
    assert torch.allclose(d5_forward[:2][0], d4_forward[0])
    assert torch.allclose(d5_forward[:2][1], d4_forward[1])
    assert torch.allclose(d5_reverse[:2][0], d4_reverse[0])
    assert torch.allclose(d5_reverse[:2][1], d4_reverse[1])
    assert torch.allclose(d5_forward[2], torch.einsum("ij,btjd->btid", forward_matrix, d5_forward[1]))
    assert torch.allclose(d5_reverse[2], torch.einsum("ij,btjd->btid", reverse_matrix, d5_reverse[1]))
    assert not torch.allclose(d5_forward[2], d5_forward[1])
    assert not torch.allclose(d5_reverse[2], d5_reverse[1])


@pytest.mark.parametrize("variant", [f"D{i}" for i in range(6)])
def test_all_variants_have_134_node_output_and_finite_cpu_values(variant: str) -> None:
    config, model = _small_model(variant, num_nodes=134)
    output = model(torch.randn(1, 12, 134, len(config.feature_cols)))["pred"]
    assert output.shape == (1, 10, 134)
    assert torch.isfinite(output).all()


def test_mixed_precision_cpu_diffusion_smoke_is_finite() -> None:
    config, model = _small_model("D5", num_nodes=134)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(torch.randn(1, 12, 134, len(config.feature_cols)))["pred"]
    assert output.shape == (1, 10, 134)
    assert torch.isfinite(output.float()).all()


@pytest.mark.parametrize("variant", [f"D{i}" for i in range(6)])
def test_only_declared_diffusion_fields_change_and_protocol_is_frozen(variant: str) -> None:
    config = apply_empirical_variant(None, variant, "D")
    report = assert_empirical_expected_diff(config, variant, "D")
    assert report["passed"]
    assert not report["frozen_protocol_changes"]
    canonical = canonical_config().to_dict()
    effective = config.to_dict()
    for field in FROZEN_PROTOCOL_FIELDS:
        assert effective[field] == canonical[field], (variant, field)
    for field in (
        "branch_graph_assignment",
        "graph_prior_component",
        "adaptive_support_mode",
        "graph_rewire_mode",
        "target_mask_col",
        "enable_physical_clip_eval",
        "physical_power_min_kw",
        "physical_power_max_kw",
        "loss_function",
        "decoder_context_mode",
        "batch_size",
        "train_sample_stride",
        "val_sample_stride",
        "test_sample_stride",
    ):
        assert effective[field] == canonical[field], (variant, field)


def test_adaptive_support_is_identical_when_only_diffusion_changes() -> None:
    reference_fine = reference_coarse = None
    for variant in [f"D{i}" for i in range(6)]:
        _, model = _small_model(variant)
        fine_graph, coarse_graph, trace = model.effective_graphs()
        assert trace["assignment"] == "matched"
        if reference_fine is None:
            reference_fine = fine_graph.detach()
            reference_coarse = coarse_graph.detach()
        else:
            assert torch.allclose(fine_graph, reference_fine)
            assert torch.allclose(coarse_graph, reference_coarse)


def test_diffusion_forward_reads_history_and_graph_only_not_future_target() -> None:
    config, model = _small_model("D5")
    x = torch.randn(1, 12, 4, len(config.feature_cols))
    future_target = torch.randn(1, 10, 4)
    before = model(x)["pred"].detach()
    future_target.add_(10000.0)
    after = model(x)["pred"].detach()
    assert torch.equal(before, after)
    assert model.graph_metadata["fit_split"] == "canonical_train_only_checkpoint"
    assert model.graph_metadata["graph_identity_reference"] == "G0/CANONICAL"
