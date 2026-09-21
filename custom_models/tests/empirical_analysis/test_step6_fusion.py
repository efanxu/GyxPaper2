from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from st_mgprompt.empirical_protocol import (
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    assert_empirical_expected_diff,
)
from st_mgprompt.experiment_protocol import canonical_config
from st_mgprompt.registry import build_model


def _graphs(num_nodes: int) -> dict:
    micro = np.eye(num_nodes, k=1, dtype=np.float32)
    micro[-1, 0] = 1.0
    macro = np.eye(num_nodes, k=-1, dtype=np.float32)
    macro[0, -1] = 1.0
    return {
        "A_macro_trend": macro,
        "A_micro_local": micro,
        "metadata": {
            "graph_uses_train_only_statistics": True,
            "fit_split": "canonical_train_only_checkpoint",
            "graph_identity_reference": "G0/CANONICAL",
        },
    }


def _small_model(variant: str, num_nodes: int = 4, diagnostics: str = "standard"):
    config = apply_empirical_variant(None, variant, "F")
    config.hidden_dim = 8
    config.num_nodes = num_nodes
    config.dropout = 0.0
    config.use_train_robust_volatility = False
    config.diagnostics_level = diagnostics
    torch.manual_seed(2026)
    model = build_model(config, input_dim=len(config.feature_cols), graph_data=_graphs(num_nodes)).eval()
    return config, model


def _forward(variant: str, num_nodes: int = 4):
    config, model = _small_model(variant, num_nodes=num_nodes)
    x = torch.randn(2, 12, num_nodes, len(config.feature_cols))
    with torch.inference_mode():
        output = model(x)
    return config, model, x, output


def test_f0_does_not_instantiate_or_call_macro_prompt_or_cross_fusion() -> None:
    _, model, _, output = _forward("F0")
    block = model.coupling_blocks[0]
    assert block.macro_prompt_encoder is None
    assert block.symmetric_cross_fusion is None
    assert block.basic_fusion_mode is None
    assert output["aux"]["uses_macro_prompt"] is False
    assert output["aux"]["uses_cross_fusion"] is False
    assert output["aux"]["macro_prompt"] is None
    assert output["aux"]["pre_fusion_cosine_similarity"] == output["aux"]["post_fusion_cosine_similarity"]
    assert output["aux"]["fine_representation_shift"] == 0.0
    assert output["aux"]["coarse_representation_shift"] == 0.0


@pytest.mark.parametrize("variant,mode", [("F1", "add"), ("F2", "concat"), ("F3", "unified_gated")])
def test_basic_fusions_do_not_call_cross_attention_or_macro_prompt(variant: str, mode: str) -> None:
    config, model, _, output = _forward(variant)
    block = model.coupling_blocks[0]
    assert config.fusion_mode == mode
    assert block.basic_fusion_mode == mode
    assert block.macro_prompt_encoder is None
    assert block.symmetric_cross_fusion is None
    assert output["aux"]["basic_fusion_called"] is True
    assert output["aux"]["uses_macro_prompt"] is False
    assert output["aux"]["uses_cross_fusion"] is False
    assert output["pred"].shape == (2, 10, 4)
    assert output["aux"]["attention_not_applicable_reason"] == "basic_fusion_has_no_cross_attention"


def test_f1_f2_dimensions_and_capacity_are_auditable() -> None:
    _, f1, _, out1 = _forward("F1")
    _, f2, _, out2 = _forward("F2")
    f1_block = f1.coupling_blocks[0]
    f2_block = f2.coupling_blocks[0]
    assert f1_block.basic_add_norm is not None
    assert f2_block.basic_concat_mlp is not None
    assert f2_block.basic_concat_mlp[0].in_features == 16
    assert f2_block.basic_concat_mlp[-2].out_features == 8
    assert sum(p.numel() for p in f2_block.basic_concat_mlp.parameters()) > sum(
        p.numel() for p in f1_block.basic_add_norm.parameters()
    )
    assert out1["aux"]["x_fine_coupled"].shape[-1] == 8
    assert out2["aux"]["x_fine_coupled"].shape[-1] == 8


def test_f3_gate_is_finite_bounded_and_declares_semantics() -> None:
    config, model, _, output = _forward("F3")
    block = model.coupling_blocks[0]
    assert block.unified_gate is not None
    assert torch.allclose(block.unified_gate.bias, torch.zeros_like(block.unified_gate.bias))
    aux = output["aux"]
    for key in ("fusion_gate_mean", "fusion_gate_std", "fusion_gate_q05", "fusion_gate_q50", "fusion_gate_q95"):
        assert math.isfinite(float(aux[key]))
    assert 0.0 <= aux["fusion_gate_q05"] <= aux["fusion_gate_q50"] <= aux["fusion_gate_q95"] <= 1.0
    assert 0.0 <= aux["fusion_gate_saturation_ratio"] <= 1.0
    assert aux["gate_input"] == "fine_coarse_concat"
    assert aux["gate_level"] == "node_time_channel"
    assert aux["gate_range"] == "[0,1]"
    assert aux["gate_stop_gradient"] is config.unified_gate_stop_gradient is False


@pytest.mark.parametrize(
    "variant,macro_enabled,fine_enabled",
    [("F4", True, False), ("F5", False, True)],
)
def test_direction_trace_matches_real_query_key_value_flow(
    variant: str,
    macro_enabled: bool,
    fine_enabled: bool,
) -> None:
    _, model, _, output = _forward(variant)
    block = model.coupling_blocks[0]
    fusion = block.symmetric_cross_fusion
    assert fusion is not None
    aux = output["aux"]
    assert (aux["macro_to_fine_query_source"] == "fine_history") is macro_enabled
    assert (aux["macro_to_fine_key_source"] == "macro_prompt") is macro_enabled
    assert (aux["fine_to_coarse_query_source"] == "coarse_history") is fine_enabled
    assert (aux["fine_to_coarse_key_source"] == "recent_fine_history") is fine_enabled
    assert (aux["macro_to_fine_attention_shape"] is not None) is macro_enabled
    assert (aux["fine_to_coarse_attention_shape"] is not None) is fine_enabled
    assert (fusion.macro_to_fine is fusion.fine_to_coarse) is False


def test_f6_shares_real_projection_parameters_and_f7_does_not() -> None:
    _, f6 = _small_model("F6")
    _, f7 = _small_model("F7")
    f6_fusion = f6.coupling_blocks[0].symmetric_cross_fusion
    f7_fusion = f7.coupling_blocks[0].symmetric_cross_fusion
    assert f6_fusion.macro_to_fine is f6_fusion.fine_to_coarse
    assert f6_fusion.macro_to_fine.in_proj_weight.data_ptr() == f6_fusion.fine_to_coarse.in_proj_weight.data_ptr()
    assert f7_fusion.macro_to_fine is not f7_fusion.fine_to_coarse
    assert f7_fusion.macro_to_fine.in_proj_weight.data_ptr() != f7_fusion.fine_to_coarse.in_proj_weight.data_ptr()


def test_f8_uses_mean_pooling_with_four_tokens_and_full_history() -> None:
    _, model, _, output = _forward("F8")
    encoder = model.coupling_blocks[0].macro_prompt_encoder
    assert encoder is not None
    assert encoder.pooling == "mean"
    assert encoder.prompt_len == 4
    assert output["aux"]["macro_prompt_shape"] == [2, 4, 4, 8]
    assert output["aux"]["macro_prompt_source_history_range"] == "[0,12)"
    assert output["aux"]["macro_prompt_attention_enabled"] is False
    assert output["aux"]["macro_prompt_attn_entropy"] is None


@pytest.mark.parametrize("variant", ["F4", "F5"])
def test_active_attention_is_normalized_and_entropy_is_finite(variant: str) -> None:
    _, _, _, output = _forward(variant)
    aux = output["aux"]
    active = []
    for entropy_key, error_key in (
        ("macro_attn_entropy", "macro_to_fine_attention_row_sum_max_error"),
        ("fine_attn_entropy", "fine_to_coarse_attention_row_sum_max_error"),
    ):
        if aux[entropy_key] is not None:
            active.append(entropy_key)
            assert math.isfinite(float(aux[entropy_key]))
            assert float(aux[error_key]) < 1e-5
    assert len(active) == 1


@pytest.mark.parametrize("variant", [f"F{i}" for i in range(9)])
def test_all_variants_keep_output_and_protocol_contract(variant: str) -> None:
    config, model = _small_model(variant, num_nodes=134, diagnostics="minimal")
    with torch.inference_mode():
        pred = model(torch.randn(1, 12, 134, len(config.feature_cols)))["pred"]
    assert pred.shape == (1, 10, 134)
    assert torch.isfinite(pred).all()
    report = assert_empirical_expected_diff(apply_empirical_variant(None, variant, "F"), variant, "F")
    assert report["passed"]
    assert not report["frozen_protocol_changes"]
    canonical = canonical_config().to_dict()
    effective = apply_empirical_variant(None, variant, "F").to_dict()
    for field in FROZEN_PROTOCOL_FIELDS:
        assert effective[field] == canonical[field], (variant, field)
    for field in (
        "target_mask_col",
        "enable_physical_clip_eval",
        "physical_power_min_kw",
        "physical_power_max_kw",
        "loss_function",
        "decoder_context_mode",
        "branch_graph_assignment",
    ):
        assert effective[field] == canonical[field], (variant, field)


def test_future_target_cannot_change_fusion_forward() -> None:
    config, model = _small_model("F3")
    x = torch.randn(1, 12, 4, len(config.feature_cols))
    future_target = torch.randn(1, 10, 4)
    with torch.inference_mode():
        before = model(x)["pred"].clone()
        future_target.add_(10000.0)
        after = model(x)["pred"].clone()
    assert torch.equal(before, after)


def test_diagnostic_export_read_does_not_change_prediction() -> None:
    config, model = _small_model("F4")
    x = torch.randn(1, 12, 4, len(config.feature_cols))
    with torch.inference_mode():
        first = model(x)
        exported = {
            key: first["aux"].get(key)
            for key in (
                "macro_attn_entropy",
                "fusion_gate_mean",
                "macro_to_fine_attention_shape",
                "macro_prompt_pairwise_cosine_mean",
            )
        }
        second = model(x)
    assert exported["macro_attn_entropy"] is not None
    assert torch.equal(first["pred"], second["pred"])
