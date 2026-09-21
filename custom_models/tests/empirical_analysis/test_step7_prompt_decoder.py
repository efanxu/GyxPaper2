from __future__ import annotations

import math

import pytest
import torch

from scripts.empirical_analysis.prompt_intervention import temporary_prompt_parameter_intervention
from st_mgprompt.decoder import (
    HorizonDirectDecoder,
    STPromptDirectDecoder,
    STPromptFullHistoryDecoder,
    STPromptHistoryPoolingDecoder,
)
from st_mgprompt.empirical_protocol import (
    EMPIRICAL_FAMILIES,
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    assert_empirical_expected_diff,
)
from st_mgprompt.prompt_alignment import STPromptEmbedding


def _history(batch: int = 2, length: int = 12, nodes: int = 3, dim: int = 8):
    generator = torch.Generator().manual_seed(2026)
    fine = torch.randn(batch, length, nodes, dim, generator=generator)
    coarse = torch.randn(batch, length, nodes, dim, generator=generator)
    macro = torch.randn(batch, nodes, 4, dim, generator=generator)
    return fine, coarse, macro


def _prompt(nodes: int = 3, horizon: int = 10, dim: int = 8) -> torch.Tensor:
    module = STPromptEmbedding(nodes, horizon, dim, dropout=0.0)
    module.eval()
    return module()


def test_n0_node_prompt_is_shared_in_eval_mode() -> None:
    module = STPromptEmbedding(
        num_nodes=4,
        max_pred_len=10,
        hidden_dim=8,
        dropout=0.0,
        use_node_identity=False,
    )
    module.eval()
    prompt = module()
    assert prompt.shape == (1, 10, 4, 8)
    assert torch.equal(prompt, prompt[:, :, :1].expand_as(prompt))


@pytest.mark.parametrize(
    ("shared_learnable", "expected_representation"),
    [(False, "shared_zero"), (True, "shared_learnable")],
)
def test_n1_n3_horizon_prompt_is_shared(
    shared_learnable: bool, expected_representation: str
) -> None:
    module = STPromptEmbedding(
        num_nodes=4,
        max_pred_len=10,
        hidden_dim=8,
        dropout=0.0,
        use_horizon_identity=False,
        use_shared_horizon_embedding=shared_learnable,
    )
    module.eval()
    prompt = module()
    assert module.horizon_representation == expected_representation
    assert torch.equal(prompt, prompt[:, :1].expand_as(prompt))


def test_n2_has_no_node_or_horizon_identity_and_keeps_fixed_type_semantics() -> None:
    config = apply_empirical_variant(None, "N2", "N")
    assert config.st_prompt_use_node_identity is False
    assert config.st_prompt_use_horizon_identity is False
    assert config.st_prompt_use_type_embedding is True
    assert config.st_prompt_type_semantics == "fixed_decoder_input_type"
    module = STPromptEmbedding(
        3,
        10,
        8,
        dropout=0.0,
        use_node_identity=False,
        use_horizon_identity=False,
        use_type_embedding=True,
        type_semantics=config.st_prompt_type_semantics,
    )
    module.eval()
    prompt = module()
    assert torch.equal(prompt, prompt[:, :1, :1].expand_as(prompt))


def test_n4_direct_horizon_heads_are_non_recursive_and_bhn() -> None:
    fine, coarse, macro = _history()
    decoder = HorizonDirectDecoder(hidden_dim=8, max_pred_len=10, dropout=0.0)
    pred, aux = decoder(fine, coarse, macro_prompt=macro, return_aux=True)
    assert pred.shape == (2, 10, 3)
    assert torch.isfinite(pred).all()
    assert decoder.metadata["head_parameterization"] == "shared_context_with_independent_output_columns"
    assert decoder.metadata["head_specific_parameter_count"] == 9
    assert aux["teacher_forcing"] is False
    assert aux["autoregressive"] is False
    assert aux["future_observed_features_used"] is False


def test_last_state_mean_and_attention_pooling_history_contracts() -> None:
    fine, coarse, macro = _history(length=12)
    prompt = _prompt()
    last = STPromptDirectDecoder(8, dropout=0.0)
    last_pred, last_aux = last(fine, coarse, prompt, macro, return_aux=True)
    assert last_pred.shape == (2, 10, 3)
    assert last_aux["decoder_type"] == "STPromptDirectDecoder"

    for pooling in ("mean", "attention"):
        decoder = STPromptHistoryPoolingDecoder(8, pooling=pooling, dropout=0.0)
        pred, aux = decoder(fine, coarse, prompt, macro, return_aux=True)
        assert pred.shape == (2, 10, 3)
        assert aux["decoder_actual_history_len"] == 12
        assert aux["decoder_history_range"] == "[0,12)"
        assert aux["decoder_used_complete_history"] is True
        assert math.isfinite(aux["fine_history_attention_entropy"])
        assert aux["fine_history_weight_sum_max_error"] < 1e-6


def test_n8_cross_attention_reads_only_complete_history() -> None:
    fine, coarse, macro = _history(length=12)
    prompt = _prompt()
    decoder = STPromptFullHistoryDecoder(8, num_heads=2, dropout=0.0, history_len=None)
    pred, aux = decoder(fine, coarse, prompt, macro, return_aux=True)
    assert pred.shape == (2, 10, 3)
    assert aux["decoder_actual_history_len"] == 12
    assert aux["decoder_used_complete_history"] is True
    assert aux["decoder_query_shape"] == [2, 10, 3, 8]
    assert len(aux["fine_history_attention_entropy_per_horizon"]) == 10
    assert aux["future_observed_features_used"] is False


def test_all_decoders_produce_one_ten_step_prediction_for_prefix_evaluation() -> None:
    fine, coarse, macro = _history()
    prompt = _prompt()
    predictions = [
        STPromptDirectDecoder(8, dropout=0.0)(fine, coarse, prompt, macro),
        STPromptHistoryPoolingDecoder(8, "mean", dropout=0.0)(fine, coarse, prompt, macro),
        STPromptHistoryPoolingDecoder(8, "attention", dropout=0.0)(fine, coarse, prompt, macro),
        STPromptFullHistoryDecoder(8, num_heads=2, dropout=0.0)(fine, coarse, prompt, macro),
        HorizonDirectDecoder(8, 10, dropout=0.0)(fine, coarse, macro_prompt=macro),
    ]
    for pred in predictions:
        assert pred.shape == (2, 10, 3)
        assert torch.isfinite(pred).all()
        assert pred[:, :3].shape[1] == 3
        assert pred[:, :6].shape[1] == 6
        assert pred[:, :10].shape[1] == 10


def test_intervention_is_repeatable_and_restores_original_prompt() -> None:
    module = STPromptEmbedding(4, 10, 8, dropout=0.0)
    module.eval()
    baseline = module().detach().clone()
    changed = []
    for _ in range(2):
        with temporary_prompt_parameter_intervention(module, "horizon_swap") as applied:
            assert applied is True
            changed.append(module().detach().clone())
        assert torch.equal(module(), baseline)
    assert torch.equal(changed[0], changed[1])
    assert not torch.equal(changed[0], baseline)


def test_n_family_unique_difference_audit_preserves_frozen_protocol() -> None:
    assert set(EMPIRICAL_FAMILIES["N"]) == {f"N{index}" for index in range(9)}
    for variant_id in EMPIRICAL_FAMILIES["N"]:
        config = apply_empirical_variant(None, variant_id, "N")
        report = assert_empirical_expected_diff(config, variant_id, "N")
        assert report["passed"] is True
        assert not set(report["actual_diff_fields"]).intersection(FROZEN_PROTOCOL_FIELDS)
    n1 = apply_empirical_variant(None, "N1", "N")
    n3 = apply_empirical_variant(None, "N3", "N")
    assert n1.st_prompt_use_shared_horizon_embedding is False
    assert n3.st_prompt_use_shared_horizon_embedding is True
