from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from st_mgprompt.empirical_protocol import (
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    assert_empirical_expected_diff,
)
from st_mgprompt.experiment_protocol import CANONICAL_ID, canonical_config
from st_mgprompt.losses import get_loss_fn
from st_mgprompt.registry import build_model
from st_mgprompt.volatility_patching import causal_downsample_upsample


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _graph(num_nodes: int) -> dict:
    eye = np.eye(num_nodes, dtype=np.float32)
    return {
        "A_macro_trend": eye,
        "A_micro_local": eye,
        "metadata": {"graph_uses_train_only_statistics": True},
    }


def _small_model(variant: str, num_nodes: int = 4):
    config = apply_empirical_variant(None, variant, "T")
    config.hidden_dim = 8
    config.num_nodes = num_nodes
    config.dropout = 0.0
    config.use_train_robust_volatility = False
    config.diagnostics_level = "minimal"
    return config, build_model(config, input_dim=16, graph_data=_graph(num_nodes))


def test_t0_matches_canonical_graph_loss_decoder_identity() -> None:
    config = apply_empirical_variant(None, "T0", "T")
    canonical = canonical_config()
    assert config.graph_operator == canonical.graph_operator == "bidirectional_diffusion"
    assert config.loss_function == canonical.loss_function == "msmg_dwu_loss"
    assert config.decoder_context_mode == canonical.decoder_context_mode == "last_state"
    assert config.vadsp_gate_mode == canonical.vadsp_gate_mode == "fixed_dual"
    assert config.protocol_profile == "STMG_FORMAL_V2"
    source = (
        PROJECT_ROOT
        / "custom_models/results/st_mgprompt_canonical"
        / CANONICAL_ID
        / "effective_config.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key in ("graph_operator", "loss_function", "decoder_context_mode", "vadsp_gate_mode"):
        assert payload[key] == getattr(config, key)


@pytest.mark.parametrize("variant,field_value", [("T1", "fine_only"), ("T2", "coarse_only")])
def test_t1_t2_only_change_gate(variant: str, field_value: str) -> None:
    config = apply_empirical_variant(None, variant, "T")
    report = assert_empirical_expected_diff(config, variant, "T")
    assert report["actual_diff_fields"] == ["vadsp_gate_mode"]
    assert not report["frozen_protocol_changes"]
    assert config.vadsp_gate_mode == field_value


def test_t3_really_shares_temporal_parameters_and_preserves_shapes() -> None:
    config, model = _small_model("T3")
    block = model.coupling_blocks[0]
    fine_transform = block.fine_micro_graph_temporal_encoder.block.causal_tcn
    coarse_transform = block.coarse_macro_graph_temporal_encoder.block.causal_tcn
    assert fine_transform is coarse_transform
    assert {id(p) for p in fine_transform.parameters()} == {id(p) for p in coarse_transform.parameters()}
    assert block.shared_temporal_output_projections is not None
    output = model(torch.randn(2, 12, 4, 16))
    assert output["pred"].shape == (2, 10, 4)
    assert output["aux"]["h_fine_shape"] == [2, 12, 4, config.hidden_dim]
    assert output["aux"]["h_coarse_shape"] == [2, 12, 4, config.hidden_dim]


def _assert_causal_resampling(x: torch.Tensor, stride: int) -> None:
    restored, trace = causal_downsample_upsample(x, stride)
    upper = trace["access_upper_bound"]
    cutoff = trace["causal_cutoff"]
    assert torch.all(upper <= cutoff)
    for position in range(x.shape[1]):
        changed = x.clone()
        changed[:, position + 1 :, :, :] += 1000.0
        changed_restored, _ = causal_downsample_upsample(changed, stride)
        assert torch.equal(restored[:, position], changed_restored[:, position])


def test_t4_is_causal_on_synthetic_and_real_small_batch() -> None:
    _assert_causal_resampling(torch.randn(2, 23, 3, 5), stride=6)
    parquet = PROJECT_ROOT / "dataset/sdwpf_model_input_base.parquet"
    pyarrow = pytest.importorskip("pyarrow.parquet")
    table = pyarrow.ParquetFile(parquet).read_row_group(
        0,
        columns=["Tmstamp", "TurbID", "Wspd", "Patv_clean_for_input"],
    )
    frame = table.to_pandas().sort_values(["Tmstamp", "TurbID"])
    turbines = sorted(frame["TurbID"].unique())[:2]
    frame = frame[frame["TurbID"].isin(turbines)]
    pivots = []
    for column in ("Wspd", "Patv_clean_for_input"):
        pivot = frame.pivot(index="Tmstamp", columns="TurbID", values=column).sort_index()
        pivots.append(pivot.iloc[:144].to_numpy(dtype=np.float32))
    real_x = torch.from_numpy(np.stack(pivots, axis=-1)[None]).nan_to_num()
    assert real_x.shape[1] == 144
    _assert_causal_resampling(real_x, stride=18)


def test_t4_never_uses_future_target_or_future_index() -> None:
    config, model = _small_model("T4")
    x = torch.randn(1, 17, 4, 16)
    future_y = torch.randn(1, 10, 4)
    before = model(x)["pred"].detach()
    future_y.add_(10000.0)
    after = model(x)["pred"].detach()
    assert torch.equal(before, after)
    trace = model(x)["aux"]["coarse_alignment_trace"]
    assert torch.all(trace["access_upper_bound"] <= trace["causal_cutoff"])


def test_t5_bypasses_macro_prompt_and_cross_fusion_and_calls_concat_mlp() -> None:
    _, model = _small_model("T5")
    block = model.coupling_blocks[0]
    assert block.macro_prompt_encoder is None
    assert block.symmetric_cross_fusion is None
    assert block.direct_concat_mlp is not None
    calls = []
    handle = block.direct_concat_mlp.register_forward_hook(lambda *_: calls.append(True))
    try:
        output = model(torch.randn(2, 12, 4, 16))
    finally:
        handle.remove()
    assert calls == [True]
    assert output["pred"].shape == (2, 10, 4)
    assert output["aux"]["direct_concat_mlp_called"] is True
    assert output["aux"]["uses_macro_prompt"] is False
    assert output["aux"]["uses_cross_fusion"] is False


def test_t5_concat_mlp_is_capacity_matched_to_t0() -> None:
    base = apply_empirical_variant(None, "T0", "T")
    controlled = apply_empirical_variant(None, "T5", "T")
    base.use_train_robust_volatility = False
    controlled.use_train_robust_volatility = False
    base_model = build_model(base, input_dim=16, graph_data=_graph(134))
    controlled_model = build_model(controlled, input_dim=16, graph_data=_graph(134))
    base_count = sum(parameter.numel() for parameter in base_model.parameters())
    controlled_count = sum(parameter.numel() for parameter in controlled_model.parameters())
    assert abs(controlled_count - base_count) / base_count < 0.01


@pytest.mark.parametrize("variant", [f"T{i}" for i in range(6)])
def test_t0_t5_finite_shape_and_all_invalid_mask_no_nan(variant: str) -> None:
    config, model = _small_model(variant)
    output = model(torch.randn(2, 12, 4, 16))["pred"]
    assert output.shape == (2, 10, 4)
    assert torch.isfinite(output).all()
    target = torch.randn_like(output)
    invalid = torch.zeros_like(output)
    loss_fn = get_loss_fn(config.loss_function, config=config, num_nodes=4)
    loss = loss_fn(output, target, invalid)
    assert loss is None or torch.isfinite(loss).all()


def test_single_branch_modes_disable_the_inactive_contribution() -> None:
    _, fine_model = _small_model("T1")
    _, coarse_model = _small_model("T2")
    x = torch.randn(2, 12, 4, 16)
    fine_aux = fine_model(x)["aux"]
    coarse_aux = coarse_model(x)["aux"]
    assert fine_aux["coarse_branch_active"] is False
    assert fine_aux["coarse_representation_norm"] == 0.0
    assert coarse_aux["fine_branch_active"] is False
    assert coarse_aux["fine_representation_norm"] == 0.0


def test_all_t_variants_preserve_frozen_protocol() -> None:
    canonical = canonical_config().to_dict()
    for variant in [f"T{i}" for i in range(6)]:
        config = apply_empirical_variant(None, variant, "T")
        report = assert_empirical_expected_diff(config, variant, "T")
        assert report["passed"]
        for field in FROZEN_PROTOCOL_FIELDS:
            assert config.to_dict()[field] == canonical[field], (variant, field)
