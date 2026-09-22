from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from st_mgprompt.a8_batch4_contract import TRAINING_PROFILE_ID, variant_contract
from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.empirical_protocol import (
    EMPIRICAL_FAMILIES,
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    assert_empirical_expected_diff,
)
from st_mgprompt.losses import LOSS_STATE_SCHEMA_VERSION, MSMGDWULoss, get_loss_fn, masked_smooth_l1
from st_mgprompt.graph_prior import _graph_paths
from st_mgprompt.train import _restore_rng_state, _rng_state_payload, _run_epoch


def _batch(batch: int = 2, horizons: int = 10, nodes: int = 4):
    generator = torch.Generator().manual_seed(2026)
    pred = torch.randn(batch, horizons, nodes, generator=generator, requires_grad=True)
    target = torch.randn(batch, horizons, nodes, generator=generator)
    mask = torch.ones_like(target)
    return pred, target, mask


def _loss(mode: str, site: str, nodes: int = 4) -> MSMGDWULoss:
    return MSMGDWULoss(
        eval_horizons=[3, 6, 10],
        num_nodes=nodes,
        base_loss="smooth_l1",
        granularity_weight_mode=mode,
        site_weight_mode=site,
        static_granularity_weights=(1.0, 2.0, 3.0),
    )


def test_l0_weights_are_one_and_loss_equals_masked_smooth_l1() -> None:
    pred, target, mask = _batch()
    loss_fn = _loss("static", "static")
    total = loss_fn(pred, target, mask)
    expected = masked_smooth_l1(pred, target, mask)
    assert total is not None and expected is not None
    assert torch.allclose(total, expected)
    assert [loss_fn.last_details[f"granularity_weight_h{h}"] for h in (3, 6, 10)] == [1.0, 1.0, 1.0]
    assert torch.equal(loss_fn.node_weight, torch.ones_like(loss_fn.node_weight))


def test_l1_node_weights_stay_one_and_l2_horizon_weights_stay_one() -> None:
    pred, target, mask = _batch()
    l1 = _loss("difficulty_rate", "static")
    assert l1(pred, target, mask) is not None
    assert torch.equal(l1.node_weight, torch.ones_like(l1.node_weight))

    l2 = _loss("static", "dynamic")
    assert l2(pred, target, mask) is not None
    assert [l2.last_details[f"granularity_weight_h{h}"] for h in (3, 6, 10)] == [1.0, 1.0, 1.0]


def test_l3_matches_canonical_loss_modes_and_state_updates() -> None:
    config = apply_empirical_variant(None, "L3", "L")
    assert config.granularity_weight_mode == "difficulty_rate"
    assert config.site_weight_mode == "dynamic"
    pred, target, mask = _batch(nodes=config.num_nodes)
    from_config = get_loss_fn(config.loss_function, config=config, num_nodes=config.num_nodes)
    direct = MSMGDWULoss(
        eval_horizons=list(config.eval_horizons),
        num_nodes=config.num_nodes,
        base_loss=config.msmg_base_loss,
        lambda_site=config.msmg_lambda_site,
        ema_alpha=config.msmg_ema_alpha,
        node_weight_clip=tuple(config.msmg_node_weight_clip),
        granularity_weight_mode="difficulty_rate",
        site_weight_mode="dynamic",
        difficulty_gamma=config.difficulty_gamma,
        difficulty_rate_gamma=config.difficulty_rate_gamma,
        difficulty_temperature=config.difficulty_temperature,
        granularity_weight_clip=tuple(config.granularity_weight_clip),
    )
    left = from_config(pred, target, mask)
    right = direct(pred.detach().clone().requires_grad_(True), target, mask)
    assert left is not None and right is not None
    assert torch.allclose(left, right)
    for key in ("ema_granularity_loss", "initial_granularity_loss", "ema_node_loss", "node_weight"):
        assert torch.equal(from_config.state_dict()[key], direct.state_dict()[key])
    left.backward()
    assert from_config.log_sigma_g.grad is None


def test_l4_static_increasing_weights_are_frozen() -> None:
    pred, target, mask = _batch()
    loss_fn = _loss("static_increasing", "static")
    before = loss_fn.static_granularity_weight.clone()
    for _ in range(3):
        assert loss_fn(pred, target, mask) is not None
    assert torch.equal(before, loss_fn.static_granularity_weight)
    assert torch.allclose(before, torch.tensor([0.5, 1.0, 1.5]))
    assert loss_fn.static_granularity_weight_source == "preset_arithmetic_progression"


def test_l5_log_sigma_is_learnable_and_restores_with_optimizer() -> None:
    pred, target, mask = _batch()
    loss_fn = _loss("uncertainty_precision", "static")
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.05)
    total = loss_fn(pred, target, mask)
    assert total is not None
    total.backward()
    assert loss_fn.log_sigma_g.grad is not None
    optimizer.step()
    saved_loss = deepcopy(loss_fn.state_dict())
    saved_optimizer = deepcopy(optimizer.state_dict())

    restored = _loss("uncertainty_precision", "static")
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.05)
    restored.load_state_dict(saved_loss)
    restored_optimizer.load_state_dict(saved_optimizer)
    assert torch.equal(restored.log_sigma_g, loss_fn.log_sigma_g)
    assert restored_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    assert torch.equal(restored.node_weight, torch.ones_like(restored.node_weight))


def test_l6_dwa_updates_only_at_epoch_end() -> None:
    pred, target, mask = _batch()
    loss_fn = _loss("dynamic_weight_average", "static")
    initial = loss_fn.dwa_weight.clone()
    loss_fn.begin_training_epoch()
    assert loss_fn(pred, target, mask) is not None
    assert torch.equal(loss_fn.dwa_weight, initial)
    first = loss_fn.end_training_epoch()
    assert first["dwa_update_count"] == 1.0
    assert torch.equal(loss_fn.dwa_weight, initial)

    loss_fn.begin_training_epoch()
    changed_target = target.clone()
    changed_target[:, 6:, :] += 4.0
    assert loss_fn(pred, changed_target, mask) is not None
    second = loss_fn.end_training_epoch()
    assert second["dwa_update_count"] == 2.0
    assert not torch.equal(loss_fn.dwa_weight, initial)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict:
        pred = x[..., 0] + self.bias
        return {"pred": pred, "aux": {"pred_bnh": pred.permute(0, 2, 1)}}


def test_all_invalid_batch_skips_optimizer_step_without_nan() -> None:
    model = _TinyModel()
    loss_fn = _loss("static", "static", nodes=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.zeros(2, 10, 4, 1)
    y = torch.zeros(2, 10, 4)
    loader = [{"x": x, "y": y, "valid_target_mask": torch.zeros_like(y)}]
    before = model.bias.detach().clone()
    value, skipped, details = _run_epoch(model, loader, torch.device("cpu"), loss_fn, optimizer)
    assert value == 0.0
    assert skipped == 1
    assert details["total_loss"] == 0.0
    assert torch.equal(before, model.bias.detach())


def test_loss_optimizer_and_rng_resume_matches_uninterrupted_path() -> None:
    torch.manual_seed(77)
    base = _loss("uncertainty_precision", "static")
    optimizer = torch.optim.Adam(base.parameters(), lr=0.01)
    pred, target, mask = _batch()
    first = base(pred, target, mask)
    assert first is not None
    first.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = {
        "loss": deepcopy(base.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        **_rng_state_payload(),
    }

    uninterrupted_random = torch.rand(3)
    second = base(pred, target, mask)
    assert second is not None
    second.backward()
    optimizer.step()
    uninterrupted_state = deepcopy(base.state_dict())

    resumed = _loss("uncertainty_precision", "static")
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
    resumed.load_state_dict(checkpoint["loss"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng_state(checkpoint)
    assert torch.equal(torch.rand(3), uninterrupted_random)
    resumed_second = resumed(pred, target, mask)
    assert resumed_second is not None
    resumed_second.backward()
    resumed_optimizer.step()
    for key, value in uninterrupted_state.items():
        assert torch.equal(value, resumed.state_dict()[key])


@pytest.mark.parametrize("mode", ["static", "static_increasing", "difficulty_rate", "uncertainty_precision", "dynamic_weight_average"])
def test_cpu_autocast_loss_is_finite_and_keeps_bhn_shape(mode: str) -> None:
    pred, target, mask = _batch()
    loss_fn = _loss(mode, "static")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        total = loss_fn(pred, target, mask)
    assert pred.shape == (2, 10, 4)
    assert total is not None and torch.isfinite(total)


def test_l_family_unique_differences_and_a8_boundary() -> None:
    assert set(EMPIRICAL_FAMILIES["L"]) == {f"L{index}" for index in range(8)}
    for variant_id in EMPIRICAL_FAMILIES["L"]:
        config = apply_empirical_variant(None, variant_id, "L")
        report = assert_empirical_expected_diff(config, variant_id, "L")
        assert report["passed"] is True
        assert not set(report["actual_diff_fields"]).intersection(FROZEN_PROTOCOL_FIELDS)
    assert apply_empirical_variant(None, "L4", "L").site_weight_mode == "static"
    assert apply_empirical_variant(None, "L6", "L").site_weight_mode == "static"
    contract = variant_contract()
    assert contract["training_profile_id"] == TRAINING_PROFILE_ID
    assert contract["batch_size"] == 4
    assert EMPIRICAL_FAMILIES["L"]["L7"].reference_only is True
    assert LOSS_STATE_SCHEMA_VERSION == "msmg_dwu_loss_state_v2"


def test_l_family_graph_artifacts_are_isolated_from_formal_graphs(tmp_path) -> None:
    config = apply_empirical_variant(None, "L0", "L")
    config.output_root = str(tmp_path)
    config.run_id = "l0_smoke"
    graph_dir, _, _ = _graph_paths(config)
    assert graph_dir == tmp_path / "l0_smoke" / "_graph_artifacts" / config.graph_tag
    assert "custom_models/graphs" not in graph_dir.as_posix()


def test_none_variant_uses_explicit_graph_output_root(tmp_path) -> None:
    config = STMGPromptConfig(graph_output_root=str(tmp_path), graph_tag="unit_graph")
    graph_dir, _, _ = _graph_paths(config)
    assert graph_dir == tmp_path / "unit_graph"
