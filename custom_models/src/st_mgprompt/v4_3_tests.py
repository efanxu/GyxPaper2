from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.coupling_block import STMGPromptCouplingBlock
from st_mgprompt.decoder import STPromptFullHistoryDecoder
from st_mgprompt.evaluate import apply_physical_clip
from st_mgprompt.graph_layers import PriorConstrainedDiffusionGraphConv, row_normalize
from st_mgprompt.graph_prior import build_macro_trend_graph, build_micro_local_graph
from st_mgprompt.losses import MSMGDWULoss
from st_mgprompt.metrics import masked_official_align_score_kw
from st_mgprompt.registry import build_model
from st_mgprompt.run_precision_ablation import read_best_val_summary
from st_mgprompt.run_st_mgprompt import load_checkpoint_model
from st_mgprompt.volatility_patching import (
    VolatilityAwareDynamicSemanticPatching,
    compute_vadsp_robust_statistics_from_timeline,
)


def _full_method_config(num_nodes: int = 134) -> STMGPromptConfig:
    cfg = STMGPromptConfig(
        model_name="STMGPrompt_Full_MSMGDWU_DiffusionHistory",
        num_nodes=num_nodes,
        use_vadsp=True,
        use_trend_prior_graph=True,
        use_adaptive_graph=True,
        use_graph_temporal_encoder=True,
        use_stmg_coupling_block=True,
        use_macro_prompt=True,
        use_st_prompt=True,
        use_msmg_dwu=True,
        loss_function="msmg_dwu_loss",
        loss_protocol="method_full",
        graph_operator="bidirectional_diffusion",
        decoder_context_mode="full_history_cross_attention",
        decoder_history_len=None,
        granularity_weight_mode="difficulty_rate",
        dropout=0.0,
    )
    cfg.validate()
    return cfg


def _synthetic_graph_data(num_nodes: int) -> dict:
    rng = np.random.default_rng(2026)
    macro = rng.random((num_nodes, num_nodes), dtype=np.float32)
    micro = rng.random((num_nodes, num_nodes), dtype=np.float32)
    np.fill_diagonal(macro, 1.0)
    np.fill_diagonal(micro, 1.0)
    return {
        "A_macro_trend": row_normalize(torch.as_tensor(macro)).numpy(),
        "A_micro_local": row_normalize(torch.as_tensor(micro.T.copy())).numpy(),
        "metadata": {
            "source": "synthetic_unit_test",
            "train_only": True,
        },
    }


class MetricsV43Tests(unittest.TestCase):
    def test_official_score_100kw_all_valid(self) -> None:
        y = np.zeros((2, 3, 134), dtype=np.float32)
        pred = np.full_like(y, 100.0)
        mask = np.ones_like(y, dtype=bool)
        self.assertAlmostEqual(masked_official_align_score_kw(y, pred, mask, num_nodes=134), 13.4, places=6)

    def test_official_score_partial_nodes_scales_to_num_nodes(self) -> None:
        y = np.zeros((1, 3, 134), dtype=np.float32)
        pred = np.full_like(y, 100.0)
        mask = np.zeros_like(y, dtype=bool)
        mask[:, :, :10] = True
        self.assertAlmostEqual(masked_official_align_score_kw(y, pred, mask, num_nodes=134), 13.4, places=6)

    def test_official_score_excludes_all_invalid_sample(self) -> None:
        y = np.zeros((2, 3, 134), dtype=np.float32)
        pred = np.full_like(y, 100.0)
        mask = np.ones_like(y, dtype=bool)
        mask[0] = False
        self.assertAlmostEqual(masked_official_align_score_kw(y, pred, mask, num_nodes=134), 13.4, places=6)


class EvaluationV43Tests(unittest.TestCase):
    def test_physical_clipping(self) -> None:
        cfg = STMGPromptConfig()
        pred = np.asarray([[[-10.0, 500.0, 1600.0]]], dtype=np.float32)
        clipped = apply_physical_clip(pred, cfg)
        np.testing.assert_allclose(clipped, np.asarray([[[0.0, 500.0, 1500.0]]], dtype=np.float32))


class DifficultyRateTests(unittest.TestCase):
    def test_difficulty_rate_weights_harder_horizon_more(self) -> None:
        loss = MSMGDWULoss(
            eval_horizons=[3, 6, 10],
            num_nodes=2,
            granularity_weight_mode="difficulty_rate",
            base_loss="mae",
            ema_alpha=0.0,
        )
        pred = torch.zeros(1, 10, 2)
        target = torch.zeros_like(pred)
        target[:, :3, :] = 1.0
        target[:, 3:6, :] = 2.0
        target[:, 6:10, :] = 5.0
        mask = torch.ones_like(pred)
        value = loss(pred, target, mask)
        self.assertIsNotNone(value)
        self.assertGreater(loss.last_details["granularity_weight_h10"], loss.last_details["granularity_weight_h3"])
        weights = [loss.last_details[f"granularity_weight_h{h}"] for h in [3, 6, 10]]
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)

    def test_all_invalid_batch_does_not_update_ema(self) -> None:
        loss = MSMGDWULoss(eval_horizons=[3, 6, 10], num_nodes=2, granularity_weight_mode="difficulty_rate")
        before = loss.ema_granularity_loss.clone()
        result = loss(torch.zeros(1, 10, 2), torch.zeros(1, 10, 2), torch.zeros(1, 10, 2))
        self.assertIsNone(result)
        self.assertTrue(torch.equal(before, loss.ema_granularity_loss))


class VADSPV43Tests(unittest.TestCase):
    def test_robust_statistics_fit_train_only_buffer(self) -> None:
        module = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=4,
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
        )
        raw = torch.randn(3, 8, 2, 2)
        module.set_robust_statistics_from_raw(raw)
        self.assertTrue(bool(module.robust_stats_fitted.item()))
        projected = torch.randn(3, 8, 2, 4)
        out = module(projected, raw)
        self.assertIn("coarse_scale_weights", out)
        self.assertTrue(torch.isfinite(out["dynamic_patch_gate"]).all())

    def test_unique_timeline_robust_statistics_shapes_and_finiteness(self) -> None:
        timeline = torch.randn(40, 5, 2)
        stats = compute_vadsp_robust_statistics_from_timeline(
            timeline,
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        self.assertEqual(tuple(stats["center"].shape), (5, 2))
        self.assertEqual(tuple(stats["scale"].shape), (5, 2))
        self.assertTrue(torch.isfinite(stats["center"]).all())
        self.assertTrue(torch.isfinite(stats["scale"]).all())
        self.assertTrue(torch.all(stats["scale"] > 0))
        self.assertFalse(stats["depends_on_window_stride"])

    def test_unique_timeline_statistics_independent_of_stride_and_batch_size(self) -> None:
        timeline = torch.randn(60, 4, 2)
        stats_by_stride = []
        for _stride in [1, 6, 12]:
            stats_by_stride.append(
                compute_vadsp_robust_statistics_from_timeline(
                    timeline,
                    feature_names=["Wspd", "Patv_clean_for_input"],
                    volatility_source_cols=["Wspd", "Patv_clean_for_input"],
                    volatility_mode="per_node",
                    robust_eps=1e-6,
                )
            )
        self.assertTrue(torch.allclose(stats_by_stride[0]["center"], stats_by_stride[1]["center"]))
        self.assertTrue(torch.allclose(stats_by_stride[0]["scale"], stats_by_stride[2]["scale"]))
        stats_batch_16 = compute_vadsp_robust_statistics_from_timeline(
            timeline,
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        stats_batch_32 = compute_vadsp_robust_statistics_from_timeline(
            timeline,
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        self.assertTrue(torch.allclose(stats_batch_16["center"], stats_batch_32["center"]))

    def test_unique_timeline_statistics_no_val_test_leakage(self) -> None:
        full = torch.randn(90, 3, 2)
        train = full[:50].clone()
        stats_before = compute_vadsp_robust_statistics_from_timeline(
            train,
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        full[50:] = 1e9
        stats_after = compute_vadsp_robust_statistics_from_timeline(
            full[:50],
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        self.assertTrue(torch.allclose(stats_before["center"], stats_after["center"]))
        self.assertTrue(torch.allclose(stats_before["scale"], stats_after["scale"]))

    def test_large_timeline_statistics_do_not_use_window_concat_shape(self) -> None:
        timeline = torch.randn(160, 134, 2)
        stats = compute_vadsp_robust_statistics_from_timeline(
            timeline,
            feature_names=["Wspd", "Patv_clean_for_input"],
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        self.assertEqual(stats["num_time_deltas"], 159)
        self.assertEqual(tuple(stats["center"].shape), (134, 2))

    def test_full_shape_vadsp_forward_backward_and_checkpoint_buffers(self) -> None:
        B, L, N, C, D = 32, 144, 134, 16, 8
        feature_names = list(STMGPromptConfig().feature_cols)
        module = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=D,
            feature_names=feature_names,
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            coarse_windows=[6, 18, 36],
        )
        train_timeline = torch.randn(180, N, C)
        stats = compute_vadsp_robust_statistics_from_timeline(
            train_timeline,
            feature_names=feature_names,
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        module.set_robust_statistics(stats)
        projected = torch.randn(B, L, N, D, requires_grad=True)
        raw = torch.randn(B, L, N, C)
        out = module(projected, raw)
        loss = out["x_fine"].mean() + out["x_coarse"].mean()
        loss.backward()
        self.assertIsNotNone(projected.grad)
        self.assertTrue(torch.isfinite(projected.grad).all())
        self.assertEqual(tuple(module.delta_abs_iqr.shape), (N, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vadsp.pt"
            torch.save(module.state_dict(), path)
            restored = VolatilityAwareDynamicSemanticPatching(
                hidden_dim=D,
                feature_names=feature_names,
                volatility_source_cols=["Wspd", "Patv_clean_for_input"],
                coarse_windows=[6, 18, 36],
            )
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
            self.assertTrue(torch.allclose(module.delta_abs_iqr.cpu(), restored.delta_abs_iqr.cpu()))
            self.assertTrue(torch.allclose(module.delta_log_median.cpu(), restored.delta_log_median.cpu()))

    def test_vadsp_forward_requires_fitted_train_robust_statistics(self) -> None:
        module = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=4,
            feature_names=list(STMGPromptConfig().feature_cols),
            volatility_source_cols=["Wspd", "Patv_clean_for_input"],
            use_train_robust_volatility=True,
        )
        with self.assertRaisesRegex(RuntimeError, "robust statistics are not fitted"):
            module(torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 16))

    def test_legacy_vadsp_checkpoint_loads_strict_and_marks_ineligible_metadata(self) -> None:
        feature_names = list(STMGPromptConfig().feature_cols)
        source_cols = ["Wspd", "Patv_clean_for_input"]
        module = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=4,
            feature_names=feature_names,
            volatility_source_cols=source_cols,
            volatility_mode="per_node",
        )
        legacy_state = module.state_dict()
        for key in [
            "delta_abs_q25",
            "delta_abs_median",
            "delta_abs_q75",
            "robust_center",
            "robust_scale",
            "robust_stats_fitted",
            "robust_stats_num_time_points",
            "robust_stats_num_deltas",
            "robust_stats_nonfinite_filtered",
            "robust_stats_degenerate_iqr_count",
        ]:
            legacy_state.pop(key)
        legacy_state["delta_abs_iqr"] = torch.tensor([0.25, 0.5])
        legacy_state["delta_log_median"] = torch.tensor([0.1, 0.2])
        legacy_state["delta_log_iqr"] = torch.tensor([0.3, 0.4])

        restored = VolatilityAwareDynamicSemanticPatching(
            hidden_dim=4,
            feature_names=feature_names,
            volatility_source_cols=source_cols,
            volatility_mode="per_node",
        )
        restored.load_state_dict(legacy_state, strict=True)
        self.assertTrue(bool(restored.robust_stats_fitted.item()))
        self.assertEqual(tuple(restored.delta_abs_iqr.shape), (1, 2))
        self.assertEqual(restored.robust_statistics_metadata["fit_source"], "legacy_checkpoint")
        self.assertFalse(bool(restored.robust_statistics_metadata["exact"]))
        self.assertTrue(bool(restored.robust_statistics_metadata["depends_on_window_stride"]))
        self.assertFalse(bool(restored.robust_statistics_metadata["eligible_for_fair_main_table"]))

        out = restored(torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 16))
        self.assertEqual(tuple(out["x_fine"].shape), (2, 8, 4, 4))

    def test_evaluate_only_checkpoint_restores_vadsp_metadata(self) -> None:
        cfg = STMGPromptConfig(
            model_name="STMGPrompt_TemporalOnly_VADSP",
            use_vadsp=True,
            num_nodes=4,
            hidden_dim=8,
            max_pred_len=10,
            dropout=0.0,
        )
        model = build_model(cfg, input_dim=len(cfg.feature_cols))
        stats = compute_vadsp_robust_statistics_from_timeline(
            torch.randn(30, cfg.num_nodes, len(cfg.feature_cols)),
            feature_names=list(cfg.feature_cols),
            volatility_source_cols=list(cfg.volatility_source_cols),
            volatility_mode=cfg.volatility_mode,
            robust_eps=cfg.volatility_scale_eps,
        )
        model.vadsp.set_robust_statistics(stats)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vadsp_robust_statistics_metadata": dict(model.vadsp.robust_statistics_metadata),
                },
                path,
            )
            data = SimpleNamespace(num_nodes=cfg.num_nodes, input_dim=len(cfg.feature_cols))
            loaded = load_checkpoint_model(cfg, data, path)
        self.assertTrue(bool(loaded.vadsp.robust_stats_fitted.item()))
        self.assertEqual(tuple(loaded.vadsp.robust_center.shape), (cfg.num_nodes, len(cfg.volatility_source_cols)))
        self.assertEqual(loaded.vadsp.robust_statistics_metadata["fit_source"], "unique_train_timeline")
        self.assertFalse(bool(loaded.vadsp.robust_statistics_metadata["depends_on_window_stride"]))


class GraphV43Tests(unittest.TestCase):
    def test_macro_micro_sources_produce_different_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loc = Path(tmp) / "loc.csv"
            loc.write_text("TurbID,x,y\n1,0,0\n2,1,0\n3,0,1\n4,1,1\n", encoding="utf-8")
            cfg = STMGPromptConfig(num_nodes=4, smoke=True)
            cfg.graph_missing_location_policy = "error"
            t = np.arange(30, dtype=np.float32)[:, None]
            nodes = np.arange(4, dtype=np.float32)[None, :]
            macro_series = np.sin(t / 5.0) + nodes
            micro_series = np.sin(t * (nodes + 1.0))
            mask = np.ones_like(macro_series, dtype=bool)
            macro = build_macro_trend_graph(macro_series, mask, loc, cfg)
            micro = build_micro_local_graph(micro_series, mask, loc, cfg)
            self.assertEqual(macro.shape, (4, 4))
            self.assertEqual(micro.shape, (4, 4))
            self.assertFalse(np.allclose(macro, micro))


class DiffusionGraphConvTests(unittest.TestCase):
    def test_shape_row_normalization_orders_identity_and_beta_zero(self) -> None:
        h = torch.randn(2, 5, 4, 8, requires_grad=True)
        A = torch.tensor(
            [
                [0.0, 2.0, 0.0, 1.0],
                [1.0, 0.0, 3.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [4.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        A_fwd = row_normalize(A)
        A_rev = row_normalize(A.T)
        self.assertTrue(torch.allclose(A_fwd.sum(dim=-1), torch.ones(4)))
        self.assertTrue(torch.allclose(A_rev.sum(dim=-1), torch.ones(4)))
        self.assertFalse(torch.allclose(
            torch.einsum("ij,btjd->btid", A_fwd, h.detach()),
            torch.einsum("ij,btjd->btid", A_rev, h.detach()),
        ))
        for order in [1, 2]:
            module = PriorConstrainedDiffusionGraphConv(8, diffusion_order=order, dropout=0.0)
            out, aux = module(h, A, return_aux=True)
            self.assertEqual(tuple(out.shape), tuple(h.shape))
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(int(aux["diffusion_order"]), order)
            out.sum().backward(retain_graph=True)
            self.assertIsNotNone(module.projection[0].weight.grad)
        identity = torch.eye(4)
        module = PriorConstrainedDiffusionGraphConv(8, diffusion_order=2, dropout=0.0)
        out = module(h.detach(), identity)
        self.assertEqual(tuple(out.shape), tuple(h.shape))
        module.beta_graph.data.zero_()
        out_zero = module(h.detach(), A)
        self.assertTrue(torch.allclose(out_zero, h.detach(), atol=1e-7))

    def test_macro_micro_diffusion_parameters_not_shared(self) -> None:
        cfg = STMGPromptConfig(
            hidden_dim=8,
            graph_operator="bidirectional_diffusion",
            diffusion_order_micro=2,
            diffusion_order_macro=2,
            use_graph_in_temporal_encoder=True,
        )
        block = STMGPromptCouplingBlock(cfg, hidden_dim=8)
        micro = block.fine_micro_graph_temporal_encoder.block.graph_conv
        macro = block.coarse_macro_graph_temporal_encoder.block.graph_conv
        self.assertIsInstance(micro, PriorConstrainedDiffusionGraphConv)
        self.assertIsInstance(macro, PriorConstrainedDiffusionGraphConv)
        self.assertIsNot(micro.projection[0].weight, macro.projection[0].weight)
        self.assertIsNot(micro.beta_graph, macro.beta_graph)


class FullHistoryDecoderTests(unittest.TestCase):
    def test_full_history_decoder_shape_attention_and_gradients(self) -> None:
        B, L, H, N, D, P = 2, 7, 10, 3, 8, 4
        z_fine = torch.randn(B, L, N, D, requires_grad=True)
        z_coarse = torch.randn(B, L, N, D, requires_grad=True)
        st_prompt = torch.randn(1, H, N, D)
        macro_prompt = torch.randn(B, N, P, D)
        decoder = STPromptFullHistoryDecoder(D, num_heads=2, dropout=0.0, history_len=None)
        pred, aux = decoder(z_fine, z_coarse, st_prompt, macro_prompt, return_aux=True)
        self.assertEqual(tuple(pred.shape), (B, H, N))
        self.assertEqual(int(aux["decoder_actual_history_len"]), L)
        self.assertTrue(bool(aux["decoder_used_complete_history"]))
        self.assertFalse(aux["teacher_forcing"])
        self.assertFalse(aux["autoregressive"])
        self.assertFalse(aux["future_observed_features_used"])
        self.assertTrue(torch.allclose(aux["fine_attention_weight_sum_mean"], torch.tensor(1.0), atol=1e-6))
        self.assertTrue(torch.allclose(aux["coarse_attention_weight_sum_mean"], torch.tensor(1.0), atol=1e-6))
        self.assertTrue(torch.allclose(aux["macro_attention_weight_sum_mean"], torch.tensor(1.0), atol=1e-6))
        self.assertEqual(tuple(pred[:, :3].shape), (B, 3, N))
        pred.sum().backward()
        self.assertIsNotNone(decoder.fine_history_attention.in_proj_weight.grad)
        self.assertIsNotNone(z_fine.grad)
        self.assertIsNot(decoder.fine_history_attention.in_proj_weight, decoder.coarse_history_attention.in_proj_weight)
        self.assertIsNot(decoder.coarse_history_attention.in_proj_weight, decoder.macro_prompt_attention.in_proj_weight)

    def test_decoder_history_len_ablation_uses_recent_configured_length(self) -> None:
        decoder = STPromptFullHistoryDecoder(8, num_heads=2, dropout=0.0, history_len=3)
        pred, aux = decoder(
            torch.randn(1, 6, 2, 8),
            torch.randn(1, 6, 2, 8),
            torch.randn(1, 4, 2, 8),
            torch.randn(1, 2, 3, 8),
            return_aux=True,
        )
        self.assertEqual(tuple(pred.shape), (1, 4, 2))
        self.assertEqual(int(aux["decoder_actual_history_len"]), 3)
        self.assertFalse(bool(aux["decoder_used_complete_history"]))


class FullMethodModelShapeTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("ST_MGPROMPT_RUN_FULL_SHAPE_TEST") == "1",
        "Set ST_MGPROMPT_RUN_FULL_SHAPE_TEST=1 to run the formal full-shape model stress test.",
    )
    def test_full_model_full_shape_forward_msmg_dwu_backward(self) -> None:
        torch.manual_seed(2026)
        B, L, N, C, H = 32, 144, 134, 16, 10
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = _full_method_config(num_nodes=N)
        self.assertEqual(cfg.hidden_dim, 64)
        self.assertEqual(cfg.lookback, L)
        self.assertEqual(cfg.max_pred_len, H)
        graph_data = _synthetic_graph_data(N)
        model = build_model(cfg, input_dim=C, graph_data=graph_data).to(device)
        stats = compute_vadsp_robust_statistics_from_timeline(
            torch.randn(160, N, C),
            feature_names=list(cfg.feature_cols),
            volatility_source_cols=list(cfg.volatility_source_cols),
            volatility_mode=cfg.volatility_mode,
            robust_eps=cfg.volatility_scale_eps,
        )
        model.vadsp.set_robust_statistics(stats)
        loss_fn = MSMGDWULoss(
            eval_horizons=list(cfg.eval_horizons),
            num_nodes=N,
            base_loss=cfg.msmg_base_loss,
            lambda_site=cfg.msmg_lambda_site,
            ema_alpha=cfg.msmg_ema_alpha,
            node_weight_clip=cfg.msmg_node_weight_clip,
            granularity_weight_mode="difficulty_rate",
            difficulty_gamma=cfg.difficulty_gamma,
            difficulty_rate_gamma=cfg.difficulty_rate_gamma,
            difficulty_temperature=cfg.difficulty_temperature,
            granularity_weight_clip=cfg.granularity_weight_clip,
        ).to(device)

        x = torch.randn(B, L, N, C, device=device)
        y = torch.randn(B, H, N, device=device)
        mask = torch.ones(B, H, N, device=device)
        out = model(x)
        self.assertEqual(tuple(out["pred"].shape), (B, H, N))
        self.assertEqual(out["aux"]["graph_operator"], "bidirectional_diffusion")
        self.assertEqual(out["aux"]["decoder_context_mode"], "full_history_cross_attention")
        self.assertTrue(bool(out["aux"]["decoder_used_complete_history"]))

        loss = loss_fn(out["pred"], y, mask)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()

        def _has_finite_grad(module: torch.nn.Module) -> bool:
            grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
            return bool(grads) and all(torch.isfinite(g).all().item() for g in grads)

        self.assertTrue(_has_finite_grad(model.vadsp))
        self.assertTrue(_has_finite_grad(model.coupling_blocks))
        self.assertTrue(_has_finite_grad(model.direct_decoder))
        first_block = model.coupling_blocks[0]
        self.assertTrue(_has_finite_grad(first_block.fine_micro_graph_temporal_encoder.block.graph_conv))
        self.assertTrue(_has_finite_grad(first_block.coarse_macro_graph_temporal_encoder.block.graph_conv))
        self.assertEqual(loss_fn.last_details["skipped_all_invalid_batch"], 0.0)


class BestEpochSelectionTests(unittest.TestCase):
    def test_precision_summary_prefers_checkpoint_best_epoch_over_first_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_dir = root / "P0" / "STMGPrompt_FairFull"
            variant_dir.mkdir(parents=True)
            with (variant_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["epoch", "is_best_epoch", "val_Score_H3", "val_Score_H6", "val_Score_H10", "val_official_score_h10"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "is_best_epoch": "true",
                        "val_Score_H3": 3.0,
                        "val_Score_H6": 6.0,
                        "val_Score_H10": 12.0,
                        "val_official_score_h10": 12.0,
                    }
                )
                writer.writerow(
                    {
                        "epoch": 3,
                        "is_best_epoch": "true",
                        "val_Score_H3": 2.0,
                        "val_Score_H6": 4.0,
                        "val_Score_H10": 10.0,
                        "val_official_score_h10": 10.0,
                    }
                )
            (variant_dir / "model_summary.json").write_text(json.dumps({"trainable_parameters": 123}), encoding="utf-8")
            torch.save(
                {
                    "best_epoch": 3,
                    "best_val_score_h3": 2.0,
                    "best_val_score_h6": 4.0,
                    "best_val_score_h10": 10.0,
                },
                variant_dir / "best_checkpoint.pt",
            )
            rows = read_best_val_summary(
                root,
                [
                    {
                        "variant": "P0",
                        "model_name": "STMGPrompt_FairFull",
                        "graph_operator": "simple",
                        "decoder_context_mode": "last_state",
                        "hidden_dim": 64,
                        "num_coupling_layers": 1,
                    }
                ],
            )
        self.assertEqual(rows[0]["best_epoch"], 3)
        self.assertEqual(float(rows[0]["best_val_Score_H10"]), 10.0)


if __name__ == "__main__":
    unittest.main()
