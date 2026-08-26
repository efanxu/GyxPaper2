from __future__ import annotations

import unittest

import torch

from st_mgprompt.artifact_status import _config_matches
from st_mgprompt.config import STMGPromptConfig, apply_component_ablation
from st_mgprompt.experiment_protocol import (
    COMPONENT_ABLATION_VARIANTS,
    PRECISION_VARIANTS,
    apply_variant,
    assert_expected_diff,
    canonical_config,
    get_variant,
    matrix_row,
)
from st_mgprompt.prompt_alignment import STPromptEmbedding
from st_mgprompt.registry import build_model


class FixedDualProtocolTests(unittest.TestCase):
    def test_exact_variant_sets(self) -> None:
        self.assertEqual(tuple(PRECISION_VARIANTS), ("P0", "P1", "P2", "P3", "P4", "P5"))
        self.assertEqual(
            tuple(COMPONENT_ABLATION_VARIANTS),
            ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"),
        )

    def test_canonical_full(self) -> None:
        cfg = canonical_config()
        self.assertEqual(cfg.vadsp_gate_mode, "fixed_dual")
        self.assertEqual(cfg.graph_operator, "bidirectional_diffusion")
        self.assertEqual(cfg.decoder_context_mode, "last_state")
        self.assertEqual(cfg.hidden_dim, 64)
        self.assertEqual(cfg.num_coupling_layers, 1)
        self.assertEqual(cfg.granularity_weight_mode, "difficulty_rate")
        self.assertEqual(cfg.site_weight_mode, "dynamic")
        self.assertEqual(cfg.loss_function, "msmg_dwu_loss")
        self.assertEqual(cfg.macro_prompt_len, 4)
        self.assertEqual(cfg.cross_fusion_recent_len, 24)
        self.assertEqual(cfg.fusion_mode, "cross")
        self.assertEqual(cfg.st_prompt_mode, "full")
        self.assertTrue(cfg.use_macro_prompt)
        self.assertFalse(cfg.disable_reverse_cross)
        self.assertTrue(cfg.use_cross_fusion)
        self.assertTrue(cfg.use_st_prompt)
        self.assertEqual(cfg.decoder_input_strategy, "direct_multi_output_prompt_query")
        cfg.validate()

    def test_reference_variants_are_not_trainable(self) -> None:
        for name in ("P0", "A0"):
            variant = get_variant(name)
            self.assertFalse(variant.trainable)
            self.assertIsNotNone(variant.canonical_reference)
            self.assertEqual(assert_expected_diff(apply_variant(STMGPromptConfig(), name), name)["effective_diff_count"], 0)

    def test_every_variant_has_only_expected_diff(self) -> None:
        for mapping in (PRECISION_VARIANTS, COMPONENT_ABLATION_VARIANTS):
            for name, variant in mapping.items():
                cfg = apply_variant(STMGPromptConfig(), name, variant.experiment_family)
                report = assert_expected_diff(cfg, name, variant.experiment_family)
                self.assertTrue(report["passed"], name)

    def test_component_meanings(self) -> None:
        expected = {
            "A1": ("use_graph_in_temporal_encoder", False),
            "A2": ("use_adaptive_graph", False),
            "A3": ("graph_operator", "simple"),
            "A4": ("macro_prompt_len", 1),
            "A5": ("cross_fusion_recent_len", 6),
            "A6": ("fusion_mode", "add"),
            "A7": ("st_prompt_mode", "horizon_only"),
            "A8": ("loss_function", "masked_score_aligned_hybrid"),
        }
        for name, (field, value) in expected.items():
            cfg = apply_component_ablation(STMGPromptConfig(), name)
            self.assertEqual(getattr(cfg, field), value, name)
            self.assertEqual(cfg.vadsp_gate_mode, "fixed_dual", name)
        a4 = apply_component_ablation(STMGPromptConfig(), "A4")
        self.assertTrue(a4.use_macro_prompt)
        self.assertTrue(a4.use_cross_fusion)
        self.assertTrue(a4.use_st_prompt)
        a5 = apply_component_ablation(STMGPromptConfig(), "A5")
        self.assertFalse(a5.disable_reverse_cross)
        a6 = apply_component_ablation(STMGPromptConfig(), "A6")
        self.assertTrue(a6.use_cross_fusion)
        a7 = apply_component_ablation(STMGPromptConfig(), "A7")
        self.assertTrue(a7.use_st_prompt)
        self.assertEqual(a7.decoder_input_strategy, "direct_multi_output_prompt_query")
        self.assertEqual(a7.decoder_context_mode, "last_state")
        self.assertEqual(apply_component_ablation(STMGPromptConfig(), "A8").loss_protocol, "fair_main")

    def test_redesigned_variants_each_have_one_effective_diff(self) -> None:
        expected_fields = {
            "A4": ["macro_prompt_len"],
            "A5": ["cross_fusion_recent_len"],
            "A6": ["fusion_mode"],
            "A7": ["st_prompt_mode"],
        }
        for name, fields in expected_fields.items():
            cfg = apply_variant(STMGPromptConfig(), name, "component_ablation")
            report = assert_expected_diff(cfg, name, "component_ablation")
            self.assertEqual(report["effective_diff_count"], 1, name)
            self.assertEqual(report["actual_diff_fields"], fields, name)

    def test_artifact_audit_rejects_pre_redesign_a4_a7_configs(self) -> None:
        old_overrides = {
            "A4": {"use_macro_prompt": False},
            "A5": {"disable_reverse_cross": True},
            "A6": {"use_cross_fusion": False},
            "A7": {
                "use_st_prompt": False,
                "decoder_input_strategy": "direct_multi_output_horizon_head",
            },
        }
        for name, overrides in old_overrides.items():
            old = canonical_config().to_dict()
            old.update(overrides)
            expected = apply_variant(STMGPromptConfig(), name, "component_ablation").to_dict()
            matches, differences = _config_matches(old, expected)
            self.assertFalse(matches, name)
            self.assertTrue(
                set(differences) & {"macro_prompt_len", "cross_fusion_recent_len", "fusion_mode", "st_prompt_mode"},
                name,
            )

        legacy_a1 = apply_variant(STMGPromptConfig(), "A1", "component_ablation").to_dict()
        legacy_a1.pop("st_prompt_mode")
        self.assertTrue(_config_matches(legacy_a1, apply_variant(STMGPromptConfig(), "A1").to_dict())[0])

    def test_obsolete_component_ids_rejected(self) -> None:
        for name in ("A9", "A10"):
            with self.assertRaisesRegex(ValueError, "Obsolete component-ablation variant"):
                get_variant(name, "component_ablation")

    def test_matrix_exposes_actual_configuration(self) -> None:
        row = matrix_row(PRECISION_VARIANTS["P5"])
        self.assertEqual(row["graph_operator"], "bi_diffusion")
        self.assertEqual(row["decoder_context_mode"], "full_history_cross_attention")
        self.assertEqual(row["hidden_dim"], 96)
        self.assertEqual(row["num_coupling_layers"], 2)
        expected = {
            "A0": (4, 24, "cross", "full"),
            "A4": (1, 24, "cross", "full"),
            "A5": (4, 6, "cross", "full"),
            "A6": (4, 24, "add", "full"),
            "A7": (4, 24, "cross", "horizon_only"),
        }
        for name, values in expected.items():
            row = matrix_row(COMPONENT_ABLATION_VARIANTS[name])
            self.assertEqual(
                (row["macro_prompt_len"], row["cross_fusion_recent_len"], row["fusion_mode"], row["st_prompt_mode"]),
                values,
                name,
            )

    def test_st_prompt_modes(self) -> None:
        full = STPromptEmbedding(num_nodes=3, max_pred_len=2, hidden_dim=4, dropout=0.0, mode="full")
        full.eval()
        full_prompt = full(num_nodes=3, horizon=2, granularity_index=0)
        node = full.node_embedding(torch.arange(3)).view(1, 1, 3, 4)
        step = full.future_step_embedding(torch.arange(2)).view(1, 2, 1, 4)
        granularity = full.granularity_embedding(torch.tensor(0)).view(1, 1, 1, 4)
        self.assertEqual(tuple(full_prompt.shape), (1, 2, 3, 4))
        self.assertTrue(torch.allclose(full_prompt, full.norm(node + step + granularity)))
        self.assertFalse(torch.allclose(full_prompt[:, :, 0], full_prompt[:, :, 1]))

        horizon_only = STPromptEmbedding(
            num_nodes=3,
            max_pred_len=2,
            hidden_dim=4,
            dropout=0.0,
            mode="horizon_only",
        )
        with torch.no_grad():
            horizon_only.future_step_embedding.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
        prompt = horizon_only(num_nodes=3, horizon=2)
        self.assertEqual(tuple(prompt.shape), (1, 2, 3, 4))
        self.assertTrue(torch.allclose(prompt[:, :, 0], prompt[:, :, 1]))
        self.assertTrue(torch.allclose(prompt[:, :, 1], prompt[:, :, 2]))
        self.assertFalse(torch.allclose(prompt[:, 0], prompt[:, 1]))

    def test_st_prompt_mode_validation(self) -> None:
        cfg = canonical_config()
        cfg.st_prompt_mode = "invalid"
        with self.assertRaisesRegex(ValueError, "st_prompt_mode"):
            cfg.validate()
        with self.assertRaisesRegex(ValueError, "mode"):
            STPromptEmbedding(num_nodes=2, max_pred_len=2, hidden_dim=4, mode="invalid")

    def test_redesigned_variants_synthetic_forward(self) -> None:
        torch.manual_seed(2026)
        num_nodes = 3
        graph = torch.eye(num_nodes).numpy()
        graph_data = {
            "A_macro_trend": graph,
            "A_micro_local": graph,
            "metadata": {"source": "synthetic_component_ablation_test", "train_only": True},
        }
        x = torch.randn(1, 36, num_nodes, len(STMGPromptConfig().feature_cols))
        for name in ("A0", "A4", "A5", "A6", "A7"):
            cfg = apply_variant(STMGPromptConfig(), name, "component_ablation")
            cfg.num_nodes = num_nodes
            cfg.hidden_dim = 8
            cfg.dropout = 0.0
            cfg.use_train_robust_volatility = False
            cfg.diagnostics_level = "none"
            model = build_model(cfg, input_dim=x.shape[-1], graph_data=graph_data).eval()
            with torch.inference_mode():
                output = model(x)
            self.assertEqual(tuple(output["pred"].shape), (1, 10, num_nodes), name)
            self.assertEqual(output["aux"]["st_prompt_mode"], cfg.st_prompt_mode, name)
            self.assertEqual(
                output["aux"]["decoder_metadata"]["decoder_input_strategy"],
                "direct_multi_output_prompt_query",
                name,
            )


if __name__ == "__main__":
    unittest.main()
