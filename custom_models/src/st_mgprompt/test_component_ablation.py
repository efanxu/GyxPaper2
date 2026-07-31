from __future__ import annotations

import unittest

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
            "A4": ("use_macro_prompt", False),
            "A5": ("disable_reverse_cross", True),
            "A6": ("use_cross_fusion", False),
            "A7": ("use_st_prompt", False),
            "A8": ("loss_function", "masked_score_aligned_hybrid"),
        }
        for name, (field, value) in expected.items():
            cfg = apply_component_ablation(STMGPromptConfig(), name)
            self.assertEqual(getattr(cfg, field), value, name)
            self.assertEqual(cfg.vadsp_gate_mode, "fixed_dual", name)
        self.assertTrue(apply_component_ablation(STMGPromptConfig(), "A4").use_graph_temporal_encoder)
        self.assertTrue(apply_component_ablation(STMGPromptConfig(), "A5").use_cross_fusion)
        self.assertTrue(apply_component_ablation(STMGPromptConfig(), "A6").use_macro_prompt)
        self.assertEqual(apply_component_ablation(STMGPromptConfig(), "A8").loss_protocol, "fair_main")

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


if __name__ == "__main__":
    unittest.main()
