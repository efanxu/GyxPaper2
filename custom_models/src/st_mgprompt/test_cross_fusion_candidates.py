from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.cross_fusion import SymmetricCrossFusion
from st_mgprompt.experiment_protocol import (
    COMPONENT_ABLATION_VARIANTS,
    CROSS_FUSION_CANDIDATE_VARIANTS,
    apply_variant,
    assert_expected_diff,
)
from st_mgprompt.formal_runner import run_family, selected_variants
from st_mgprompt.registry import build_model


class CrossFusionCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.h_fine = torch.randn(2, 144, 3, 8)
        self.h_coarse = torch.randn(2, 144, 3, 8)
        self.macro_prompt = torch.randn(2, 3, 4, 8)
        self.kwargs = {
            "hidden_dim": 8,
            "num_heads": 2,
            "dropout": 0.0,
            "recent_len": 24,
            "fusion_mode": "cross",
            "diagnostics_level": "standard",
        }

    def test_formal_default_ids_remain_a0_through_a7(self) -> None:
        self.assertEqual(tuple(COMPONENT_ABLATION_VARIANTS), tuple(f"A{i}" for i in range(8)))
        self.assertEqual(
            tuple(item.variant_id for item in selected_variants("component_ablation", None)),
            tuple(f"A{i}" for i in range(8)),
        )
        self.assertEqual(tuple(CROSS_FUSION_CANDIDATE_VARIANTS), ("A6-C1", "A6-C2", "A6-C3"))

    def test_candidate_effective_diffs_are_single_and_independent(self) -> None:
        expected = {
            "A6-C1": ["macro_to_fine_exclude_recent_len"],
            "A6-C2": ["macro_to_fine_mode"],
            "A6-C3": ["share_cross_attention_projections"],
        }
        for variant, fields in expected.items():
            cfg = apply_variant(STMGPromptConfig(), variant, "component_ablation")
            report = assert_expected_diff(cfg, variant, "component_ablation")
            self.assertEqual(report["actual_diff_fields"], fields, variant)
            self.assertEqual(report["effective_diff_count"], 1, variant)

        a5 = apply_variant(STMGPromptConfig(), "A5", "component_ablation")
        a6 = apply_variant(STMGPromptConfig(), "A6", "component_ablation")
        c1 = apply_variant(STMGPromptConfig(), "A6-C1", "component_ablation")
        self.assertEqual(a5.cross_fusion_recent_len, 6)
        self.assertEqual(a5.macro_to_fine_exclude_recent_len, 0)
        self.assertTrue(a6.disable_macro_to_fine_cross)
        self.assertFalse(c1.disable_macro_to_fine_cross)
        self.assertEqual(c1.cross_fusion_recent_len, 24)
        self.assertEqual(c1.macro_to_fine_exclude_recent_len, 24)

    def test_candidate_config_fields_are_validated(self) -> None:
        cfg = apply_variant(STMGPromptConfig(), "A6-C1", "component_ablation")
        cfg.macro_to_fine_exclude_recent_len = -1
        with self.assertRaisesRegex(ValueError, "macro_to_fine_exclude_recent_len"):
            cfg.validate()

        cfg = apply_variant(STMGPromptConfig(), "A6-C2", "component_ablation")
        cfg.macro_to_fine_mode = "invalid"
        with self.assertRaisesRegex(ValueError, "macro_to_fine_mode"):
            cfg.validate()

        cfg = apply_variant(STMGPromptConfig(), "A6-C3", "component_ablation")
        cfg.share_cross_attention_projections = 1
        with self.assertRaisesRegex(ValueError, "share_cross_attention_projections"):
            cfg.validate()

    def test_c1_excludes_recent_queries_and_preserves_both_directions(self) -> None:
        module = SymmetricCrossFusion(
            **self.kwargs,
            macro_to_fine_exclude_recent_len=24,
        ).eval()
        calls: dict[str, tuple[int, ...]] = {}

        def macro_hook(_module, inputs, _output):
            calls["macro_query"] = tuple(inputs[0].shape)

        def reverse_hook(_module, inputs, _output):
            calls["reverse_query"] = tuple(inputs[0].shape)
            calls["reverse_key"] = tuple(inputs[1].shape)

        handles = [
            module.macro_to_fine.register_forward_hook(macro_hook),
            module.fine_to_coarse.register_forward_hook(reverse_hook),
        ]
        with torch.inference_mode():
            fine, coarse, aux = module(self.h_fine, self.h_coarse, self.macro_prompt)
        for handle in handles:
            handle.remove()

        self.assertEqual(calls["macro_query"], (6, 120, 8))
        self.assertEqual(calls["reverse_query"], (6, 144, 8))
        self.assertEqual(calls["reverse_key"], (6, 24, 8))
        self.assertEqual(tuple(fine.shape), tuple(self.h_fine.shape))
        self.assertEqual(tuple(coarse.shape), tuple(self.h_coarse.shape))
        self.assertEqual(aux["macro_to_fine_query_count"], 120)
        self.assertEqual(aux["macro_to_fine_excluded_recent_count"], 24)
        self.assertFalse(aux["disable_macro_to_fine_cross"])
        self.assertFalse(aux["disable_reverse_cross"])

        expected_recent = module.fine_norm(self.h_fine[:, -24:])
        self.assertTrue(torch.allclose(fine[:, -24:], expected_recent))
        self.assertFalse(torch.allclose(fine[:, :-24], module.fine_norm(self.h_fine[:, :-24])))

    def test_c2_uses_node_specific_static_prompt_mean_without_attention_call(self) -> None:
        module = SymmetricCrossFusion(**self.kwargs, macro_to_fine_mode="static_mean").eval()
        macro_calls = 0

        def macro_hook(_module, _inputs, _output):
            nonlocal macro_calls
            macro_calls += 1

        handle = module.macro_to_fine.register_forward_hook(macro_hook)
        with torch.inference_mode():
            fine, _, aux = module(self.h_fine, self.h_coarse, self.macro_prompt)
        handle.remove()

        static = self.macro_prompt.mean(dim=2).unsqueeze(1).expand_as(self.h_fine)
        recent = self.h_fine[:, -24:].permute(0, 2, 1, 3).reshape(6, 24, 8)
        coarse_seq = self.h_coarse.permute(0, 2, 1, 3).reshape(6, 144, 8)
        with torch.inference_mode():
            reverse_seq, _ = module.fine_to_coarse(coarse_seq, recent, recent, need_weights=False)
            reverse = reverse_seq.reshape(2, 3, 144, 8).permute(0, 2, 1, 3)
            gate = module.gate(torch.cat([static, reverse, self.h_fine], dim=-1))
            expected = module.fine_norm(self.h_fine + gate * static)

        self.assertEqual(macro_calls, 0)
        self.assertTrue(torch.allclose(fine, expected))
        self.assertEqual(aux["coarse_to_fine_source"], "macro_prompt_static_mean")
        self.assertIsNone(aux["macro_attn_entropy"])

    def test_c3_shares_attention_module_and_parameters_by_identity(self) -> None:
        a0 = SymmetricCrossFusion(**self.kwargs).eval()
        c3 = SymmetricCrossFusion(
            **self.kwargs,
            share_cross_attention_projections=True,
        ).eval()
        self.assertIs(c3.macro_to_fine, c3.fine_to_coarse)
        self.assertIs(c3.macro_to_fine.in_proj_weight, c3.fine_to_coarse.in_proj_weight)
        self.assertLess(
            sum(parameter.numel() for parameter in c3.parameters()),
            sum(parameter.numel() for parameter in a0.parameters()),
        )

        calls = 0

        def hook(_module, _inputs, _output):
            nonlocal calls
            calls += 1

        handle = c3.macro_to_fine.register_forward_hook(hook)
        with torch.inference_mode():
            fine, coarse, aux = c3(self.h_fine, self.h_coarse, self.macro_prompt)
        handle.remove()
        self.assertEqual(calls, 2)
        self.assertEqual(tuple(fine.shape), tuple(self.h_fine.shape))
        self.assertEqual(tuple(coarse.shape), tuple(self.h_coarse.shape))
        self.assertTrue(aux["share_cross_attention_projections"])

    def test_each_candidate_has_observable_forward_difference_from_a0(self) -> None:
        base = SymmetricCrossFusion(**self.kwargs).eval()
        candidates = {
            "A6-C1": SymmetricCrossFusion(
                **self.kwargs, macro_to_fine_exclude_recent_len=24
            ).eval(),
            "A6-C2": SymmetricCrossFusion(
                **self.kwargs, macro_to_fine_mode="static_mean"
            ).eval(),
            "A6-C3": SymmetricCrossFusion(
                **self.kwargs, share_cross_attention_projections=True
            ).eval(),
        }
        with torch.inference_mode():
            base_fine, base_coarse, _ = base(self.h_fine, self.h_coarse, self.macro_prompt)
        for name, candidate in candidates.items():
            candidate.load_state_dict(base.state_dict(), strict=True)
            with torch.inference_mode():
                fine, coarse, _ = candidate(self.h_fine, self.h_coarse, self.macro_prompt)
            self.assertEqual(tuple(fine.shape), tuple(base_fine.shape), name)
            self.assertEqual(tuple(coarse.shape), tuple(base_coarse.shape), name)
            self.assertFalse(
                torch.allclose(fine, base_fine) and torch.allclose(coarse, base_coarse),
                name,
            )

    def test_candidate_full_model_synthetic_forward(self) -> None:
        num_nodes = 3
        graph = torch.eye(num_nodes).numpy()
        graph_data = {
            "A_macro_trend": graph,
            "A_micro_local": graph,
            "metadata": {"source": "cross_fusion_candidate_test", "train_only": True},
        }
        x = torch.randn(1, 36, num_nodes, len(STMGPromptConfig().feature_cols))

        base_cfg = apply_variant(STMGPromptConfig(), "A0", "component_ablation")
        base_cfg.num_nodes = num_nodes
        base_cfg.hidden_dim = 8
        base_cfg.dropout = 0.0
        base_cfg.use_train_robust_volatility = False
        base_cfg.diagnostics_level = "none"
        base = build_model(base_cfg, input_dim=x.shape[-1], graph_data=graph_data).eval()
        with torch.inference_mode():
            base_pred = base(x)["pred"]

        for name in CROSS_FUSION_CANDIDATE_VARIANTS:
            cfg = apply_variant(STMGPromptConfig(), name, "component_ablation")
            cfg.num_nodes = num_nodes
            cfg.hidden_dim = 8
            cfg.dropout = 0.0
            cfg.use_train_robust_volatility = False
            cfg.diagnostics_level = "none"
            model = build_model(cfg, input_dim=x.shape[-1], graph_data=graph_data).eval()
            model.load_state_dict(base.state_dict(), strict=True)
            with torch.inference_mode():
                output = model(x)
            self.assertEqual(tuple(output["pred"].shape), (1, 10, num_nodes), name)
            self.assertFalse(torch.allclose(output["pred"], base_pred), name)
            self.assertEqual(output["aux"]["macro_to_fine_mode"], cfg.macro_to_fine_mode)
            self.assertEqual(
                output["aux"]["macro_to_fine_exclude_recent_len"],
                cfg.macro_to_fine_exclude_recent_len,
            )
            self.assertEqual(
                output["aux"]["share_cross_attention_projections"],
                cfg.share_cross_attention_projections,
            )

    def test_runner_keeps_candidate_artifacts_and_summary_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_family(
                "component_ablation",
                [
                    "--variants",
                    "A6-C1",
                    "A6-C2",
                    "A6-C3",
                    "--dry-run",
                    "--output-root",
                    tmp,
                    "--run-id",
                    "candidate_test",
                ],
            )
            root = Path(result["root"])
            for name in CROSS_FUSION_CANDIDATE_VARIANTS:
                candidate_dir = root / name
                self.assertTrue((candidate_dir / "effective_config.json").is_file())
                self.assertTrue((candidate_dir / "effective_config_diff.json").is_file())
                self.assertTrue((candidate_dir / "candidate_manifest.json").is_file())
                config = json.loads(
                    (candidate_dir / "effective_config.json").read_text(encoding="utf-8")
                )
                self.assertEqual(config["component_ablation"], name)
            self.assertTrue((root / "cross_fusion_candidate_config_matrix.csv").is_file())
            self.assertTrue((root / "cross_fusion_candidate_metrics_summary.json").is_file())
            formal_manifest = json.loads(
                (root / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["variant_id"] for item in formal_manifest["variants"]],
                [f"A{i}" for i in range(8)],
            )


if __name__ == "__main__":
    unittest.main()
