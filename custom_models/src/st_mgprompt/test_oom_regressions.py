from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from st_mgprompt.cross_fusion import SymmetricCrossFusion, _attention_entropy
from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.data import make_dataloaders
from st_mgprompt.graph_layers import row_normalize
from st_mgprompt.registry import build_model
from st_mgprompt.run_precision_ablation import (
    _mark_dependents_blocked_by_failed_baseline,
    _safe_optional_float,
    _should_train_variant,
    read_best_val_summary,
)
from st_mgprompt.train import _run_epoch
from st_mgprompt.volatility_patching import compute_vadsp_robust_statistics_from_timeline


class AttentionDiagnosticsRegressionTests(unittest.TestCase):
    def test_attention_entropy_is_detached_finite_and_accepts_row_cap(self) -> None:
        weights = torch.tensor(
            [[[0.0, 0.25, 0.75], [0.5, 0.5, 0.0]]],
            dtype=torch.float16,
            requires_grad=True,
        )
        before = weights.detach().clone()

        entropy = _attention_entropy(weights, max_rows=1)

        self.assertTrue(torch.isfinite(entropy))
        self.assertFalse(entropy.requires_grad)
        self.assertEqual(entropy.dtype, torch.float32)
        self.assertTrue(torch.equal(weights.detach(), before))

    def test_minimal_diagnostics_does_not_request_attention_weights(self) -> None:
        module = SymmetricCrossFusion(
            hidden_dim=8,
            num_heads=2,
            dropout=0.0,
            recent_len=3,
            diagnostics_level="minimal",
        )
        requested: list[bool] = []

        def record_request(_module, args, kwargs):
            requested.append(bool(kwargs.get("need_weights", True)))

        handles = [
            module.macro_to_fine.register_forward_pre_hook(record_request, with_kwargs=True),
            module.fine_to_coarse.register_forward_pre_hook(record_request, with_kwargs=True),
        ]
        try:
            with patch("st_mgprompt.cross_fusion._attention_entropy") as entropy_mock:
                _, _, aux = module(
                    torch.randn(2, 5, 3, 8),
                    torch.randn(2, 5, 3, 8),
                    torch.randn(2, 3, 2, 8),
                )
                entropy_mock.assert_not_called()
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(requested, [False, False])
        self.assertIsNone(aux["macro_attn_entropy"])
        self.assertIsNone(aux["fine_attn_entropy"])

    def test_minimal_diagnostics_preserves_model_predictions(self) -> None:
        def config(level: str) -> STMGPromptConfig:
            return STMGPromptConfig(
                model_name="STMGPrompt_FairFull",
                num_nodes=3,
                hidden_dim=8,
                lookback=8,
                max_pred_len=4,
                eval_horizons=[2, 3, 4],
                primary_val_horizon=4,
                checkpoint_selection_metric="val_official_score_h4",
                use_vadsp=True,
                use_trend_prior_graph=True,
                use_adaptive_graph=True,
                use_graph_temporal_encoder=True,
                use_stmg_coupling_block=True,
                use_macro_prompt=True,
                use_st_prompt=True,
                dropout=0.0,
                diagnostics_level=level,
            )

        graph = {
            "A_macro_trend": row_normalize(torch.eye(3)).numpy(),
            "A_micro_local": row_normalize(torch.eye(3)).numpy(),
            "metadata": {},
        }
        standard = build_model(config("standard"), input_dim=16, graph_data=graph)
        minimal = build_model(config("minimal"), input_dim=16, graph_data=graph)
        stats = compute_vadsp_robust_statistics_from_timeline(
            torch.randn(24, 3, 16),
            feature_names=list(config("minimal").feature_cols),
            volatility_source_cols=list(config("minimal").volatility_source_cols),
            volatility_mode="per_node",
            robust_eps=1e-6,
        )
        standard.vadsp.set_robust_statistics(stats)
        minimal.load_state_dict(standard.state_dict())
        standard.eval()
        minimal.eval()
        x = torch.randn(2, 8, 3, 16)
        with torch.inference_mode():
            standard_out = standard(x)
            minimal_out = minimal(x)

        self.assertTrue(torch.allclose(standard_out["pred"], minimal_out["pred"], atol=1e-5, rtol=1e-5))
        self.assertTrue(standard_out["aux"]["attention_weights_requested"])
        self.assertFalse(minimal_out["aux"]["attention_weights_requested"])
        self.assertFalse(minimal_out["aux"]["entropy_called"])


class ValidationInferenceRegressionTests(unittest.TestCase):
    def test_independent_loader_batch_sizes_are_honored(self) -> None:
        cfg = STMGPromptConfig(
            smoke=True,
            lookback=12,
            max_pred_len=10,
            smoke_num_time_steps=600,
            train_batch_size=5,
            val_batch_size=3,
            test_batch_size=2,
        )
        bundle = make_dataloaders(cfg)["bundle"]
        self.assertEqual(next(iter(bundle.train_loader))["x"].shape[0], 5)
        self.assertEqual(next(iter(bundle.val_loader))["x"].shape[0], 3)
        self.assertEqual(next(iter(bundle.test_loader))["x"].shape[0], 2)

    def test_validation_forward_runs_with_grad_disabled(self) -> None:
        class RecordingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))
                self.grad_states: list[bool] = []

            def forward(self, x):
                self.grad_states.append(torch.is_grad_enabled())
                pred = x[:, :2, :, 0] * self.scale
                return {"pred": pred, "aux": {"pred_bnh": pred.transpose(1, 2)}}

        class MaskedMSE(torch.nn.Module):
            def forward(self, pred, target, mask):
                return (((pred - target) ** 2) * mask).sum() / mask.sum()

        model = RecordingModel()
        before = model.scale.detach().clone()
        loader = [
            {
                "x": torch.randn(2, 3, 4, 1),
                "y": torch.randn(2, 2, 4),
                "valid_target_mask": torch.ones(2, 2, 4),
            }
            for _ in range(2)
        ]

        _, _, details = _run_epoch(model, loader, torch.device("cpu"), MaskedMSE())

        self.assertEqual(model.grad_states, [False, False])
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(model.scale.detach(), before))
        self.assertTrue(all(not torch.is_tensor(value) for value in details.values()))


class PrecisionAblationFailureRegressionTests(unittest.TestCase):
    def test_safe_optional_float_rejects_missing_invalid_and_nonfinite(self) -> None:
        for value in [None, "", " ", "not-a-number", float("nan"), float("inf")]:
            self.assertIsNone(_safe_optional_float(value))
        self.assertEqual(_safe_optional_float("1.25"), 1.25)

    def test_failed_p0_with_null_metric_can_still_be_summarized(self) -> None:
        manifest = [
            {
                "variant": "P0",
                "model_name": "STMGPrompt_FairFull",
                "graph_operator": "simple",
                "decoder_context_mode": "last_state",
                "hidden_dim": 64,
                "num_coupling_layers": 1,
            }
        ]
        status = {
            "P0": {
                "status": "failed",
                "train_status": "failed",
                "best_val_score_h10": None,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            rows = read_best_val_summary(Path(tmp), manifest, status)

        self.assertEqual(rows[0]["status"], "failed")
        self.assertIsNone(rows[0]["best_val_Score_H10"])
        self.assertEqual(rows[0]["relative_improvement_vs_P0_H10"], "")

    def test_failed_p0_blocks_dependents_and_writes_failure_summary(self) -> None:
        manifest = [
            {"variant": name, "model_name": "STMGPrompt_FairFull"}
            for name in ["P0", "P1", "P2", "P3", "P4", "P5"]
        ]
        status = {
            "P0": {
                "status": "failed",
                "failure_stage": "validation",
                "error_type": "OutOfMemoryError",
                "error": "CUDA out of memory",
                "run_command": ["python", "run_st_mgprompt.py"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _mark_dependents_blocked_by_failed_baseline(root, manifest, status)
            self.assertTrue((root / "precision_ablation_failure_summary.json").exists())
        for name in ["P1", "P2", "P3", "P4", "P5"]:
            self.assertEqual(status[name]["status"], "blocked_by_failed_baseline")

    def test_resume_retries_failed_p0_but_skip_completed_skips_success(self) -> None:
        args = SimpleNamespace(force_rerun=False, resume=True, skip_completed=True)
        self.assertTrue(_should_train_variant(args, {"completed": False, "source": "missing"}))
        self.assertTrue(_should_train_variant(args, {"completed": False, "source": "last_checkpoint"}))
        self.assertFalse(_should_train_variant(args, {"completed": True, "source": "train_complete.json"}))


if __name__ == "__main__":
    unittest.main()
