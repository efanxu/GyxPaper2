from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from st_mgprompt.data import make_dataloaders
from st_mgprompt.empirical_protocol import (
    EMPIRICAL_FAMILIES,
    EMPIRICAL_VARIANTS,
    FROZEN_PROTOCOL_FIELDS,
    apply_empirical_variant,
    assert_empirical_expected_diff,
    dry_run_report,
)
from st_mgprompt.experiment_protocol import (
    CANONICAL_ID,
    COMPONENT_ABLATION_VARIANTS,
    PRECISION_VARIANTS,
    canonical_config,
)
from st_mgprompt.graph_prior import prepare_graph_artifacts
from st_mgprompt.losses import get_loss_fn
from st_mgprompt.metrics import evaluate_prefix_horizons
from st_mgprompt.registry import build_model
from st_mgprompt.run_empirical import run_empirical
from st_mgprompt.summarize_empirical import summarize_empirical


class EmpiricalProtocolTests(unittest.TestCase):
    def test_registry_has_requested_namespaces_and_declared_differences(self) -> None:
        self.assertEqual(tuple(EMPIRICAL_FAMILIES), ("T", "G", "D", "F", "N", "L", "R"))
        self.assertEqual(tuple(EMPIRICAL_FAMILIES["T"]), tuple(f"T{i}" for i in range(6)))
        self.assertEqual(tuple(EMPIRICAL_FAMILIES["G"]), tuple(f"G{i}" for i in range(9)))
        self.assertEqual(tuple(EMPIRICAL_FAMILIES["F"]), tuple(f"F{i}" for i in range(9)))
        self.assertEqual(tuple(EMPIRICAL_FAMILIES["N"]), tuple(f"N{i}" for i in range(9)))
        self.assertEqual(tuple(EMPIRICAL_FAMILIES["L"]), tuple(f"L{i}" for i in range(8)))
        self.assertEqual(tuple(EMPIRICAL_FAMILIES["R"]), ("R0",))
        for variant in EMPIRICAL_VARIANTS.values():
            report = dry_run_report(variant)
            self.assertTrue(report["passed"], variant.variant_id)
            if variant.ready:
                config = apply_empirical_variant(None, variant.variant_id, variant.family)
                audit = assert_empirical_expected_diff(config, variant.variant_id, variant.family)
                self.assertEqual(audit["actual_diff_fields"], sorted(variant.expected_unique_diff))
                self.assertFalse(set(audit["actual_diff_fields"]).intersection(FROZEN_PROTOCOL_FIELDS))
            else:
                self.assertFalse(report["execution_ready"])

    def test_formal_registries_and_canonical_identity_are_unchanged(self) -> None:
        self.assertEqual(tuple(PRECISION_VARIANTS), ("P0", "P1", "P2", "P3", "P4", "P5"))
        self.assertEqual(tuple(COMPONENT_ABLATION_VARIANTS), tuple(f"A{i}" for i in range(9)))
        self.assertEqual(PRECISION_VARIANTS["P0"].canonical_reference, CANONICAL_ID)
        self.assertEqual(COMPONENT_ABLATION_VARIANTS["A0"].canonical_reference, CANONICAL_ID)
        self.assertEqual(COMPONENT_ABLATION_VARIANTS["A6"].config_overrides, {"macro_to_fine_exclude_recent_len": 24})
        self.assertEqual(COMPONENT_ABLATION_VARIANTS["A7"].config_overrides, {"share_cross_attention_projections": True})
        self.assertEqual(
            COMPONENT_ABLATION_VARIANTS["A8"].config_overrides,
            {"use_msmg_dwu": False, "loss_function": "masked_score_aligned_hybrid", "loss_protocol": "fair_main"},
        )
        self.assertFalse(any(name.startswith(("P", "A", "E")) for name in EMPIRICAL_VARIANTS))

    def test_ready_variants_forward_backward_and_metrics_are_finite(self) -> None:
        graph = {
            "A_macro_trend": np.eye(4, dtype=np.float32),
            "A_micro_local": np.eye(4, dtype=np.float32),
            "metadata": {"graph_uses_train_only_statistics": True},
        }
        x = torch.randn(2, 12, 4, 16)
        target = torch.randn(2, 10, 4)
        mask = torch.ones_like(target)
        for variant in EMPIRICAL_VARIANTS.values():
            if not variant.trainable or not variant.ready:
                continue
            config = apply_empirical_variant(None, variant.variant_id, variant.family)
            config.hidden_dim = 8
            config.dropout = 0.0
            config.num_nodes = 4
            model = build_model(config, input_dim=16, graph_data=graph)
            output = model(x)["pred"]
            self.assertEqual(tuple(output.shape), (2, 10, 4), variant.variant_id)
            self.assertTrue(torch.isfinite(output).all(), variant.variant_id)
            loss_fn = get_loss_fn(config.loss_function, config=config, num_nodes=4)
            loss = loss_fn(output, target, mask)
            self.assertIsNotNone(loss, variant.variant_id)
            self.assertTrue(torch.isfinite(loss).all(), variant.variant_id)
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()), variant.variant_id)
            rows = evaluate_prefix_horizons(
                output.detach().numpy(), target.numpy(), mask.numpy(), [3, 6, 10], num_nodes=4
            )
            self.assertTrue(all(math.isfinite(float(row["Score"])) for row in rows), variant.variant_id)

    def test_formal_node_count_output_shape(self) -> None:
        config = apply_empirical_variant(None, "T1", "T")
        config.hidden_dim = 8
        config.dropout = 0.0
        graph = {
            "A_macro_trend": np.eye(134, dtype=np.float32),
            "A_micro_local": np.eye(134, dtype=np.float32),
            "metadata": {"graph_uses_train_only_statistics": True},
        }
        model = build_model(config, input_dim=16, graph_data=graph).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 12, 134, 16))["pred"]
        self.assertEqual(tuple(output.shape), (1, 10, 134))

    def test_windows_are_causal_and_train_only_fit_metadata_is_explicit(self) -> None:
        config = canonical_config()
        config.smoke = True
        config.smoke_use_synthetic = True
        config.lookback = 12
        config.hidden_dim = 8
        config.batch_size = 2
        config.eval_batch_size = 2
        config.num_workers = 0
        data = make_dataloaders(config)["bundle"]
        sample = data.train_loader.dataset[0]
        prediction_start = sample["prediction_start_index"]
        self.assertEqual(sample["timestamps"], data.train_loader.dataset.timestamps[prediction_start - 12 : prediction_start])
        self.assertEqual(sample["target_timestamps"], data.train_loader.dataset.timestamps[prediction_start : prediction_start + 10])
        self.assertTrue(set(sample["timestamps"]).isdisjoint(sample["target_timestamps"]))
        self.assertEqual(data.metadata["window_rule"], "[t-lookback,t)->[t,t+max_pred_len)")
        self.assertEqual(data.metadata["input_scaler_fit_split"], "train_only")
        self.assertEqual(data.metadata["target_scaler_fit_split"], "train_only")
        self.assertEqual(data.metadata["vadsp_statistics_fit_split"], "train")
        self.assertTrue(data.metadata["on_the_fly_features_history_only"])
        with tempfile.TemporaryDirectory() as temporary:
            config.graph_output_root = temporary
            config.graph_tag = "train_only_test"
            graph = prepare_graph_artifacts(data, config)
            self.assertTrue(graph["metadata"]["graph_uses_train_only_statistics"])
            self.assertFalse(graph["metadata"]["graph_uses_test_statistics"])
            self.assertFalse(graph["metadata"]["graph_uses_future_target"])

    def test_dry_run_and_summary_do_not_zero_fill_missing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_empirical(
                [
                    "--family",
                    "T",
                    "--variants",
                    "T0",
                    "T1",
                    "T3",
                    "--dry-run",
                    "--output-root",
                    temporary,
                ]
            )
            self.assertEqual(len(result["statuses"]), 3)
            self.assertTrue(all(row["status"] == "DRY_RUN" for row in result["statuses"]))
            manifest = summarize_empirical(temporary)
            self.assertTrue(manifest["missing_values_are_not_zero_filled"])
            long_path = Path(temporary) / "summary" / "empirical_metrics_long.csv"
            with long_path.open("r", newline="", encoding="utf-8-sig") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            matrix = Path(temporary) / "variant_config_matrix.csv"
            self.assertTrue(matrix.is_file())
            json.loads((Path(temporary) / "T" / "t1_empirical_v1" / "dry_run_report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
