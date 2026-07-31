import copy
import unittest

import torch

from benchmark_v2.contracts import BenchmarkBatch
from benchmark_v2.errors import ContractError
from benchmark_v2.losses import masked_mse
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream import ALLOWED_TSLIB_MODELS


MODEL_IDS = ("transformer", "patchtst", "itransformer", "timexer")


def make_batch(b=1, n=2, seed=2026):
    generator = torch.Generator().manual_seed(seed)
    return BenchmarkBatch(
        x=torch.randn(b, 144, n, 16, generator=generator),
        target=torch.randn(b, n, 10, generator=generator),
        mask=torch.ones(b, n, 10, dtype=torch.bool),
        sample_ids=list(range(b)),
        window_end_indices=list(range(b)),
        node_ids=list(range(n)),
        metadata={"contains_future_target": False},
    )


class E2AModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def build(self, model_id):
        runtime = build_model_runtime(model_id, self.protocol, run_mode="smoke")
        runtime.model.eval()
        return runtime

    def test_allowlist_is_exactly_eighteen(self):
        self.assertEqual(
            ALLOWED_TSLIB_MODELS,
            (
                "dlinear", "lightts", "tide", "segrnn",
                "transformer", "patchtst", "itransformer", "timexer",
                "timesnet", "micn", "wpmixer", "multipatchformer",
                "timemixer", "tsmixer", "frets",
                "crossformer", "msgnet", "timefilter",
            ),
        )

    def test_registry_status_and_statistics(self):
        registry = load_registry()
        for model_id in MODEL_IDS:
            entry = registry.get(model_id)
            self.assertTrue(entry.supports_train)
            self.assertEqual(entry.node_semantics, "node_shared")
        stats = registry.statistics()
        self.assertEqual(stats["total_entries"], 28)
        self.assertEqual(stats["available_with_hardware_preflight"], 10)
        self.assertEqual(stats["non_trainable_available"], 2)

    def test_shapes_backward_and_no_target_or_mask_leakage(self):
        for model_id in MODEL_IDS:
            with self.subTest(model_id=model_id):
                runtime = self.build(model_id)
                batch = make_batch()
                output = runtime.adapter(runtime.model, batch)
                self.assertEqual(tuple(output.prediction.shape), (1, 2, 10))
                self.assertTrue(torch.isfinite(output.prediction).all())
                changed = copy.copy(batch)
                changed.target = torch.randn_like(batch.target) * 1000
                changed.mask = torch.zeros_like(batch.mask)
                second = runtime.adapter(runtime.model, changed)
                self.assertTrue(torch.equal(output.prediction, second.prediction))
                loss = masked_mse(output.prediction, batch.target, batch.mask)
                loss.backward()
                self.assertTrue(any(
                    p.grad is not None for p in runtime.model.parameters()
                    if p.requires_grad
                ))

    def test_transformer_decoder_future_and_marks_are_zero(self):
        runtime = self.build("transformer")
        batch = make_batch()
        output = runtime.adapter(runtime.model, batch)
        trace = output.aux["decoder_trace"]
        self.assertEqual(trace["label_len"], 48)
        self.assertEqual(trace["decoder_shape"], (2, 58, 16))
        self.assertTrue(trace["history_matches_observed_x"])
        self.assertTrue(trace["future_decoder_all_zero"])
        self.assertTrue(trace["encoder_marks_all_zero"])
        self.assertTrue(trace["decoder_marks_all_zero"])

    def test_patchtst_exact_patch_geometry(self):
        runtime = self.build("patchtst")
        trace = runtime.adapter(runtime.model, make_batch()).aux["patch_trace"]
        self.assertEqual(trace["original_length"], 144)
        self.assertEqual(trace["padding_length"], 8)
        self.assertEqual(trace["patch_num"], 18)
        self.assertEqual(trace["final_patch_shape"], "(B*N*16,18,512)")

    def test_itransformer_has_sixteen_feature_tokens(self):
        runtime = self.build("itransformer")
        output = runtime.adapter(runtime.model, make_batch())
        self.assertEqual(output.aux["variable_token_count"], 16)
        self.assertEqual(
            output.aux["variable_token_semantics"],
            "features_within_one_turbine",
        )

    def test_timexer_endogenous_exogenous_and_future_rejection(self):
        runtime = self.build("timexer")
        output = runtime.adapter(runtime.model, make_batch())
        trace = output.semantic_trace
        self.assertEqual(trace["endogenous_feature"], "Patv_clean_for_input")
        self.assertEqual(trace["endogenous_feature_index"], 15)
        self.assertEqual(len(trace["exogenous_features"]), 15)
        self.assertIsNone(trace["future_exogenous_tensor"])
        with self.assertRaisesRegex(ContractError, "future"):
            runtime.adapter(
                runtime.model,
                make_batch(),
                future_exogenous=torch.ones(2, 10, 15),
            )
        with self.assertRaisesRegex(ContractError, "future exogenous"):
            runtime.adapter(
                runtime.model,
                make_batch(),
                future_exogenous=torch.zeros(2, 10, 15),
            )


if __name__ == "__main__":
    unittest.main()
