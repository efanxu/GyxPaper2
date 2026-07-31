import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from benchmark_v2.artifacts import validate_run, write_status
from benchmark_v2.checkpointing import CheckpointManager
from benchmark_v2.experiments.e5_common_loss import a8_reference, aggregation
from benchmark_v2.experiments.e5_common_loss.a8_reference import (
    validate_a8_reference,
)
from benchmark_v2.experiments.e5_common_loss.contracts import (
    BATCH4_A8_REFERENCE_ID,
)
from benchmark_v2.experiments.e5_common_loss.readiness import build_readiness
from benchmark_v2.hardware_preflight import preflight_identity
from benchmark_v2.model_cli import _write_common
from benchmark_v2.model_runtime import build_model_runtime
from benchmark_v2.protocol import load_protocol
from benchmark_v2.registry import load_registry
from benchmark_v2.runtime import ProviderBatchIterable
from benchmark_v2.training_profiles import (
    PROFILE_ALLOWLIST,
    PROFILE_SCHEMA_VERSION,
    UNIFORM_BATCH4_PROFILE_ID,
    apply_training_profile,
    load_training_profile,
    resolved_batch_sizes,
)


PROFILE_ID = UNIFORM_BATCH4_PROFILE_ID


class UniformBatch4ProfileTests(unittest.TestCase):
    def test_schema_hash_allowlist_and_fixed_values(self):
        first = load_training_profile(PROFILE_ID)
        second = load_training_profile(PROFILE_ID)
        self.assertEqual(first.payload["schema_version"], PROFILE_SCHEMA_VERSION)
        self.assertEqual(PROFILE_ALLOWLIST, (PROFILE_ID,))
        self.assertEqual(first.profile_hash, second.profile_hash)
        self.assertEqual(
            first.profile_hash,
            "f58bbc161dfba0f00774879fdf2f78ec7c59a7faff9ef28a1733a2ed6835fe66",
        )
        self.assertEqual(first.train_batch_size, 4)
        self.assertEqual(first.val_batch_size, 4)
        self.assertEqual(first.test_batch_size, 4)
        self.assertEqual(first.payload["gradient_accumulation_steps"], 1)
        self.assertEqual(first.payload["effective_train_batch_size"], 4)
        with self.assertRaises(Exception):
            load_training_profile("uniform_train_batch8_v1")

    def test_backward_compatibility_and_registry_counts(self):
        protocol = load_protocol()
        self.assertEqual(protocol.protocol_hash, "0140d8774e2cc189a8bd99f1fc9c8a120b565265bf9a7c729ed1d47a0b1c069b")
        self.assertEqual(
            resolved_batch_sizes(protocol, None),
            {"train": 32, "val": 4, "test": 4},
        )
        registry = load_registry()
        self.assertEqual(len(registry.list()), 28)
        self.assertEqual(
            sum(
                entry.values.get("source_type") == "tslib_source"
                for entry in registry.list()
            ),
            18,
        )

    def test_provider_batches_are_exactly_four_without_sample_change(self):
        class FakeProvider:
            def __init__(self):
                self.starts = {"train": list(range(10))}

            def batches(self, split, batch_size):
                return [list(self.starts[split])]

        provider = FakeProvider()
        batches = list(ProviderBatchIterable(provider, "train", 4))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(provider.starts["train"], list(range(10)))

    def test_model_initialization_forward_and_optimizer_are_invariant(self):
        from benchmark_v2.experiments.e5_common_loss.config_diff import (
            optimizer_group_signature,
            state_dict_hash,
        )
        from benchmark_v2.model_cli import _synthetic_batch, _synthetic_scalers
        from benchmark_v2.seeds import seed_everything

        protocol = load_protocol()
        input_scaler, target_scaler = _synthetic_scalers()
        batch = _synthetic_batch(batch_size=4, time_steps=144, nodes=4)
        seed_everything(2026)
        base = build_model_runtime(
            "gru",
            protocol,
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        base_hash = state_dict_hash(base.model)
        base_output = base.adapter(
            base.model, batch, expected_horizon=10, expected_features=16
        ).prediction.detach()
        base_optimizer = torch.optim.Adam(base.model.parameters(), lr=0.001)
        seed_everything(2026)
        profiled = build_model_runtime(
            "gru",
            protocol,
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        apply_training_profile(profiled, PROFILE_ID)
        profiled_output = profiled.adapter(
            profiled.model, batch, expected_horizon=10, expected_features=16
        ).prediction.detach()
        profiled_optimizer = torch.optim.Adam(profiled.model.parameters(), lr=0.001)
        self.assertEqual(list(base.model.state_dict()), list(profiled.model.state_dict()))
        self.assertEqual(base_hash, state_dict_hash(profiled.model))
        self.assertEqual(base.parameter_count, profiled.parameter_count)
        self.assertEqual(
            base.trainable_parameter_count, profiled.trainable_parameter_count
        )
        self.assertTrue(torch.equal(base_output, profiled_output))
        self.assertEqual(
            optimizer_group_signature(base_optimizer),
            optimizer_group_signature(profiled_optimizer),
        )

    def test_preflight_identity_rejects_other_batch_shapes_and_hashes(self):
        batch4 = preflight_identity("gru", training_profile=PROFILE_ID)
        batch32 = preflight_identity("gru")
        self.assertEqual(batch4["B"], 4)
        self.assertEqual(batch32["B"], 32)
        self.assertNotEqual(batch4, batch32)
        for batch in (32, 16, 8):
            candidate = dict(batch4)
            candidate["B"] = batch
            self.assertNotEqual(candidate, batch4)
        for key in (
            "training_batch_profile_hash",
            "model_source_closure_hash",
            "model_config_hash",
            "graph_protocol_hash",
        ):
            candidate = dict(batch4)
            candidate[key] = "mismatch"
            self.assertNotEqual(candidate, batch4)

    def test_artifact_and_checkpoint_write_batch_identity(self):
        from benchmark_v2.model_cli import _synthetic_scalers

        protocol = load_protocol()
        input_scaler, target_scaler = _synthetic_scalers()
        runtime = build_model_runtime(
            "gru",
            protocol,
            run_mode="smoke",
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        )
        apply_training_profile(runtime, PROFILE_ID)
        profile = load_training_profile(PROFILE_ID)
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            resolved, effective = _write_common(
                run_dir,
                runtime,
                protocol,
                profile="REFERENCE_ONLY",
                run_mode="smoke",
                data_signature={},
                formal_training=False,
            )
            manifest = json.loads(
                (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["training_batch_profile_id"], PROFILE_ID)
            manager = CheckpointManager(
                run_dir,
                protocol_hash=protocol.protocol_hash,
                model_id="gru",
                resolved_config=resolved,
                effective_config=effective,
            )
            optimizer = torch.optim.Adam(runtime.model.parameters(), lr=0.001)
            path = manager.save(
                "identity_checkpoint.pt",
                epoch=1,
                global_step=1,
                monitor_value=1.0,
                model=runtime.model,
                optimizer=optimizer,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                checkpoint["batch_identity"]["training_batch_profile_hash"],
                profile.profile_hash,
            )
            write_status(
                run_dir,
                status="REFERENCE_ONLY",
                run_mode="smoke",
                artifact_profile="REFERENCE_ONLY",
            )
            with self.assertRaises(Exception):
                validate_run(
                    run_dir,
                    expected_training_batch_profile_id="wrong_profile",
                )

    def test_readiness_and_aggregation_reject_mixed_batch(self):
        profile = load_training_profile(PROFILE_ID)
        manifest = {
            "entries": [
                {
                    "entry_id": "e5_gru",
                    "model_id": "gru",
                    "entry_type": "TRAIN_COMMON_LOSS",
                    "e5_run_id": "gru_bs4",
                    "loss_id": "masked_score_aligned_hybrid",
                    "loss_source_hash": "loss-source",
                    "loss_profile_hash": "loss-profile",
                    "e5_common_loss_protocol_hash": "e5-protocol",
                    "base_model_config_hash": "config",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "gru_bs4"
            run.mkdir()
            (run / "run_status.json").write_text(
                json.dumps({"status": "COMPLETED", "run_mode": "formal"}),
                encoding="utf-8",
            )
            effective = {
                "loss": {
                    "id": "masked_score_aligned_hybrid",
                    "source_hash": "loss-source",
                    "profile_hash": "loss-profile",
                },
                "provenance": {
                    "e5_common_loss_protocol_hash": "e5-protocol",
                    "base_model_config_hash": "config",
                },
                **profile.identity(),
                "training_batch_profile_hash": "wrong",
            }
            (run / "effective_config.json").write_text(
                json.dumps(effective), encoding="utf-8"
            )
            for horizon in (3, 6, 10):
                (run / f"metrics_eval_h{horizon}.json").write_text(
                    json.dumps({"horizon": horizon}), encoding="utf-8"
                )
            with patch(
                "benchmark_v2.experiments.e5_common_loss.readiness.build_variant_manifest",
                return_value=manifest,
            ), patch(
                "benchmark_v2.experiments.e5_common_loss.readiness.validate_a8_reference",
                return_value={"status": "VALID"},
            ):
                report = build_readiness(
                    output_root=td, training_profile=PROFILE_ID
                )
            self.assertFalse(report["entries"][0]["batch_profile_match"])
            self.assertIn(
                "BLOCKED_MIXED_BATCH_PROFILE",
                report["entries"][0]["blocked_reason"],
            )
        with patch.object(
            aggregation,
            "build_readiness",
            return_value={"status": "NOT_READY", "ready_entries": 0},
        ):
            with self.assertRaises(RuntimeError):
                aggregation.aggregate(
                    output_root=".",
                    require_complete=True,
                    training_profile=PROFILE_ID,
                )

    def test_batch4_a8_rejects_old_and_accepts_matching_new_reference(self):
        blocked = validate_a8_reference(training_profile=PROFILE_ID)
        self.assertEqual(blocked["reference_id"], BATCH4_A8_REFERENCE_ID)
        self.assertNotEqual(blocked.get("status"), "VALID")
        profile = load_training_profile(PROFILE_ID)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = (
                root
                / "custom_models/results/st_mgprompt_uniform_bs4"
                / "component_ablation_a8_bs4_seed2026"
                / "STMGPrompt_ComponentAblation"
            )
            run.mkdir(parents=True)
            config = {
                "component_ablation": "A8",
                "loss_function": "masked_score_aligned_hybrid",
                "loss_protocol": "fair_main",
                "use_msmg_dwu": False,
                "seed": 2026,
                "lookback": 144,
                "max_pred_len": 10,
                **profile.identity(),
            }
            json_files = {
                "effective_config.json": config,
                "effective_config_diff.json": {"passed": True},
                "training_batch_profile.json": profile.identity(),
                "artifact_manifest.json": profile.identity(),
                "run_status.json": {"current_stage": "PROCESS_FINISHED"},
                "train_complete.json": {"status": "completed"},
                "evaluation_complete.json": {"status": "completed"},
                "prediction_metadata.json": {},
                "protocol_check.json": {"passed": True},
            }
            for name, payload in json_files.items():
                (run / name).write_text(json.dumps(payload), encoding="utf-8")
            for horizon in (3, 6, 10):
                (run / f"metrics_eval_h{horizon}.json").write_text(
                    json.dumps(
                        {
                            "horizon": horizon,
                            "Score": 1.0,
                            "MAE": 1.0,
                            "RMSE": 1.0,
                            "R2": 0.0,
                        }
                    ),
                    encoding="utf-8",
                )
            torch.save({}, run / "best_checkpoint.pt")
            with patch.object(a8_reference, "PROJECT_ROOT", root):
                accepted = validate_a8_reference(training_profile=PROFILE_ID)
            self.assertEqual(accepted["status"], "VALID")
            self.assertEqual(accepted["training_batch_profile_hash"], profile.profile_hash)


if __name__ == "__main__":
    unittest.main()
