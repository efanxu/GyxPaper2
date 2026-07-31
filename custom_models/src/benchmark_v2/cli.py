from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .adapters import NodeSharedAdapter
from .artifacts import atomic_write_csv, atomic_write_json, is_formal_discoverable, safe_run_dir, validate_run, write_status
from .contracts import BenchmarkBatch
from .data import SDWPFDataProvider
from .engine import Evaluator, Trainer
from .errors import BenchmarkV2Error, ModelUnavailableError
from .experiments.e5_common_loss.a8_reference import create_a8_reference
from .experiments.e5_common_loss.aggregation import aggregate as e5_aggregate
from .experiments.e5_common_loss.contracts import FORMAL_OUTPUT_ROOT_RELATIVE
from .experiments.e5_common_loss.loss_profile import (
    ALLOWLIST as EXPERIMENT_PROFILE_ALLOWLIST,
    CLI_PROFILE_ID as E5_PROFILE_ID,
    check_loss_profile,
)
from .experiments.e5_common_loss.readiness import build_readiness
from .experiments.e5_common_loss.scope27_contract import E5_SCOPE27_ID
from .hardware_preflight import (
    launch_formal_train,
    launch_preflight,
    run_preflight_worker,
)
from .losses import get_loss
from .model_cli import (
    DEFAULT_INPUT_PATH,
    DEFAULT_TARGET_PATH,
    formal_evaluate_only,
    formal_train,
    full_shape_model_smoke,
    model_smoke,
    real_data_model_smoke,
)
from .protocol import check_protocol, load_protocol
from .registry import load_registry
from .runtime import FORMAL_ROOT, PROJECT_ROOT, SMOKE_ROOT, environment_snapshot
from .training_profiles import PROFILE_ALLOWLIST


def _tiny_batch(*, batch_size: int, time_steps: int, nodes: int, features: int, horizon: int, seed: int = 2026) -> BenchmarkBatch:
    import torch
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, time_steps, nodes, features, generator=generator)
    target = torch.randn(batch_size, nodes, horizon, generator=generator) * 200 + 700
    mask = torch.ones(batch_size, nodes, horizon, dtype=torch.bool)
    mask[:, 0, 0] = False
    return BenchmarkBatch(x=x, target=target, target_raw_or_inverse_transform=target.clone(), mask=mask, sample_ids=[f"smoke-{i}" for i in range(batch_size)], window_end_indices=list(range(batch_size)), node_ids=list(range(nodes)), split="train", metadata={"contains_future_target": False, "target_model_space": "identity", "target_raw_space": "kW"})


def _write_common(run_dir: Path, protocol, effective: dict, profile: str) -> None:
    atomic_write_json(run_dir / "resolved_config.json", {"protocol_hash": protocol.protocol_hash, "model_id": effective["model_id"], "config_source": "benchmark_protocol_v1"})
    atomic_write_json(run_dir / "effective_config.json", effective)
    atomic_write_json(run_dir / "protocol_check.json", check_protocol(protocol, mode="smoke"))
    atomic_write_json(run_dir / "model_summary.json", {"model_id": effective["model_id"], "formal_registry_entry": False, "profile": profile})
    atomic_write_json(run_dir / "artifact_manifest.json", {"schema_version": "artifact_schema_v1", "artifact_profile": profile, "protocol_hash": protocol.protocol_hash, "run_mode": "smoke"})


def framework_smoke() -> dict:
    import torch
    protocol = load_protocol()
    run_dir = safe_run_dir(SMOKE_ROOT, "e0_b_framework")
    run_dir.mkdir(parents=True)
    effective = {"model_id": "__framework_test_only__", "formal_registry_entry": False, "run_mode": "smoke", "artifact_profile": "SMOKE", "seed": 2026, "epochs": 2, "patience": 6, "min_delta": 0.01, "learning_rate": 1e-3, "protocol_hash": protocol.protocol_hash}
    _write_common(run_dir, protocol, effective, "SMOKE")
    batch = _tiny_batch(batch_size=2, time_steps=12, nodes=4, features=16, horizon=10)
    batch.validate(expected_nodes=4)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(16, 10)
        def forward(self, value):
            return self.head(value[:, -1, :])
    model = Tiny()
    trainer = Trainer(model=model, adapter=NodeSharedAdapter(), loss_fn=get_loss("masked_mse"), protocol=protocol, run_dir=run_dir, model_id="__framework_test_only__", resolved_config={"model_id":"__framework_test_only__"}, effective_config=effective)
    history = trainer.fit([batch], [batch])
    metrics = Evaluator(trainer).evaluate([batch])
    atomic_write_json(run_dir / "metrics_eval_h3.json", metrics[0])
    atomic_write_json(run_dir / "metrics_eval_h6.json", metrics[1])
    atomic_write_json(run_dir / "metrics_eval_h10.json", metrics[2])
    atomic_write_csv(run_dir / "metrics.csv", ["horizon", "MAE", "RMSE", "R2", "Score", "score", "valid_target_count"], metrics)
    atomic_write_json(run_dir / "prediction_metadata.json", {"run_mode":"smoke", "model_id":"__framework_test_only__", "formal_registry_entry":False, "protocol_hash":protocol.protocol_hash, "prediction_shape":[2,4,10], "source_checkpoint":"best_checkpoint.pt"})
    write_status(run_dir, status="COMPLETED", run_mode="smoke", artifact_profile="SMOKE", exit_code=0)
    validation = validate_run(run_dir, expected_protocol_hash=protocol.protocol_hash)
    return {"status":"PASS", "run_dir":str(run_dir), "history":history, "metrics":metrics, "artifact_validation":validation, "run_mode":"smoke", "model_id":"__framework_test_only__", "formal_registry_entry":False}


def full_shape_framework_smoke() -> dict:
    import time
    import torch
    protocol = load_protocol()
    actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"run_mode":"smoke", "model_id":"__framework_test_only__", "formal_registry_entry":False, "requested_shape":{"B":32,"T":144,"N":134,"C":16,"H":10}, "device":str(actual_device), "cuda_available":bool(torch.cuda.is_available())}
    try:
        start = time.perf_counter()
        batch = _tiny_batch(batch_size=32, time_steps=144, nodes=134, features=16, horizon=10)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.head = torch.nn.Linear(16, 10)
            def forward(self, value): return self.head(value[:, -1, :])
        model, adapter = Tiny().to(actual_device), NodeSharedAdapter()
        batch.x = batch.x.to(actual_device)
        batch.target = batch.target.to(actual_device)
        batch.mask = batch.mask.to(actual_device)
        batch.target_raw_or_inverse_transform = batch.target_raw_or_inverse_transform.to(actual_device)
        out = adapter(model, batch, expected_horizon=10, expected_features=16)
        loss = get_loss("masked_mse")(out.prediction, batch.target, batch.mask)
        if loss is None: raise RuntimeError("full-shape smoke unexpectedly has zero valid targets")
        loss.backward()
        result.update({"status":"PASS", "output_shape":list(out.prediction.shape), "loss":float(loss.detach()), "elapsed_seconds":time.perf_counter()-start, "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
    except Exception as exc:
        result.update({"status":"FAIL", "error_type":type(exc).__name__, "error_message":str(exc), "traceback_tail":traceback.format_exc().splitlines()[-20:]})
    atomic_write_json(SMOKE_ROOT / "e0_b_full_shape_framework_smoke.json", result)
    return result


def data_contract_smoke() -> dict:
    from .protocol import load_protocol
    protocol = load_protocol()
    try:
        provider = SDWPFDataProvider.from_files(PROJECT_ROOT / "dataset/sdwpf_model_input_base.parquet", PROJECT_ROOT / "dataset/sdwpf_eval_target.parquet", protocol=protocol)
        result = {"status":"PASS", "limited":True, "dataset_signature":provider.signature(), "windows_per_split":{s:min(len(provider.windows(s)),2) for s in ("train","val","test")}, "batch_size":2, "trained":False, "formal_output_written":False}
    except Exception as exc:
        result = {"status":"NOT_RUN", "limited":True, "reason":f"{type(exc).__name__}: {exc}", "trained":False, "formal_output_written":False}
    path = SMOKE_ROOT / "e0_b_framework" / "data_contract_smoke_results.json"
    atomic_write_json(path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("registry-list")
    show = sub.add_parser("registry-show"); show.add_argument("--model", required=True)
    sub.add_parser("protocol-show")
    sub.add_parser("protocol-check")
    sub.add_parser("graph-protocol-check")
    validate = sub.add_parser("artifact-validate"); validate.add_argument("--run-dir", required=True)
    sub.add_parser("framework-smoke")
    sub.add_parser("full-shape-framework-smoke")
    sub.add_parser("data-contract-smoke")
    def add_experiment_profile(target) -> None:
        target.add_argument(
            "--experiment-profile",
            choices=EXPERIMENT_PROFILE_ALLOWLIST,
            default=None,
        )
    def add_training_profile(target) -> None:
        target.add_argument(
            "--training-profile",
            choices=PROFILE_ALLOWLIST,
            default=None,
        )
    def add_formal_scope(target) -> None:
        target.add_argument(
            "--formal-scope-id",
            choices=(E5_SCOPE27_ID,),
            default=None,
        )

    model_smoke_parser = sub.add_parser("model-smoke")
    model_smoke_parser.add_argument("--model", required=True)
    model_smoke_parser.add_argument("--output-root")
    add_experiment_profile(model_smoke_parser)
    add_training_profile(model_smoke_parser)
    full_model_smoke = sub.add_parser("full-shape-model-smoke")
    full_model_smoke.add_argument("--model", required=True)
    full_model_smoke.add_argument("--output-root")
    add_experiment_profile(full_model_smoke)
    add_training_profile(full_model_smoke)
    real_model_smoke = sub.add_parser("real-data-model-smoke")
    real_model_smoke.add_argument("--model", required=True)
    real_model_smoke.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    real_model_smoke.add_argument("--target-path", default=str(DEFAULT_TARGET_PATH))
    real_model_smoke.add_argument("--attempt-tag", default="")
    real_model_smoke.add_argument("--output-root")
    add_experiment_profile(real_model_smoke)
    add_training_profile(real_model_smoke)
    evaluate = sub.add_parser("evaluate-only")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    evaluate.add_argument("--target-path", default=str(DEFAULT_TARGET_PATH))
    evaluate.add_argument("--output-root", default=str(FORMAL_ROOT / "basic_lightweight_seed2026"))
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--device", default="cpu")
    add_experiment_profile(evaluate)
    add_training_profile(evaluate)
    add_formal_scope(evaluate)
    train = sub.add_parser("train")
    train.add_argument("--model", required=True)
    train.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    train.add_argument("--target-path", default=str(DEFAULT_TARGET_PATH))
    train.add_argument("--output-root", default=str(FORMAL_ROOT / "basic_lightweight_seed2026"))
    train.add_argument("--run-id")
    train.add_argument("--device", default="cuda")
    train.add_argument("--preflight-root")
    add_experiment_profile(train)
    add_training_profile(train)
    add_formal_scope(train)
    preflight = sub.add_parser("hardware-preflight")
    preflight.add_argument("--model", required=True)
    preflight.add_argument("--preflight-root")
    add_experiment_profile(preflight)
    add_training_profile(preflight)
    preflight_worker = sub.add_parser("_hardware-preflight-worker")
    preflight_worker.add_argument("--model", required=True)
    preflight_worker.add_argument("--preflight-root")
    add_experiment_profile(preflight_worker)
    add_training_profile(preflight_worker)
    train_worker = sub.add_parser("_formal-train-worker")
    train_worker.add_argument("--model", required=True)
    train_worker.add_argument("--input-path", required=True)
    train_worker.add_argument("--target-path", required=True)
    train_worker.add_argument("--output-root", required=True)
    train_worker.add_argument("--run-id", required=True)
    train_worker.add_argument("--device", default="cuda")
    train_worker.add_argument("--preflight-root")
    add_experiment_profile(train_worker)
    add_training_profile(train_worker)
    add_formal_scope(train_worker)
    loss_profile_check = sub.add_parser("loss-profile-check")
    add_experiment_profile(loss_profile_check)
    readiness = sub.add_parser("e5-readiness")
    readiness.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / FORMAL_OUTPUT_ROOT_RELATIVE),
    )
    add_training_profile(readiness)
    readiness.add_argument(
        "--report-path",
        default=str(
            PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/E5/E5_RESULT_READINESS.json"
        ),
    )
    aggregate = sub.add_parser("e5-aggregate")
    aggregate.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / FORMAL_OUTPUT_ROOT_RELATIVE),
    )
    aggregate.add_argument("--require-complete", action="store_true")
    add_training_profile(aggregate)
    a8_reference = sub.add_parser("e5-reference-a8")
    a8_reference.add_argument(
        "--output-path",
        default=str(
            PROJECT_ROOT
            / "custom_models/docs/benchmark_v2/E5/E5_A8_REFERENCE.json"
        ),
    )
    add_training_profile(a8_reference)
    args = parser.parse_args(argv)
    try:
        if args.command == "registry-list": print(json.dumps([e.to_dict() for e in load_registry().list()], ensure_ascii=False, indent=2)); return 0
        if args.command == "registry-show": print(json.dumps(load_registry().get(args.model).to_dict(), ensure_ascii=False, indent=2)); return 0
        if args.command == "protocol-show": print(json.dumps(load_protocol().to_dict(), ensure_ascii=False, indent=2)); return 0
        if args.command == "protocol-check": print(json.dumps(check_protocol(load_protocol(), mode="formal"), ensure_ascii=False, indent=2)); return 0
        if args.command == "graph-protocol-check":
            from .graph import graph_protocol_check

            print(
                json.dumps(
                    graph_protocol_check(), ensure_ascii=False, indent=2
                )
            )
            return 0
        if args.command == "artifact-validate": print(json.dumps(validate_run(args.run_dir, expected_protocol_hash=load_protocol().protocol_hash), ensure_ascii=False, indent=2)); return 0
        if args.command == "framework-smoke": print(json.dumps(framework_smoke(), ensure_ascii=False, indent=2)); return 0
        if args.command == "full-shape-framework-smoke": print(json.dumps(full_shape_framework_smoke(), ensure_ascii=False, indent=2)); return 0
        if args.command == "data-contract-smoke": print(json.dumps(data_contract_smoke(), ensure_ascii=False, indent=2)); return 0
        if args.command == "model-smoke": print(json.dumps(model_smoke(args.model, root=None if args.output_root is None else Path(args.output_root), experiment_profile=args.experiment_profile, training_profile=args.training_profile), ensure_ascii=False, indent=2)); return 0
        if args.command == "full-shape-model-smoke": print(json.dumps(full_shape_model_smoke(args.model, root=None if args.output_root is None else Path(args.output_root), experiment_profile=args.experiment_profile, training_profile=args.training_profile), ensure_ascii=False, indent=2)); return 0
        if args.command == "real-data-model-smoke": print(json.dumps(real_data_model_smoke(args.model, input_path=args.input_path, target_path=args.target_path, root=None if args.output_root is None else Path(args.output_root), attempt_tag=args.attempt_tag, experiment_profile=args.experiment_profile, training_profile=args.training_profile), ensure_ascii=False, indent=2)); return 0
        if args.command == "evaluate-only": print(json.dumps(formal_evaluate_only(args.model, input_path=args.input_path, target_path=args.target_path, output_root=args.output_root, run_id=args.run_id, device=args.device, experiment_profile=args.experiment_profile, training_profile=args.training_profile, formal_scope_id=args.formal_scope_id), ensure_ascii=False, indent=2)); return 0
        if args.command == "loss-profile-check":
            print(json.dumps(check_loss_profile(args.experiment_profile or E5_PROFILE_ID), ensure_ascii=False, indent=2))
            return 0
        if args.command == "e5-readiness":
            report = build_readiness(output_root=args.output_root, report_path=args.report_path, training_profile=args.training_profile)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "e5-aggregate":
            print(json.dumps(e5_aggregate(output_root=args.output_root, require_complete=args.require_complete, training_profile=args.training_profile), ensure_ascii=False, indent=2))
            return 0
        if args.command == "e5-reference-a8":
            reference = create_a8_reference(args.output_path, training_profile=args.training_profile)
            print(json.dumps(reference, ensure_ascii=False, indent=2))
            return 0 if reference.get("status") == "VALID" else 4
        if args.command == "hardware-preflight":
            return launch_preflight(
                args.model,
                root=args.preflight_root,
                experiment_profile=args.experiment_profile,
                training_profile=args.training_profile,
            )
        if args.command == "_hardware-preflight-worker":
            result = run_preflight_worker(
                args.model,
                root=args.preflight_root,
                experiment_profile=args.experiment_profile,
                training_profile=args.training_profile,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 3
        if args.command == "_formal-train-worker":
            print(json.dumps(formal_train(
                args.model,
                input_path=args.input_path,
                target_path=args.target_path,
                output_root=args.output_root,
                run_id=args.run_id,
                device=args.device,
                preflight_root=args.preflight_root,
                experiment_profile=args.experiment_profile,
                training_profile=args.training_profile,
                formal_scope_id=args.formal_scope_id,
            ), ensure_ascii=False, indent=2))
            return 0
        if args.command == "train":
            entry = load_registry().get(args.model)
            if entry.supports_train and not args.run_id:
                raise ValueError(
                    "--run-id is required for formal trainable-model training."
                )
            return launch_formal_train(
                args.model,
                input_path=args.input_path,
                target_path=args.target_path,
                output_root=args.output_root,
                run_id=args.run_id or "",
                device=args.device,
                preflight_root=args.preflight_root,
                experiment_profile=args.experiment_profile,
                training_profile=args.training_profile,
                formal_scope_id=args.formal_scope_id,
            )
    except (BenchmarkV2Error, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
