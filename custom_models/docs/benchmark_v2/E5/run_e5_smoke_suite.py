from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "custom_models/src"
RUNNER = SOURCE_ROOT / "benchmark_v2/run_benchmark.py"
SMOKE_ROOT = (
    PROJECT_ROOT / "custom_models/results_smoke/benchmark_v2/e5_common_loss"
)
E5_PROTOCOL = (
    SOURCE_ROOT
    / "benchmark_v2/experiments/e5_common_loss/e5_common_loss_protocol_v1.json"
)
LOSS_ID = json.loads(E5_PROTOCOL.read_text(encoding="utf-8"))["loss_id"]
TRAINABLE = (
    "gru dlinear lightts tide segrnn transformer patchtst itransformer timexer "
    "timesnet micn wpmixer multipatchformer timemixer tsmixer frets crossformer "
    "msgnet timefilter gcn stgcn dcrnn graph_wavenet mtgnn agcrn stid"
).split()
NONTRAINABLE = ["persistence", "moving_average"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordinary_existing(model_id: str) -> dict | None:
    for status_path in SMOKE_ROOT.rglob("run_status.json"):
        if status_path.parent.name != model_id:
            continue
        effective_path = status_path.parent / "effective_config.json"
        if not effective_path.is_file():
            continue
        status, effective = _load(status_path), _load(effective_path)
        if (
            status.get("status") == "COMPLETED"
            and effective.get("experiment_profile_id")
            == "e5_common_loss_architecture_v1"
        ):
            parity_path = status_path.parent / "initialization_parity.json"
            return {
                "status": "PASS",
                "source": "existing_validated_artifact",
                "run_dir": str(status_path.parent),
                "initialization_parity": (
                    _load(parity_path) if parity_path.is_file() else None
                ),
            }
    return None


def _existing(mode: str, model_id: str) -> dict | None:
    if mode == "ordinary":
        return _ordinary_existing(model_id)
    if mode == "full_shape":
        path = SMOKE_ROOT / "full_shape" / f"{model_id}.json"
    else:
        candidates = list((SMOKE_ROOT / "real_data").glob(f"{model_id}*/run_status.json"))
        for path in candidates:
            if _load(path).get("status") == "COMPLETED":
                return {
                    "status": "PASS",
                    "source": "existing_validated_artifact",
                    "run_dir": str(path.parent),
                }
        return None
    if path.is_file():
        payload = _load(path)
        if payload.get("status") in {"PASS", "FAIL_OOM"}:
            accepted = payload.get("status") == "PASS" or mode == "full_shape"
            return {
                "source": "existing_validated_artifact",
                **payload,
                "loss_id": LOSS_ID,
                "accepted": accepted,
                "e5_variant_status": (
                    "E5_AVAILABLE_TRAINABLE"
                    if payload.get("status") == "PASS"
                    else "E5_AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED"
                ),
            }
    return None


def _command(mode: str, model_id: str) -> list[str]:
    command_by_mode = {
        "ordinary": "model-smoke",
        "full_shape": "full-shape-model-smoke",
        "real_data": "real-data-model-smoke",
    }
    command = [
        str(Path(sys.executable).resolve()),
        str(RUNNER),
        command_by_mode[mode],
        "--model",
        model_id,
        "--experiment-profile",
        "e5_common_loss_v1",
    ]
    if mode == "real_data":
        existing = list((SMOKE_ROOT / "real_data").glob(f"{model_id}*"))
        if existing:
            command.extend(["--attempt-tag", f"e5_retry{len(existing)}"])
    return command


def _run(mode: str, model_id: str) -> dict:
    existing = _existing(mode, model_id)
    if existing is not None:
        return {"model_id": model_id, **existing}
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(SOURCE_ROOT),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    completed = subprocess.run(
        _command(mode, model_id),
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "FAIL",
            "stdout_tail": completed.stdout.splitlines()[-50:],
        }
    payload.update(
        {
            "model_id": model_id,
            "exit_code": completed.returncode,
            "stderr_tail": completed.stderr.splitlines()[-50:],
            "source": "executed_independent_process",
            "loss_id": LOSS_ID,
        }
    )
    raw_status = payload.get("status")
    accepted = completed.returncode == 0 and (
        raw_status == "PASS" or (mode == "full_shape" and raw_status == "FAIL_OOM")
    )
    if completed.returncode != 0:
        payload["status"] = "FAIL"
    payload["accepted"] = accepted
    payload["e5_variant_status"] = (
        "E5_AVAILABLE_TRAINABLE"
        if raw_status == "PASS"
        else (
            "E5_AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED"
            if mode == "full_shape" and raw_status == "FAIL_OOM"
            else "E5_BLOCKED_NON_OOM"
        )
    )
    return payload


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("ordinary", "full_shape", "real_data"), required=True
    )
    args = parser.parse_args()
    models = TRAINABLE + (NONTRAINABLE if args.mode == "ordinary" else [])
    results = []
    for index, model_id in enumerate(models, start=1):
        print(f"[{index}/{len(models)}] {args.mode}: {model_id}", flush=True)
        result = _run(args.mode, model_id)
        results.append(result)
        print(
            f"  status={result.get('status')} source={result.get('source')}",
            flush=True,
        )
        if not result.get("accepted", result.get("status") == "PASS"):
            break
    payload = {
        "schema_version": "e5_smoke_suite_v1",
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_models": len(models),
        "completed_models": len(results),
        "passed_models": sum(item.get("status") == "PASS" for item in results),
        "oom_routed_models": sum(
            item.get("status") == "FAIL_OOM" for item in results
        ),
        "accepted_models": sum(
            item.get("accepted", item.get("status") == "PASS")
            for item in results
        ),
        "status": (
            (
                "PASS_WITH_OOM_ROUTING"
                if any(item.get("status") == "FAIL_OOM" for item in results)
                else "PASS"
            )
            if len(results) == len(models)
            and all(
                item.get("accepted", item.get("status") == "PASS")
                for item in results
            )
            else "FAIL"
        ),
        "one_model_per_independent_process": True,
        "formal_training_started": False,
        "results": results,
    }
    output_name = {
        "ordinary": "model_smoke_results.json",
        "full_shape": "full_shape_model_smoke_results.json",
        "real_data": "real_data_smoke_results.json",
    }[args.mode]
    _write(DOC_ROOT / output_name, payload)
    if args.mode == "ordinary":
        _write(
            DOC_ROOT / "nontrainable_smoke_results.json",
            {
                **payload,
                "expected_models": 2,
                "completed_models": sum(
                    item["model_id"] in NONTRAINABLE for item in results
                ),
                "passed_models": sum(
                    item["model_id"] in NONTRAINABLE
                    and item.get("status") == "PASS"
                    for item in results
                ),
                "results": [
                    item for item in results if item["model_id"] in NONTRAINABLE
                ],
            },
        )
    print(json.dumps({"status": payload["status"], "output": str(DOC_ROOT / output_name)}))
    return 0 if payload["status"] in {"PASS", "PASS_WITH_OOM_ROUTING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
