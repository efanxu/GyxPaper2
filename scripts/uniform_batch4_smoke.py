from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
DOC_ROOT = (
    PROJECT_ROOT
    / "custom_models/docs/benchmark_v2/UNIFORM_BATCH4_PROTOCOL"
)
PROFILE = "uniform_train_batch4_v1"
TRAINABLE = (
    "gru", "dlinear", "lightts", "tide", "segrnn", "transformer",
    "patchtst", "itransformer", "timexer", "timesnet", "micn", "wpmixer",
    "multipatchformer", "timemixer", "tsmixer", "frets", "crossformer",
    "msgnet", "timefilter", "gcn", "stgcn", "dcrnn", "graph_wavenet",
    "mtgnn", "agcrn", "stid",
)
ALL_MODELS = ("persistence", "moving_average", *TRAINABLE)


def run(command: list[str], log_path: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    payload = None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pass
    return {
        "exit_code": completed.returncode,
        "status": (
            payload.get("status", "PASS")
            if completed.returncode == 0 and isinstance(payload, dict)
            else "FAIL"
        ),
        "result": payload,
        "log": str(log_path),
        "stderr_tail": completed.stderr.splitlines()[-10:],
    }


def benchmark_suite(
    kind: str,
    *,
    selected_models: tuple[str, ...] | None = None,
    attempt_tag: str | None = None,
) -> dict:
    rows = []
    e5 = kind == "e5"
    real = kind == "real"
    models = ALL_MODELS if not real else TRAINABLE
    if selected_models is not None:
        unknown = sorted(set(selected_models) - set(models))
        if unknown:
            raise ValueError(f"Unknown models for {kind}: {unknown}")
        models = selected_models
    namespace = kind if attempt_tag is None else f"{kind}_{attempt_tag}"
    root = (
        PROJECT_ROOT
        / "custom_models/results_smoke/benchmark_v2_uniform_bs4"
        / namespace
    )
    for model in models:
        command = [
            sys.executable,
            str(SRC_ROOT / "benchmark_v2/run_benchmark.py"),
            "real-data-model-smoke" if real else "model-smoke",
            "--model",
            model,
            "--training-profile",
            PROFILE,
            "--output-root",
            str(root),
        ]
        if e5:
            command.extend(["--experiment-profile", "e5_common_loss_v1"])
        row = run(
            command,
            PROJECT_ROOT
            / f"custom_models/logs/uniform_bs4/smoke_{namespace}/{model}.log",
        )
        rows.append({"model_id": model, **row})
    return {
        "schema_version": "uniform_batch4_smoke_inventory_v1",
        "suite": kind,
        "attempt_tag": attempt_tag,
        "profile_id": PROFILE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "counts": {
            "PASS": sum(row["exit_code"] == 0 for row in rows),
            "FAIL": sum(row["exit_code"] != 0 for row in rows),
        },
    }


def st_suite() -> dict:
    rows = []
    for variant, run_id in (
        ("FULL", "ordinary_full_bs4_retry1"),
        ("A8", "ordinary_a8_bs4_retry1"),
    ):
        command = [
            sys.executable,
            str(SRC_ROOT / "st_mgprompt/run_st_mgprompt.py"),
            "--smoke",
            "--training-profile",
            PROFILE,
            "--run-id",
            run_id,
            "--output-root",
            "custom_models/results_smoke/st_mgprompt_uniform_bs4",
            "--windows-safe-mode",
            "--amp",
        ]
        if variant == "FULL":
            command.insert(3, "--canonical-full")
        else:
            command[3:3] = ["--experiment-variant", variant]
        row = run(
            command,
            PROJECT_ROOT
            / f"custom_models/logs/uniform_bs4/smoke_st/{variant}.log",
        )
        rows.append({"variant": variant, **row})
    return {
        "schema_version": "uniform_batch4_st_smoke_inventory_v1",
        "suite": "st_mgprompt",
        "profile_id": PROFILE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "counts": {
            "PASS": sum(row["exit_code"] == 0 for row in rows),
            "FAIL": sum(row["exit_code"] != 0 for row in rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "suite", choices=("original", "e5", "real", "st")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TRAINABLE,
        help="Run only the selected benchmark models.",
    )
    parser.add_argument(
        "--attempt-tag",
        help="Use a distinct output/log/report namespace for a retry.",
    )
    args = parser.parse_args()
    if args.suite == "st" and (args.models or args.attempt_tag):
        parser.error("--models/--attempt-tag are benchmark-suite options")
    report = (
        st_suite()
        if args.suite == "st"
        else benchmark_suite(
            args.suite,
            selected_models=(
                tuple(args.models) if args.models is not None else None
            ),
            attempt_tag=args.attempt_tag,
        )
    )
    target = {
        "original": "BATCH4_ORDINARY_SMOKE_RESULTS.json",
        "e5": "BATCH4_E5_ORDINARY_SMOKE_RESULTS.json",
        "real": "BATCH4_REAL_DATA_SMOKE_RESULTS.json",
        "st": "BATCH4_ST_MGPROMPT_ORDINARY_SMOKE_RESULTS.json",
    }[args.suite]
    if args.attempt_tag is not None:
        stem = Path(target).stem
        target = f"{stem}_{args.attempt_tag}.json"
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / target).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0 if report["counts"]["FAIL"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
