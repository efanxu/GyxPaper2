from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "custom_models" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from st_mgprompt.config import STMGPromptConfig, resolve_project_path
from st_mgprompt.data import make_dataloaders
from st_mgprompt.metrics import regression_metrics_kw
from st_mgprompt.registry import build_model


INTERVENTIONS = (
    "horizon_zero",
    "horizon_swap",
    "horizon_mean",
    "node_zero",
    "node_swap",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _find_run_file(run: Path, name: str) -> Path:
    direct = run / name
    if direct.is_file():
        return direct
    matches = list(run.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} under {run}, found {len(matches)}.")
    return matches[0]


def _load_config(path: Path) -> STMGPromptConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(STMGPromptConfig)}
    config = STMGPromptConfig(**{key: value for key, value in payload.items() if key in allowed})
    config.validate()
    return config


@contextmanager
def temporary_prompt_parameter_intervention(prompt, intervention: str) -> Iterator[bool]:
    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention {intervention!r}.")
    parameter = None
    operation = intervention.rsplit("_", 1)[-1]
    if intervention.startswith("horizon_"):
        if prompt.future_step_embedding is not None:
            parameter = prompt.future_step_embedding.weight
        elif prompt.shared_horizon_embedding is not None:
            parameter = prompt.shared_horizon_embedding
    elif prompt.node_embedding is not None:
        parameter = prompt.node_embedding.weight
    if parameter is None:
        yield False
        return
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            if operation == "zero":
                parameter.zero_()
            elif operation == "swap":
                parameter.copy_(torch.roll(original, shifts=1, dims=0))
            elif operation == "mean":
                parameter.copy_(original.mean(dim=0, keepdim=True).expand_as(original))
        yield True
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def _modes(value: str) -> list[str]:
    requested = {item.strip() for item in value.split(",") if item.strip()}
    expanded: list[str] = []
    if "zero" in requested:
        expanded.extend(["horizon_zero", "node_zero"])
    if "swap" in requested:
        expanded.extend(["horizon_swap", "node_swap"])
    if "mean" in requested:
        expanded.append("horizon_mean")
    expanded.extend(item for item in INTERVENTIONS if item in requested)
    return list(dict.fromkeys(expanded))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference-only ST Prompt sensitivity interventions.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", default="zero,swap,mean")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="custom_models/results/empirical_analysis_v1/N/N_INTERVENTION_SUMMARY.csv",
    )
    return parser


def run_interventions(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    run = resolve_project_path(args.run)
    checkpoint_path = _find_run_file(run, "best_checkpoint.pt")
    config_path = _find_run_file(run, "effective_config.json")
    config = _load_config(config_path)
    config.device = args.device
    config.diagnostics_level = "minimal"
    device = _resolve_device(args.device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    graph = {
        "A_macro_trend": state["A_macro_prior"].detach().cpu().numpy().astype(np.float32),
        "A_micro_local": state["A_micro_prior"].detach().cpu().numpy().astype(np.float32),
        "metadata": {"graph_uses_train_only_statistics": True, "fit_split": "checkpoint_train_only"},
    }
    model = build_model(config, input_dim=len(config.feature_cols), graph_data=graph).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.st_prompt is None:
        raise ValueError("The selected run has no ST Prompt to intervene on.")

    data = make_dataloaders(config)
    scaler = data["bundle"].scalers["target"]
    batches = []
    for index, batch in enumerate(data[args.split]):
        if index >= args.batches:
            break
        batches.append(batch)
    x = torch.cat([batch["x"] for batch in batches], dim=0).to(device)
    target_raw = np.concatenate([batch["y_raw"].numpy() for batch in batches], axis=0)
    mask = np.concatenate([batch["valid_target_mask"].numpy() for batch in batches], axis=0)
    prediction_start_indices = [int(value) for batch in batches for value in batch["prediction_start_index"]]

    with torch.inference_mode():
        baseline_norm = model(x)["pred"].detach().cpu().numpy()
    baseline = scaler.inverse_transform(baseline_norm)
    if config.enable_physical_clip_eval:
        baseline = np.clip(baseline, config.physical_power_min_kw, config.physical_power_max_kw)

    rows: list[dict[str, object]] = []
    modes = _modes(args.mode)
    for intervention in modes:
        with temporary_prompt_parameter_intervention(model.st_prompt, intervention) as applied:
            if not applied:
                rows.append(
                    {
                        "variant_id": config.variant or config.model_name,
                        "status": "NOT_APPLICABLE",
                        "intervention_type": intervention,
                        "analysis_label": "internal_sensitivity_analysis",
                        "reason": "selected_prompt_component_is_absent",
                    }
                )
                continue
            with torch.inference_mode():
                changed_norm = model(x)["pred"].detach().cpu().numpy()
        changed = scaler.inverse_transform(changed_norm)
        if config.enable_physical_clip_eval:
            changed = np.clip(changed, config.physical_power_min_kw, config.physical_power_max_kw)
        difference = changed - baseline
        for horizon in config.eval_horizons:
            base_metric = regression_metrics_kw(
                baseline[:, :horizon], target_raw[:, :horizon], mask[:, :horizon], num_nodes=config.num_nodes,
                physical_clip_applied=config.enable_physical_clip_eval,
            )
            changed_metric = regression_metrics_kw(
                changed[:, :horizon], target_raw[:, :horizon], mask[:, :horizon], num_nodes=config.num_nodes,
                physical_clip_applied=config.enable_physical_clip_eval,
            )
            rows.append(
                {
                    "variant_id": config.variant or config.model_name,
                    "status": "completed",
                    "intervention_type": intervention,
                    "analysis_label": "internal_sensitivity_analysis",
                    "horizon": horizon,
                    "seed": config.seed,
                    "split": args.split,
                    "sample_selection_rule": "first_batches_in_chronological_order",
                    "sample_count": int(x.shape[0]),
                    "node_selection": "all_nodes",
                    "prediction_difference_mae_kw": float(np.mean(np.abs(difference[:, :horizon]))),
                    "prediction_difference_max_kw": float(np.max(np.abs(difference[:, :horizon]))),
                    "baseline_MAE": base_metric["MAE"],
                    "intervention_MAE": changed_metric["MAE"],
                    "MAE_delta": changed_metric["MAE"] - base_metric["MAE"],
                    "baseline_Score": base_metric["Score"],
                    "intervention_Score": changed_metric["Score"],
                    "Score_delta": changed_metric["Score"] - base_metric["Score"],
                    "reason": "",
                }
            )

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields_out = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "analysis_label": "internal_sensitivity_analysis",
        "real_world_counterfactual": False,
        "source_run": str(run.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "seed": config.seed,
        "split": args.split,
        "sample_selection_rule": "first_batches_in_chronological_order",
        "prediction_start_indices": prediction_start_indices,
        "node_selection": "all_nodes",
        "horizons": list(config.eval_horizons),
        "interventions": modes,
        "checkpoint_modified": False,
        "backpropagation_used": False,
        "formal_metrics_overwritten": False,
        "created_at": _utc_now(),
    }
    manifest_path = output.with_name("N_INTERVENTION_MANIFEST.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"summary": str(output.resolve()), "manifest": str(manifest_path.resolve()), "rows": len(rows)}


def main() -> None:
    print(json.dumps(run_interventions(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
