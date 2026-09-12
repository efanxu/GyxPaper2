from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from st_mgprompt.config import STMGPromptConfig
from st_mgprompt.data import make_dataloaders
from st_mgprompt.experiment_protocol import apply_variant, assert_expected_diff, write_json
from st_mgprompt.losses import get_loss_fn
from st_mgprompt.registry import build_model
from st_mgprompt.run_st_mgprompt import (
    _autocast_context,
    _build_graph_data,
    _make_grad_scaler,
    _set_model_vadsp_statistics,
    resolve_device,
    set_seed,
    validate_batch_shapes,
    validate_model_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_VARIANTS = (
    "P1", "P2", "P4", "P5",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7",
)


def _preflight_autocast_context(
    device: torch.device,
    amp_enabled: bool,
    cpu_autocast_bfloat16: bool,
):
    if device.type == "cpu" and cpu_autocast_bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return _autocast_context(device, amp_enabled)


def run(
    variants: list[str],
    device_name: str = "auto",
    cpu_autocast_bfloat16: bool = False,
) -> dict:
    base = apply_variant(STMGPromptConfig(), "P1", "precision")
    base.device = device_name
    base.diagnostics_level = "minimal"
    base.prediction_accumulation = "streaming"
    base.validate()
    set_seed(base.seed)
    data = make_dataloaders(base)["bundle"]
    train_batch = next(iter(data.train_loader))
    val_batch = next(iter(data.val_loader))
    validate_batch_shapes(train_batch, 144, 10)
    expected = (32, 144, 134, 16)
    if tuple(train_batch["x"].shape) != expected:
        raise RuntimeError(f"Formal full-shape batch must be {expected}, got {tuple(train_batch['x'].shape)}")
    graph_data = _build_graph_data(base, data)
    device = resolve_device(device_name)
    results = []
    for name in variants:
        family = "precision" if name.startswith("P") else "component_ablation"
        cfg = apply_variant(STMGPromptConfig(), name, family)
        cfg.device = device_name
        cfg.diagnostics_level = "minimal"
        cfg.prediction_accumulation = "streaming"
        cfg.validate()
        assert_expected_diff(cfg, name, family)
        set_seed(cfg.seed)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(cfg, data.input_dim, graph_data=graph_data).to(device)
        _set_model_vadsp_statistics(cfg, model, data)
        loss_fn = get_loss_fn(cfg.loss_function, config=cfg, num_nodes=data.num_nodes)
        if hasattr(loss_fn, "to"):
            loss_fn = loss_fn.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = _make_grad_scaler(bool(cfg.amp_enabled))
        x = train_batch["x"].to(device).float()
        y = train_batch["y"].to(device).float()
        mask = train_batch["valid_target_mask"].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with _preflight_autocast_context(
            device,
            bool(cfg.amp_enabled),
            cpu_autocast_bfloat16,
        ):
            output = model(x)
        validate_model_output(output, train_batch | {"y": y})
        with torch.autocast(device_type=device.type, enabled=False):
            loss = loss_fn(output["pred"].float(), y.float(), mask.float())
        if loss is None or not torch.isfinite(loss).all().item():
            raise RuntimeError(f"{name} produced an invalid loss.")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        finite_grad = any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        model.eval()
        with torch.inference_mode():
            vx = val_batch["x"].to(device).float()
            with _preflight_autocast_context(
                device,
                bool(cfg.amp_enabled),
                cpu_autocast_bfloat16,
            ):
                val_output = model(vx)
            expected_val = (val_batch["x"].shape[0], 10, 134)
            if tuple(val_output["pred"].shape) != expected_val:
                raise RuntimeError(f"{name} evaluation shape {tuple(val_output['pred'].shape)} != {expected_val}")
        peak_mib = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else None
        )
        results.append(
            {
                "variant": name,
                "status": "passed",
                "device": str(device),
                "autocast_dtype": (
                    "bfloat16"
                    if device.type == "cpu" and cpu_autocast_bfloat16
                    else ("float16" if device.type == "cuda" and cfg.amp_enabled else "float32")
                ),
                "train_x_shape": list(train_batch["x"].shape),
                "train_y_shape": list(train_batch["y"].shape),
                "prediction_shape": list(output["pred"].shape),
                "evaluation_shape": list(val_output["pred"].shape),
                "loss": float(loss.detach().cpu()),
                "loss_finite": True,
                "gradient_finite": finite_grad,
                "optimizer_step": True,
                "hidden_dim": cfg.hidden_dim,
                "num_coupling_layers": cfg.num_coupling_layers,
                "graph_operator": cfg.graph_operator,
                "decoder_context_mode": cfg.decoder_context_mode,
                "macro_prompt_len": cfg.macro_prompt_len,
                "macro_prompt_pooling": cfg.macro_prompt_pooling,
                "cross_fusion_gate_strategy": cfg.cross_fusion_gate_strategy,
                "st_prompt_mode": cfg.st_prompt_mode,
                "st_prompt_use_node_identity": bool(cfg.st_prompt_use_node_identity),
                "loss_name": cfg.loss_function,
                "peak_allocated_mib": peak_mib,
            }
        )
        del model, loss_fn, optimizer, x, y, mask, output, loss, vx, val_output
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "passed": len(results) == len(variants) and all(item["status"] == "passed" for item in results),
        "formal_shape": {"B": 32, "N": 134, "C": 16, "lookback": 144, "max_pred_len": 10},
        "variants": results,
    }
    output = (
        PROJECT_ROOT
        / "custom_models/results_smoke/st_mgprompt_fixed_dual_refactor_full_shape"
        / "full_shape_smoke_summary.json"
    )
    merged = report
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        by_variant = {item["variant"]: item for item in previous.get("variants", [])}
        by_variant.update({item["variant"]: item for item in results})
        missing_required_variants = [name for name in REQUIRED_VARIANTS if name not in by_variant]
        merged = {
            "passed": not missing_required_variants
            and all(by_variant[name].get("status") == "passed" for name in REQUIRED_VARIANTS),
            "formal_shape": report["formal_shape"],
            "required_variants": list(REQUIRED_VARIANTS),
            "missing_required_variants": missing_required_variants,
            "variants": [by_variant[name] for name in sorted(by_variant)],
        }
    write_json(output, merged)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fixed-Dual formal full-shape variant matrix smoke.")
    parser.add_argument("--variants", nargs="+", default=list(REQUIRED_VARIANTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cpu-autocast-bfloat16",
        action="store_true",
        help="Use CPU bfloat16 autocast for the isolated full-shape preflight.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                [name.upper() for name in args.variants],
                args.device,
                cpu_autocast_bfloat16=args.cpu_autocast_bfloat16,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
