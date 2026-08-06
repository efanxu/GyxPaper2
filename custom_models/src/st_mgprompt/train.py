from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import STMGPromptConfig
from .data import STMGPromptDataBundle
from .diagnostics import save_loss_diagnostics, validate_model_output
from .graph_prior import prepare_graph_artifacts
from .losses import get_loss_fn
from .metrics import evaluate_prefix_horizons
from .registry import build_model


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_details(loss_fn) -> dict[str, float]:
    details = getattr(loss_fn, "last_details", None)
    if not details:
        return {}
    return {key: value for key, value in details.items() if isinstance(value, (int, float))}


STRICT_RESUME_KEYS = [
    "model_name",
    "hidden_dim",
    "num_coupling_layers",
    "graph_operator",
    "decoder_context_mode",
    "lookback",
    "max_pred_len",
    "feature_cols",
    "split_ratios",
    "seed",
    "loss_function",
    "granularity_weight_mode",
    "site_weight_mode",
    "vadsp_gate_mode",
]


def _training_batch_identity(config: STMGPromptConfig) -> dict:
    keys = (
        "training_batch_profile_id",
        "train_batch_size",
        "val_batch_size",
        "test_batch_size",
        "gradient_accumulation_steps",
        "effective_train_batch_size",
    )
    return {
        key: getattr(config, key)
        for key in keys
        if hasattr(config, key)
    }


def _atomic_torch_save(payload: dict, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _rng_state_payload() -> dict:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(checkpoint: dict) -> None:
    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("numpy_random_state") is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
    if checkpoint.get("torch_random_state") is not None:
        torch_state = checkpoint["torch_random_state"]
        if hasattr(torch_state, "cpu"):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state_all") is not None:
        cuda_states = []
        for state in checkpoint["cuda_random_state_all"]:
            cuda_states.append(state.cpu() if hasattr(state, "cpu") else state)
        torch.cuda.set_rng_state_all(cuda_states)


def _strict_resume_differences(config: STMGPromptConfig, checkpoint: dict) -> dict[str, dict[str, object]]:
    saved = checkpoint.get("config") or {}
    current = config.to_dict()
    diffs: dict[str, dict[str, object]] = {}
    for key in STRICT_RESUME_KEYS:
        if saved.get(key) != current.get(key):
            diffs[key] = {"checkpoint": saved.get(key), "current": current.get(key)}
    return diffs


def _make_grad_scaler(enabled: bool):
    if not (enabled and torch.cuda.is_available()):
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def _autocast_context(device: torch.device, enabled: bool):
    if not (enabled and device.type == "cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def cuda_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "current_allocated": None,
            "current_reserved": None,
            "peak_allocated": None,
            "peak_reserved": None,
        }
    return {
        "current_allocated": int(torch.cuda.memory_allocated(device)),
        "current_reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _log_cuda_memory(stage: str, device: torch.device, epoch: int | None = None, batch_index: int | None = None) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    payload = {
        "memory_event": stage,
        "memory_unit": "bytes",
        "epoch": epoch,
        "batch_index": batch_index,
        **cuda_memory_snapshot(device),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _attention_diagnostics(model) -> dict:
    blocks = getattr(model, "coupling_blocks", None)
    if not blocks:
        return {}
    fusion = getattr(blocks[-1], "symmetric_cross_fusion", None)
    return dict(getattr(fusion, "last_attention_diagnostics", {}) or {})


@torch.no_grad()
def _fit_vadsp_robust_statistics(model, data: STMGPromptDataBundle) -> None:
    vadsp = getattr(model, "vadsp", None)
    if vadsp is None:
        return
    if not getattr(vadsp, "requires_robust_statistics", True):
        return
    if not hasattr(vadsp, "set_robust_statistics"):
        raise RuntimeError("VADSP module does not expose set_robust_statistics.")
    if data.vadsp_robust_statistics is None:
        raise RuntimeError(
            "VADSP robust statistics are missing. Formal training must fit them from the unique train timeline, "
            "not from sliding-window train_loader batches."
        )
    vadsp.set_robust_statistics(data.vadsp_robust_statistics)
    shape = data.vadsp_robust_statistics["shape"]
    print(
        "[VADSP] fitted robust statistics from unique train timeline: "
        f"T={data.vadsp_robust_statistics['num_time_points']}, "
        f"N={data.vadsp_robust_statistics['num_nodes']}, "
        f"K={data.vadsp_robust_statistics['num_sources']}, "
        f"mode={data.vadsp_robust_statistics['mode']}, "
        f"center_shape={tuple(shape['center'])}, scale_shape={tuple(shape['scale'])}"
    )


@torch.inference_mode()
def _validation_metrics(model, loader, target_scaler, device, config: STMGPromptConfig) -> dict[str, float]:
    model.eval()
    pred_parts = []
    y_parts = []
    mask_parts = []
    for batch in loader:
        with _autocast_context(device, bool(config.amp_enabled)):
            out = model(batch["x"].to(device).float())
        pred_norm = out["pred"].detach().float().cpu().numpy()
        pred_kw = target_scaler.inverse_transform(pred_norm)
        if config.enable_physical_clip_eval:
            pred_kw = np.clip(pred_kw, config.physical_power_min_kw, config.physical_power_max_kw)
        pred_parts.append(pred_kw)
        y_parts.append(batch["y_raw"].numpy())
        mask_parts.append(batch["valid_target_mask"].numpy())
    rows = evaluate_prefix_horizons(
        np.concatenate(pred_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(mask_parts, axis=0),
        config.eval_horizons,
        num_nodes=config.num_nodes,
        physical_clip_applied=bool(config.enable_physical_clip_eval),
    )
    metrics: dict[str, float] = {}
    for row in rows:
        horizon = int(row["horizon"])
        metrics[f"val_MAE_H{horizon}"] = float(row["MAE"])
        metrics[f"val_RMSE_H{horizon}"] = float(row["RMSE"])
        metrics[f"val_R2_H{horizon}"] = float(row["R2"])
        metrics[f"val_Score_H{horizon}"] = float(row["Score"])
        metrics[f"val_official_score_h{horizon}"] = float(row["official_align_score"])
    metrics["val_score_primary"] = metrics[f"val_official_score_h{config.primary_val_horizon}"]
    return metrics


def _run_epoch(
    model,
    loader,
    device,
    loss_fn,
    optimizer=None,
    gradient_clip_val: float = 5.0,
    scaler=None,
    amp_enabled: bool = False,
    epoch: int | None = None,
) -> tuple[float, int, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    if hasattr(loss_fn, "train"):
        loss_fn.train(is_train)
    total_loss = 0.0
    counted_batches = 0
    skipped = 0
    detail_sums: dict[str, float] = {}
    detail_counts: dict[str, int] = {}
    stage = "train" if is_train else "validation"
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    _log_cuda_memory(f"{stage}_start", device, epoch=epoch)
    for batch_index, batch in enumerate(loader):
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        mask = batch["valid_target_mask"].to(device).float()
        if mask.sum().item() <= 0:
            skipped += 1
            continue
        try:
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, amp_enabled):
                    forward_grad_enabled = torch.is_grad_enabled()
                    out = model(x)
                validate_model_output(out, batch | {"y": y})
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = loss_fn(out["pred"].float(), y.float(), mask.float())
            else:
                with torch.inference_mode():
                    with _autocast_context(device, amp_enabled):
                        forward_grad_enabled = torch.is_grad_enabled()
                        out = model(x)
                    validate_model_output(out, batch | {"y": y})
                    with torch.autocast(device_type=device.type, enabled=False):
                        loss = loss_fn(out["pred"].float(), y.float(), mask.float())
        except torch.OutOfMemoryError as exc:
            exc.stmg_oom_context = {
                "stage": stage,
                "epoch": epoch,
                "batch_index": batch_index,
                "input_shape": list(x.shape),
                "target_shape": list(y.shape),
                "mask_shape": list(mask.shape),
                "amp_enabled": bool(amp_enabled),
                "grad_enabled": bool(forward_grad_enabled),
                "attention_shapes": _attention_diagnostics(model),
                **cuda_memory_snapshot(device),
            }
            raise
        if batch_index == 0:
            _log_cuda_memory(f"{stage}_first_batch_forward_finished", device, epoch=epoch, batch_index=batch_index)
        if loss is None:
            skipped += 1
            continue
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError("Loss produced NaN or Inf.")
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if gradient_clip_val and gradient_clip_val > 0:
                clip_params = list(model.parameters())
                if isinstance(loss_fn, torch.nn.Module):
                    clip_params.extend(list(loss_fn.parameters()))
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, gradient_clip_val)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if batch_index == 0:
                _log_cuda_memory("train_first_batch_backward_finished", device, epoch=epoch, batch_index=batch_index)
        total_loss += float(loss.detach().cpu().item())
        counted_batches += 1
        for key, value in _loss_details(loss_fn).items():
            if np.isfinite(value):
                detail_sums[key] = detail_sums.get(key, 0.0) + float(value)
                detail_counts[key] = detail_counts.get(key, 0) + 1
        del out, loss, x, y, mask
    _log_cuda_memory(f"{stage}_end", device, epoch=epoch)
    mean_loss = total_loss / max(counted_batches, 1)
    details = {key: detail_sums[key] / max(detail_counts[key], 1) for key in detail_sums}
    details["total_loss"] = mean_loss
    return mean_loss, skipped, details


def save_train_log(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_MAE_H3",
        "val_RMSE_H3",
        "val_R2_H3",
        "val_Score_H3",
        "val_MAE_H6",
        "val_RMSE_H6",
        "val_R2_H6",
        "val_Score_H6",
        "val_MAE_H10",
        "val_RMSE_H10",
        "val_R2_H10",
        "val_Score_H10",
        "val_score_primary",
        "is_best_epoch",
        "bad_epochs",
        "learning_rate",
        "epoch_time_sec",
        "total_loss",
        "granularity_loss_h3",
        "granularity_loss_h6",
        "granularity_loss_h10",
        "granularity_weight_h3",
        "granularity_weight_h6",
        "granularity_weight_h10",
        "site_weight_mean",
        "site_weight_max",
        "site_weight_min",
        "skipped_train_batches_epoch",
        "skipped_val_batches_epoch",
        "skipped_train_batches_total",
        "skipped_val_batches_total",
        "skipped_batches_total_run",
    ]
    extra_fieldnames = sorted({key for row in rows for key in row if key not in base_fieldnames})
    fieldnames = base_fieldnames + extra_fieldnames
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_train_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_model(
    config: STMGPromptConfig,
    data: STMGPromptDataBundle,
    run_dir: Path,
    resume_from: str | Path | None = None,
    allow_warm_start_only: bool = False,
):
    set_seed(config.seed)
    device = resolve_device(config.device)
    graph_data = None
    if (
        config.use_trend_prior_graph
        or config.use_graph_temporal_encoder
        or config.use_stmg_coupling_block
        or config.model_name in {
            "STMGPrompt_TrendPriorGraph",
            "STMGPrompt_GraphTemporalSmoke",
            "STMGPrompt_FairFull",
            "STMGPrompt_FairFull_Diffusion",
            "STMGPrompt_FairFull_HistoryDecoder",
            "STMGPrompt_FairFull_DiffusionHistory",
            "STMGPrompt_CouplingCrossFusion",
            "STMGPrompt_Full_MSMGDWU",
            "STMGPrompt_Full_MSMGDWU_DiffusionHistory",
        }
    ):
        graph_data = prepare_graph_artifacts(data, config)
    model = build_model(config, data.input_dim, graph_data=graph_data).to(device)
    _fit_vadsp_robust_statistics(model, data)
    loss_fn = get_loss_fn(config.loss_function, config=config, num_nodes=data.num_nodes)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    optim_params = list(model.parameters())
    if isinstance(loss_fn, torch.nn.Module):
        optim_params.extend(list(loss_fn.parameters()))
    optimizer = torch.optim.Adam(optim_params, lr=config.lr, weight_decay=config.weight_decay)
    scaler = _make_grad_scaler(bool(config.amp_enabled))
    checkpoint_path = run_dir / "best_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    train_log_path = run_dir / "train_log.csv"
    best_val = float("inf")
    best_epoch = 0
    best_scores: dict[str, float] = {}
    bad_epochs = 0
    logs: list[dict] = []
    skipped_train_total = 0
    skipped_val_total = 0
    start_epoch = 1

    resume_checkpoint_path = Path(resume_from) if resume_from else None
    if resume_checkpoint_path is not None:
        if str(resume_checkpoint_path).lower() == "auto":
            resume_checkpoint_path = last_checkpoint_path
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        diffs = _strict_resume_differences(config, checkpoint)
        if diffs and not allow_warm_start_only:
            print(json.dumps({"resume_rejected": True, "differences": diffs}, indent=2, ensure_ascii=False), flush=True)
            raise RuntimeError("Strict resume rejected because checkpoint config differs from current config.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if isinstance(loss_fn, torch.nn.Module) and checkpoint.get("loss_state_dict") is not None:
            loss_fn.load_state_dict(checkpoint["loss_state_dict"])
        if diffs and allow_warm_start_only:
            print(json.dumps({"warm_start_only": True, "differences": diffs}, indent=2, ensure_ascii=False), flush=True)
            start_epoch = 1
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scaler is not None and checkpoint.get("grad_scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            _restore_rng_state(checkpoint)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint.get("best_epoch") or 0)
            best_val = float(checkpoint.get("best_val_score_h10") or checkpoint.get("best_metric_value") or float("inf"))
            best_scores = {
                "val_official_score_h3": checkpoint.get("best_val_score_h3"),
                "val_official_score_h6": checkpoint.get("best_val_score_h6"),
                "val_official_score_h10": checkpoint.get("best_val_score_h10"),
            }
            best_scores = {k: float(v) for k, v in best_scores.items() if v is not None}
            bad_epochs = int(checkpoint.get("early_stopping_counter") or 0)
            skipped_train_total = int(checkpoint.get("skipped_train_batches_total") or 0)
            skipped_val_total = int(checkpoint.get("skipped_val_batches_total") or 0)
            logs = _load_existing_train_log(train_log_path)
            logs = [row for row in logs if int(row.get("epoch", 0) or 0) < start_epoch]
            print(
                f"resume_from={resume_checkpoint_path} start_epoch={start_epoch} "
                f"best_epoch={best_epoch} best_val_score_h10={best_val:.6f}",
                flush=True,
            )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, skipped_train, train_details = _run_epoch(
            model,
            data.train_loader,
            device,
            loss_fn,
            optimizer,
            config.gradient_clip_val,
            scaler=scaler,
            amp_enabled=bool(config.amp_enabled),
            epoch=epoch,
        )
        val_loss, skipped_val, val_details = _run_epoch(
            model,
            data.val_loader,
            device,
            loss_fn,
            None,
            scaler=None,
            amp_enabled=bool(config.amp_enabled),
            epoch=epoch,
        )
        val_metric_details = _validation_metrics(model, data.val_loader, data.scalers["target"], device, config)
        skipped_train_total += skipped_train
        skipped_val_total += skipped_val
        selection_value = float(val_metric_details["val_score_primary"])
        is_best = selection_value < (best_val - float(config.early_stopping_min_delta))
        epoch_time_sec = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metric_details,
            "is_best_epoch": bool(is_best),
            "bad_epochs": bad_epochs,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": epoch_time_sec,
            "total_loss": train_details.get("total_loss", train_loss),
            "skipped_train_batches_epoch": skipped_train,
            "skipped_val_batches_epoch": skipped_val,
            "skipped_train_batches_total": skipped_train_total,
            "skipped_val_batches_total": skipped_val_total,
            "skipped_batches_total_run": skipped_train_total + skipped_val_total,
        }
        for key, value in train_details.items():
            if key != "total_loss":
                row[key] = value
        for key, value in val_details.items():
            if key != "total_loss":
                row[f"val_{key}"] = value
        logs.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_score_h{config.primary_val_horizon}={selection_value:.6f}",
            flush=True,
        )
        if is_best:
            best_val = selection_value
            best_epoch = epoch
            best_scores = dict(val_metric_details)
            bad_epochs = 0
        else:
            bad_epochs += 1
            row["bad_epochs"] = bad_epochs
        checkpoint_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric_name": config.checkpoint_selection_metric,
            "best_metric_value": best_val,
            "best_val_score_h3": best_scores.get("val_official_score_h3"),
            "best_val_score_h6": best_scores.get("val_official_score_h6"),
            "best_val_score_h10": best_scores.get("val_official_score_h10"),
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss_fn.state_dict() if isinstance(loss_fn, torch.nn.Module) else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "early_stopping_counter": bad_epochs,
            "config": config.to_dict(),
            "batch_identity": _training_batch_identity(config),
            "input_dim": data.input_dim,
            "num_nodes": data.num_nodes,
            "feature_cols": data.feature_cols,
            "turbine_ids": data.turbine_ids,
            "scalers": {k: v.to_dict() for k, v in data.scalers.items()},
            "graph_metadata": graph_data.get("metadata", {}) if graph_data else {},
            "vadsp_robust_statistics_metadata": (
                dict(getattr(model.vadsp, "robust_statistics_metadata", {}))
                if getattr(model, "vadsp", None) is not None
                else None
            ),
            "skipped_batches_at_checkpoint": skipped_train_total + skipped_val_total,
            "skipped_train_batches_total": skipped_train_total,
            "skipped_val_batches_total": skipped_val_total,
            "amp_enabled": bool(config.amp_enabled),
            "amp_dtype": config.amp_dtype if config.amp_enabled else None,
            "checkpoint_selection": {
                "metric": config.checkpoint_selection_metric,
                "mode": config.checkpoint_selection_mode,
                "primary_val_horizon": config.primary_val_horizon,
                "single_checkpoint_for_all_horizons": True,
            },
            **_rng_state_payload(),
        }
        _atomic_torch_save(checkpoint_payload, last_checkpoint_path)
        if is_best:
            _atomic_torch_save(checkpoint_payload, checkpoint_path)
        else:
            if bad_epochs >= config.patience:
                print(f"early_stopping epoch={epoch}", flush=True)
                save_train_log(logs, train_log_path)
                break
        save_train_log(logs, train_log_path)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if isinstance(loss_fn, torch.nn.Module) and checkpoint.get("loss_state_dict") is not None:
            loss_fn.load_state_dict(checkpoint["loss_state_dict"])
    save_train_log(logs, train_log_path)
    save_loss_diagnostics(loss_fn, logs, run_dir / "diagnostics")
    train_complete = {
        "status": "completed",
        "last_epoch": int(logs[-1]["epoch"]) if logs else 0,
        "best_epoch": int(best_epoch),
        "best_val_score_h10": float(best_scores.get("val_official_score_h10", best_val)),
        "best_checkpoint": str(checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "amp_enabled": bool(config.amp_enabled),
        "amp_dtype": config.amp_dtype if config.amp_enabled else None,
    }
    (run_dir / "train_complete.json").write_text(
        json.dumps(train_complete, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model, logs, checkpoint_path, skipped_train_total + skipped_val_total
