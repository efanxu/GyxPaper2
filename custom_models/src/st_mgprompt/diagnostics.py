from __future__ import annotations

import csv
import dataclasses
import struct
import zlib
import json
from pathlib import Path
from typing import Any

import torch
import numpy as np

from .data import STMGPromptDataBundle
from .volatility_patching import save_vadsp_diagnostics_from_aux


def _scalar_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def find_cuda_tensors(obj: Any, path: str = "root") -> list[dict[str, Any]]:
    """Return paths and metadata for CUDA tensors reachable from a result object."""
    found: list[dict[str, Any]] = []
    if torch.is_tensor(obj):
        if obj.is_cuda:
            found.append(
                {
                    "path": path,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "requires_grad": bool(obj.requires_grad),
                    "has_grad_fn": obj.grad_fn is not None,
                }
            )
        return found
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return find_cuda_tensors(dataclasses.asdict(obj), path)
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_cuda_tensors(value, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(find_cuda_tensors(value, f"{path}[{index}]"))
    return found


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_batch_shapes(batch: dict[str, Any], lookback: int, horizon: int) -> None:
    x = batch["x"]
    y = batch["y"]
    mask = batch["valid_target_mask"]
    if x.ndim != 4:
        raise ValueError(f"x must be [B,L,N,C], got {tuple(x.shape)}.")
    if y.ndim != 3:
        raise ValueError(f"y must be [B,H,N], got {tuple(y.shape)}.")
    if x.shape[1] != lookback:
        raise ValueError(f"x lookback must be {lookback}, got {x.shape[1]}.")
    if y.shape[1] != horizon:
        raise ValueError(f"y horizon must be {horizon}, got {y.shape[1]}.")
    if mask.shape != y.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match y {tuple(y.shape)}.")


def validate_model_output(out: dict[str, Any], batch: dict[str, Any]) -> None:
    if "pred" not in out or "aux" not in out:
        raise ValueError('model forward must return {"pred": Tensor[B,H,N], "aux": dict}.')
    pred = out["pred"]
    if pred.shape != batch["y"].shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} must match y {tuple(batch['y'].shape)}.")
    if "pred_bnh" not in out["aux"]:
        raise ValueError("aux must include pred_bnh for [B,N,H] protocol conversion.")
    if out["aux"].get("uses_vadsp"):
        diagnostics_level = out["aux"].get("diagnostics_level", "standard")
        if diagnostics_level in {"none", "minimal"}:
            pass
        else:
            for key in ["volatility", "rolling_std", "dynamic_patch_gate", "x_fine", "x_coarse"]:
                if key not in out["aux"]:
                    raise ValueError(f"VADSP aux missing key: {key}")
            gate = out["aux"]["dynamic_patch_gate"]
            if torch.min(gate).item() < -1e-6 or torch.max(gate).item() > 1.0 + 1e-6:
                raise ValueError("dynamic_patch_gate must be in [0,1].")
    elif out["aux"].get("uses_fixed_dual_granularity"):
        gate = out["aux"].get("dynamic_patch_gate")
        if gate is not None and not torch.allclose(gate, torch.full_like(gate, 0.5)):
            raise ValueError("Fixed dual-granularity ablation must use a 0.5/0.5 gate.")
    if out["aux"].get("uses_graph_temporal_encoder"):
        aux = out["aux"]
        diagnostics_level = aux.get("diagnostics_level", "standard")
        if diagnostics_level in {"none", "minimal"}:
            required_graph_keys = [] if diagnostics_level == "none" else ["h_fine_norm_mean", "h_coarse_norm_mean"]
            for key in required_graph_keys:
                if key not in aux or not torch.isfinite(torch.as_tensor(aux[key])).all().item():
                    raise ValueError(f"GraphTemporal diagnostic is missing or non-finite: {key}")
        else:
            for key in [
                "h_fine",
                "h_coarse",
                "h_fine_norm_mean",
                "h_fine_norm_std",
                "h_coarse_norm_mean",
                "h_coarse_norm_std",
                "h_fine_spatial_delta_norm",
                "h_coarse_spatial_delta_norm",
            ]:
                if key not in aux:
                    raise ValueError(f"GraphTemporal aux missing key: {key}")
            if aux["h_fine"].shape != aux["x_fine"].shape:
                raise ValueError("h_fine shape must match x_fine shape.")
            if aux["h_coarse"].shape != aux["x_coarse"].shape:
                raise ValueError("h_coarse shape must match x_coarse shape.")
            for key in [
                "h_fine_norm_mean",
                "h_fine_norm_std",
                "h_coarse_norm_mean",
                "h_coarse_norm_std",
                "h_fine_spatial_delta_norm",
                "h_coarse_spatial_delta_norm",
            ]:
                if not torch.isfinite(torch.as_tensor(aux[key])).all().item():
                    raise ValueError(f"GraphTemporal diagnostic is not finite: {key}")
    if out["aux"].get("uses_stmg_coupling_block"):
        aux = out["aux"]
        diagnostics_level = aux.get("diagnostics_level", "standard")
        uses_macro = bool(aux.get("uses_macro_prompt"))
        uses_cross = bool(aux.get("uses_cross_fusion"))
        uses_st = bool(aux.get("uses_st_prompt"))
        required_coupling_keys = ["coupling_metadata"]
        if uses_macro:
            required_coupling_keys.append("macro_prompt_norm_mean")
        if uses_st:
            required_coupling_keys.append("st_prompt_norm_mean")
        if uses_cross:
            required_coupling_keys.extend(
                ["macro_attn_entropy", "fine_attn_entropy", "fusion_gate_mean", "fusion_gate_std"]
            )
        if diagnostics_level not in {"none", "minimal"}:
            if uses_macro:
                required_coupling_keys.append("macro_prompt")
            if uses_st:
                required_coupling_keys.append("st_prompt")
        for key in required_coupling_keys:
            if key not in aux:
                raise ValueError(f"CouplingBlock aux missing key: {key}")
        if diagnostics_level not in {"none", "minimal"} and uses_macro:
            if aux["macro_prompt"].ndim != 4:
                raise ValueError("macro_prompt must be [B,N,P,D].")
        if diagnostics_level not in {"none", "minimal"} and uses_st:
            if aux["st_prompt"].ndim != 4:
                raise ValueError("st_prompt must be [1,H,N,D].")
        if uses_macro and not aux.get("macro_prompt_from_spatial_enhanced_coarse"):
            raise ValueError("macro_prompt must come from graph-temporal enhanced h_coarse.")
        if uses_cross and not aux.get("cross_fusion_uses_spatial_enhanced_features"):
            raise ValueError("cross_fusion must use graph-temporal enhanced h_fine/h_coarse.")
        for key in [
            "macro_prompt_norm_mean",
            "st_prompt_norm_mean",
            "macro_attn_entropy",
            "fine_attn_entropy",
            "fusion_gate_mean",
            "fusion_gate_std",
        ]:
            if key not in aux:
                continue
            value = aux[key]
            if diagnostics_level == "none":
                continue
            if value is None and (
                diagnostics_level in {"none", "minimal"}
                or not uses_macro
                or not uses_cross
                or not uses_st
            ):
                continue
            if not torch.isfinite(torch.as_tensor(value)).all().item():
                raise ValueError(f"Coupling diagnostic is not finite: {key}")
    if out["aux"].get("uses_direct_decoder"):
        aux = out["aux"]
        metadata = aux.get("decoder_metadata", {})
        if metadata.get("decoder_input_strategy") not in {
            "direct_multi_output_prompt_query",
            "direct_multi_output_horizon_head",
        }:
            raise ValueError("Unsupported direct decoder strategy.")
        if metadata.get("teacher_forcing") is not False or metadata.get("autoregressive") is not False:
            raise ValueError("STPromptDirectDecoder must not use teacher forcing or autoregression.")
        for key in [
            "decoder_context_norm_mean",
            "decoder_macro_summary_norm_mean",
            "decoder_st_prompt_norm_mean",
            "decoder_context_future_norm_mean",
        ]:
            if key not in aux:
                raise ValueError(f"Direct decoder aux missing key: {key}")
            if aux.get("diagnostics_level") == "none":
                continue
            if key == "decoder_st_prompt_norm_mean" and not aux.get("uses_st_prompt"):
                continue
            if not torch.isfinite(torch.as_tensor(aux[key])).all().item():
                raise ValueError(f"Direct decoder diagnostic is not finite: {key}")


def model_summary(model: torch.nn.Module, data: STMGPromptDataBundle) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    non_trainable = total - trainable
    shared_parameter_count = 0
    branch_output_dimensions = None
    temporal_branch_transform = getattr(getattr(model, "config", None), "temporal_branch_transform", None)
    coupling_blocks = getattr(model, "coupling_blocks", None)
    if coupling_blocks:
        first_block = coupling_blocks[0]
        fine_temporal = first_block.fine_micro_graph_temporal_encoder.block.causal_tcn
        coarse_temporal = first_block.coarse_macro_graph_temporal_encoder.block.causal_tcn
        if fine_temporal is coarse_temporal:
            shared_parameter_count = sum(parameter.numel() for parameter in fine_temporal.parameters())
        hidden_dim = int(getattr(model, "hidden_dim", 0))
        branch_output_dimensions = {"fine": hidden_dim, "coarse": hidden_dim}
    return {
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "total_parameters": total,
        "shared_parameter_count": shared_parameter_count,
        "temporal_branch_transform": temporal_branch_transform,
        "branch_output_dimensions": branch_output_dimensions,
        "parameter_memory_mb": float(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)),
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "feature_cols": data.feature_cols,
        "window_counts": {k: len(v) for k, v in data.window_indices.items()},
        "raw_split_valid_ratio": data.metadata.get("raw_split_valid_ratio", {}),
        "window_expanded_valid_ratio": data.metadata.get("window_expanded_valid_ratio", {}),
    }


def _save_heatmap(array: np.ndarray, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(array, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception:
        values = np.asarray(array, dtype=np.float32)
        values = values - np.nanmin(values)
        denom = float(np.nanmax(values)) or 1.0
        values = (255.0 * values / denom).astype(np.uint8)
        h, w = values.shape
        canvas = bytearray()
        for v in values.reshape(-1):
            canvas.extend([int(v), int(v), int(v)])
        _write_png_rgb(path, w, h, canvas)


@torch.no_grad()
def save_graph_snapshots(model: torch.nn.Module, output_dir, checkpoint_name: str = "best_checkpoint.pt") -> dict[str, Any] | None:
    if not all(hasattr(model, name) for name in ["A_macro_prior", "A_micro_prior", "macro_graph_builder", "micro_graph_builder"]):
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    A_macro_prior = model.A_macro_prior.detach().cpu().numpy()
    A_micro_prior = model.A_micro_prior.detach().cpu().numpy()
    assignment_trace = None
    if hasattr(model, "effective_graphs"):
        A_micro_final_t, A_macro_final_t, assignment_trace = model.effective_graphs()
    else:
        A_macro_final_t = model.macro_graph_builder(model.A_macro_prior)
        A_micro_final_t = model.micro_graph_builder(model.A_micro_prior)
    A_macro_final = A_macro_final_t.detach().cpu().numpy()
    A_micro_final = A_micro_final_t.detach().cpu().numpy()
    macro_adaptive = A_macro_final
    micro_adaptive = A_micro_final
    files = {
        "macro_prior_adjacency.npy": A_macro_prior,
        "micro_prior_adjacency.npy": A_micro_prior,
        "macro_adaptive_final.npy": macro_adaptive,
        "micro_adaptive_final.npy": micro_adaptive,
        "macro_final_adjacency.npy": A_macro_final,
        "micro_final_adjacency.npy": A_micro_final,
        "coarse_branch_final_adjacency.npy": A_macro_final,
        "fine_branch_final_adjacency.npy": A_micro_final,
    }
    for filename, array in files.items():
        np.save(output_dir / filename, array)
    _save_heatmap(A_macro_prior, output_dir / "macro_prior_heatmap.png")
    _save_heatmap(A_micro_prior, output_dir / "micro_prior_heatmap.png")
    _save_heatmap(A_macro_final, output_dir / "macro_final_heatmap.png")
    _save_heatmap(A_micro_final, output_dir / "micro_final_heatmap.png")
    macro_support = A_macro_final > 1e-8
    micro_support = A_micro_final > 1e-8
    prior_macro_support = A_macro_prior > 1e-8
    prior_micro_support = A_micro_prior > 1e-8
    summary = {
        "artifact_source_checkpoint": checkpoint_name,
        "macro_micro_prior_support_jaccard": float(
            np.logical_and(prior_macro_support, prior_micro_support).sum()
            / max(int(np.logical_or(prior_macro_support, prior_micro_support).sum()), 1)
        ),
        "macro_micro_final_support_jaccard": float(
            np.logical_and(macro_support, micro_support).sum() / max(int(np.logical_or(macro_support, micro_support).sum()), 1)
        ),
        "macro_micro_prior_weight_correlation": float(np.corrcoef(A_macro_prior.reshape(-1), A_micro_prior.reshape(-1))[0, 1]),
        "macro_micro_final_weight_correlation": float(np.corrcoef(A_macro_final.reshape(-1), A_micro_final.reshape(-1))[0, 1]),
        "macro_topology_edge_count": int(macro_support.sum()),
        "micro_topology_edge_count": int(micro_support.sum()),
        "macro_self_weight_mean": float(np.diag(A_macro_final).mean()),
        "micro_self_weight_mean": float(np.diag(A_micro_final).mean()),
        "branch_graph_assignment_trace": assignment_trace,
    }
    (output_dir / "graph_snapshot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _draw_line_rgb(canvas: bytearray, width: int, height: int, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            offset = (y0 * width + x0) * 3
            canvas[offset : offset + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _write_png_rgb(path: Path, width: int, height: int, canvas: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = [b"\x00" + bytes(canvas[y * width * 3 : (y + 1) * width * 3]) for y in range(height)]
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _write_simple_granularity_weight_png(logs: list[dict[str, Any]], horizons: list[int], path: Path) -> None:
    width, height = 640, 400
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 24, 48
    canvas = bytearray([255] * width * height * 3)
    axis_color = (50, 50, 50)
    _draw_line_rgb(canvas, width, height, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), axis_color)
    _draw_line_rgb(canvas, width, height, (margin_left, margin_top), (margin_left, height - margin_bottom), axis_color)
    epochs = [float(row.get("epoch", idx + 1) or idx + 1) for idx, row in enumerate(logs)]
    series: list[tuple[int, list[float]]] = []
    all_values: list[float] = []
    for horizon in horizons:
        values = []
        for row in logs:
            value = row.get(f"granularity_weight_h{horizon}")
            if value is not None:
                values.append(float(value))
        if values:
            series.append((horizon, values))
            all_values.extend(values)
    if not series:
        _write_png_rgb(path, width, height, canvas)
        return
    x_min, x_max = min(epochs), max(epochs)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    y_min, y_max = min(all_values), max(all_values)
    pad = max((y_max - y_min) * 0.1, 1e-3)
    y_min -= pad
    y_max += pad
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    for idx, (_, values) in enumerate(series):
        points: list[tuple[int, int]] = []
        for epoch, value in zip(epochs, values):
            x = int(margin_left + (epoch - x_min) / (x_max - x_min) * plot_w)
            y = int(height - margin_bottom - (value - y_min) / (y_max - y_min) * plot_h)
            points.append((x, y))
        for p0, p1 in zip(points, points[1:]):
            _draw_line_rgb(canvas, width, height, p0, p1, colors[idx % len(colors)])
        for x, y in points:
            _draw_line_rgb(canvas, width, height, (x - 3, y), (x + 3, y), colors[idx % len(colors)])
            _draw_line_rgb(canvas, width, height, (x, y - 3), (x, y + 3), colors[idx % len(colors)])
    _write_png_rgb(path, width, height, canvas)


def _write_curve_png(series: dict[str, list[float]], path: Path) -> None:
    rows = [{"epoch": idx + 1, **{name: values[idx] for name, values in series.items()}} for idx in range(max((len(v) for v in series.values()), default=0))]
    _write_simple_granularity_weight_png(rows, list(range(1, len(series) + 1)), path)


@torch.no_grad()
def save_vadsp_diagnostics(model: torch.nn.Module, loader, output_dir, max_batches: int = 8) -> dict[str, Any] | None:
    if not getattr(model, "vadsp", None):
        return None
    device = next(model.parameters()).device
    model.eval()
    aux_batches = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch["x"].to(device).float()
        out = model(x)
        aux = out["aux"]
        aux_batches.append(
            {
                "volatility": aux["volatility"].detach().cpu().numpy(),
                "rolling_std": aux["rolling_std"].detach().cpu().numpy(),
                "dynamic_patch_gate": aux["dynamic_patch_gate"].detach().cpu().numpy(),
                "delta_source_abs": aux["volatility"].detach().cpu().numpy(),
                "fine_feature_norm": float(aux.get("fine_feature_norm", torch.tensor(float("nan"))).detach().cpu())
                if torch.is_tensor(aux.get("fine_feature_norm"))
                else aux.get("fine_feature_norm"),
                "coarse_feature_norm": float(aux.get("coarse_feature_norm", torch.tensor(float("nan"))).detach().cpu())
                if torch.is_tensor(aux.get("coarse_feature_norm"))
                else aux.get("coarse_feature_norm"),
                "weighted_fine_norm": float(aux.get("weighted_fine_norm", torch.tensor(float("nan"))).detach().cpu())
                if torch.is_tensor(aux.get("weighted_fine_norm"))
                else aux.get("weighted_fine_norm"),
                "weighted_coarse_norm": float(aux.get("weighted_coarse_norm", torch.tensor(float("nan"))).detach().cpu())
                if torch.is_tensor(aux.get("weighted_coarse_norm"))
                else aux.get("weighted_coarse_norm"),
                "source_weights": aux["source_weights"].detach().cpu().numpy() if torch.is_tensor(aux.get("source_weights")) else aux.get("source_weights"),
                "temperature": float(aux["temperature"].detach().cpu()) if torch.is_tensor(aux.get("temperature")) else aux.get("temperature"),
                "beta_fine": float(aux["beta_fine"].detach().cpu()) if torch.is_tensor(aux.get("beta_fine")) else aux.get("beta_fine"),
                "beta_coarse": float(aux["beta_coarse"].detach().cpu()) if torch.is_tensor(aux.get("beta_coarse")) else aux.get("beta_coarse"),
                "coarse_scale_weights": aux["coarse_scale_weights"].detach().cpu().numpy()
                if torch.is_tensor(aux.get("coarse_scale_weights"))
                else aux.get("coarse_scale_weights"),
            }
        )
    if not aux_batches:
        return None
    return save_vadsp_diagnostics_from_aux(
        aux_batches,
        output_dir,
        volatility_source_cols=list(model.vadsp.volatility_source_cols),
        robust_statistics_metadata=getattr(model.vadsp, "robust_statistics_metadata", None),
    )


@torch.no_grad()
def save_graph_temporal_diagnostics(model: torch.nn.Module, loader, output_dir, max_batches: int = 8) -> dict[str, Any] | None:
    if not getattr(model, "graph_temporal_encoder", None):
        return None
    device = next(model.parameters()).device
    model.eval()
    rows: list[dict[str, float]] = []
    keys = [
        "h_fine_norm_mean",
        "h_fine_norm_std",
        "h_coarse_norm_mean",
        "h_coarse_norm_std",
        "h_fine_spatial_delta_norm",
        "h_coarse_spatial_delta_norm",
    ]
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        out = model(batch["x"].to(device).float())
        aux = out["aux"]
        rows.append({key: float(aux[key].detach().cpu()) for key in keys})
    if not rows:
        return None
    summary = {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in keys
    }
    summary.update(
        {
            "num_batches": len(rows),
            "use_graph": bool(getattr(model.config, "use_graph_in_temporal_encoder", True)),
            "use_temporal_attention": bool(getattr(model.config, "use_temporal_attention", False)),
            "fine_tcn_dilations": list(getattr(model.config, "fine_tcn_dilations", [])),
            "coarse_tcn_dilations": list(getattr(model.config, "coarse_tcn_dilations", [])),
        }
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graph_temporal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def save_loss_diagnostics(loss_fn: Any, logs: list[dict[str, Any]], output_dir) -> dict[str, Any] | None:
    """Persist MS-MG-DWU loss weights and curves when that loss is active."""

    if not hasattr(loss_fn, "diagnostics_state"):
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = loss_fn.diagnostics_state()
    horizons = [int(h) for h in state["eval_horizons"]]
    weights = state["granularity_weight"].numpy().tolist()
    log_sigma = state["log_sigma_g"].numpy().tolist()
    summary = {
        "loss_function": "msmg_dwu_loss",
        "base_loss": state["base_loss"],
        "lambda_site": state["lambda_site"],
        "ema_alpha": state["ema_alpha"],
        "node_weight_clip": list(state["node_weight_clip"]),
        "granularity_weight_mode": state.get("granularity_weight_mode"),
        "site_weight_mode": state.get("site_weight_mode"),
        "eval_horizons": horizons,
        "final_granularity_weight": {f"h{h}": float(w) for h, w in zip(horizons, weights)},
        "final_log_sigma_g": {f"h{h}": float(s) for h, s in zip(horizons, log_sigma)},
        "artifact_source_checkpoint": "best_checkpoint.pt",
    }

    weight_fieldnames = ["epoch"]
    for horizon in horizons:
        weight_fieldnames.extend(
            [
                f"raw_loss_h{horizon}",
                f"ema_loss_h{horizon}",
                f"initial_loss_h{horizon}",
                f"difficulty_level_h{horizon}",
                f"relative_training_rate_h{horizon}",
                f"granularity_loss_h{horizon}",
                f"granularity_weight_h{horizon}",
                f"weighted_contribution_h{horizon}",
            ]
        )
    weight_fieldnames.extend(["site_weight_mean", "site_weight_min", "site_weight_max", "total_loss"])
    with (output_dir / "loss_weights.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=weight_fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow({key: row.get(key) for key in weight_fieldnames})

    node_weight = state["node_weight"].numpy().tolist()
    ema_node_loss = state["ema_node_loss"].numpy().tolist()
    with (output_dir / "site_weight_final.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_index", "site_weight", "ema_node_loss"])
        writer.writeheader()
        for node_idx, (weight, ema_loss) in enumerate(zip(node_weight, ema_node_loss)):
            writer.writerow(
                {
                    "node_index": node_idx,
                    "site_weight": float(weight),
                    "ema_node_loss": float(ema_loss),
                }
            )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        epochs = [row.get("epoch") for row in logs]
        for horizon in horizons:
            values = [row.get(f"granularity_weight_h{horizon}") for row in logs]
            if any(value is not None for value in values):
                ax.plot(epochs, values, marker="o", label=f"H={horizon}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("granularity weight")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "granularity_weight_curve.png", dpi=150)
        plt.close(fig)
        summary["granularity_weight_curve_path"] = str(output_dir / "granularity_weight_curve.png")
    except Exception as exc:  # pragma: no cover - diagnostic fallback only
        fallback_path = output_dir / "granularity_weight_curve.png"
        _write_simple_granularity_weight_png(logs, horizons, fallback_path)
        summary["granularity_weight_curve_path"] = str(fallback_path)
        summary["granularity_weight_curve_fallback"] = "stdlib_png"
        summary["granularity_weight_curve_error"] = str(exc)

    (output_dir / "loss_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def save_coupling_diagnostics(model: torch.nn.Module, loader, output_dir, max_batches: int = 8) -> dict[str, Any] | None:
    if not getattr(model, "coupling_blocks", None):
        return None
    device = next(model.parameters()).device
    model.eval()
    prompt_rows: list[dict[str, float | int]] = []
    attention_batch_rows: list[dict[str, float | int]] = []
    attention_layer_rows: list[dict[str, float | int]] = []
    decoder_attention_rows: list[dict[str, float | int]] = []
    decoder_context_rows: list[dict[str, float | int]] = []
    diffusion_rows: list[dict[str, float | int | str]] = []
    scalar_keys = [
        "macro_prompt_norm_mean",
        "st_prompt_norm_mean",
        "macro_attn_entropy",
        "fine_attn_entropy",
        "fusion_gate_mean",
        "fusion_gate_std",
        "h_fine_spatial_delta_norm",
        "h_coarse_spatial_delta_norm",
        "h_fine_spatial_delta_ratio",
        "h_coarse_spatial_delta_ratio",
    ]
    layer_prefixes = [f"coupling_layer_{i}" for i in range(len(model.coupling_blocks))]
    layer_keys = [
        "macro_prompt_norm_mean",
        "macro_attn_entropy",
        "fine_attn_entropy",
        "fusion_gate_mean",
        "fusion_gate_std",
        "h_fine_spatial_delta_norm",
        "h_coarse_spatial_delta_norm",
        "h_fine_spatial_delta_ratio",
        "h_coarse_spatial_delta_ratio",
    ]
    rows: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        out = model(batch["x"].to(device).float())
        aux = out["aux"]
        rows.append({key: _scalar_float(aux.get(key)) for key in scalar_keys})
        prompt_rows.append(
            {
                "batch": batch_idx,
                "macro_prompt_norm_mean": _scalar_float(aux.get("macro_prompt_norm_mean")),
                "st_prompt_norm_mean": _scalar_float(aux.get("st_prompt_norm_mean")),
            }
        )
        attention_batch_rows.append(
            {
                "batch": batch_idx,
                "macro_attn_entropy": _scalar_float(aux.get("macro_attn_entropy")),
                "fine_attn_entropy": _scalar_float(aux.get("fine_attn_entropy")),
                "fusion_gate_mean": _scalar_float(aux.get("fusion_gate_mean")),
                "fusion_gate_std": _scalar_float(aux.get("fusion_gate_std")),
            }
        )
        if aux.get("decoder_context_mode") == "full_history_cross_attention" and aux.get("st_prompt") is not None:
            H = int(aux["st_prompt"].shape[1])
            for horizon_idx in range(H):
                decoder_attention_rows.append(
                    {
                        "batch": batch_idx,
                        "horizon": horizon_idx + 1,
                        "fine_history_attention_entropy": float(aux["fine_history_attention_entropy_per_horizon"][horizon_idx].detach().cpu()),
                        "coarse_history_attention_entropy": float(aux["coarse_history_attention_entropy_per_horizon"][horizon_idx].detach().cpu()),
                        "macro_attention_entropy": float(aux["macro_attention_entropy_per_horizon"][horizon_idx].detach().cpu()),
                        "fine_effective_history_steps": float(aux["fine_effective_history_steps_per_horizon"][horizon_idx].detach().cpu()),
                        "coarse_effective_history_steps": float(aux["coarse_effective_history_steps_per_horizon"][horizon_idx].detach().cpu()),
                        "fine_expected_lag": float(aux["fine_expected_lag_per_horizon"][horizon_idx].detach().cpu()),
                        "coarse_expected_lag": float(aux["coarse_expected_lag_per_horizon"][horizon_idx].detach().cpu()),
                        "fine_context_norm": float(aux["fine_context_norm_per_horizon"][horizon_idx].detach().cpu()),
                        "coarse_context_norm": float(aux["coarse_context_norm_per_horizon"][horizon_idx].detach().cpu()),
                        "macro_context_norm": float(aux["macro_context_norm_per_horizon"][horizon_idx].detach().cpu()),
                    }
                )
                decoder_context_rows.append(
                    {
                        "batch": batch_idx,
                        "horizon": horizon_idx + 1,
                        "context_weight_fine": float(aux["context_weight_fine_per_horizon"][horizon_idx].detach().cpu()),
                        "context_weight_coarse": float(aux["context_weight_coarse_per_horizon"][horizon_idx].detach().cpu()),
                        "context_weight_macro": float(aux["context_weight_macro_per_horizon"][horizon_idx].detach().cpu()),
                    }
                )
        for branch in ["micro", "macro"]:
            if aux.get(f"{branch}_graph_operator") == "bidirectional_diffusion":
                row = {
                    "batch": batch_idx,
                    "branch": branch,
                    "diffusion_order": int(aux[f"{branch}_diffusion_order"].detach().cpu()),
                    "beta_graph": float(aux[f"{branch}_beta_graph"].detach().cpu()),
                    "input_node_cosine_similarity": float(aux[f"{branch}_input_node_cosine_similarity"].detach().cpu()),
                    "hop1_node_cosine_similarity": float(aux[f"{branch}_hop1_node_cosine_similarity"].detach().cpu()),
                    "hop2_node_cosine_similarity": float(aux[f"{branch}_hop2_node_cosine_similarity"].detach().cpu()),
                    "output_node_cosine_similarity": float(aux[f"{branch}_output_node_cosine_similarity"].detach().cpu()),
                    "graph_delta_norm": float(aux[f"{branch}_graph_delta_norm"].detach().cpu()),
                    "graph_delta_ratio": float(aux[f"{branch}_graph_delta_ratio"].detach().cpu()),
                }
                for hop in range(1, int(row["diffusion_order"]) + 1):
                    row[f"forward_hop{hop}_norm"] = float(aux[f"{branch}_forward_hop{hop}_norm"].detach().cpu())
                    row[f"backward_hop{hop}_norm"] = float(aux[f"{branch}_backward_hop{hop}_norm"].detach().cpu())
                diffusion_rows.append(row)
        for prefix in layer_prefixes:
            layer_row = {"batch": batch_idx, "layer": int(prefix.rsplit("_", 1)[-1])}
            has_layer = False
            for key in layer_keys:
                aux_key = f"{prefix}_{key}"
                if aux_key in aux:
                    layer_row[key] = _scalar_float(aux[aux_key])
                    has_layer = True
            if has_layer:
                attention_layer_rows.append(layer_row)
    if not rows:
        return None

    summary = {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in scalar_keys
    }
    metadata = dict(getattr(model, "coupling_metadata", {}))
    metadata.update(
        {
            "fusion_mode": getattr(model.config, "fusion_mode", None),
            "disable_reverse_cross": bool(getattr(model.config, "disable_reverse_cross", False)),
            "disable_macro_to_fine_cross": bool(
                getattr(model.config, "disable_macro_to_fine_cross", False)
            ),
            "macro_prompt_len": int(getattr(model.config, "macro_prompt_len", 0)),
            "macro_prompt_pooling": getattr(model.config, "macro_prompt_pooling", "attention"),
            "cross_fusion_recent_len": int(getattr(model.config, "cross_fusion_recent_len", 0)),
            "cross_fusion_uses_spatial_enhanced_features": bool(
                getattr(model.config, "use_cross_fusion", True)
            ),
            "macro_prompt_from_spatial_enhanced_coarse": bool(
                getattr(model.config, "use_macro_prompt", True)
            ),
            "st_prompt_direct_decoder_enabled": bool(getattr(model, "direct_decoder", None) is not None),
            "decoder_input_strategy": getattr(model.config, "decoder_input_strategy", None),
            "decoder_context_mode": getattr(model.config, "decoder_context_mode", None),
            "decoder_history_len": getattr(model.config, "decoder_history_len", None),
            "st_prompt_use_node_identity": bool(getattr(model.config, "st_prompt_use_node_identity", True)),
            "st_prompt_information": (
                "node+future+granularity"
                if bool(getattr(model.config, "st_prompt_use_node_identity", True))
                else "future+granularity"
            ),
            "graph_operator": getattr(model.config, "graph_operator", None),
            "diffusion_order_micro": getattr(model.config, "diffusion_order_micro", None),
            "diffusion_order_macro": getattr(model.config, "diffusion_order_macro", None),
            "teacher_forcing": False,
            "future_observed_features_used": False,
        }
    )
    summary.update(
        {
            "num_batches": len(rows),
            "metadata": metadata,
            "cross_fusion_uses_spatial_enhanced_features": bool(
                getattr(model.config, "use_cross_fusion", True)
            ),
            "macro_prompt_from_spatial_enhanced_coarse": bool(
                getattr(model.config, "use_macro_prompt", True)
            ),
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prompt_norm.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "macro_prompt_norm_mean",
                "st_prompt_norm_mean",
                "macro_prompt_token_cosine_mean",
                "macro_prompt_token_cosine_max",
                "macro_prompt_node_variance",
                "macro_prompt_batch_variance",
                "st_prompt_node_variance",
            ],
        )
        writer.writeheader()
        writer.writerows(prompt_rows)
    batch_fieldnames = [
        "batch",
        "macro_attn_entropy",
        "fine_attn_entropy",
        "fusion_gate_mean",
        "fusion_gate_std",
    ]
    with (output_dir / "cross_attention_batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=batch_fieldnames)
        writer.writeheader()
        for row in attention_batch_rows:
            writer.writerow({key: row.get(key) for key in batch_fieldnames})
    layer_fieldnames = [
        "batch",
        "layer",
        "macro_attn_entropy",
        "fine_attn_entropy",
        "fusion_gate_mean",
        "fusion_gate_std",
        "macro_prompt_norm_mean",
        "h_fine_spatial_delta_norm",
        "h_coarse_spatial_delta_norm",
        "h_fine_spatial_delta_ratio",
        "h_coarse_spatial_delta_ratio",
    ]
    with (output_dir / "cross_attention_layer_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fieldnames)
        writer.writeheader()
        for row in attention_layer_rows:
            writer.writerow({key: row.get(key) for key in layer_fieldnames})
    if diffusion_rows:
        diffusion_fieldnames = sorted({key for row in diffusion_rows for key in row})
        with (output_dir / "diffusion_hop_norms.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=diffusion_fieldnames)
            writer.writeheader()
            writer.writerows(diffusion_rows)
        oversmoothing_fields = [
            "batch",
            "branch",
            "input_node_cosine_similarity",
            "hop1_node_cosine_similarity",
            "hop2_node_cosine_similarity",
            "output_node_cosine_similarity",
            "graph_delta_ratio",
            "beta_graph",
        ]
        with (output_dir / "diffusion_oversmoothing.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=oversmoothing_fields)
            writer.writeheader()
            for row in diffusion_rows:
                writer.writerow({key: row.get(key) for key in oversmoothing_fields})
        summary["diffusion_graph_summary"] = {
            "graph_operator": getattr(model.config, "graph_operator", None),
            "diffusion_order_micro": getattr(model.config, "diffusion_order_micro", None),
            "diffusion_order_macro": getattr(model.config, "diffusion_order_macro", None),
            "diffusion_use_bidirectional": getattr(model.config, "diffusion_use_bidirectional", None),
            "mean_graph_delta_ratio": float(np.mean([float(row["graph_delta_ratio"]) for row in diffusion_rows])),
            "mean_beta_graph": float(np.mean([float(row["beta_graph"]) for row in diffusion_rows])),
            "oversmoothing_warning": bool(
                np.mean([float(row["hop2_node_cosine_similarity"]) for row in diffusion_rows])
                - np.mean([float(row["input_node_cosine_similarity"]) for row in diffusion_rows])
                > 0.25
            ),
        }
        (output_dir / "diffusion_graph_summary.json").write_text(
            json.dumps(summary["diffusion_graph_summary"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if decoder_attention_rows:
        attention_fields = list(decoder_attention_rows[0].keys())
        with (output_dir / "decoder_attention_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=attention_fields)
            writer.writeheader()
            writer.writerows(decoder_attention_rows)
        context_fields = list(decoder_context_rows[0].keys())
        with (output_dir / "decoder_context_weight_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=context_fields)
            writer.writeheader()
            writer.writerows(decoder_context_rows)
        try:
            import matplotlib.pyplot as plt

            for filename, rows_for_plot, keys, ylabel in [
                (
                    "decoder_expected_lag_curve.png",
                    decoder_attention_rows,
                    ["fine_expected_lag", "coarse_expected_lag"],
                    "expected lag",
                ),
                (
                    "decoder_context_weight_curve.png",
                    decoder_context_rows,
                    ["context_weight_fine", "context_weight_coarse", "context_weight_macro"],
                    "context weight",
                ),
            ]:
                fig, ax = plt.subplots(figsize=(6, 4))
                horizons = sorted({int(row["horizon"]) for row in rows_for_plot})
                for key in keys:
                    values = [
                        float(np.mean([float(row[key]) for row in rows_for_plot if int(row["horizon"]) == horizon]))
                        for horizon in horizons
                    ]
                    ax.plot(horizons, values, marker="o", label=key)
                ax.set_xlabel("horizon")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / filename, dpi=150)
                plt.close(fig)
        except Exception as exc:  # pragma: no cover - plotting is diagnostic only
            summary["decoder_curve_warning"] = str(exc)
    (output_dir / "coupling_block_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
