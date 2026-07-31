from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DEFAULT_16_FEATURES, is_forbidden_input_col


def _feature_indices(feature_names: list[str], source_cols: list[str]) -> list[int]:
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in source_cols if name not in index]
    if missing:
        raise ValueError(f"volatility_source_cols not found in feature_names: {missing}")
    forbidden = [name for name in source_cols if is_forbidden_input_col(name)]
    if forbidden:
        raise ValueError(f"Forbidden volatility source columns: {forbidden}")
    not_base = [name for name in source_cols if name not in DEFAULT_16_FEATURES]
    if not_base:
        raise ValueError(f"volatility_source_cols must come from base 16 features: {not_base}")
    return [index[name] for name in source_cols]


def _as_cpu_float_tensor(values: Tensor | np.ndarray) -> Tensor:
    if torch.is_tensor(values):
        return values.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(values, device="cpu", dtype=torch.float32)


def _column_quantiles(values: Tensor, label: str) -> tuple[Tensor, int]:
    finite = values[torch.isfinite(values)]
    filtered = int(values.numel() - finite.numel())
    if finite.numel() == 0:
        raise ValueError(f"No finite values available for VADSP robust statistics column {label}.")
    quantiles = torch.quantile(
        finite.to(device="cpu", dtype=torch.float32),
        torch.tensor([0.25, 0.50, 0.75], dtype=torch.float32),
    )
    return quantiles, filtered


def compute_vadsp_robust_statistics_from_timeline(
    timeline: Tensor | np.ndarray,
    feature_names: list[str],
    volatility_source_cols: list[str],
    volatility_mode: str,
    robust_eps: float,
    fit_split: str = "train",
    fit_source: str = "unique_train_timeline",
    feature_space: str = "model_input_scaled",
) -> dict[str, Any]:
    """Fit exact CPU robust stats from a unique, unwindowed [T,N,C] timeline.

    per_node stats have shape [N,K]. global stats have shape [1,K]. No lookback
    dimension is retained, so the result is independent of window stride/batch size.
    """

    if volatility_mode not in {"per_node", "global"}:
        raise ValueError("volatility_mode must be per_node or global.")
    if robust_eps <= 0:
        raise ValueError("robust_eps must be positive.")
    source_indices = _feature_indices(feature_names, volatility_source_cols)
    x = _as_cpu_float_tensor(timeline)
    if x.ndim != 3:
        raise ValueError(f"timeline must be [T,N,C], got {tuple(x.shape)}.")
    T, N, _ = x.shape
    K = len(source_indices)
    if T <= 1:
        raise ValueError("Need at least two train time points to fit VADSP robust statistics.")
    sources = x[..., source_indices]
    delta_abs = (sources[1:] - sources[:-1]).abs().contiguous()
    q_shape = (N, K) if volatility_mode == "per_node" else (1, K)
    q25 = torch.empty(q_shape, dtype=torch.float32)
    q50 = torch.empty(q_shape, dtype=torch.float32)
    q75 = torch.empty(q_shape, dtype=torch.float32)
    nonfinite_filtered = 0
    degenerate_iqr_count = 0

    if volatility_mode == "per_node":
        for node_idx in range(N):
            for source_idx in range(K):
                quantiles, filtered = _column_quantiles(
                    delta_abs[:, node_idx, source_idx],
                    f"node={node_idx},source={volatility_source_cols[source_idx]}",
                )
                q25[node_idx, source_idx], q50[node_idx, source_idx], q75[node_idx, source_idx] = quantiles
                nonfinite_filtered += filtered
                if float(quantiles[2] - quantiles[0]) < robust_eps:
                    degenerate_iqr_count += 1
    else:
        for source_idx in range(K):
            quantiles, filtered = _column_quantiles(
                delta_abs[:, :, source_idx].reshape(-1),
                f"global,source={volatility_source_cols[source_idx]}",
            )
            q25[0, source_idx], q50[0, source_idx], q75[0, source_idx] = quantiles
            nonfinite_filtered += filtered
            if float(quantiles[2] - quantiles[0]) < robust_eps:
                degenerate_iqr_count += 1

    robust_scale = (q75 - q25).clamp_min(float(robust_eps))
    if volatility_mode == "per_node":
        delta_log = torch.log1p(delta_abs / robust_scale.view(1, N, K))
    else:
        delta_log = torch.log1p(delta_abs / robust_scale.view(1, 1, K))

    lq25 = torch.empty(q_shape, dtype=torch.float32)
    lq50 = torch.empty(q_shape, dtype=torch.float32)
    lq75 = torch.empty(q_shape, dtype=torch.float32)
    if volatility_mode == "per_node":
        for node_idx in range(N):
            for source_idx in range(K):
                quantiles, filtered = _column_quantiles(
                    delta_log[:, node_idx, source_idx],
                    f"log,node={node_idx},source={volatility_source_cols[source_idx]}",
                )
                lq25[node_idx, source_idx], lq50[node_idx, source_idx], lq75[node_idx, source_idx] = quantiles
                nonfinite_filtered += filtered
    else:
        for source_idx in range(K):
            quantiles, filtered = _column_quantiles(
                delta_log[:, :, source_idx].reshape(-1),
                f"log,global,source={volatility_source_cols[source_idx]}",
            )
            lq25[0, source_idx], lq50[0, source_idx], lq75[0, source_idx] = quantiles
            nonfinite_filtered += filtered

    log_iqr = (lq75 - lq25).clamp_min(float(robust_eps))
    return {
        "q25": q25,
        "median": q50,
        "q75": q75,
        "center": q50,
        "scale": robust_scale,
        "delta_abs_iqr": robust_scale,
        "delta_log_median": lq50,
        "delta_log_iqr": log_iqr,
        "source_cols": list(volatility_source_cols),
        "source_indices": list(source_indices),
        "mode": volatility_mode,
        "fit_split": fit_split,
        "fit_source": fit_source,
        "feature_space": feature_space,
        "exact": True,
        "device": "cpu",
        "num_time_points": int(T),
        "num_time_deltas": int(T - 1),
        "num_nodes": int(N),
        "num_sources": int(K),
        "shape": {
            "q25": list(q25.shape),
            "median": list(q50.shape),
            "q75": list(q75.shape),
            "center": list(q50.shape),
            "scale": list(robust_scale.shape),
        },
        "nonfinite_filtered": int(nonfinite_filtered),
        "degenerate_iqr_count": int(degenerate_iqr_count),
        "depends_on_window_stride": False,
    }


def causal_rolling_std(values: Tensor, window: int, eps: float = 1e-6) -> Tensor:
    """Causal rolling std over dim=1 for values [B,L,N]."""

    if values.ndim != 3:
        raise ValueError(f"values must be [B,L,N], got {tuple(values.shape)}.")
    if window <= 0:
        raise ValueError("window must be positive.")
    B, L, N = values.shape
    csum = torch.cumsum(values, dim=1)
    csum2 = torch.cumsum(values.square(), dim=1)
    padded = F.pad(csum, (0, 0, 1, 0))
    padded2 = F.pad(csum2, (0, 0, 1, 0))
    idx = torch.arange(L, device=values.device)
    start = torch.clamp(idx + 1 - window, min=0)
    end = idx + 1
    sums = padded[:, end, :] - padded[:, start, :]
    sums2 = padded2[:, end, :] - padded2[:, start, :]
    counts = torch.clamp(idx + 1, max=window).to(values.dtype).view(1, L, 1)
    mean = sums / counts
    var = torch.clamp(sums2 / counts - mean.square(), min=0.0)
    return torch.sqrt(var + eps).reshape(B, L, N)


def causal_rolling_mean(features: Tensor, window: int) -> Tensor:
    """Causal rolling mean over dim=1 for features [B,L,N,D]."""

    if features.ndim != 4:
        raise ValueError(f"features must be [B,L,N,D], got {tuple(features.shape)}.")
    if window <= 0:
        raise ValueError("window must be positive.")
    B, L, N, D = features.shape
    csum = torch.cumsum(features, dim=1)
    padded = F.pad(csum, (0, 0, 0, 0, 1, 0))
    idx = torch.arange(L, device=features.device)
    start = torch.clamp(idx + 1 - window, min=0)
    end = idx + 1
    sums = padded[:, end, :, :] - padded[:, start, :, :]
    counts = torch.clamp(idx + 1, max=window).to(features.dtype).view(1, L, 1, 1)
    return (sums / counts).reshape(B, L, N, D)


class CausalDepthwiseTemporalConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=self.kernel_size,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, L, N, D = x.shape
        series = x.permute(0, 2, 3, 1).reshape(B * N, D, L)
        series = F.pad(series, (self.kernel_size - 1, 0))
        out = self.pointwise(torch.relu(self.depthwise(series)))
        out = out.reshape(B, N, D, L).permute(0, 3, 1, 2).contiguous()
        return self.dropout(self.norm(out))


class VolatilityAwareDynamicSemanticPatching(nn.Module):
    """波动感知动态语义分块.

    Gate statistics are computed only from the current historical input window.
    The target, target mask, anomaly labels, and audit columns are never used.
    """

    def __init__(
        self,
        hidden_dim: int,
        feature_names: list[str],
        volatility_source_cols: list[str] | None = None,
        volatility_mode: str = "per_node",
        vol_window: int = 6,
        fine_kernel_size: int = 3,
        coarse_windows: list[int] | None = None,
        dropout: float = 0.1,
        use_train_robust_volatility: bool = True,
        volatility_scale_eps: float = 1e-6,
        tau_min: float = 0.5,
        tau_max: float = 5.0,
        residual_scale: float = 0.1,
        branch_beta_init: float = 0.05,
        diagnostics_level: str = "standard",
        dynamic_gating: bool = True,
        gate_mode: str | None = None,
        random_gate_seed: int = 2026,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_names = list(feature_names)
        self.volatility_source_cols = list(volatility_source_cols or ["Wspd", "Patv_clean_for_input"])
        self.source_indices = _feature_indices(self.feature_names, self.volatility_source_cols)
        self.volatility_mode = volatility_mode
        if volatility_mode not in {"per_node", "global"}:
            raise ValueError("volatility_mode must be per_node or global.")
        self.vol_window = int(vol_window)
        self.coarse_windows = list(coarse_windows or [6, 18, 36])
        self.use_train_robust_volatility = bool(use_train_robust_volatility)
        self.volatility_scale_eps = float(volatility_scale_eps)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.residual_scale = float(residual_scale)
        self.diagnostics_level = diagnostics_level
        self.dynamic_gating = bool(dynamic_gating)
        self.gate_mode = str(gate_mode or ("dynamic" if dynamic_gating else "fixed_dual"))
        if self.gate_mode not in {
            "dynamic",
            "fine_only",
            "coarse_only",
            "fixed_dual",
            "random_gate",
            "shuffled_volatility",
        }:
            raise ValueError(f"Unsupported VADSP gate_mode: {self.gate_mode}")
        self.random_gate_seed = int(random_gate_seed)
        self.requires_robust_statistics = bool(
            self.gate_mode in {"dynamic", "random_gate", "shuffled_volatility"}
            and self.use_train_robust_volatility
        )
        self.source_weight_logits = nn.Parameter(torch.zeros(len(self.source_indices)))
        self.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.raw_vol_scale = nn.Parameter(torch.tensor(0.5))
        self.raw_std_scale = nn.Parameter(torch.tensor(0.5))
        self.raw_temperature = nn.Parameter(torch.tensor(1.0))
        self.residual_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        self.scale_gate = nn.Linear(2, len(self.coarse_windows))
        self.beta_fine = nn.Parameter(torch.tensor(float(branch_beta_init)))
        self.beta_coarse = nn.Parameter(torch.tensor(float(branch_beta_init)))
        self.fine_conv = CausalDepthwiseTemporalConv(hidden_dim, fine_kernel_size, dropout)
        self.coarse_norm = nn.LayerNorm(hidden_dim)
        self.coarse_dropout = nn.Dropout(dropout)
        num_sources = len(self.source_indices)
        self.register_buffer("delta_abs_q25", torch.zeros(1, num_sources))
        self.register_buffer("delta_abs_median", torch.zeros(1, num_sources))
        self.register_buffer("delta_abs_q75", torch.ones(1, num_sources))
        self.register_buffer("robust_center", torch.zeros(1, num_sources))
        self.register_buffer("robust_scale", torch.ones(1, num_sources))
        self.register_buffer("delta_abs_iqr", torch.ones(1, num_sources))
        self.register_buffer("delta_log_median", torch.zeros(1, num_sources))
        self.register_buffer("delta_log_iqr", torch.ones(1, num_sources))
        self.register_buffer("robust_stats_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("robust_stats_num_time_points", torch.tensor(0, dtype=torch.long))
        self.register_buffer("robust_stats_num_deltas", torch.tensor(0, dtype=torch.long))
        self.register_buffer("robust_stats_nonfinite_filtered", torch.tensor(0, dtype=torch.long))
        self.register_buffer("robust_stats_degenerate_iqr_count", torch.tensor(0, dtype=torch.long))
        self.robust_statistics_metadata: dict[str, Any] = {
            "fit_split": None,
            "fit_source": None,
            "exact": None,
            "device": None,
            "mode": self.volatility_mode,
            "source_cols": list(self.volatility_source_cols),
            "depends_on_window_stride": None,
        }

    def _stats_view(self, values: Tensor, batch_size: int, seq_len: int, num_nodes: int) -> Tensor:
        if values.ndim == 1:
            values = values.view(1, -1)
        if values.ndim != 2:
            raise ValueError(f"VADSP robust statistic must be [N,K] or [1,K], got {tuple(values.shape)}.")
        if values.shape[0] == 1:
            return values.view(1, 1, 1, values.shape[1])
        if values.shape[0] != num_nodes:
            raise ValueError(f"VADSP per-node statistic has N={values.shape[0]}, expected {num_nodes}.")
        return values.view(1, 1, num_nodes, values.shape[1])

    @torch.no_grad()
    def set_robust_statistics(self, statistics: dict[str, Any]) -> None:
        required = ["q25", "median", "q75", "center", "scale", "delta_abs_iqr", "delta_log_median", "delta_log_iqr"]
        missing = [key for key in required if key not in statistics]
        if missing:
            raise KeyError(f"VADSP robust statistics missing keys: {missing}")

        def _set_buffer(name: str, value: Any) -> None:
            tensor = _as_cpu_float_tensor(value).to(device=self.robust_stats_fitted.device)
            if tensor.ndim == 1:
                tensor = tensor.view(1, -1)
            if tensor.ndim != 2 or tensor.shape[1] != len(self.source_indices):
                raise ValueError(f"{name} must be [N,K] or [1,K], got {tuple(tensor.shape)}.")
            setattr(self, name, tensor.contiguous())

        _set_buffer("delta_abs_q25", statistics["q25"])
        _set_buffer("delta_abs_median", statistics["median"])
        _set_buffer("delta_abs_q75", statistics["q75"])
        _set_buffer("robust_center", statistics["center"])
        _set_buffer("robust_scale", statistics["scale"])
        _set_buffer("delta_abs_iqr", statistics["delta_abs_iqr"])
        _set_buffer("delta_log_median", statistics["delta_log_median"])
        _set_buffer("delta_log_iqr", statistics["delta_log_iqr"])
        self.robust_stats_fitted.copy_(torch.tensor(True, device=self.robust_stats_fitted.device))
        self.robust_stats_num_time_points.copy_(torch.tensor(int(statistics.get("num_time_points", 0)), device=self.robust_stats_num_time_points.device))
        self.robust_stats_num_deltas.copy_(torch.tensor(int(statistics.get("num_time_deltas", 0)), device=self.robust_stats_num_deltas.device))
        self.robust_stats_nonfinite_filtered.copy_(torch.tensor(int(statistics.get("nonfinite_filtered", 0)), device=self.robust_stats_nonfinite_filtered.device))
        self.robust_stats_degenerate_iqr_count.copy_(torch.tensor(int(statistics.get("degenerate_iqr_count", 0)), device=self.robust_stats_degenerate_iqr_count.device))
        self.robust_statistics_metadata = {
            "fit_split": statistics.get("fit_split"),
            "fit_source": statistics.get("fit_source"),
            "exact": statistics.get("exact"),
            "device": statistics.get("device"),
            "mode": statistics.get("mode", self.volatility_mode),
            "source_cols": list(statistics.get("source_cols", self.volatility_source_cols)),
            "num_time_points": int(statistics.get("num_time_points", 0)),
            "num_time_deltas": int(statistics.get("num_time_deltas", 0)),
            "shape": dict(statistics.get("shape", {})),
            "nonfinite_filtered": int(statistics.get("nonfinite_filtered", 0)),
            "degenerate_iqr_count": int(statistics.get("degenerate_iqr_count", 0)),
            "depends_on_window_stride": bool(statistics.get("depends_on_window_stride", False)),
            "feature_space": statistics.get("feature_space"),
        }

    @torch.no_grad()
    def set_robust_statistics_from_raw(self, raw_x: Tensor) -> None:
        """Compatibility/debug fallback for small window tensors; formal training does not call this."""
        if raw_x.ndim != 4:
            raise ValueError(f"raw_x must be [B,L,N,C], got {tuple(raw_x.shape)}.")
        B, L, N, C = raw_x.shape
        timeline = raw_x.detach().to(device="cpu", dtype=torch.float32).reshape(B * L, N, C)
        statistics = compute_vadsp_robust_statistics_from_timeline(
            timeline=timeline,
            feature_names=self.feature_names,
            volatility_source_cols=self.volatility_source_cols,
            volatility_mode=self.volatility_mode,
            robust_eps=self.volatility_scale_eps,
            fit_split="unknown",
            fit_source="loader_bounded_fallback",
            feature_space="model_input_scaled",
        )
        statistics["exact"] = False
        statistics["depends_on_window_stride"] = True
        self.set_robust_statistics(statistics)

    def compute_volatility(self, raw_x: Tensor) -> tuple[Tensor, Tensor]:
        if raw_x.ndim != 4:
            raise ValueError(f"raw_x must be [B,L,N,C], got {tuple(raw_x.shape)}.")
        if self.use_train_robust_volatility and not bool(self.robust_stats_fitted.item()):
            raise RuntimeError(
                "VADSP train-robust volatility is enabled but robust statistics are not fitted. "
                "Call set_robust_statistics() with unique-train-timeline statistics before forward()."
            )
        sources = raw_x[..., self.source_indices]
        delta = torch.zeros_like(sources)
        delta[:, 1:, :, :] = (sources[:, 1:, :, :] - sources[:, :-1, :, :]).abs()
        if self.use_train_robust_volatility:
            B, L, N, _ = delta.shape
            abs_iqr = self._stats_view(self.delta_abs_iqr, B, L, N)
            log_median = self._stats_view(self.delta_log_median, B, L, N)
            log_iqr = self._stats_view(self.delta_log_iqr, B, L, N)
            delta_log = torch.log1p(delta / (abs_iqr + self.volatility_scale_eps))
            normalized_delta = (
                delta_log - log_median
            ) / (log_iqr + self.volatility_scale_eps)
        else:
            normalized_delta = delta
        source_weights = torch.softmax(self.source_weight_logits, dim=0)
        volatility = (normalized_delta * source_weights.view(1, 1, 1, -1)).sum(dim=-1)
        if self.volatility_mode == "global":
            volatility = volatility.mean(dim=2, keepdim=True).expand_as(volatility)
        rolling_std = causal_rolling_std(volatility, self.vol_window)
        return volatility, rolling_std

    def _dynamic_gate(self, volatility: Tensor, rolling_std: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gate_input = torch.stack([volatility, rolling_std], dim=-1)
        base_logit = (
            self.gate_bias
            + F.softplus(self.raw_vol_scale) * volatility
            + F.softplus(self.raw_std_scale) * rolling_std
        )
        residual_logit = self.residual_mlp(gate_input).squeeze(-1)
        temperature = torch.clamp(F.softplus(self.raw_temperature), min=self.tau_min, max=self.tau_max)
        gate = torch.sigmoid((base_logit + self.residual_scale * residual_logit) / temperature).unsqueeze(-1)
        return gate, torch.softmax(self.scale_gate(gate_input), dim=-1), temperature

    def _fixed_random_permutation(self, values: Tensor) -> Tensor:
        flat = values.reshape(-1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_gate_seed + int(flat.numel()))
        permutation = torch.randperm(flat.numel(), generator=generator, device="cpu").to(flat.device)
        return flat.index_select(0, permutation).reshape_as(values)

    def forward(self, projected_x: Tensor, raw_x: Tensor) -> dict[str, Tensor]:
        if projected_x.ndim != 4:
            raise ValueError(f"projected_x must be [B,L,N,D], got {tuple(projected_x.shape)}.")
        if self.gate_mode in {"dynamic", "random_gate", "shuffled_volatility"}:
            volatility, rolling_std = self.compute_volatility(raw_x)
            conditioning_volatility = volatility
            conditioning_rolling_std = rolling_std
            if self.gate_mode == "shuffled_volatility":
                if volatility.shape[0] > 1:
                    conditioning_volatility = torch.roll(volatility, shifts=1, dims=0)
                    conditioning_rolling_std = torch.roll(rolling_std, shifts=1, dims=0)
                else:
                    conditioning_volatility = torch.flip(volatility, dims=(1,))
                    conditioning_rolling_std = torch.flip(rolling_std, dims=(1,))
            dynamic_patch_gate, coarse_scale_weights, temperature = self._dynamic_gate(
                conditioning_volatility,
                conditioning_rolling_std,
            )
            if self.gate_mode == "random_gate":
                dynamic_patch_gate = self._fixed_random_permutation(dynamic_patch_gate)
        else:
            # Clean w/o VADSP: retain the same fine/coarse branch modules and
            # shapes, but remove all volatility dependence and learned gates.
            B, L, N, _ = projected_x.shape
            volatility = projected_x.new_zeros((B, L, N))
            rolling_std = projected_x.new_zeros((B, L, N))
            gate_input = projected_x.new_zeros((B, L, N, 2))
            gate_value = {"fine_only": 1.0, "coarse_only": 0.0, "fixed_dual": 0.5}[self.gate_mode]
            dynamic_patch_gate = projected_x.new_full((B, L, N, 1), gate_value)
            coarse_scale_weights = projected_x.new_full(
                (B, L, N, len(self.coarse_windows)),
                1.0 / len(self.coarse_windows),
            )
            temperature = projected_x.new_tensor(1.0)
        fine_delta = self.fine_conv(projected_x)
        fine_feature = projected_x + self.beta_fine * fine_delta
        coarse_features = [causal_rolling_mean(projected_x, window) for window in self.coarse_windows]
        stacked_coarse = torch.stack(coarse_features, dim=-2)
        coarse_delta = (stacked_coarse * coarse_scale_weights.unsqueeze(-1)).sum(dim=-2)
        coarse_delta = self.coarse_dropout(self.coarse_norm(coarse_delta))
        coarse_feature = projected_x + self.beta_coarse * coarse_delta
        x_fine = dynamic_patch_gate * fine_feature
        x_coarse = (1.0 - dynamic_patch_gate) * coarse_feature
        result = {
            "x_fine": x_fine,
            "x_coarse": x_coarse,
            "vadsp_mode": self.gate_mode,
            "x_fine_shape": list(x_fine.shape),
            "x_coarse_shape": list(x_coarse.shape),
        }
        if self.diagnostics_level in {"none", "minimal"}:
            if self.diagnostics_level == "minimal":
                with torch.no_grad():
                    result.update(
                        {
                            "dynamic_patch_gate_min": float(dynamic_patch_gate.detach().min().cpu().item()),
                            "dynamic_patch_gate_max": float(dynamic_patch_gate.detach().max().cpu().item()),
                            "dynamic_patch_gate_mean": float(dynamic_patch_gate.detach().mean().cpu().item()),
                        }
                    )
            return result
        with torch.no_grad():
            result.update({
            "fine_feature": fine_feature,
            "coarse_feature": coarse_feature,
            "volatility": volatility,
            "rolling_std": rolling_std,
            "dynamic_patch_gate": dynamic_patch_gate,
            "dynamic_patch_gate_min": float(dynamic_patch_gate.detach().min().cpu().item()),
            "dynamic_patch_gate_max": float(dynamic_patch_gate.detach().max().cpu().item()),
            "dynamic_patch_gate_mean": float(dynamic_patch_gate.detach().mean().cpu().item()),
            "source_weights": torch.softmax(self.source_weight_logits.detach(), dim=0),
            "temperature": temperature.detach(),
            "beta_fine": self.beta_fine.detach(),
            "beta_coarse": self.beta_coarse.detach(),
            "coarse_scale_weights": coarse_scale_weights.detach(),
            "fine_feature_norm": fine_feature.detach().float().norm(dim=-1).mean().cpu(),
            "coarse_feature_norm": coarse_feature.detach().float().norm(dim=-1).mean().cpu(),
            "weighted_fine_norm": x_fine.detach().float().norm(dim=-1).mean().cpu(),
            "weighted_coarse_norm": x_coarse.detach().float().norm(dim=-1).mean().cpu(),
            "robust_stats_fitted": self.robust_stats_fitted.detach(),
            "vadsp_statistics_num_time_points": self.robust_stats_num_time_points.detach(),
            "vadsp_statistics_num_deltas": self.robust_stats_num_deltas.detach(),
            "vadsp_statistics_nonfinite_filtered": self.robust_stats_nonfinite_filtered.detach(),
            "vadsp_statistics_degenerate_iqr_count": self.robust_stats_degenerate_iqr_count.detach(),
            })
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        num_sources = len(self.source_indices)
        legacy_abs_iqr_key = prefix + "delta_abs_iqr"
        legacy_log_median_key = prefix + "delta_log_median"
        legacy_log_iqr_key = prefix + "delta_log_iqr"
        legacy_abs_iqr = state_dict.get(legacy_abs_iqr_key)
        legacy_log_median = state_dict.get(legacy_log_median_key)
        legacy_log_iqr = state_dict.get(legacy_log_iqr_key)
        if torch.is_tensor(legacy_abs_iqr) and legacy_abs_iqr.ndim == 1:
            legacy_abs_iqr = legacy_abs_iqr.view(1, -1)
        if torch.is_tensor(legacy_log_median) and legacy_log_median.ndim == 1:
            legacy_log_median = legacy_log_median.view(1, -1)
        if torch.is_tensor(legacy_log_iqr) and legacy_log_iqr.ndim == 1:
            legacy_log_iqr = legacy_log_iqr.view(1, -1)
        legacy_checkpoint = any(
            prefix + name not in state_dict
            for name in ["delta_abs_q25", "delta_abs_median", "delta_abs_q75", "robust_center", "robust_scale"]
        )
        for name in [
            "delta_abs_q25",
            "delta_abs_median",
            "delta_abs_q75",
            "robust_center",
            "robust_scale",
            "delta_abs_iqr",
            "delta_log_median",
            "delta_log_iqr",
        ]:
            key = prefix + name
            if key in state_dict:
                value = state_dict[key]
                if torch.is_tensor(value) and value.ndim == 1:
                    value = value.view(1, -1)
                    state_dict[key] = value
                if torch.is_tensor(value) and tuple(getattr(self, name).shape) != tuple(value.shape):
                    setattr(self, name, torch.zeros_like(value, device=getattr(self, name).device))
            else:
                if name in {"delta_abs_iqr", "robust_scale"} and torch.is_tensor(legacy_abs_iqr):
                    state_dict[key] = legacy_abs_iqr.clone()
                elif name == "delta_log_median" and torch.is_tensor(legacy_log_median):
                    state_dict[key] = legacy_log_median.clone()
                elif name == "delta_log_iqr" and torch.is_tensor(legacy_log_iqr):
                    state_dict[key] = legacy_log_iqr.clone()
                elif name in {"delta_abs_median", "robust_center"}:
                    source = torch.zeros_like(legacy_abs_iqr) if torch.is_tensor(legacy_abs_iqr) else torch.zeros(1, num_sources)
                    state_dict[key] = source
                elif name == "delta_abs_q75" and torch.is_tensor(legacy_abs_iqr):
                    state_dict[key] = legacy_abs_iqr.clone()
                else:
                    default = getattr(self, name).detach().clone()
                    if default.ndim == 1:
                        default = default.view(1, num_sources)
                    state_dict[key] = default
        for name in [
            "robust_stats_num_time_points",
            "robust_stats_num_deltas",
            "robust_stats_nonfinite_filtered",
            "robust_stats_degenerate_iqr_count",
        ]:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        if prefix + "robust_stats_fitted" not in state_dict:
            state_dict[prefix + "robust_stats_fitted"] = torch.tensor(torch.is_tensor(legacy_abs_iqr), dtype=torch.bool)
        if legacy_checkpoint:
            self.robust_statistics_metadata = {
                "fit_split": "unknown",
                "fit_source": "legacy_checkpoint",
                "exact": False,
                "device": "unknown",
                "mode": self.volatility_mode,
                "source_cols": list(self.volatility_source_cols),
                "num_time_points": 0,
                "num_time_deltas": 0,
                "shape": {
                    "center": list(state_dict[prefix + "robust_center"].shape),
                    "scale": list(state_dict[prefix + "robust_scale"].shape),
                },
                "nonfinite_filtered": 0,
                "degenerate_iqr_count": 0,
                "depends_on_window_stride": True,
                "feature_space": "legacy_unknown",
                "eligible_for_fair_main_table": False,
            }
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    a = a[valid]
    b = b[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def save_vadsp_diagnostics_from_aux(
    aux_batches: list[dict[str, Any]],
    output_dir: str | Path,
    volatility_source_cols: list[str],
    robust_statistics_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    volatility = np.concatenate([item["volatility"] for item in aux_batches], axis=0)
    rolling_std = np.concatenate([item["rolling_std"] for item in aux_batches], axis=0)
    gate = np.concatenate([item["dynamic_patch_gate"] for item in aux_batches], axis=0)
    delta = np.concatenate([item["delta_source_abs"] for item in aux_batches], axis=0)
    gate_flat = gate.reshape(-1)
    vol_flat = volatility.reshape(-1)
    std_flat = rolling_std.reshape(-1)
    q = np.nanpercentile(gate_flat, [1, 10, 25, 50, 75, 90, 99])
    low_thr, high_thr = np.nanpercentile(vol_flat, [10, 90])
    low_mask = vol_flat <= low_thr
    mid_mask = (vol_flat > low_thr) & (vol_flat < high_thr)
    high_mask = vol_flat >= high_thr
    weighted_fine = [item.get("weighted_fine_norm") for item in aux_batches if "weighted_fine_norm" in item]
    weighted_coarse = [item.get("weighted_coarse_norm") for item in aux_batches if "weighted_coarse_norm" in item]
    scale_weights = [item.get("coarse_scale_weights") for item in aux_batches if "coarse_scale_weights" in item]
    source_weights = aux_batches[-1].get("source_weights") if aux_batches and "source_weights" in aux_batches[-1] else None
    summary = {
        "volatility_mean": float(np.mean(volatility)),
        "volatility_std": float(np.std(volatility)),
        "rolling_std_mean": float(np.mean(rolling_std)),
        "rolling_std_std": float(np.std(rolling_std)),
        "gate_mean": float(np.mean(gate)),
        "gate_std": float(np.std(gate)),
        "gate_min": float(np.min(gate)),
        "gate_max": float(np.max(gate)),
        "gate_p01": float(q[0]),
        "gate_p10": float(q[1]),
        "gate_p25": float(q[2]),
        "gate_p50": float(q[3]),
        "gate_p75": float(q[4]),
        "gate_p90": float(q[5]),
        "gate_p99": float(q[6]),
        "gate_inter_quantile_range": float(q[4] - q[2]),
        "gate_p99_minus_p01": float(q[6] - q[0]),
        "gate_mean_lowest_10pct_volatility": float(np.nanmean(gate_flat[low_mask])) if np.any(low_mask) else float("nan"),
        "gate_mean_middle_volatility": float(np.nanmean(gate_flat[mid_mask])) if np.any(mid_mask) else float("nan"),
        "gate_mean_highest_10pct_volatility": float(np.nanmean(gate_flat[high_mask])) if np.any(high_mask) else float("nan"),
        "gate_high_minus_low": (
            float(np.nanmean(gate_flat[high_mask]) - np.nanmean(gate_flat[low_mask]))
            if np.any(high_mask) and np.any(low_mask)
            else float("nan")
        ),
        "corr_gate_volatility": _safe_corr(gate_flat, vol_flat),
        "corr_gate_rolling_std": _safe_corr(gate_flat, std_flat),
        "corr_gate_abs_delta_source": _safe_corr(gate, delta),
        "fine_feature_norm": float(np.nanmean([item["fine_feature_norm"] for item in aux_batches if "fine_feature_norm" in item]))
        if any("fine_feature_norm" in item for item in aux_batches)
        else float("nan"),
        "coarse_feature_norm": float(np.nanmean([item["coarse_feature_norm"] for item in aux_batches if "coarse_feature_norm" in item]))
        if any("coarse_feature_norm" in item for item in aux_batches)
        else float("nan"),
        "weighted_fine_norm": float(np.nanmean(weighted_fine)) if weighted_fine else float("nan"),
        "weighted_coarse_norm": float(np.nanmean(weighted_coarse)) if weighted_coarse else float("nan"),
        "fine_to_coarse_weighted_norm_ratio": (
            float(np.nanmean(weighted_fine) / max(np.nanmean(weighted_coarse), 1e-12)) if weighted_fine and weighted_coarse else float("nan")
        ),
        "source_weights": source_weights.tolist() if isinstance(source_weights, np.ndarray) else source_weights,
        "temperature": float(aux_batches[-1].get("temperature", np.nan)),
        "beta_fine": float(aux_batches[-1].get("beta_fine", np.nan)),
        "beta_coarse": float(aux_batches[-1].get("beta_coarse", np.nan)),
        "coarse_scale_weight_mean_per_scale": (
            np.nanmean(np.concatenate(scale_weights, axis=0), axis=(0, 1, 2)).tolist() if scale_weights else []
        ),
        "volatility_source_cols": list(volatility_source_cols),
        "statistics_fit_split": "train_only",
        "history_only": True,
        "uses_target_mask": False,
        "uses_future_inputs": False,
    }
    if robust_statistics_metadata:
        summary.update(
            {
                "vadsp_robust_statistics_enabled": True,
                "vadsp_statistics_fit_split": robust_statistics_metadata.get("fit_split"),
                "vadsp_statistics_source": robust_statistics_metadata.get("fit_source"),
                "vadsp_statistics_exact": robust_statistics_metadata.get("exact"),
                "vadsp_statistics_device": robust_statistics_metadata.get("device"),
                "vadsp_statistics_mode": robust_statistics_metadata.get("mode"),
                "vadsp_statistics_source_cols": robust_statistics_metadata.get("source_cols"),
                "vadsp_statistics_num_time_points": robust_statistics_metadata.get("num_time_points"),
                "vadsp_statistics_num_deltas": robust_statistics_metadata.get("num_time_deltas"),
                "vadsp_statistics_shape": robust_statistics_metadata.get("shape"),
                "vadsp_statistics_nonfinite_filtered": robust_statistics_metadata.get("nonfinite_filtered"),
                "vadsp_statistics_degenerate_iqr_count": robust_statistics_metadata.get("degenerate_iqr_count"),
                "vadsp_statistics_depends_on_window_stride": robust_statistics_metadata.get("depends_on_window_stride"),
                "vadsp_statistics_feature_space": robust_statistics_metadata.get("feature_space"),
            }
        )
    (output_dir / "vadsp_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(gate.reshape(-1), bins=40)
        plt.xlabel("dynamic_patch_gate")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / "vadsp_gate_hist.png", dpi=140)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.scatter(vol_flat[:: max(1, len(vol_flat) // 2000)], gate_flat[:: max(1, len(gate_flat) // 2000)], s=4, alpha=0.35)
        plt.xlabel("volatility")
        plt.ylabel("dynamic_patch_gate")
        plt.tight_layout()
        plt.savefig(output_dir / "vadsp_gate_vs_volatility.png", dpi=140)
        plt.close()

        order = np.argsort(vol_flat)
        bins = np.array_split(order, 10)
        means = [float(np.nanmean(gate_flat[idx])) for idx in bins if len(idx)]
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(means) + 1), means, marker="o")
        plt.xlabel("volatility decile")
        plt.ylabel("gate mean")
        plt.tight_layout()
        plt.savefig(output_dir / "vadsp_gate_by_volatility_quantile.png", dpi=140)
        plt.close()

        if scale_weights:
            weights = np.nanmean(np.concatenate(scale_weights, axis=0), axis=(0, 1, 2))
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(weights) + 1), weights, marker="o")
            plt.xlabel("coarse scale index")
            plt.ylabel("mean scale weight")
            plt.tight_layout()
            plt.savefig(output_dir / "coarse_scale_weight_curve.png", dpi=140)
            plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(volatility.reshape(-1), bins=40)
        plt.xlabel("volatility")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / "volatility_hist.png", dpi=140)
        plt.close()
    except Exception as exc:
        summary["plot_error"] = str(exc)
        (output_dir / "vadsp_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary
