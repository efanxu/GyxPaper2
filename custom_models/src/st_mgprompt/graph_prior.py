from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import STMGPromptConfig
from .data import STMGPromptDataBundle


EPS = 1e-8


def _as_2d_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"train_series must be [T,N], got {arr.shape}.")
    return arr


def _row_normalize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    A = np.where(np.isfinite(A), A, 0.0)
    row_sum = A.sum(axis=1, keepdims=True)
    bad_rows = row_sum.squeeze(-1) <= EPS
    if np.any(bad_rows):
        A[bad_rows] = 0.0
        A[bad_rows, bad_rows] = 1.0
        row_sum = A.sum(axis=1, keepdims=True)
    return (A / np.maximum(row_sum, EPS)).astype(np.float32)


def _top_k(A: np.ndarray, k: int, keep_self: bool) -> np.ndarray:
    N = A.shape[0]
    k = min(max(int(k), 1), N)
    sparse = np.zeros_like(A, dtype=np.float32)
    for i in range(N):
        row = A[i].copy()
        if not keep_self:
            row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        sparse[i, idx] = np.maximum(A[i, idx], 0.0)
        if keep_self:
            sparse[i, i] = max(float(A[i, i]), 1.0)
    return sparse


def _causal_moving_average(series: np.ndarray, window: int) -> np.ndarray:
    T, N = series.shape
    out = np.zeros((T, N), dtype=np.float32)
    counts = np.zeros((T, N), dtype=np.float32)
    values = np.where(np.isfinite(series), series, 0.0).astype(np.float32)
    valid = np.isfinite(series).astype(np.float32)
    cs_values = np.cumsum(values, axis=0)
    cs_valid = np.cumsum(valid, axis=0)
    for t in range(T):
        start = max(0, t - window + 1)
        total = cs_values[t] - (cs_values[start - 1] if start > 0 else 0.0)
        count = cs_valid[t] - (cs_valid[start - 1] if start > 0 else 0.0)
        out[t] = total / np.maximum(count, 1.0)
        counts[t] = count
    out[counts <= 0] = np.nan
    return out


def _causal_ema(series: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(series, dtype=np.float32)
    running = np.zeros(series.shape[1], dtype=np.float32)
    seen = np.zeros(series.shape[1], dtype=bool)
    for t in range(series.shape[0]):
        current = series[t]
        valid = np.isfinite(current)
        running[valid & ~seen] = current[valid & ~seen]
        seen |= valid
        running[valid] = alpha * current[valid] + (1.0 - alpha) * running[valid]
        out[t] = np.where(seen, running, np.nan)
    return out


def _extract_trend(series: np.ndarray, config: STMGPromptConfig) -> np.ndarray:
    if config.trend_method == "causal_moving_average":
        return _causal_moving_average(series, config.trend_window)
    if config.trend_method == "causal_ema":
        return _causal_ema(series, config.trend_ema_alpha)
    if config.trend_method == "almon_smooth":
        raise NotImplementedError("almon_smooth is reserved for ablation and is not a Step 2 default.")
    raise ValueError(f"Unsupported trend_method: {config.trend_method}")


def _fill_nan_by_column(series: np.ndarray) -> np.ndarray:
    filled = np.asarray(series, dtype=np.float32).copy()
    col_mean = np.nanmean(filled, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    row_idx, col_idx = np.where(~np.isfinite(filled))
    filled[row_idx, col_idx] = col_mean[col_idx]
    return filled


def _pairwise_similarity_graph(series: np.ndarray, metric: str, top_k: int, keep_self: bool) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(series, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"series must be [T,N], got {values.shape}.")
    N = values.shape[1]
    sim = np.zeros((N, N), dtype=np.float32)
    valid_counts = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            valid_counts[i, j] = int(valid.sum())
            if valid_counts[i, j] < 3:
                continue
            a = values[valid, i].astype(np.float64)
            b = values[valid, j].astype(np.float64)
            if metric == "pearson":
                a = a - a.mean()
                b = b - b.mean()
            elif metric != "cosine":
                raise ValueError(f"Unsupported graph similarity metric: {metric}")
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > EPS:
                sim[i, j] = max(float(np.dot(a, b) / denom), 0.0)
    if keep_self:
        np.fill_diagonal(sim, 1.0)
    else:
        np.fill_diagonal(sim, 0.0)
    graph = _row_normalize(_top_k(sim, top_k, keep_self))
    metadata = {
        "pairwise_valid_count_min": int(valid_counts.min()),
        "pairwise_valid_count_median": float(np.median(valid_counts)),
        "pairwise_valid_count_max": int(valid_counts.max()),
        "pairwise_reliability_mean": float(np.mean(valid_counts / max(values.shape[0], 1))),
    }
    return graph, metadata


def _similarity_graph(
    trend_series: np.ndarray,
    config: STMGPromptConfig,
    metric: str | None = None,
    top_k: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _pairwise_similarity_graph(
        trend_series,
        metric or config.graph_similarity_metric,
        int(top_k if top_k is not None else config.graph_top_k),
        bool(config.graph_self_loop),
    )


def _location_frame(location: Any) -> pd.DataFrame | None:
    if location is None:
        return None
    if isinstance(location, pd.DataFrame):
        return location.copy()
    path = Path(location)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _location_coordinates(
    location: Any,
    num_nodes: int,
    turbine_ids: list[int] | None = None,
    config: STMGPromptConfig | None = None,
) -> np.ndarray | None:
    df = _location_frame(location)
    if df is None:
        return None
    lower_to_col = {c.lower(): c for c in df.columns}
    if "turbid" in lower_to_col:
        id_col = lower_to_col["turbid"]
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        if turbine_ids is not None:
            df = df.set_index(id_col).reindex(turbine_ids).reset_index()
    requested = list(getattr(config, "graph_coordinate_cols", ["x", "y"]))
    if bool(getattr(config, "graph_use_elevation", False)):
        requested.append("elevation")
    coord_cols = []
    for name in [*requested, "x", "y"]:
        if name in lower_to_col and lower_to_col[name] in df.columns:
            coord_cols.append(lower_to_col[name])
    coord_cols = list(dict.fromkeys(coord_cols))
    if len(coord_cols) < 2:
        numeric_cols = [
            c
            for c in df.columns
            if c.lower() not in {"turbid", "turbine_id"} and pd.api.types.is_numeric_dtype(df[c])
        ]
        coord_cols = numeric_cols[:3]
    if len(coord_cols) < 2:
        return None
    coords = df[coord_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    coords = coords[:num_nodes]
    if coords.shape[0] != num_nodes or not np.isfinite(coords).all():
        return None
    return coords


def build_distance_prior_graph(
    location,
    config: STMGPromptConfig,
    *,
    top_k: int | None = None,
) -> np.ndarray:
    num_nodes = int(getattr(config, "_graph_num_nodes", config.num_nodes))
    turbine_ids = getattr(config, "_graph_turbine_ids", None)
    coords = _location_coordinates(location, num_nodes=num_nodes, turbine_ids=turbine_ids, config=config)
    if coords is None:
        if config.smoke and config.graph_missing_location_policy == "identity_for_synthetic_smoke":
            return np.eye(num_nodes, dtype=np.float32)
        raise ValueError("Location coordinates are required for graph construction; no silent identity fallback.")
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    nonzero = dist2[dist2 > EPS]
    sigma = config.graph_distance_sigma
    if sigma is None:
        sigma = float(np.sqrt(np.median(nonzero))) if nonzero.size else 1.0
    A = np.exp(-dist2 / max(float(sigma) ** 2, EPS)).astype(np.float32)
    if config.graph_self_loop:
        np.fill_diagonal(A, 1.0)
    else:
        np.fill_diagonal(A, 0.0)
    return _row_normalize(
        _top_k(A, int(config.graph_top_k if top_k is None else top_k), config.graph_self_loop)
    )


def degree_preserving_random_rewire(
    adjacency: np.ndarray,
    *,
    seed: int,
    keep_self: bool,
) -> np.ndarray:
    """Randomize row-wise support while preserving every row's nonzero count."""

    source = np.asarray(adjacency, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError(f"adjacency must be square, got {source.shape}.")
    node_count = source.shape[0]
    rng = np.random.default_rng(int(seed))
    rewired = np.zeros_like(source)
    for row_idx in range(node_count):
        support = np.flatnonzero(source[row_idx] > EPS)
        has_self = bool(row_idx in support)
        if has_self != bool(keep_self):
            raise ValueError(
                f"Row {row_idx} self-loop state {has_self} does not match declared keep_self={keep_self}."
            )
        offdiag_support = support[support != row_idx]
        candidates = np.delete(np.arange(node_count, dtype=np.int64), row_idx)
        if offdiag_support.size > candidates.size:
            raise ValueError("Cannot preserve row degree during random rewiring.")
        chosen = rng.choice(candidates, size=offdiag_support.size, replace=False)
        weights = source[row_idx, offdiag_support].copy()
        rng.shuffle(weights)
        rewired[row_idx, chosen] = weights
        if has_self:
            rewired[row_idx, row_idx] = source[row_idx, row_idx]
    before = np.count_nonzero(source > EPS, axis=1)
    after = np.count_nonzero(rewired > EPS, axis=1)
    if not np.array_equal(before, after):
        raise RuntimeError("Degree-preserving rewiring changed at least one row degree.")
    return _row_normalize(rewired)


def _select_prior_component(
    statistics_graph: np.ndarray,
    distance_graph: np.ndarray,
    *,
    component: str,
    alpha: float,
) -> tuple[np.ndarray, list[str]]:
    if component == "distance_only":
        return distance_graph.copy(), ["distance"]
    if component == "statistics_only":
        return statistics_graph.copy(), ["train_statistics"]
    if component == "fused":
        return _row_normalize(alpha * statistics_graph + (1.0 - alpha) * distance_graph), [
            "distance",
            "train_statistics",
        ]
    raise ValueError(f"Unsupported graph_prior_component: {component}")


def reconstruct_variant_graphs_from_fused(
    macro_fused: np.ndarray,
    micro_fused: np.ndarray,
    location,
    config: STMGPromptConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct Step-4 priors from canonical fused checkpoint buffers."""

    distance = build_distance_prior_graph(location, config)
    if config.macro_alpha <= 0 or config.micro_alpha <= 0:
        raise ValueError("Statistics-only reconstruction requires positive canonical graph alpha values.")
    macro_statistics = _row_normalize(
        np.maximum(
            (np.asarray(macro_fused, dtype=np.float32) - (1.0 - config.macro_alpha) * distance)
            / config.macro_alpha,
            0.0,
        )
    )
    micro_statistics = _row_normalize(
        np.maximum(
            (np.asarray(micro_fused, dtype=np.float32) - (1.0 - config.micro_alpha) * distance)
            / config.micro_alpha,
            0.0,
        )
    )
    macro, macro_sources = _select_prior_component(
        macro_statistics,
        distance,
        component=config.graph_prior_component,
        alpha=config.macro_alpha,
    )
    micro, micro_sources = _select_prior_component(
        micro_statistics,
        distance,
        component=config.graph_prior_component,
        alpha=config.micro_alpha,
    )
    if config.graph_rewire_mode == "degree_preserving_random":
        macro = degree_preserving_random_rewire(
            macro, seed=int(config.seed) + 1009, keep_self=bool(config.graph_self_loop)
        )
        micro = degree_preserving_random_rewire(
            micro, seed=int(config.seed) + 2017, keep_self=bool(config.graph_self_loop)
        )
    return macro, micro, {
        "macro_source_components": macro_sources,
        "micro_source_components": micro_sources,
        "reconstructed_from_fused_checkpoint": True,
        "graph_rewire_seed": int(config.seed),
    }


def build_trend_prior_graph(
    train_series,
    train_mask=None,
    location=None,
    config: STMGPromptConfig | None = None,
) -> np.ndarray:
    config = config or STMGPromptConfig()
    series = _as_2d_series(train_series).copy()
    if train_mask is not None:
        mask = np.asarray(train_mask).astype(bool)
        if mask.shape != series.shape:
            raise ValueError(f"train_mask shape {mask.shape} must match train_series {series.shape}.")
        series = np.where(mask, series, np.nan)
    trend = _extract_trend(series, config)
    graph, _ = _similarity_graph(trend, config)
    return graph


def build_macro_trend_graph(train_series, train_mask, location, config: STMGPromptConfig) -> np.ndarray:
    series = _as_2d_series(train_series).copy()
    if train_mask is not None:
        series = np.where(np.asarray(train_mask).astype(bool), series, np.nan)
    trend = _extract_trend(series, config)
    A_trend, _ = _similarity_graph(
        trend,
        config,
        metric=config.macro_similarity_metric,
        top_k=config.macro_top_k,
    )
    A_dist = build_distance_prior_graph(location, config)
    graph, _ = _select_prior_component(
        A_trend,
        A_dist,
        component=config.graph_prior_component,
        alpha=config.macro_alpha,
    )
    return graph


def build_micro_local_graph(train_series, train_mask, location, config: STMGPromptConfig) -> np.ndarray:
    A_dist = build_distance_prior_graph(location, config)
    if train_series is None:
        return A_dist
    series = _as_2d_series(train_series).copy()
    if train_mask is not None:
        series = np.where(np.asarray(train_mask).astype(bool), series, np.nan)
    if config.micro_graph_source == "distance_delta_wspd":
        micro_series = np.zeros_like(series, dtype=np.float32)
        micro_series[1:] = series[1:] - series[:-1]
        micro_series[~np.isfinite(micro_series)] = np.nan
    else:
        trend = _causal_moving_average(series, max(int(config.macro_trend_window), 1))
        micro_series = series - trend
    A_local, _ = _similarity_graph(
        micro_series,
        config,
        metric=config.micro_similarity_metric,
        top_k=config.micro_top_k,
    )
    graph, _ = _select_prior_component(
        A_local,
        A_dist,
        component=config.graph_prior_component,
        alpha=config.micro_alpha,
    )
    return graph


def build_wake_prior_graph(*args, **kwargs):
    raise NotImplementedError("Wake/delay prior graph is not implemented in Step 2.")


def _train_graph_series(data: STMGPromptDataBundle, config: STMGPromptConfig) -> tuple[np.ndarray, np.ndarray | None]:
    dataset = data.train_loader.dataset
    S, E = data.split_indices["train"]
    if config.trend_source_col == config.target_col:
        series = dataset.y_raw[S:E]
        mask = dataset.mask[S:E].astype(bool)
        return series, mask
    if config.trend_source_col not in data.feature_cols:
        raise ValueError(f"trend_source_col not available: {config.trend_source_col}")
    idx = data.feature_cols.index(config.trend_source_col)
    series = dataset.x[S:E, :, idx]
    return series, np.isfinite(series)


def _train_feature_series(data: STMGPromptDataBundle, col: str) -> tuple[np.ndarray, np.ndarray | None]:
    if col not in data.feature_cols:
        raise ValueError(f"Graph feature source not available: {col}")
    dataset = data.train_loader.dataset
    S, E = data.split_indices["train"]
    idx = data.feature_cols.index(col)
    series = dataset.x[S:E, :, idx]
    return series, np.isfinite(series)


def _graph_paths(config: STMGPromptConfig) -> tuple[Path, Path, Path]:
    variant_value = getattr(config, "variant", None)
    variant = str(variant_value).upper() if variant_value else ""
    if variant[:1] in {"T", "G", "D", "F", "N", "L", "R"}:
        graph_dir = config.resolve_path(config.output_root) / str(config.run_id) / "_graph_artifacts" / config.graph_tag
    else:
        graph_dir = config.resolve_path(config.graph_output_root) / config.graph_tag
    return graph_dir, graph_dir / "macro_trend_adjacency.npy", graph_dir / "micro_local_adjacency.npy"


def _row_degree_summary(adjacency: np.ndarray) -> dict[str, float | int]:
    degree = np.count_nonzero(np.asarray(adjacency) > EPS, axis=1)
    return {
        "min": int(degree.min()),
        "median": float(np.median(degree)),
        "mean": float(degree.mean()),
        "max": int(degree.max()),
    }


def summarize_graph_identity(
    macro: np.ndarray,
    micro: np.ndarray,
    config: STMGPromptConfig,
    *,
    node_order: list[int] | None = None,
    location_path: str | Path | None = None,
    source_components: list[str] | None = None,
    fit_split: str = "train_only",
) -> dict[str, Any]:
    macro = np.asarray(macro, dtype=np.float32)
    micro = np.asarray(micro, dtype=np.float32)
    if macro.shape != micro.shape or macro.ndim != 2 or macro.shape[0] != macro.shape[1]:
        raise ValueError(f"Macro and Micro graphs must be matching square matrices, got {macro.shape}, {micro.shape}.")
    count = int(macro.shape[0])
    order = node_order if node_order is not None else list(range(1, count + 1))
    return {
        "node_count": count,
        "node_order_source": "STMGPromptDataBundle.turbine_ids" if node_order is not None else "canonical_1_based_order",
        "node_order": [int(value) for value in order],
        "coordinate_columns": list(config.graph_coordinate_cols),
        "location_file": str(Path(location_path or config.resolve_path(config.location_path)).resolve()),
        "distance_definition": "exp(-squared_euclidean_distance/sigma^2)",
        "micro_top_k": int(config.micro_top_k),
        "macro_top_k": int(config.macro_top_k),
        "distance_top_k": int(config.graph_top_k),
        "self_loop": bool(config.graph_self_loop),
        "normalization": "row_sum_one",
        "elevation_used": bool(config.graph_use_elevation),
        "micro_source": config.micro_graph_source,
        "macro_source": config.macro_graph_source,
        "source_components": source_components
        if source_components is not None
        else {
            "fused": ["distance", "train_statistics"],
            "distance_only": ["distance"],
            "statistics_only": ["train_statistics"],
        }[config.graph_prior_component],
        "fit_split": fit_split,
        "edge_count": {"micro": int((micro > EPS).sum()), "macro": int((macro > EPS).sum())},
        "row_degree_summary": {
            "micro": _row_degree_summary(micro),
            "macro": _row_degree_summary(macro),
        },
        "branch_graph_assignment": config.branch_graph_assignment,
        "adaptive_support_mode": config.adaptive_support_mode,
        "graph_prior_component": config.graph_prior_component,
        "graph_rewire_mode": config.graph_rewire_mode,
        "graph_rewire_seed": int(config.seed) if config.graph_rewire_mode != "none" else None,
        "directionality": "directed_rowwise_top_k",
        "not_applicable_reason": None,
    }


def build_graph_artifacts(data: STMGPromptDataBundle, config: STMGPromptConfig) -> dict[str, Any]:
    graph_dir, macro_path, micro_path = _graph_paths(config)
    graph_dir.mkdir(parents=True, exist_ok=True)

    setattr(config, "_graph_num_nodes", data.num_nodes)
    setattr(config, "_graph_turbine_ids", data.turbine_ids)
    train_series, train_mask = _train_graph_series(data, config)
    micro_series, micro_mask = _train_feature_series(data, "Wspd")
    location_path = config.resolve_path(config.location_path)
    A_macro = build_macro_trend_graph(train_series, train_mask, location_path, config)
    A_micro = build_micro_local_graph(micro_series, micro_mask, location_path, config)
    if config.graph_rewire_mode == "degree_preserving_random":
        macro_degree_before = np.count_nonzero(A_macro > EPS, axis=1)
        micro_degree_before = np.count_nonzero(A_micro > EPS, axis=1)
        A_macro = degree_preserving_random_rewire(
            A_macro,
            seed=int(config.seed) + 1009,
            keep_self=bool(config.graph_self_loop),
        )
        A_micro = degree_preserving_random_rewire(
            A_micro,
            seed=int(config.seed) + 2017,
            keep_self=bool(config.graph_self_loop),
        )
        if not np.array_equal(macro_degree_before, np.count_nonzero(A_macro > EPS, axis=1)):
            raise RuntimeError("Macro graph rewiring failed the row-degree contract.")
        if not np.array_equal(micro_degree_before, np.count_nonzero(A_micro > EPS, axis=1)):
            raise RuntimeError("Micro graph rewiring failed the row-degree contract.")
    np.save(macro_path, A_macro)
    np.save(micro_path, A_micro)
    macro_support = A_macro > EPS
    micro_support = A_micro > EPS
    union = np.logical_or(macro_support, micro_support).sum()
    support_jaccard = float(np.logical_and(macro_support, micro_support).sum() / max(int(union), 1))
    weight_corr = float(np.corrcoef(A_macro.reshape(-1), A_micro.reshape(-1))[0, 1]) if A_macro.size > 1 else float("nan")

    graph_path = str(graph_dir).replace("\\", "/") + "/"
    component_sources = {
        "fused": ["distance", "train_statistics"],
        "distance_only": ["distance"],
        "statistics_only": ["train_statistics"],
    }[config.graph_prior_component]
    graph_identity = summarize_graph_identity(
        A_macro,
        A_micro,
        config,
        node_order=data.turbine_ids,
        location_path=location_path,
        source_components=component_sources,
    )
    metadata = {
        "uses_custom_graph": True,
        "graph_type": "trend_prior_adaptive_dual_graph",
        "graph_tag": config.graph_tag,
        "graph_path": graph_path,
        "graph_uses_train_only_statistics": True,
        "graph_uses_test_statistics": False,
        "graph_uses_future_target": False,
        "edge_weight_used": True,
        "graph_normalize": True,
        "uses_macro_trend_graph": True,
        "uses_micro_local_graph": True,
        "macro_graph_source": config.macro_graph_source,
        "micro_graph_source": config.micro_graph_source,
        "trend_extraction_method": config.trend_method,
        "trend_statistics_fit_split": "train_only",
        "uses_adaptive_graph": bool(config.use_adaptive_graph and config.adaptive_support_mode != "fixed"),
        "adaptive_graph_constrained_by_prior": bool(
            config.use_adaptive_graph and config.adaptive_support_mode == "prior_constrained"
        ),
        "adaptive_support_mode": config.adaptive_support_mode,
        "branch_graph_assignment": config.branch_graph_assignment,
        "graph_prior_component": config.graph_prior_component,
        "graph_rewire_mode": config.graph_rewire_mode,
        "source_components": component_sources,
        "graph_fusion_mode": config.graph_fusion_mode,
        "target_mask_used_for_graph_statistics": bool(config.trend_source_col == config.target_col),
        "target_mask_used_as_model_input": False,
        "macro_graph_path": str(macro_path).replace("\\", "/"),
        "micro_graph_path": str(micro_path).replace("\\", "/"),
        "macro_row_sum_min": float(A_macro.sum(axis=1).min()),
        "macro_row_sum_max": float(A_macro.sum(axis=1).max()),
        "micro_row_sum_min": float(A_micro.sum(axis=1).min()),
        "micro_row_sum_max": float(A_micro.sum(axis=1).max()),
        "num_nodes": int(data.num_nodes),
        "trend_source_col": config.trend_source_col,
        "graph_top_k": int(config.graph_top_k),
        "macro_top_k": int(config.macro_top_k),
        "micro_top_k": int(config.micro_top_k),
        "macro_alpha": float(config.macro_alpha),
        "micro_alpha": float(config.micro_alpha),
        "macro_micro_prior_support_jaccard": support_jaccard,
        "macro_micro_prior_weight_correlation": weight_corr,
        "graph_self_loop": bool(config.graph_self_loop),
        "graph_coordinate_cols": list(config.graph_coordinate_cols),
        "graph_use_elevation": bool(config.graph_use_elevation),
        "graph_identity": graph_identity,
    }
    (graph_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"A_macro_trend": A_macro, "A_micro_local": A_micro, "metadata": metadata}


def prepare_graph_artifacts(data: STMGPromptDataBundle, config: STMGPromptConfig) -> dict[str, Any] | None:
    """Return graph inputs without constructing a trend prior for prior-free runs."""

    graph_required = bool(
        config.use_trend_prior_graph
        or config.use_graph_temporal_encoder
        or config.use_stmg_coupling_block
    )
    if not graph_required:
        return None
    if config.use_trend_prior_graph:
        artifacts = build_graph_artifacts(data, config)
        artifacts["metadata"]["uses_adaptive_graph"] = bool(
            config.use_adaptive_graph and config.adaptive_support_mode != "fixed"
        )
        artifacts["metadata"]["adaptive_graph_constrained_by_prior"] = bool(
            config.use_adaptive_graph and config.adaptive_support_mode == "prior_constrained"
        )
        return artifacts

    identity = np.eye(int(data.num_nodes), dtype=np.float32)
    metadata = {
        "uses_custom_graph": False,
        "graph_type": "prior_free_learnable_dual_graph",
        "graph_tag": "prior_free",
        "graph_path": None,
        "graph_uses_train_only_statistics": False,
        "graph_uses_test_statistics": False,
        "graph_uses_future_target": False,
        "edge_weight_used": True,
        "graph_normalize": True,
        "uses_macro_trend_graph": False,
        "uses_micro_local_graph": False,
        "adaptive_graph_constrained_by_prior": False,
        "uses_adaptive_graph": bool(config.use_adaptive_graph),
        "graph_fusion_mode": "prior_free_softmax",
        "target_mask_used_for_graph_statistics": False,
        "target_mask_used_as_model_input": False,
        "num_nodes": int(data.num_nodes),
        "prior_placeholder": "identity_not_used_by_adaptive_builder",
    }
    return {
        "A_macro_trend": identity.copy(),
        "A_micro_local": identity.copy(),
        "metadata": metadata,
    }
