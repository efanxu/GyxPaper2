from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#2F5D8A",
    "orange": "#D97925",
    "green": "#3A7D44",
    "red": "#B44343",
    "purple": "#76558F",
    "gray": "#7A7A7A",
    "light": "#D9E2EC",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / f"{stem}.png", figure_dir / f"{stem}.svg"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _plot_split_distribution(data_dir: Path, figure_dir: Path) -> list[Path]:
    overview = pd.read_csv(data_dir / "data_overview.csv")
    turbine = pd.read_csv(data_dir / "turbine_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    axes[0].bar(overview["split"], overview["time_points"], color=colors, width=0.65)
    axes[0].set_title("Chronological split size")
    axes[0].set_ylabel("Time points (10-min)")
    for index, row in overview.iterrows():
        axes[0].text(index, row["time_points"], f"{int(row['window_count']):,} windows", ha="center", va="bottom", fontsize=8)
    axes[1].bar(overview["split"], overview["valid_target_ratio"], color=colors, width=0.65)
    axes[1].axhline(
        turbine.loc[turbine["signal"] == "Patv_raw", "valid_ratio"].mean(),
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1,
        label="Mean train turbine validity",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Valid target share")
    axes[1].set_ylabel("Share")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("SDWPF data split and target availability", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save(fig, figure_dir, "figure_01_data_split_distribution")


def _plot_temporal_scales(data_dir: Path, figure_dir: Path) -> list[Path]:
    curve = pd.read_csv(data_dir / "alignment_typical_event_curve.csv")
    x = curve["relative_step"]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(
        x,
        curve["Patv_clean_for_input_farm_mean_kw"],
        color="#222222",
        linewidth=1.8,
        label="Patv clean input",
    )
    for column, label, color in (
        ("Coarse_ma6_kw", "Coarse MA6", PALETTE["blue"]),
        ("Coarse_ma18_kw", "Coarse MA18", PALETTE["orange"]),
        ("Coarse_ma36_kw", "Coarse MA36", PALETTE["green"]),
    ):
        axes[0].plot(x, curve[column], linewidth=1.5, label=label, color=color)
    axes[0].axvline(0, color=PALETTE["red"], linestyle="--", linewidth=1)
    axes[0].set_ylabel("Farm-mean power (kW)")
    axes[0].set_title("Coaxial causal trends at the pre-defined maximum input ramp")
    axes[0].legend(ncol=4, loc="upper center", fontsize=8)
    axes[1].plot(x, curve["Fine_local_residual_ma6_kw"], color=PALETTE["purple"], label="Raw − MA6")
    axes[1].plot(x, curve["Fine_first_difference_kw"], color=PALETTE["red"], alpha=0.8, label="First difference")
    axes[1].axhline(0, color=PALETTE["gray"], linewidth=0.7)
    axes[1].axvline(0, color=PALETTE["red"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("Relative 10-min step")
    axes[1].set_ylabel("Local change (kW)")
    axes[1].set_title("Fine local variation")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return _save(fig, figure_dir, "figure_02_temporal_scales_coaxial")


def _plot_spectrum_acf(data_dir: Path, figure_dir: Path) -> list[Path]:
    source = pd.read_csv(data_dir / "spectrum_acf_source.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    color_map = {"raw": "#222222", "trend_ma36": PALETTE["green"], "local_residual_ma6": PALETTE["purple"]}
    label_map = {"raw": "Raw", "trend_ma36": "Trend MA36", "local_residual_ma6": "Local residual MA6"}
    for representation, group in source.groupby("representation"):
        spectrum = group[group["curve_type"] == "spectrum"]
        acf = group[group["curve_type"] == "acf"]
        axes[0].plot(spectrum["x"], spectrum["y"], label=label_map[representation], color=color_map[representation])
        axes[1].plot(acf["x"], acf["y"], label=label_map[representation], color=color_map[representation])
    axes[0].axvline(1 / 18, color=PALETTE["red"], linestyle="--", linewidth=1, label="HF cutoff 1/18")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Frequency (cycles / 10-min step)")
    axes[0].set_ylabel("Normalized spectral energy")
    axes[0].set_title("Frequency-domain separation")
    axes[1].axhline(np.exp(-1), color=PALETTE["red"], linestyle="--", linewidth=1, label="1/e")
    axes[1].set_xlabel("Lag (10-min steps)")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].set_title("ACF decay")
    for axis in axes:
        axis.legend(fontsize=8)
    turbine_id = int(source["TurbID"].iloc[0])
    fig.suptitle(f"Spectrum and ACF for train-volatility-selected turbine {turbine_id}", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save(fig, figure_dir, "figure_03_spectrum_acf")


def _plot_alignment(data_dir: Path, figure_dir: Path) -> list[Path]:
    curve = pd.read_csv(data_dir / "alignment_typical_event_curve.csv")
    metrics = pd.read_csv(data_dir / "alignment_event_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    x = curve["relative_step"]
    axes[0].plot(
        x,
        curve["Patv_clean_for_input_farm_mean_kw"],
        color="#222222",
        linewidth=1.8,
        label="Patv clean input",
    )
    axes[0].plot(x, curve["Coarse_ma36_kw"], color=PALETTE["green"], linewidth=1.6, label="Coaxial MA36")
    axes[0].plot(
        x,
        curve["Downsample_upsample_ma36_kw"],
        color=PALETTE["orange"],
        linewidth=1.6,
        label="Causal down/up MA36",
    )
    axes[0].axvline(0, color=PALETTE["red"], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Relative 10-min step")
    axes[0].set_ylabel("Farm-mean power (kW)")
    axes[0].set_title("Event position at maximum train ramp")
    axes[0].legend(fontsize=8)
    comparison = metrics[metrics["representation"].isin(["coaxial_ma36", "causal_downsample_upsample_ma36"])]
    labels = ["Coaxial MA36", "Down/up MA36"]
    data = [
        comparison.loc[comparison["representation"] == name, "peak_timing_error_steps"].abs().to_numpy()
        for name in ("coaxial_ma36", "causal_downsample_upsample_ma36")
    ]
    boxes = axes[1].boxplot(data, labels=labels, patch_artist=True, widths=0.55)
    for box, color in zip(boxes["boxes"], [PALETTE["green"], PALETTE["orange"]]):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    axes[1].axhline(0, color=PALETTE["gray"], linewidth=0.7)
    axes[1].set_ylabel("Absolute peak timing error (steps)")
    axes[1].set_title("Ramp-event timing error")
    fig.tight_layout()
    return _save(fig, figure_dir, "figure_04_alignment_event_position")


def _plot_graph_semantics(data_dir: Path, figure_dir: Path) -> list[Path]:
    fit = pd.read_csv(data_dir / "graph_semantic_fit.csv")
    overlap = pd.read_csv(data_dir / "graph_neighbor_overlap.csv")
    selected = fit[fit["graph"].isin(["Micro", "Macro"])].copy()
    selected["label"] = selected["graph"] + " / " + selected["semantic"].map(
        {"delta_wspd_cosine": "ΔWspd", "ma36_patv_correlation": "MA36 Patv"}
    )
    selected = selected.sort_values(["assignment", "graph"])
    colors = [PALETTE["green"] if value == "matched" else PALETTE["orange"] for value in selected["assignment"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    axes[0].barh(selected["label"], selected["semantic_fit_score"], color=colors)
    axes[0].axvline(0, color=PALETTE["gray"], linewidth=0.8)
    axes[0].set_xlabel("Semantic-fit lift over distance graph")
    axes[0].set_title("Matched vs crossed graph semantics")
    axes[0].axvline(0, color=PALETTE["gray"], linewidth=0.7)
    axes[1].hist(overlap["neighbor_jaccard_micro_macro"], bins=12, color=PALETTE["blue"], alpha=0.8)
    global_value = overlap["global_support_jaccard_micro_macro"].iloc[0]
    axes[1].axvline(global_value, color=PALETTE["red"], linestyle="--", linewidth=1.3, label=f"Global = {global_value:.2f}")
    axes[1].set_xlabel("Per-node Micro/Macro neighbor Jaccard")
    axes[1].set_ylabel("Turbine count")
    axes[1].set_title("Distinct graph support")
    axes[1].legend(fontsize=8)
    fig.suptitle("Dual-graph statistical semantic evidence", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save(fig, figure_dir, "figure_05_graph_semantic_matching")


def _plot_difficulty(data_dir: Path, figure_dir: Path) -> list[Path]:
    horizon = pd.read_csv(data_dir / "baseline_horizon_difficulty.csv")
    turbine = pd.read_csv(data_dir / "baseline_turbine_difficulty.csv")
    prefix = horizon[(horizon["split"] == "test") & (horizon["aggregation_level"] == "prefix_horizon")]
    step = horizon[(horizon["split"] == "test") & (horizon["aggregation_level"] == "future_step")]
    test_h10 = turbine[(turbine["split"] == "test") & (turbine["horizon"] == 10)]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    axes[0].plot(step["future_step"], step["Score"], marker="o", color=PALETTE["blue"], label="Per future step")
    axes[0].scatter(prefix["horizon"], prefix["Score"], color=PALETTE["red"], marker="s", s=38, label="Prefix H3/H6/H10")
    axes[0].set_xticks(range(1, 11))
    axes[0].set_xlabel("Future step / prefix horizon")
    axes[0].set_ylabel("Official Score (lower is better)")
    axes[0].set_title("Horizon difficulty")
    axes[0].legend(fontsize=8)
    normal = test_h10[~test_h10["is_top10pct_hard_turbine"].astype(bool)]["Score"]
    hard = test_h10[test_h10["is_top10pct_hard_turbine"].astype(bool)]["Score"]
    bins = np.histogram_bin_edges(test_h10["Score"].dropna(), bins=14)
    axes[1].hist(normal, bins=bins, color=PALETTE["blue"], alpha=0.7, label="Other turbines")
    axes[1].hist(hard, bins=bins, color=PALETTE["red"], alpha=0.8, label="Validation-fixed top 10%")
    axes[1].set_xlabel("Test H10 turbine Score")
    axes[1].set_ylabel("Turbine count")
    axes[1].set_title("Turbine difficulty heterogeneity")
    axes[1].legend(fontsize=8)
    fig.suptitle("MovingAverage formal baseline difficulty", y=1.02, fontsize=12)
    fig.tight_layout()
    return _save(fig, figure_dir, "figure_06_horizon_turbine_difficulty")


def plot_all(data_dir: Path) -> list[Path]:
    _style()
    data_dir = Path(data_dir)
    figure_dir = data_dir / "figures"
    paths: list[Path] = []
    for builder in (
        _plot_split_distribution,
        _plot_temporal_scales,
        _plot_spectrum_acf,
        _plot_alignment,
        _plot_graph_semantics,
        _plot_difficulty,
    ):
        paths.extend(builder(data_dir, figure_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot empirical step-2 data evidence.")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    for path in plot_all(args.data_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
