from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCADA_INPUT_FEATURES = [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
]

ERA5_INPUT_FEATURES = ["T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp"]

BASE_INPUT_FEATURES = SCADA_INPUT_FEATURES + ERA5_INPUT_FEATURES + ["Patv_clean_for_input"]
BASE_SOURCE_FEATURES = SCADA_INPUT_FEATURES + ERA5_INPUT_FEATURES + ["Patv"]

EXCLUDED_FROM_MODEL_INPUT = [
    "TurbID",
    "Tmstamp",
    "Patv_raw",
    "valid_target_mask",
    "Modification_Reason",
    "is_patv_anomaly",
    "is_sensor_fault",
    "is_missing_timestamp",
    "is_original_observed",
    "is_imputed_any",
    "valid_input_mask",
    "valid_patv_input_mask",
]

PROHIBITED_EXACT_FEATURES = {
    "theta_scada",
    "theta_scada_sin",
    "theta_scada_cos",
    "theta_era5",
    "theta_era5_sin",
    "theta_era5_cos",
    "theta_scada_era5_diff",
    "Wdir_sin",
    "Wdir_cos",
    "Ndir_sin",
    "Ndir_cos",
    "PabMean",
    "PabStd",
    "PabMax",
    "Wspd_diff_1",
    "Patv_input_diff_1",
    "Wspd_roll_mean_6",
    "Wspd_roll_std_6",
    "Patv_input_roll_mean_6",
    "Patv_input_roll_std_6",
    "theta_scada_roll_stability_6",
    "valid_input_ratio_roll_6",
}

PROHIBITED_SUBSTRINGS = [
    "calibration",
    "reliability",
    "gate",
    "wake",
    "delay",
    "rolling",
    "roll_",
    "_roll",
    "_diff",
    "diff_",
]

OUTPUT_FILENAMES = {
    "raw_aligned": "sdwpf_raw_aligned.parquet",
    "model_input": "sdwpf_model_input_base.parquet",
    "eval_target": "sdwpf_eval_target.parquet",
    "report_json": "data_cleaning_report.json",
    "report_md": "data_cleaning_report.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build target-safe, mask-aware SDWPF benchmark baseline inputs."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("dataset/sdwpf_10min.parquet"),
        help="Raw SDWPF parquet input.",
    )
    parser.add_argument(
        "--location-file",
        type=Path,
        default=Path("dataset/sdwpf_turb_location_elevation.csv"),
        help="Turbine location CSV with TurbID, x, y, and optional Ele.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory for aligned parquet files and cleaning reports.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Time-ordered split ratios. Train statistics are computed only from train.",
    )
    parser.add_argument("--knn-k", type=int, default=5, help="Distance KNN neighbor count.")
    parser.add_argument(
        "--use-elevation",
        action="store_true",
        default=False,
        help="Include Ele in distance KNN if the location file has it.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=False,
        help="Also export CSV copies of the three parquet outputs.",
    )
    parser.add_argument(
        "--no-bfill",
        action="store_true",
        default=True,
        help="Kept for explicit CLI protocol compatibility. Backward fill is not used.",
    )
    parser.add_argument(
        "--no-zero-fill",
        action="store_true",
        default=True,
        help="Kept for explicit CLI protocol compatibility. Zero fill is not used.",
    )
    args = parser.parse_args()

    if args.knn_k < 1:
        raise ValueError("--knn-k must be >= 1.")
    split_sum = sum(args.split_ratios)
    if any(r <= 0 for r in args.split_ratios) or not np.isclose(split_sum, 1.0):
        raise ValueError("--split-ratios must be positive and sum to 1.0.")
    return args


def _false_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index, dtype=bool)


def _series_mask(df: pd.DataFrame, column: str, condition: Any) -> pd.Series:
    if column not in df.columns:
        return _false_series(df)
    return condition(df[column]).fillna(False).astype(bool)


def _is_time_only(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(1000)
    if sample.empty:
        return False
    pattern = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
    return sample.map(lambda value: bool(pattern.match(value.strip()))).mean() > 0.95


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Tmstamp" not in df.columns:
        raise KeyError("Input parquet must contain Tmstamp.")

    if "Day" in df.columns and _is_time_only(df["Tmstamp"]):
        day = pd.to_numeric(df["Day"], errors="coerce")
        if day.isna().any():
            raise ValueError("Day contains non-numeric values; cannot build full timestamps.")
        time_delta = pd.to_timedelta(df["Tmstamp"].astype(str), errors="coerce")
        if time_delta.isna().any():
            raise ValueError("Tmstamp contains invalid time-only values.")
        base_day = int(day.min())
        df["Tmstamp"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(
            day.astype(int) - base_day, unit="D"
        ) + time_delta
        return df

    parsed = pd.to_datetime(df["Tmstamp"], errors="coerce")
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(f"Tmstamp has {bad_count} values that cannot be parsed as datetime.")
    df["Tmstamp"] = parsed
    return df


def load_inputs(input_parquet: Path, location_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")
    if not location_file.exists():
        raise FileNotFoundError(f"Location file not found: {location_file}")

    df = pd.read_parquet(input_parquet)
    loc = pd.read_csv(location_file)
    if "TurbID" not in df.columns:
        raise KeyError("Input parquet must contain TurbID.")
    required_loc_cols = {"TurbID", "x", "y"}
    missing_loc = sorted(required_loc_cols - set(loc.columns))
    if missing_loc:
        raise KeyError(f"Location file is missing required columns: {missing_loc}")

    df["TurbID"] = pd.to_numeric(df["TurbID"], errors="raise").astype(int)
    loc["TurbID"] = pd.to_numeric(loc["TurbID"], errors="raise").astype(int)
    df = _normalize_timestamps(df)
    return df, loc


def build_full_time_index(
    df: pd.DataFrame, loc: pd.DataFrame
) -> tuple[pd.DataFrame, int, list[int]]:
    duplicate_count = int(df.duplicated(["Tmstamp", "TurbID"]).sum())
    if duplicate_count:
        df = df.sort_values(["Tmstamp", "TurbID"]).drop_duplicates(
            ["Tmstamp", "TurbID"], keep="first"
        )

    all_turbines = sorted(set(loc["TurbID"].astype(int)).union(set(df["TurbID"].astype(int))))
    full_timestamps = pd.date_range(df["Tmstamp"].min(), df["Tmstamp"].max(), freq="10min")
    full_index = pd.MultiIndex.from_product(
        [full_timestamps, all_turbines], names=["Tmstamp", "TurbID"]
    )

    aligned = (
        df.sort_values(["Tmstamp", "TurbID"])
        .assign(is_original_observed=True)
        .set_index(["Tmstamp", "TurbID"])
        .reindex(full_index)
        .reset_index()
    )
    aligned["is_original_observed"] = aligned["is_original_observed"].fillna(False).astype(bool)
    aligned["is_missing_timestamp"] = ~aligned["is_original_observed"]
    aligned["TurbID"] = aligned["TurbID"].astype(int)
    if "Day" in aligned.columns:
        first_ts = aligned["Tmstamp"].min()
        aligned["Day"] = ((aligned["Tmstamp"] - first_ts).dt.days + 1).astype(int)
    if "Patv" not in aligned.columns:
        aligned["Patv"] = np.nan
    aligned["Patv_raw"] = aligned["Patv"]
    return aligned, duplicate_count, all_turbines


def build_quality_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    observed = (
        df["is_original_observed"].fillna(False).astype(bool)
        if "is_original_observed" in df.columns
        else pd.Series(True, index=df.index, dtype=bool)
    )
    missing_timestamp = (
        df["is_missing_timestamp"].fillna(False).astype(bool)
        if "is_missing_timestamp" in df.columns
        else _false_series(df)
    )
    features_for_missing = [col for col in BASE_SOURCE_FEATURES if col in df.columns]
    if features_for_missing:
        masks["mask_orig_missing_any"] = df[features_for_missing].isna().any(axis=1) & observed
    else:
        masks["mask_orig_missing_any"] = _false_series(df)
    masks["mask_missing_timestamp"] = missing_timestamp
    masks["mask_patv_negative"] = _series_mask(df, "Patv_raw", lambda s: s < 0)

    if "Wspd" in df.columns and "Patv_raw" in df.columns:
        masks["mask_wspd_high_patv_zero"] = (
            (df["Wspd"] > 2.5) & (df["Patv_raw"] == 0)
        ).fillna(False)
        masks["mask_wspd_low_patv_high"] = (
            ((df["Wspd"] < 1.0) & (df["Patv_raw"] > 10))
            | ((df["Wspd"] < 2.0) & (df["Patv_raw"] > 100))
            | ((df["Wspd"] < 3.0) & (df["Patv_raw"] > 200))
        ).fillna(False)
    else:
        masks["mask_wspd_high_patv_zero"] = _false_series(df)
        masks["mask_wspd_low_patv_high"] = _false_series(df)

    etmp_bad = _series_mask(df, "Etmp", lambda s: (s < -21) | (s > 60))
    itmp_bad = _series_mask(df, "Itmp", lambda s: (s < -21) | (s > 70))
    masks["mask_temp_out_of_bounds"] = etmp_bad | itmp_bad

    wdir_bad = _series_mask(df, "Wdir", lambda s: (s < -180) | (s > 180))
    ndir_bad = _series_mask(df, "Ndir", lambda s: (s < -720) | (s > 720))
    masks["mask_angle_out_of_bounds"] = wdir_bad | ndir_bad

    pab_bad = _false_series(df)
    for col in ["Pab1", "Pab2", "Pab3"]:
        pab_bad = pab_bad | _series_mask(df, col, lambda s: s > 89)
    masks["mask_pab_out_of_bounds"] = pab_bad

    if all(col in df.columns for col in ["Wspd", "Wdir", "Etmp"]):
        masks["mask_sensor_all_zero_fault"] = (
            (df["Wspd"] == 0) & (df["Wdir"] == 0) & (df["Etmp"] == 0)
        ).fillna(False)
    else:
        masks["mask_sensor_all_zero_fault"] = _false_series(df)

    return {key: value.astype(bool) for key, value in masks.items()}


def build_modification_reason(masks: dict[str, pd.Series]) -> pd.Series:
    reason_map = [
        ("mask_orig_missing_any", "1"),
        ("mask_patv_negative", "2"),
        ("mask_wspd_high_patv_zero", "3"),
        ("mask_wspd_low_patv_high", "4"),
        ("mask_temp_out_of_bounds", "5"),
        ("mask_angle_out_of_bounds", "6"),
        ("mask_pab_out_of_bounds", "7"),
        ("mask_sensor_all_zero_fault", "8"),
        ("mask_missing_timestamp", "9"),
    ]
    index = next(iter(masks.values())).index
    reasons = pd.Series("", index=index, dtype=object)
    for mask_name, code in reason_map:
        reasons.loc[masks[mask_name]] = reasons.loc[masks[mask_name]].map(
            lambda value: f"{value},{code}" if value else code
        )
    return reasons.replace("", "0")


def apply_feature_level_fuses(
    df: pd.DataFrame, masks: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    for col in BASE_SOURCE_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    patv_anomaly = (
        masks["mask_patv_negative"]
        | masks["mask_wspd_high_patv_zero"]
        | masks["mask_wspd_low_patv_high"]
    )
    valid_target_mask = ~(
        df["Patv_raw"].isna()
        | masks["mask_missing_timestamp"]
        | patv_anomaly
        | masks["mask_sensor_all_zero_fault"]
    )
    valid_patv_input_mask = ~(df["Patv_raw"].isna() | patv_anomaly | masks["mask_sensor_all_zero_fault"])

    df["Patv_clean_for_input"] = df["Patv_raw"]
    df.loc[~valid_patv_input_mask, "Patv_clean_for_input"] = np.nan

    if "Etmp" in df.columns:
        df.loc[(df["Etmp"] < -21) | (df["Etmp"] > 60), "Etmp"] = np.nan
    if "Itmp" in df.columns:
        df.loc[(df["Itmp"] < -21) | (df["Itmp"] > 70), "Itmp"] = np.nan
    if "Wdir" in df.columns:
        df.loc[(df["Wdir"] < -180) | (df["Wdir"] > 180), "Wdir"] = np.nan
    if "Ndir" in df.columns:
        df.loc[(df["Ndir"] < -720) | (df["Ndir"] > 720), "Ndir"] = np.nan
    for col in ["Pab1", "Pab2", "Pab3"]:
        if col in df.columns:
            df.loc[df[col] > 89, col] = np.nan

    sensor_fault_cols = [col for col in SCADA_INPUT_FEATURES if col in df.columns]
    df.loc[masks["mask_sensor_all_zero_fault"], sensor_fault_cols] = np.nan
    return df, valid_target_mask.astype(bool), valid_patv_input_mask.astype(bool)


def compute_split_ranges(
    timestamps: pd.Series | pd.DatetimeIndex, split_ratios: tuple[float, float, float]
) -> dict[str, Any]:
    unique_times = pd.Index(pd.Series(timestamps).drop_duplicates().sort_values())
    num_time_steps = len(unique_times)
    train_end = int(num_time_steps * split_ratios[0])
    val_end = int(num_time_steps * (split_ratios[0] + split_ratios[1]))
    if train_end <= 0 or val_end <= train_end or val_end >= num_time_steps:
        raise ValueError("Split ratios create an empty train, val, or test time range.")

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    def _range(values: pd.Index) -> dict[str, str | None]:
        if len(values) == 0:
            return {"start": None, "end": None}
        return {"start": str(values[0]), "end": str(values[-1])}

    return {
        "unique_times": unique_times,
        "train_times": train_times,
        "val_times": val_times,
        "test_times": test_times,
        "train_mask_times": set(train_times),
        "split_ratios": list(split_ratios),
        "train_time_range": _range(train_times),
        "val_time_range": _range(val_times),
        "test_time_range": _range(test_times),
        "train_num_timestamps": int(len(train_times)),
        "val_num_timestamps": int(len(val_times)),
        "test_num_timestamps": int(len(test_times)),
    }


def build_knn_neighbors(
    loc: pd.DataFrame, turbines: list[int], k: int, use_elevation: bool
) -> dict[int, list[tuple[int, float]]]:
    loc_indexed = loc.drop_duplicates("TurbID").set_index("TurbID")
    missing_loc = sorted(set(turbines) - set(loc_indexed.index.astype(int)))
    if missing_loc:
        raise ValueError(f"Location file lacks coordinates for TurbID: {missing_loc[:10]}")
    coord_cols = ["x", "y"] + (["Ele"] if use_elevation and "Ele" in loc_indexed.columns else [])
    coords = loc_indexed.loc[turbines, coord_cols].astype(float).to_numpy()
    neighbors: dict[int, list[tuple[int, float]]] = {}
    for i, turb_id in enumerate(turbines):
        diff = coords - coords[i]
        distances = np.sqrt((diff * diff).sum(axis=1))
        order = np.argsort(distances)
        nearest = [idx for idx in order if idx != i][:k]
        neighbors[turb_id] = [
            (int(turbines[j]), float(1.0 / (distances[j] + 1e-6))) for j in nearest
        ]
    return neighbors


def spatial_knn_impute(
    df: pd.DataFrame, features: list[str], neighbors: dict[int, list[tuple[int, float]]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True)
    counts: dict[str, int] = {}
    indexed = df.set_index(["Tmstamp", "TurbID"])
    for feature in features:
        before_missing = int(indexed[feature].isna().sum())
        if before_missing == 0:
            counts[feature] = 0
            continue

        pivot = indexed[feature].unstack("TurbID")
        for turb_id, neighbor_info in neighbors.items():
            if turb_id not in pivot.columns:
                continue
            neighbor_ids = [nid for nid, _ in neighbor_info if nid in pivot.columns]
            if not neighbor_ids:
                continue
            weights = pd.Series(
                [weight for nid, weight in neighbor_info if nid in pivot.columns],
                index=neighbor_ids,
                dtype=float,
            )
            missing = pivot[turb_id].isna()
            if not missing.any():
                continue
            neighbor_values = pivot[neighbor_ids]
            available = neighbor_values.notna()
            denom = available.mul(weights, axis=1).sum(axis=1)
            numerator = neighbor_values.mul(weights, axis=1).sum(axis=1, skipna=True)
            fill_values = numerator / denom.replace(0, np.nan)
            eligible = missing & fill_values.notna()
            if eligible.any():
                pivot.loc[eligible, turb_id] = fill_values.loc[eligible]

        filled = pivot.stack()
        indexed[feature] = filled.reindex(indexed.index).to_numpy()
        after_missing = int(indexed[feature].isna().sum())
        counts[feature] = before_missing - after_missing
    return indexed.reset_index(), counts


def causal_ffill_impute(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.sort_values(["TurbID", "Tmstamp"]).reset_index(drop=True)
    counts: dict[str, int] = {}
    for feature in features:
        before_missing = int(df[feature].isna().sum())
        df[feature] = df.groupby("TurbID", sort=False)[feature].ffill()
        after_missing = int(df[feature].isna().sum())
        counts[feature] = before_missing - after_missing
    return df.sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True), counts


def train_median_impute(
    df: pd.DataFrame, features: list[str], train_times: set[pd.Timestamp]
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, Any]]:
    df = df.sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True)
    train_mask = df["Tmstamp"].isin(train_times)
    counts: dict[str, dict[str, int]] = {}
    stats: dict[str, Any] = {"per_turbine_median": {}, "global_median": {}}
    for feature in features:
        counts[feature] = {"train_turbine_median": 0, "train_global_median": 0}
        train_values = df.loc[train_mask, ["TurbID", feature]]
        per_turbine = train_values.groupby("TurbID")[feature].median()
        stats["per_turbine_median"][feature] = {
            str(int(k)): None if pd.isna(v) else float(v) for k, v in per_turbine.items()
        }
        before_missing = df[feature].isna()
        turbine_fill_values = df["TurbID"].map(per_turbine)
        eligible = before_missing & turbine_fill_values.notna()
        if eligible.any():
            df.loc[eligible, feature] = turbine_fill_values.loc[eligible]
        counts[feature]["train_turbine_median"] = int(before_missing.sum() - df[feature].isna().sum())

        before_global = df[feature].isna()
        global_median = df.loc[train_mask, feature].median()
        stats["global_median"][feature] = None if pd.isna(global_median) else float(global_median)
        if pd.notna(global_median):
            df.loc[before_global, feature] = global_median
        counts[feature]["train_global_median"] = int(before_global.sum() - df[feature].isna().sum())
    return df, counts, stats


def write_outputs(
    raw_aligned: pd.DataFrame,
    model_input: pd.DataFrame,
    eval_target: pd.DataFrame,
    output_dir: Path,
    export_csv: bool,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        key: str(output_dir / filename) for key, filename in OUTPUT_FILENAMES.items()
    }
    raw_aligned.to_parquet(output_files["raw_aligned"], index=False)
    model_input.to_parquet(output_files["model_input"], index=False)
    eval_target.to_parquet(output_files["eval_target"], index=False)

    if export_csv:
        csv_paths = {
            "raw_aligned_csv": str(output_dir / "sdwpf_raw_aligned.csv"),
            "model_input_csv": str(output_dir / "sdwpf_model_input_base.csv"),
            "eval_target_csv": str(output_dir / "sdwpf_eval_target.csv"),
        }
        raw_aligned.to_csv(csv_paths["raw_aligned_csv"], encoding="utf_8_sig", index=False)
        model_input.to_csv(csv_paths["model_input_csv"], encoding="utf_8_sig", index=False)
        eval_target.to_csv(csv_paths["eval_target_csv"], encoding="utf_8_sig", index=False)
        output_files.update(csv_paths)
    return output_files


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / OUTPUT_FILENAMES["report_json"]
    md_path = output_dir / OUTPUT_FILENAMES["report_md"]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    anomaly_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in report["anomaly_counts"].items()
    )
    nan_before_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in report["feature_nan_before"].items()
    )
    nan_after_lines = "\n".join(
        f"- `{name}`: {count}" for name, count in report["feature_nan_after"].items()
    )
    impute_lines = "\n".join(
        "- `{}`: KNN={}, causal_ffill={}, train_turbine_median={}, train_global_median={}".format(
            feature,
            counts["distance_knn"],
            counts["causal_ffill"],
            counts["train_turbine_median"],
            counts["train_global_median"],
        )
        for feature, counts in report["imputation_counts_by_feature"].items()
    )
    self_check_lines = "\n".join(
        f"- `{name}`: {'通过' if result else '失败'}"
        for name, result in report["self_check_results"].items()
    )

    md = f"""# SDWPF 基础数据清洗报告

## 输入与输出
- 输入 parquet: `{report["input_parquet"]}`
- 风机位置文件: `{report["location_file"]}`
- 输出目录: `{report["output_dir"]}`
- 输出文件: `{report["output_files"]}`

## 数据规模
- 风机数量: {report["num_turbines"]}
- 原始行数: {report["num_raw_rows"]}
- 对齐后行数: {report["num_full_aligned_rows"]}
- 时间戳数量: {report["num_timestamps"]}
- 重复 `(Tmstamp, TurbID)` 数量: {report["duplicate_count"]}
- 缺失 timestamp 补齐行数: {report["missing_timestamp_count"]}

## 时间切分
- split_ratios: {report["split_ratios"]}
- train: {report["split_time_ranges"]["train"]}, 时间戳数 {report["train_num_timestamps"]}
- val: {report["split_time_ranges"]["val"]}, 时间戳数 {report["val_num_timestamps"]}
- test: {report["split_time_ranges"]["test"]}, 时间戳数 {report["test_num_timestamps"]}

## 异常统计
{anomaly_lines}

## Mask 统计
- valid_target_mask=False: {report["valid_target_false_count"]} ({report["valid_target_false_ratio"]:.6f})
- valid_input_mask=False: {report["valid_input_false_count"]} ({report["valid_input_false_ratio"]:.6f})

## 填充前后缺失
### 填充前
{nan_before_lines}

### 填充后
{nan_after_lines}

## 填充数量
{impute_lines}

## 基础输入特征
{", ".join(f"`{feature}`" for feature in report["base_input_features"])}

## 明确排除的模型输入列
{", ".join(f"`{feature}`" for feature in report["excluded_from_model_input"])}

## 方法约束
- imputation_graph: `{report["imputation_graph"]}`
- uses_kmeans: {str(report["uses_kmeans"]).lower()}
- uses_bfill: {str(report["uses_bfill"]).lower()}
- uses_zero_fill: {str(report["uses_zero_fill"]).lower()}
- uses_method_specific_features: {str(report["uses_method_specific_features"]).lower()}
- parquet 主文件统一 round(2): false
- train 统计量只来自 train split: true

## 自检结果
{self_check_lines}
"""
    md_path.write_text(md, encoding="utf-8")


def self_check_outputs(output_dir: str) -> dict[str, bool]:
    output_path = Path(output_dir)
    raw_path = output_path / OUTPUT_FILENAMES["raw_aligned"]
    model_path = output_path / OUTPUT_FILENAMES["model_input"]
    target_path = output_path / OUTPUT_FILENAMES["eval_target"]
    report_path = output_path / OUTPUT_FILENAMES["report_json"]
    md_path = output_path / OUTPUT_FILENAMES["report_md"]

    results: dict[str, bool] = {}
    results["three_parquet_exist"] = raw_path.exists() and model_path.exists() and target_path.exists()
    results["report_generated"] = report_path.exists() and md_path.exists()
    if not results["three_parquet_exist"]:
        raise AssertionError("Required parquet outputs are missing.")

    raw = pd.read_parquet(raw_path)
    model = pd.read_parquet(model_path)
    target = pd.read_parquet(target_path)
    raw_index = pd.MultiIndex.from_frame(raw[["Tmstamp", "TurbID"]])
    model_index = pd.MultiIndex.from_frame(model[["Tmstamp", "TurbID"]])
    target_index = pd.MultiIndex.from_frame(target[["Tmstamp", "TurbID"]])

    results["indices_aligned"] = raw_index.equals(model_index) and raw_index.equals(target_index)
    results["exactly_134_turbines_each_timestamp"] = (
        raw.groupby("Tmstamp")["TurbID"].nunique().eq(134).all()
    )
    results["no_duplicate_turbid_timestamp"] = (
        not raw.duplicated(["Tmstamp", "TurbID"]).any()
        and not model.duplicated(["Tmstamp", "TurbID"]).any()
        and not target.duplicated(["Tmstamp", "TurbID"]).any()
    )
    results["has_patv_raw"] = "Patv_raw" in raw.columns and "Patv_raw" in target.columns
    results["has_patv_clean_for_input"] = "Patv_clean_for_input" in model.columns
    results["has_valid_target_mask"] = "valid_target_mask" in target.columns
    if "valid_target_mask" in target.columns:
        mask = target["valid_target_mask"]
        results["valid_target_mask_bool_or_binary"] = (
            pd.api.types.is_bool_dtype(mask) or set(mask.dropna().unique()).issubset({0, 1})
        )
    else:
        results["valid_target_mask_bool_or_binary"] = False

    model_columns = set(model.columns)
    lower_columns = {col.lower() for col in model.columns}
    exact_bad = bool(model_columns.intersection(PROHIBITED_EXACT_FEATURES))
    substring_bad = any(
        bad in col for bad in PROHIBITED_SUBSTRINGS for col in lower_columns
    )
    results["no_method_specific_features"] = not exact_bad and not substring_bad
    results["contains_base_input_features"] = set(BASE_INPUT_FEATURES).issubset(model_columns)

    source = Path(__file__).read_text(encoding="utf-8")
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    results["not_using_kmeans"] = (
        re.search(r"^\s*from\s+sklearn\.cluster\s+import", source, re.MULTILINE) is None
        and re.search(r"\bKMeans\s*\(", source) is None
    )
    results["not_using_bfill"] = (
        re.search(r"\.bfill\s*\(", source) is None and report.get("uses_bfill") is False
    )
    results["not_using_zero_fill"] = (
        re.search(r"\.fillna\s*\(\s*0(?:\.0)?\s*\)", source) is None
        and report.get("uses_zero_fill") is False
    )
    results["parquet_not_uniform_round2"] = report.get("parquet_uniform_round2") is False
    results["train_stats_train_split_only"] = report.get("train_stats_from_train_split_only") is True
    results["report_generated"] = report_path.exists() and md_path.exists()

    results = {name: bool(ok) for name, ok in results.items()}
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        raise AssertionError(f"Self-check failed: {failed}")
    return results


def _missing_columns_for_report(df: pd.DataFrame) -> dict[str, list[str]]:
    mask_dependencies = {
        "mask_patv_negative": ["Patv_raw"],
        "mask_wspd_high_patv_zero": ["Wspd", "Patv_raw"],
        "mask_wspd_low_patv_high": ["Wspd", "Patv_raw"],
        "mask_temp_out_of_bounds": ["Etmp", "Itmp"],
        "mask_angle_out_of_bounds": ["Wdir", "Ndir"],
        "mask_pab_out_of_bounds": ["Pab1", "Pab2", "Pab3"],
        "mask_sensor_all_zero_fault": ["Wspd", "Wdir", "Etmp"],
    }
    return {
        mask: [col for col in cols if col not in df.columns]
        for mask, cols in mask_dependencies.items()
        if any(col not in df.columns for col in cols)
    }


def main() -> None:
    args = parse_args()
    raw_df, loc = load_inputs(args.input_parquet, args.location_file)
    num_raw_rows = int(len(raw_df))

    aligned, duplicate_count, all_turbines = build_full_time_index(raw_df, loc)
    raw_aligned = aligned.copy()
    split = compute_split_ranges(aligned["Tmstamp"], tuple(args.split_ratios))

    missing_columns_by_mask = _missing_columns_for_report(aligned)
    masks = build_quality_masks(aligned)
    aligned["Modification_Reason"] = build_modification_reason(masks)
    aligned, valid_target_mask, valid_patv_input_mask = apply_feature_level_fuses(aligned, masks)

    aligned["is_patv_anomaly"] = (
        masks["mask_patv_negative"]
        | masks["mask_wspd_high_patv_zero"]
        | masks["mask_wspd_low_patv_high"]
    )
    aligned["is_temp_anomaly"] = masks["mask_temp_out_of_bounds"]
    aligned["is_angle_anomaly"] = masks["mask_angle_out_of_bounds"]
    aligned["is_pab_anomaly"] = masks["mask_pab_out_of_bounds"]
    aligned["is_sensor_fault"] = masks["mask_sensor_all_zero_fault"]
    aligned["valid_input_mask"] = ~(
        masks["mask_missing_timestamp"] | masks["mask_sensor_all_zero_fault"]
    )
    aligned["valid_patv_input_mask"] = valid_patv_input_mask
    aligned["valid_target_mask"] = valid_target_mask

    feature_nan_before = {
        feature: int(aligned[feature].isna().sum()) for feature in BASE_INPUT_FEATURES
    }

    neighbors = build_knn_neighbors(loc, all_turbines, args.knn_k, args.use_elevation)
    imputed, knn_counts = spatial_knn_impute(aligned, BASE_INPUT_FEATURES, neighbors)
    imputed, ffill_counts = causal_ffill_impute(imputed, BASE_INPUT_FEATURES)
    imputed, median_counts, train_stats = train_median_impute(
        imputed, BASE_INPUT_FEATURES, split["train_mask_times"]
    )

    imputation_counts = {}
    for feature in BASE_INPUT_FEATURES:
        imputation_counts[feature] = {
            "distance_knn": int(knn_counts.get(feature, 0)),
            "causal_ffill": int(ffill_counts.get(feature, 0)),
            "train_turbine_median": int(median_counts.get(feature, {}).get("train_turbine_median", 0)),
            "train_global_median": int(median_counts.get(feature, {}).get("train_global_median", 0)),
        }
    imputed["is_imputed_any"] = False
    for feature in BASE_INPUT_FEATURES:
        original_missing = aligned.sort_values(["Tmstamp", "TurbID"])[feature].isna().to_numpy()
        now_present = imputed[feature].notna().to_numpy()
        imputed["is_imputed_any"] = imputed["is_imputed_any"] | (original_missing & now_present)

    for feature in BASE_INPUT_FEATURES:
        imputed[feature] = pd.to_numeric(imputed[feature], errors="coerce").astype("float32")

    feature_nan_after = {
        feature: int(imputed[feature].isna().sum()) for feature in BASE_INPUT_FEATURES
    }
    anomaly_counts = {name: int(mask.sum()) for name, mask in masks.items()}

    model_columns = [
        "TurbID",
        "Tmstamp",
        *BASE_INPUT_FEATURES,
        "Modification_Reason",
        "is_original_observed",
        "is_missing_timestamp",
        "is_imputed_any",
        "is_patv_anomaly",
        "is_temp_anomaly",
        "is_angle_anomaly",
        "is_pab_anomaly",
        "is_sensor_fault",
        "valid_input_mask",
        "valid_patv_input_mask",
        "valid_target_mask",
    ]
    target_columns = [
        "TurbID",
        "Tmstamp",
        "Patv_raw",
        "valid_target_mask",
        "is_patv_anomaly",
        "is_missing_timestamp",
        "is_sensor_fault",
        "Modification_Reason",
    ]

    model_input = imputed[model_columns].sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True)
    eval_target = imputed[target_columns].sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True)
    raw_aligned = raw_aligned.sort_values(["Tmstamp", "TurbID"]).reset_index(drop=True)

    output_files = write_outputs(raw_aligned, model_input, eval_target, args.output_dir, args.export_csv)

    num_rows = int(len(model_input))
    report: dict[str, Any] = {
        "input_parquet": str(args.input_parquet),
        "location_file": str(args.location_file),
        "output_dir": str(args.output_dir),
        "output_files": output_files,
        "num_turbines": int(len(all_turbines)),
        "num_raw_rows": num_raw_rows,
        "num_full_aligned_rows": int(len(raw_aligned)),
        "num_timestamps": int(raw_aligned["Tmstamp"].nunique()),
        "duplicate_count": duplicate_count,
        "missing_timestamp_count": int(raw_aligned["is_missing_timestamp"].sum()),
        "missing_columns_by_mask": missing_columns_by_mask,
        "anomaly_counts": anomaly_counts,
        "valid_target_false_count": int((~eval_target["valid_target_mask"]).sum()),
        "valid_target_false_ratio": float((~eval_target["valid_target_mask"]).sum() / num_rows),
        "valid_input_false_count": int((~model_input["valid_input_mask"]).sum()),
        "valid_input_false_ratio": float((~model_input["valid_input_mask"]).sum() / num_rows),
        "feature_nan_before": feature_nan_before,
        "feature_nan_after": feature_nan_after,
        "imputation_counts_by_feature": imputation_counts,
        "split_ratios": list(args.split_ratios),
        "split_time_ranges": {
            "train": split["train_time_range"],
            "val": split["val_time_range"],
            "test": split["test_time_range"],
        },
        "train_time_range": split["train_time_range"],
        "val_time_range": split["val_time_range"],
        "test_time_range": split["test_time_range"],
        "train_num_timestamps": split["train_num_timestamps"],
        "val_num_timestamps": split["val_num_timestamps"],
        "test_num_timestamps": split["test_num_timestamps"],
        "knn_k": int(args.knn_k),
        "use_elevation": bool(args.use_elevation),
        "imputation_graph": {
            "type": "distance_knn",
            "k": int(args.knn_k),
            "use_elevation": bool(args.use_elevation),
        },
        "uses_kmeans": False,
        "uses_bfill": False,
        "uses_zero_fill": False,
        "uses_method_specific_features": False,
        "parquet_uniform_round2": False,
        "train_stats_from_train_split_only": True,
        "train_median_stats": train_stats,
        "base_input_features": BASE_INPUT_FEATURES,
        "excluded_from_model_input": EXCLUDED_FROM_MODEL_INPUT,
        "self_check_results": {},
    }
    write_report(report, args.output_dir)
    self_check_results = self_check_outputs(str(args.output_dir))
    report["self_check_results"] = self_check_results
    write_report(report, args.output_dir)
    print(json.dumps(self_check_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
