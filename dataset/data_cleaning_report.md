# SDWPF 基础数据清洗报告

## 输入与输出
- 输入 parquet: `dataset\sdwpf_10min.parquet`
- 风机位置文件: `dataset\sdwpf_turb_location_elevation.csv`
- 输出目录: `dataset`
- 输出文件: `{'raw_aligned': 'dataset\\sdwpf_raw_aligned.parquet', 'model_input': 'dataset\\sdwpf_model_input_base.parquet', 'eval_target': 'dataset\\sdwpf_eval_target.parquet', 'report_json': 'dataset\\data_cleaning_report.json', 'report_md': 'dataset\\data_cleaning_report.md'}`

## 数据规模
- 风机数量: 134
- 原始行数: 7042906
- 对齐后行数: 7042906
- 时间戳数量: 52559
- 重复 `(Tmstamp, TurbID)` 数量: 0
- 缺失 timestamp 补齐行数: 0

## 时间切分
- split_ratios: [0.8, 0.1, 0.1]
- train: {'start': '2021-01-01 00:10:00', 'end': '2021-10-19 23:50:00'}, 时间戳数 42047
- val: {'start': '2021-10-20 00:00:00', 'end': '2021-11-25 11:50:00'}, 时间戳数 5256
- test: {'start': '2021-11-25 12:00:00', 'end': '2021-12-31 23:50:00'}, 时间戳数 5256

## 异常统计
- `mask_orig_missing_any`: 353292
- `mask_missing_timestamp`: 0
- `mask_patv_negative`: 1856385
- `mask_wspd_high_patv_zero`: 29849
- `mask_wspd_low_patv_high`: 971
- `mask_temp_out_of_bounds`: 99208
- `mask_angle_out_of_bounds`: 0
- `mask_pab_out_of_bounds`: 1501171
- `mask_sensor_all_zero_fault`: 747

## Mask 统计
- valid_target_mask=False: 2241243 (0.318227)
- valid_input_mask=False: 747 (0.000106)

## 填充前后缺失
### 填充前
- `Wspd`: 354039
- `Wdir`: 354039
- `Etmp`: 452386
- `Itmp`: 370138
- `Ndir`: 354039
- `Pab1`: 1853028
- `Pab2`: 1839123
- `Pab3`: 1839355
- `Prtv`: 354039
- `T2m`: 0
- `Sp`: 0
- `RelH`: 0
- `Wspd_w`: 0
- `Wdir_w`: 0
- `Tp`: 0
- `Patv_clean_for_input`: 2241243

### 填充后
- `Wspd`: 0
- `Wdir`: 0
- `Etmp`: 0
- `Itmp`: 0
- `Ndir`: 0
- `Pab1`: 0
- `Pab2`: 0
- `Pab3`: 0
- `Prtv`: 0
- `T2m`: 0
- `Sp`: 0
- `RelH`: 0
- `Wspd_w`: 0
- `Wdir_w`: 0
- `Tp`: 0
- `Patv_clean_for_input`: 0

## 填充数量
- `Wspd`: KNN=24403, causal_ffill=329636, train_turbine_median=0, train_global_median=0
- `Wdir`: KNN=24403, causal_ffill=329636, train_turbine_median=0, train_global_median=0
- `Etmp`: KNN=118412, causal_ffill=333974, train_turbine_median=0, train_global_median=0
- `Itmp`: KNN=36164, causal_ffill=333974, train_turbine_median=0, train_global_median=0
- `Ndir`: KNN=24403, causal_ffill=329636, train_turbine_median=0, train_global_median=0
- `Pab1`: KNN=925543, causal_ffill=927485, train_turbine_median=0, train_global_median=0
- `Pab2`: KNN=915908, causal_ffill=923215, train_turbine_median=0, train_global_median=0
- `Pab3`: KNN=912300, causal_ffill=927055, train_turbine_median=0, train_global_median=0
- `Prtv`: KNN=24403, causal_ffill=329636, train_turbine_median=0, train_global_median=0
- `T2m`: KNN=0, causal_ffill=0, train_turbine_median=0, train_global_median=0
- `Sp`: KNN=0, causal_ffill=0, train_turbine_median=0, train_global_median=0
- `RelH`: KNN=0, causal_ffill=0, train_turbine_median=0, train_global_median=0
- `Wspd_w`: KNN=0, causal_ffill=0, train_turbine_median=0, train_global_median=0
- `Wdir_w`: KNN=0, causal_ffill=0, train_turbine_median=0, train_global_median=0
- `Tp`: KNN=0, causal_ffill=0, train_turbine_median=0, train_global_median=0
- `Patv_clean_for_input`: KNN=1586390, causal_ffill=654853, train_turbine_median=0, train_global_median=0

## 基础输入特征
`Wspd`, `Wdir`, `Etmp`, `Itmp`, `Ndir`, `Pab1`, `Pab2`, `Pab3`, `Prtv`, `T2m`, `Sp`, `RelH`, `Wspd_w`, `Wdir_w`, `Tp`, `Patv_clean_for_input`

## 明确排除的模型输入列
`TurbID`, `Tmstamp`, `Patv_raw`, `valid_target_mask`, `Modification_Reason`, `is_patv_anomaly`, `is_sensor_fault`, `is_missing_timestamp`, `is_original_observed`, `is_imputed_any`, `valid_input_mask`, `valid_patv_input_mask`

## 方法约束
- imputation_graph: `{'type': 'distance_knn', 'k': 5, 'use_elevation': False}`
- uses_kmeans: false
- uses_bfill: false
- uses_zero_fill: false
- uses_method_specific_features: false
- parquet 主文件统一 round(2): false
- train 统计量只来自 train split: true

## 自检结果
- `three_parquet_exist`: 通过
- `report_generated`: 通过
- `indices_aligned`: 通过
- `exactly_134_turbines_each_timestamp`: 通过
- `no_duplicate_turbid_timestamp`: 通过
- `has_patv_raw`: 通过
- `has_patv_clean_for_input`: 通过
- `has_valid_target_mask`: 通过
- `valid_target_mask_bool_or_binary`: 通过
- `no_method_specific_features`: 通过
- `contains_base_input_features`: 通过
- `not_using_kmeans`: 通过
- `not_using_bfill`: 通过
- `not_using_zero_fill`: 通过
- `parquet_not_uniform_round2`: 通过
- `train_stats_train_split_only`: 通过
