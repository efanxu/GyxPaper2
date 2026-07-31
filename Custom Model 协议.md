# Custom Model 协议：ST-MGPrompt Coupled Mask-Aware Fair-Comparison v4.2

> 协议版本：`custom_model_stmgprompt_coupled_mask_aware_fair_v4_2`  
> 适用任务：SDWPF 风电功率多步预测自定义模型构建。本文模型以风速、风向、ERA5 气象变量和历史功率作为驱动信息，最终预测目标为 `Patv_raw`。论文表述中若使用“风速非平稳性”“风速空间传播”“风速多粒度”，应理解为风电功率预测中的物理驱动机制，而不是将 `Wspd` 作为最终 target。  
> 适用目标：使 ST-MGPrompt 及其他 Custom Model 能够和统一 Benchmark 模型进行公平对比，同时保留本文完整方法的模块消融、诊断和鲁棒性实验。  
> 核心原则：Custom Model 可以实现自定义结构、图模块、门控模块、Prompt 模块、注意力模块和物理先验模块，但必须严格遵守相同的数据、划分、输入输出、训练预算、loss、metrics、保存格式和防泄露规则。

本协议是一份独立执行说明。新的对话框只需要提供本协议和具体 Custom Model 架构 prompt，不需要额外提供 Benchmark 协议。

---

## 0.0 Python 解释器策略（强制）

Windows 当前环境中的 Custom Model 训练、评价、消融、审计、smoke、dry-run、单元测试和 `compileall` 推荐使用以下解释器：

```text
D:\Apps\Miniconda3\envs\env_tslib\python.exe
```

通用 Python 执行代码不得硬编码任何 Windows 绝对解释器路径。父 runner 启动子 runner 时必须使用当前父进程的 `Path(sys.executable).resolve()`；Linux 通过当前环境的 `python -m ...` 启动时，子进程同样继承 Linux 当前解释器。禁止把 Windows 路径带入 Linux 命令。

Windows 当前运行配置中禁止使用或写入：

```text
D:\Apps\Miniconda3\python.exe
D:\Apps\Miniconda3\python.exe
```

执行规范：

1. Windows 所有正式 Full 命令必须以 `D:\Apps\Miniconda3\envs\env_tslib\python.exe` 启动；Linux 使用当前环境的 `python`。
2. runner 的 `--python` 默认使用当前父进程解释器；显式传入时必须存在并与当前父进程解释器解析到同一路径。
3. PyCharm 的 Project Interpreter 和 Run Configuration Interpreter 必须选择 `env_tslib`，其实际解释器路径必须核对为上述绝对路径。
4. Script parameters 只填写 runner 的命令行参数，不得把 Python 解释器路径误写进 Script parameters；通常不需要手工传入 `--python`。
5. Windows 输出正式命令或 PyCharm 配置前，必须逐字检查路径包含 `Miniconda3\envs\env_tslib`，不得依赖 base 环境或其他 Miniconda3 根解释器；Linux 不得出现 Windows 路径。
6. metadata 中的 `python_executable` 必须记录运行时解析后的实际解释器路径；若子进程与父进程不一致，应在启动前失败，不得静默继续正式训练。

---

## 0. 协议定位与本文框架关系

本协议不是单纯的模型结构说明，而是 Custom Model 的公平实验规则。它规定：

```text
所有模型必须吃同一份数据；
使用同一个时间划分；
使用同一套基础输入特征；
预测同一个目标 Patv_raw；
使用同一个 valid_target_mask；
使用同一套训练预算；
使用同一套 loss 与 metrics；
用同一个 summary 排序逻辑；
不得偷看未来；
smoke test 不得污染正式结果。
```

本文 ST-MGPrompt 的模型主线不是“先把空间关系算完，再单独做粗细粒度融合”的串行流程，而是：

```text
X 历史输入
→ Volatility-Aware Dynamic Semantic Patching, VADSP
→ 得到 X_fine / X_coarse 两条粒度分支
→ 若干层 ST-MG Coupling Block
   ├─ Fine/Micro Graph-Temporal Encoding: A_micro_local 对 X_fine 做局部空间增强
   ├─ Coarse/Macro Graph-Temporal Encoding: A_macro_trend 对 X_coarse 做宏观空间增强
   ├─ Macro-Trend Prompt: 由空间增强后的 h_coarse 生成
   └─ Symmetric Cross-Fusion: 使用空间增强后的 h_fine / h_coarse 双向更新
→ ST-Prompt Direct Multi-output Decoder
→ Multi-step Power Prediction
→ Fair main loss 或 Method-full MS-MG-DWU loss
```

核心约束：空间联系与粗细粒度联系必须在 `ST-MG Coupling Block` 中协同建模。也就是说，Macro-Trend Prompt 不能直接由原始 `X_coarse` 生成，Cross-Fusion 也不能直接使用未经过空间图增强的 `X_fine/X_coarse`。完整模型必须使用空间增强后的 `h_coarse` 生成宏观趋势提示词，并使用空间增强后的 `h_fine/h_coarse` 进行对称交叉融合。若实现多层 `num_coupling_layers > 1`，每层都应重复“图空间增强 + Prompt 生成 + 对称交叉融合”的联合更新过程。

为了和 Benchmark 严格公平对比，本文工程实现默认采用 direct multi-output decoder：一次输出未来 `max_pred_len=10` 个预测步，再按 `eval_horizons=[3,6,10]` 前缀裁剪评估。Prompt 可以作为未来步长查询向量，但不得使用预测窗口真实值进行自回归 teacher forcing。

其中：

1. `STMGPrompt_FairFull` 用于和 Benchmark 严格公平对比，仍使用默认 `masked_score_aligned_hybrid` loss。
2. `STMGPrompt_Full_MSMGDWU` 是本文完整方法版本，启用 MS-MG-DWU loss，可用于方法完整表和 loss 消融，但不能混入 strict fair main table 排名。

---

## 1. 总体原则

Custom Model 必须满足：

1. 使用相同基础清洗数据。
2. 使用相同时间顺序 split。
3. 使用相同基础输入特征。
4. 使用相同 target 和 target mask。
5. 使用相同 scaler 拟合规则。
6. 使用相同 split-contained window 规则。
7. 使用相同 history-only forecasting 设定。
8. 使用相同 lookback、max_pred_len 和 eval_horizons。
9. 使用相同 mask-aware loss 进入公平主表。
10. 使用相同 mask-aware metrics。
11. 使用相同 train/val/test window sampling。
12. 使用相同训练预算。
13. 使用相同 official score 公式。
14. 使用相同 summary 排序逻辑。
15. 不允许未来信息泄露。
16. 不允许 silent fallback。
17. 不允许使用 test split 选择模型、参数、图结构或模块配置。
18. 不允许将 smoke test 结果写入正式结果。
19. 不允许修改基础数据文件。
20. 不允许把 `valid_target_mask`、`Patv_raw`、`Modification_Reason` 或审计 mask 列作为模型输入。
21. 不允许为了某个 Custom Model 单独改变 split、window、scaler、loss、metrics、epoch、batch size 或 early stopping 规则。
22. 如果启用非公平主表 loss，例如 `msmg_dwu_loss`，必须在 metadata 中显式标记 `eligible_for_fair_main_table=false`。

Custom Model 可以新增：

```text
模型结构
图构建模块
门控模块
注意力模块
Prompt 模块
物理先验模块
多分支融合模块
消融实验分支
鲁棒性评估分支
```

但这些新增模块必须遵守：

```text
只使用训练允许的信息；
不改变统一评估协议；
不改变 target/mask 定义；
不改变 split/window/scaler/loss/metrics；
所有新增模块必须记录 metadata；
所有新增特征或图必须有 feature_protocol / graph_protocol 记录；
所有 on-the-fly 特征必须由当前历史窗口在线生成，不允许提前使用全时间序列生成。
```

---

## 2. 数据文件协议

Custom Model 默认使用以下基础数据文件：

```yaml
data:
  raw_aligned_parquet: dataset/sdwpf_raw_aligned.parquet
  model_input_parquet: dataset/sdwpf_model_input_base.parquet
  eval_target_parquet: dataset/sdwpf_eval_target.parquet

  turbine_location_file: dataset/sdwpf_turb_location_elevation.csv
  column_description_file: dataset/列名含义.csv

  turbine_id_col: TurbID
  timestamp_col: Tmstamp

  target_col: Patv_raw
  target_mask_col: valid_target_mask
  input_patv_col: Patv_clean_for_input

  num_nodes: 134
```

要求：

1. `model_input_parquet` 提供模型输入 `X`。
2. `eval_target_parquet` 提供预测目标 `Patv_raw` 和目标有效性掩码 `valid_target_mask`。
3. `raw_aligned_parquet` 只用于审计、诊断或必要的数据对齐检查。
4. 不允许修改、覆盖、移动这三个基础数据文件。
5. 如果文件不存在，必须明确报错，不允许 silent fallback。
6. 不允许回退到旧版 `sdwpf_10min_cleaned.parquet` 进入正式结果。
7. 所有 metrics 必须在 `Patv_raw` 反归一化回 kW 后计算。
8. `Patv_clean_for_input` 只作为历史输入特征，不是评估目标。
9. `valid_target_mask` 只能用于 loss 和 metrics，不能作为模型输入。
10. `Wspd`、`Wdir`、`Ndir`、ERA5 风速/风向等变量可以作为物理驱动信息，但不能替代最终 target。

---

## 3. 基础输入特征协议

Custom Model 的默认基础输入特征为 16 个：

```text
Wspd
Wdir
Etmp
Itmp
Ndir
Pab1
Pab2
Pab3
Prtv
T2m
Sp
RelH
Wspd_w
Wdir_w
Tp
Patv_clean_for_input
```

说明：

1. `Wspd/Wdir/Etmp/Itmp/Ndir/Pab1/Pab2/Pab3/Prtv` 为基础 SCADA 字段。
2. `T2m/Sp/RelH/Wspd_w/Wdir_w/Tp` 为基础 ERA5 字段。
3. `Patv_clean_for_input` 只用于历史输入窗口。
4. 预测目标为 `Patv_raw`。
5. 目标有效性由 `valid_target_mask` 控制。
6. 输入特征只能来自历史窗口 `[t-lookback, t)`。
7. 禁止使用预测窗口 `[t, t+max_pred_len)` 内的任何真实 SCADA、ERA5 或 `Patv` 特征。

默认禁止作为输入的列：

```text
TurbID
Tmstamp
Patv_raw
valid_target_mask
Modification_Reason
is_patv_anomaly
is_temp_anomaly
is_angle_anomaly
is_pab_anomaly
is_sensor_fault
is_missing_timestamp
is_original_observed
is_imputed_any
valid_input_mask
valid_patv_input_mask
```

metadata 必须保存：

```json
{
  "input_feature_set": "base_official_scada_era5",
  "feature_names": [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "Tp",
    "Patv_clean_for_input"
  ],
  "num_features": 16,
  "target_col": "Patv_raw",
  "target_mask_col": "valid_target_mask",
  "input_patv_col": "Patv_clean_for_input",
  "final_prediction_target_is_power": true,
  "wind_speed_variables_used_as_physical_drivers": true
}
```

---

## 4. ST-MGPrompt 专属模块边界

Custom Model 可以实现以下 ST-MGPrompt 专属模块。

### 4.1 Volatility-Aware Dynamic Semantic Patching, VADSP

中文名：波动感知动态语义分块。

作用：基于历史窗口内 `Wspd` 和/或 `Patv_clean_for_input` 的一阶差分波动率，动态生成 fine branch 与 coarse branch。

要求：

1. 只能使用当前历史输入窗口 `[t-lookback, t)`。
2. 不允许使用预测窗口 `[t, t+H)` 的真实值。
3. 不允许使用 `valid_target_mask`、异常标签或审计标签作为门控输入。
4. 默认波动率来源列为 `Wspd` 与 `Patv_clean_for_input`；可选加入 `Wspd_w`，但不得使用 `Patv_raw`、未来 `Wspd` 或未来 ERA5 真值。
5. `volatility_source_cols` 必须写入 metadata，且所有来源列都必须属于基础 16 输入特征或由历史窗口在线生成。
6. 建议输出：

```text
volatility: [B, L, N]
rolling_std: [B, L, N]
dynamic_patch_gate: [B, L, N, 1]
x_fine: [B, L, N, D]
x_coarse: [B, L, N, D]
```

### 4.2 Trend-Prior Adaptive Dual Graph

中文名：趋势先验约束的自适应双图。

作用：基于 train split 的低频趋势相似性构造趋势先验图，并与自适应图融合，减少高频伪相关。

必须支持两类图：

1. `A_macro_trend`：用于 coarse branch，表达宏观气象联动。
2. `A_micro_local`：用于 fine branch，表达局部扰动、近邻传播或可选 wake/delay 先验。

要求：

1. 趋势提取必须 causal，不能使用中心窗口。
2. 统计量只能来自 train split。
3. 不允许使用 val/test 统计量构图。
4. 不允许使用预测窗口真实 target 构图。
5. 如果趋势先验使用 `Patv_raw` 的 train 历史统计量，只能使用 `valid_target_mask=True` 的 train 位置；`valid_target_mask` 只可用于统计过滤，不能作为模型输入。
6. 默认趋势提取方法为 `causal_moving_average` 或 `causal_ema`；`almon_smooth` 只能作为可选消融，不作为默认主模型方法。
7. 自适应图 `A_adaptive` 不能无约束作为主模型默认图，必须由先验图约束。
8. 学习型 `A_adaptive` 需要在训练后保存最终邻接矩阵快照，并记录 epoch/checkpoint 来源。

### 4.3 ST-MG Coupling Block

中文名：时空多粒度耦合块。

作用：把双图空间增强、Macro-Trend Prompt 生成和粗细粒度对称交叉融合放在同一个可堆叠模块中，实现“空间关系”和“粗细粒度关系”的层内协同建模，而不是把空间建模当成一次性前处理。

推荐输入输出：

```text
输入：
  x_fine:   [B, L, N, D]
  x_coarse: [B, L, N, D]
  A_micro_local: [N, N]
  A_macro_trend: [N, N]

块内中间表示：
  h_fine:   [B, L, N, D]  # 经过 A_micro_local 空间增强后的细粒度表示
  h_coarse: [B, L, N, D]  # 经过 A_macro_trend 空间增强后的粗粒度表示
  macro_prompt: [B, N, P, D]

输出：
  x_fine_next:   [B, L, N, D] 或 [B, H, N, D]
  x_coarse_next: [B, L, N, D] 或 summary/context 表示
```

强制要求：

1. Fine branch 必须先经过 `A_micro_local` 的图时序编码，得到空间增强后的 `h_fine`。
2. Coarse branch 必须先经过 `A_macro_trend` 的图时序编码，得到空间增强后的 `h_coarse`。
3. `Macro-Trend Prompt` 必须由空间增强后的 `h_coarse` 生成，不能直接由原始 `x_coarse` 生成。
4. `Symmetric Cross-Fusion` 必须使用空间增强后的 `h_fine/h_coarse`，不能直接使用未经过图传播的 `x_fine/x_coarse` 作为完整主模型输入。
5. `ST-MG Coupling Block` 支持堆叠，参数 `num_coupling_layers` 默认 1，可选 2 或 3；若大于 1，上一层输出的 fine/coarse 表示必须进入下一层继续进行空间增强与粗细粒度交叉更新。
6. 若实现“先图后融合”的简单串行结构，只能作为 `STMGPrompt_SerialGraphThenFusion` 消融，不得作为 `STMGPrompt_FairFull` 默认实现。

推荐伪代码：

```python
for block in coupling_blocks:
    h_fine = block.fine_graph_temporal(x_fine, A_micro_local)
    h_coarse = block.coarse_graph_temporal(x_coarse, A_macro_trend)
    macro_prompt = block.macro_prompt(h_coarse)
    x_fine, x_coarse = block.symmetric_cross_fusion(
        h_fine=h_fine,
        h_coarse=h_coarse,
        macro_prompt=macro_prompt,
    )
```

诊断要求：

```text
stmg_coupling_block_enabled
num_coupling_layers
macro_prompt_from_spatial_enhanced_coarse
cross_fusion_uses_spatial_enhanced_features
h_fine_spatial_delta_norm
h_coarse_spatial_delta_norm
coupling_fusion_gate_mean
coupling_fusion_gate_std
```

### 4.4 Macro-Trend Prompt Token

中文名：宏观趋势提示词。

作用：将空间增强后的 coarse branch 表示 `h_coarse` 压缩为宏观趋势提示词，不允许通过零填充方式强制对齐粗粒度序列。完整主模型中不得直接使用原始 `x_coarse` 生成 Macro-Trend Prompt。

建议张量：

```text
macro_prompt: [B, N, P, D]
```

其中 `P` 是 prompt token 数量，默认 `P=4`。

### 4.5 ST-Prompt Token

中文名：时空预测提示词。

作用：使用 node embedding、horizon embedding、granularity embedding 构造未来预测提示词。

必须包含：

```text
node_embedding
future-step embedding, 即 1..max_pred_len 的未来步长提示
granularity_embedding
```

注意：`eval_horizons=[3,6,10]` 只用于评估裁剪，不应作为唯一 horizon embedding；模型内部必须能为 `1..max_pred_len` 的每个未来步生成查询向量。

建议张量：

```text
st_prompt: [1, H, N, D]
future_query: [B, H, N, D]
```

### 4.6 Symmetric Spatio-Temporal Cross-Fusion

中文名：对称时空多粒度交叉融合。

作用：在 ST-MG Coupling Block 内实现空间增强后的 `h_fine/h_coarse` 之间的双向交叉注意力耦合，使局部空间扰动和宏观空间趋势同步参与粗细粒度交互。

至少支持：

1. `fusion_mode=add`：直接相加。
2. `fusion_mode=concat`：拼接后线性映射。
3. `fusion_mode=cross`：完整交叉注意力，默认用于本文主模型。
4. `disable_reverse_cross=True/False`：是否关闭反向细节修正分支。

建议输出：

```text
fused_future: [B, H, N, D]
fusion_gate: [B, H, N, D] or summary statistics
```

强制要求：

1. `fusion_mode=cross` 的主模型输入必须是 `ST-MG Coupling Block` 内的 `h_fine/h_coarse`。
2. 分支 A 使用 `future_query` attend to `macro_prompt`，其中 `macro_prompt` 来自空间增强后的 `h_coarse`。
3. 分支 B 使用 `future_query` 或 coarse query attend to `h_fine_recent`，其中 `h_fine_recent` 必须来自空间增强后的 `h_fine`。
4. 直接 `concat/add` 只能作为消融或轻量 baseline；完整主模型默认使用 cross。

### 4.7 MS-MG-DWU Loss

中文名：多站点-多粒度动态联合不确定性损失。

作用：同时处理多预测视界/多粒度任务不平衡与多风机/多站点难度不平衡。

本文工程协议中，“多粒度任务”默认定义为 `eval_horizons=[3,6,10]` 对应的前缀预测任务，即 H=3、H=6、H=10 三个 horizon-level losses。若额外实现 coarse auxiliary head，只能由同一预测窗口内部的 `y[:, :, :H]` 聚合得到，不得引入预测窗口之外的真实未来信息；该 auxiliary head 默认不进入公平主表 metrics。

边界：

1. `MS-MG-DWU` 不是公平主表默认 loss。
2. 启用该 loss 的模型必须标记：

```json
{
  "loss_function": "msmg_dwu_loss",
  "loss_changed_from_fair_protocol": true,
  "eligible_for_fair_main_table": false,
  "eligible_for_method_full_table": true
}
```

### 4.8 Robust Missing-Data Prompt Evaluation

中文名：缺失输入鲁棒性评估。

作用：人为遮蔽 test 历史输入窗口中的一部分输入特征，验证 Prompt 对齐机制对传感器缺失的鲁棒性。

边界：

1. 只用于鲁棒性分析，不进入正式主表排名。
2. 不允许遮蔽、修改或重新定义 `y_true`。
3. 不允许修改 `valid_target_mask`。
4. 不允许遮蔽后重新 fit scaler。

### 4.9 模块通用约束

所有模块必须满足：

1. 不能修改基础数据文件。
2. 不能修改 `Patv_raw`。
3. 不能修改 `valid_target_mask`。
4. 不能把未来窗口真实特征作为输入。
5. 不能用 test split 统计量构造图、scaler、阈值或校准参数。
6. 不能把 target mask、异常标签、审计标签直接作为模型输入。
7. 任何增强特征必须保存到 Custom 专属目录，或显式声明为 on-the-fly historical features。
8. 任何图结构必须有独立 graph protocol。
9. 任何模块是否启用必须写入 metadata。

推荐目录：

```text
custom_models/
  configs/
  src/
    st_mgprompt/
  features/
    angle_calibration/
    reliability_gate/
    wake_delay/
    semantic_graph/
    trend_prior_graph/
  graphs/
  checkpoints/
  results/
  logs/
  reports/
```

正式 Custom 结果中必须记录：

```json
{
  "uses_custom_features": true,
  "custom_feature_protocol": "...",
  "custom_feature_path": "...",
  "uses_custom_graph": true,
  "custom_graph_protocol": "...",
  "custom_graph_path": "...",
  "custom_module_names": [
    "VADSP",
    "TrendPriorAdaptiveDualGraph",
    "STMGPromptCouplingBlock",
    "MacroTrendPrompt",
    "STPrompt",
    "SymmetricCrossFusion"
  ],
  "custom_modules_use_future_information": false,
  "custom_modules_use_test_statistics": false
}
```

如果某次实验只使用基础 16 特征，则记录：

```json
{
  "uses_custom_features": false,
  "custom_feature_protocol": null,
  "custom_feature_path": null
}
```

---

## 5. 时间顺序数据划分

默认 split：

```yaml
split:
  split_mode: chronological_ratio
  split_ratios: [0.8, 0.1, 0.1]
```

要求：

1. train/val/test 必须严格按时间顺序划分。
2. 不允许随机打乱时间轴后划分。
3. 不允许任何 target 跨 split 边界。
4. 不允许 validation/test 输入窗口从前一个 split 借历史窗口。
5. 不允许使用 test split 选择模型、图参数、门控参数、阈值、loss 或 early stopping。
6. 所有 Custom 变体必须使用完全相同的 split indices。
7. `X_all`、`y_all`、`mask_all` 必须使用相同的时间索引和风机索引。

推荐实现：

```python
def compute_split_indices(num_time_steps, split_ratios):
    train_ratio, val_ratio, test_ratio = split_ratios
    assert len(split_ratios) == 3
    assert all(r > 0 for r in split_ratios)
    assert abs(sum(split_ratios) - 1.0) < 1e-6

    train_start = 0
    train_end = int(num_time_steps * train_ratio)
    val_start = train_end
    val_end = int(num_time_steps * (train_ratio + val_ratio))
    test_start = val_end
    test_end = num_time_steps

    return {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }
```

---

## 6. Split-contained window 规则

对于任意 split 区间：

```text
[S, E)
```

设：

```text
lookback = L
max_pred_len = H
预测起点 = t
```

则：

```text
input indices:  [t - L, t)
target indices: [t, t + H)
mask indices:   [t, t + H)
```

合法样本必须满足：

```text
S <= t - L
t + H <= E
```

因此：

```text
t_first = S + lookback
t_last = E - max_pred_len
```

窗口数量：

```python
num_windows = max(0, (E - S) - lookback - max_pred_len + 1)
```

要求：

1. train 样本的输入窗口、预测目标和 target mask 必须完全位于 train split 内。
2. validation 样本的输入窗口、预测目标和 target mask 必须完全位于 validation split 内。
3. test 样本的输入窗口、预测目标和 target mask 必须完全位于 test split 内。
4. 不允许跨 split 借历史窗口。
5. 如果某个窗口存在部分 invalid target，不删除窗口，通过 `y_mask` 控制 loss 和 metrics。
6. 如果某个训练 batch 全部 target invalid，则跳过反向传播并记录数量。

---

## 7. Window sampling 协议

为了降低训练成本，允许 train/val 使用样本起点 stride。

默认：

```yaml
window_sampling:
  train_window_sampling: stride
  train_sample_stride: 6
  val_window_sampling: stride
  val_sample_stride: 3
  test_window_sampling: full
```

注意：

```text
train_sample_stride / val_sample_stride 是样本起点 stride，不是窗口内部时间下采样。
lookback=144 始终表示每个输入样本包含连续 144 个时间步。
禁止写成 X[t-lookback:t:stride]。
正确写法是先按 stride 选择预测起点 t，再取连续输入窗口 X[t-lookback:t]。
```

推荐实现：

```python
def build_window_start_indices(S, E, lookback, max_pred_len, sample_stride):
    first_t = S + lookback
    last_t = E - max_pred_len
    if last_t < first_t:
        return []
    return list(range(first_t, last_t + 1, sample_stride))
```

构造样本：

```python
for t in start_indices:
    x = X_all[t - lookback : t]          # 连续 lookback 个时间步
    y = y_all[t : t + max_pred_len]
    m = mask_all[t : t + max_pred_len]
```

要求：

1. train 使用 sample-start stride=6。
2. validation 使用 sample-start stride=3。
3. test 使用 full windows，即 stride=1。
4. test 不允许使用 stride 抽样作为正式结果。
5. 所有 Custom 变体必须使用相同 window sampling。
6. metadata 必须记录 train/val/test 的 window 数量和 valid target ratio。

---

## 8. 归一化协议

### 8.1 输入特征归一化

默认采用：

```text
Global per-feature normalization across training time × all turbines
```

对于输入特征 `f`，scaler 只能在 train split 上 fit：

```python
values_f = X_train[:, :, f].reshape(-1)
```

然后对 validation/test 使用相同 scaler 参数。

默认 scaler：

```text
StandardScaler
```

公式：

```text
x_norm = (x - mean_train_f) / (std_train_f + eps)
```

其中：

```text
eps = 1e-6
```

### 8.2 目标归一化

target scaler 只能使用 train split 中 `valid_target_mask=True` 的 `Patv_raw` 拟合：

```python
target_values_for_scaler = y_train_raw[train_valid_target_mask]
```

要求：

1. 无效 target 不参与 target scaler 拟合。
2. validation/test 不参与 target scaler 拟合。
3. Custom 预测输出在计算 metrics 前必须反归一化回 kW。
4. loss 默认在 normalized target 空间计算。
5. metrics 默认在 kW 空间计算。

必须保存：

```json
{
  "input_scaler_fit_split": "train_only",
  "input_scaler_fit_axes": ["time", "turbine"],
  "target_scaler_fit_split": "train_only",
  "target_scaler_fit_mask": "valid_target_mask",
  "normalization_granularity": "global_per_feature",
  "target_inverse_transform_before_metrics": true
}
```

---

## 9. 预测任务定义

任务为：

```text
history-only multi-step forecasting
```

模型输入：

```python
X.shape == (B, lookback, N, C)
```

预测输出：

```python
y_pred.shape == (B, N, max_pred_len)
```

真实目标：

```python
y_true.shape == (B, N, max_pred_len)
```

目标 mask：

```python
y_mask.shape == (B, N, max_pred_len)
```

默认参数：

```yaml
protocol:
  lookback: 144
  max_pred_len: 10
  eval_horizons: [3, 6, 10]
  result_protocol: max_pred_len_prefix_eval
```

注意：

```text
在 SDWPF 10min 采样设置下，eval_horizons=[3,6,10] 表示预测第 3/6/10 个 10min 步长，
对应约 30/60/100 分钟，而不是 3h/6h/10h。

论文和结果表中应写作：
H=3 steps, H=6 steps, H=10 steps
或
30min, 60min, 100min ahead。
```

评估逻辑：

```python
for H in eval_horizons:
    y_true_eval = y_true_full[:, :, :H]
    y_pred_eval = y_pred_full[:, :, :H]
    y_mask_eval = y_mask_full[:, :, :H]
    metrics = regression_metrics(y_true_eval, y_pred_eval, y_mask_eval)
```

默认 decoder 策略：

```text
decoder_input_strategy = direct_multi_output_prompt_query
```

说明：

1. 模型一次性输出 `max_pred_len` 个未来步。
2. `ST-Prompt` 只能提供 node/future-step/granularity 的可学习查询向量。
3. 不允许 teacher forcing。
4. 不允许把未来真实 `Patv_raw`、`Patv_clean_for_input`、SCADA 或 ERA5 作为 decoder 输入。
5. 若实现 autoregressive decoder，只能使用模型上一时刻预测值作为输入，并必须作为单独 ablation，不能进入默认公平主表。

禁止：

1. 为 H=3、H=6、H=10 分别训练不同模型。
2. 针对不同 horizon 使用不同模型参数。
3. 使用预测期真实外生变量。
4. 使用预测期真实 `Patv_raw`。
5. 使用预测期真实 `Patv_clean_for_input`。
6. 使用预测期 `valid_target_mask` 作为模型输入。

---

## 10. Custom 模型结构协议

Custom Model 应统一封装为：

```python
class CustomModel(torch.nn.Module):
    def __init__(self, config):
        ...

    def forward(self, x, graph_data=None, aux_data=None):
        """
        x: (B, lookback, N, C)
        graph_data: optional custom graph artifacts
        aux_data: optional historical-only auxiliary data
        return: (B, N, max_pred_len)
        """
        ...
```

强制要求：

```python
assert y_pred.ndim == 3
assert y_pred.shape[1] == num_nodes
assert y_pred.shape[2] == max_pred_len
```

如果内部使用其他格式，只能在模型 wrapper 内转换，不能改变全局 Dataset、DataLoader、loss、metrics 协议。

ST-MGPrompt 内部允许使用以下中间张量格式：

```text
x_fine:        [B, L, N, D]
x_coarse:      [B, L, N, D]
h_fine:        [B, L, N, D]  # A_micro_local 空间增强后的细粒度表示
h_coarse:      [B, L, N, D]  # A_macro_trend 空间增强后的粗粒度表示
macro_prompt:  [B, N, P, D]  # 必须由 h_coarse 生成
st_prompt:     [1, H, N, D]
future_query:  [B, H, N, D]
fused_future:  [B, H, N, D]
```

但模型 wrapper 的最终输出必须统一转换为：

```text
y_pred: [B, N, max_pred_len]
```

禁止修改 Dataset、DataLoader、loss、metrics 的全局 shape 协议。

ST-MGPrompt 完整主模型推荐封装为：

```python
x_fine, x_coarse = vadsp(x)
for block in coupling_blocks:
    x_fine, x_coarse = block(x_fine, x_coarse, A_micro_local, A_macro_trend)
future_query = st_prompt_decoder.build_query(x_fine, x_coarse)
y_pred = prediction_head(future_query)  # [B, N, max_pred_len]
```

其中 `block` 必须是 `STMGPromptCouplingBlock`，而不是单独的 GraphEncoder 后接独立 CrossFusion。

---

## 11. Mask-aware loss 协议

### 11.1 公平主表默认 loss

默认公平主表 loss：

```text
masked_score_aligned_hybrid
```

loss 默认在 normalized target 空间计算。

该 loss 用于：

1. Benchmark vs Custom 的公平主表。
2. `STMGPrompt_TemporalOnly`。
3. `STMGPrompt_FixedPatching`。
4. `STMGPrompt_DynamicPatching`。
5. `STMGPrompt_TrendPriorGraph`。
6. `STMGPrompt_MacroPrompt`。
7. `STMGPrompt_CrossFusion`。
8. `STMGPrompt_FairFull`。

推荐实现：

```python
def masked_score_aligned_hybrid_loss(y_pred, y_true, y_mask, eps=1e-6):
    """
    y_pred: (B, N, H), normalized target space
    y_true: (B, N, H), normalized target space
    y_mask: (B, N, H), bool or 0/1
    """
    mask = y_mask.float()
    err = y_pred - y_true

    abs_err = torch.abs(err) * mask
    sq_err = (err ** 2) * mask

    denom = torch.sum(mask, dim=-1).clamp_min(1.0)

    mae = torch.sum(abs_err, dim=-1) / denom
    rmse = torch.sqrt(torch.sum(sq_err, dim=-1) / denom + eps)

    valid_node = (torch.sum(mask, dim=-1) > 0).float()
    node_loss = 0.5 * rmse + 0.5 * mae

    valid_node_count = valid_node.sum()
    if valid_node_count.item() == 0:
        return None

    return torch.sum(node_loss * valid_node) / valid_node_count
```

训练要求：

1. `y_mask=False` 的 target 不参与 loss。
2. 如果 batch 内所有 target 均 invalid，跳过 backward 和 optimizer step。
3. 记录 `skipped_all_invalid_batches`。
4. 不允许某个进入公平主表的 Custom 变体单独更换 loss。
5. 若需要研究其他 loss，必须作为单独 ablation 标记，不能进入 strict fair main table。

metadata：

```json
{
  "loss_function": "masked_score_aligned_hybrid",
  "loss_compute_space": "normalized_target",
  "loss_rmse_weight": 0.5,
  "loss_mae_weight": 0.5,
  "target_mask_used_in_loss": true,
  "target_mask_col": "valid_target_mask",
  "skipped_all_invalid_batches": 0,
  "eligible_for_fair_main_table": true,
  "fair_loss_comparison": true
}
```

### 11.2 允许的 loss 列表

```yaml
allowed_loss_functions:
  - masked_score_aligned_hybrid
  - msmg_dwu_loss
```

### 11.3 MS-MG-DWU loss 边界

如果启用 MS-MG-DWU，则必须标记为：

```json
{
  "custom_protocol_variant": "method_full_with_msmg_dwu",
  "loss_function": "msmg_dwu_loss",
  "loss_ablation": true,
  "loss_changed_from_fair_protocol": true,
  "fair_loss_comparison": false,
  "eligible_for_fair_main_table": false,
  "eligible_for_method_full_table": true
}
```

MS-MG-DWU 结果可以作为论文方法完整模型结果和 loss 消融结果展示，但不能与 fixed-loss benchmark 混在同一个“严格公平主表”中排名。

---

## 12. Mask-aware metrics 协议

所有 metrics 输入必须为反归一化后的 kW：

```python
y_true.shape == (T, N, H)
y_pred.shape == (T, N, H)
y_mask.shape == (T, N, H)
```

所有指标只在：

```text
y_mask == True
```

的位置计算。

必须计算：

```text
mae_single_turbine_kw
rmse_single_turbine_kw
r2_single_turbine
smape
mape
official_align_score
score
valid_target_count
total_target_count
valid_target_ratio
```

必须保存：

```json
{
  "masked_metrics": true,
  "target_mask_col": "valid_target_mask",
  "valid_target_count": 123,
  "total_target_count": 456,
  "valid_target_ratio": 0.987
}
```

不得保存：

```text
*_3sig
metrics_raw
*_percent
*_percentage
smape_single_turbine_percent
mape_single_turbine_percent
```

所有正式指标保留小数点后三位。

---

## 13. Masked official score 公式

误差先从 kW 转 MW：

```python
err_mw = (y_pred - y_true) / 1000.0
```

对每个测试样本 `t`、每台风机 `i`，在 horizon `H` 内只使用有效 mask 点：

```math
M_{t,i,h} \in \{0, 1\}
```

有效 horizon 数：

```math
K_{t,i} = \sum_{h=1}^{H} M_{t,i,h}
```

若：

```math
K_{t,i} > 0
```

则：

```math
RMSE_{t,i} =
\sqrt{
\frac{
\sum_{h=1}^{H} M_{t,i,h} \cdot err_{t,i,h}^{2}
}{
K_{t,i}
}}
```

```math
MAE_{t,i} =
\frac{
\sum_{h=1}^{H} M_{t,i,h} \cdot |err_{t,i,h}|
}{
K_{t,i}
}
```

```math
s_{t,i} = 0.5 \cdot (RMSE_{t,i} + MAE_{t,i})
```

若：

```math
K_{t,i} = 0
```

则该 `t,i` 不参与该样本 score 聚合。

为了保持与全风机求和 score 的尺度一致，对每个测试样本 `t`：

```math
V_t = \{ i \mid K_{t,i} > 0 \}
```

```math
S_t =
N \cdot
\frac{
\sum_{i \in V_t} s_{t,i}
}{
|V_t|
}
```

其中：

```text
N = 134
```

若某个样本：

```math
|V_t| = 0
```

则该样本不参与最终 score。

最终：

```math
official\_align\_score =
\frac{1}{|\mathcal{T}_{valid}|}
\sum_{t \in \mathcal{T}_{valid}} S_t
```

强制要求：

```python
score == official_align_score
```

该 score 是误差型指标：

```text
lower is better
```

metadata：

```json
{
  "official_score_masked": true,
  "score_scaled_to_num_nodes": true,
  "num_nodes_for_score_scaling": 134,
  "score_lower_is_better": true
}
```

---

## 14. 训练预算协议

默认训练配置：

```yaml
train:
  seed: 2026
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
  optimizer: Adam
  weight_decay: 0.0
  loss_function: masked_score_aligned_hybrid
  loss_compute_space: normalized_target
  early_stopping_patience: 10
  early_stopping_metric: val_masked_official_align_score
  early_stopping_mode: min
  gradient_clip_val: 5.0
  device: auto
  progress: auto
  log_interval: 50
```

要求：

1. 主对比实验使用 fixed-base 配置。
2. 不进行多候选配置搜索。
3. 不允许为了 Custom Model 单独增加 epoch、减小 stride 或改变 batch size。
4. 若显存不足，可按统一规则降级 batch size。
5. batch size 降级时默认 learning rate 不变。
6. 若出现数值不稳定，可启用统一 safe lr scaling，并记录原因。

batch size 降级顺序：

```text
32 -> 16 -> 8 -> 4
```

safe lr scaling：

```text
lr_actual = lr_base * sqrt(actual_batch_size / configured_batch_size)
```

---

## 15. Custom 图构建协议

Custom Model 可以使用图结构，但必须明确图来源。

允许图来源：

```text
距离图
语义图
差分语义图
learned graph
趋势先验图
物理先验图
多图融合
动态历史图
Trend-Prior Adaptive Dual Graph
```

禁止：

1. 用 test split 统计量构图。
2. 用未来窗口真实 target 构图。
3. 用 evaluation target mask 构图并作为输入。
4. 在不同模型之间悄悄改变基础 split 或 metrics。
5. silent fallback 到默认图。

### 15.1 基础距离图默认配置

如果使用基础距离图，默认配置：

```yaml
graph:
  graph_type: distance_knn_gaussian
  location_file: dataset/sdwpf_turb_location_elevation.csv
  coordinate_cols: [x, y]
  elevation_col: Ele
  use_elevation: false
  k: 5
  sigma: auto
  self_loop: true
  normalize: true
  directed: false
```

### 15.2 ST-MGPrompt 默认图：Trend-Prior Adaptive Dual Graph

ST-MGPrompt 默认图结构为：

```text
Trend-Prior Adaptive Dual Graph
```

必须至少支持两类图：

1. `A_macro_trend`
   - 用于 coarse branch。
   - 由 train split 中的低频趋势相似性构造。
   - 趋势提取必须 causal，不能使用中心窗口。
   - 可与距离图融合。
   - 主要表达宏观气象联动。

2. `A_micro_local`
   - 用于 fine branch。
   - 默认由距离 KNN 图构造。
   - 可选加入 wake/delay 物理先验。
   - 主要表达局部扰动、尾流或近邻传播。

自适应图 `A_adaptive` 只能与 `A_macro_trend` / `A_micro_local` 融合，不能完全无约束学习作为主模型默认图。

默认融合方式：

```text
A_final = normalize(A_prior * A_adaptive)
```

可选融合方式：

```text
multiply
add
mask
```

所有趋势相关图统计量只能使用 train split。若趋势序列来自 `Patv_raw`，必须在 train split 内按 `valid_target_mask` 过滤无效点；若趋势序列来自 `Wspd` 或 `Wspd_w`，只能使用历史输入特征，不得使用预测窗口未来风速。图构建时使用 target mask 做统计过滤不等于把 mask 输入模型，metadata 中必须区分：

```json
{
  "target_mask_used_for_graph_statistics": true,
  "target_mask_used_as_model_input": false
}
```

若使用 Custom 图，必须保存：

```text
custom_models/graphs/{graph_tag}/adjacency.npy
custom_models/graphs/{graph_tag}/edge_index.npy
custom_models/graphs/{graph_tag}/edge_weight.npy
custom_models/graphs/{graph_tag}/graph_config.json
custom_models/graphs/{graph_tag}/graph_diagnostics.json
```

ST-MGPrompt 双图还必须保存：

```text
custom_models/graphs/{graph_tag}/macro_trend_adjacency.npy
custom_models/graphs/{graph_tag}/micro_local_adjacency.npy
custom_models/graphs/{graph_tag}/adaptive_adjacency.npy
custom_models/graphs/{graph_tag}/macro_graph_heatmap.png
custom_models/graphs/{graph_tag}/micro_graph_heatmap.png
custom_models/graphs/{graph_tag}/graph_topk_edges.csv
```

metadata：

```json
{
  "uses_custom_graph": true,
  "graph_type": "trend_prior_adaptive_dual_graph",
  "graph_tag": "...",
  "graph_path": "...",
  "graph_uses_train_only_statistics": true,
  "graph_uses_test_statistics": false,
  "graph_uses_future_target": false,
  "edge_weight_used": true,
  "graph_normalize": true,
  "uses_macro_trend_graph": true,
  "uses_micro_local_graph": true,
  "trend_extraction_method": "causal_moving_average",
  "trend_statistics_fit_split": "train_only",
  "adaptive_graph_constrained_by_prior": true,
  "graph_fusion_mode": "multiply",
  "macro_graph_path": "...",
  "micro_graph_path": "..."
}
```

---

## 16. Custom 特征生成协议

若 Custom Model 需要额外离线特征，应保存到：

```text
custom_models/features/{feature_tag}/
```

必须包含：

```text
feature_config.json
feature_diagnostics.json
feature_protocol.md
```

要求：

1. 特征只能由历史窗口或 train-only 统计量构造。
2. 不允许使用未来 prediction window 真实特征。
3. 不允许使用 test split 统计量。
4. 不允许直接使用 `Patv_raw` 的未来信息。
5. 不允许使用 `valid_target_mask` 作为模型输入。
6. 不允许覆盖基础 `sdwpf_model_input_base.parquet`。
7. Custom 特征必须有开关。
8. 每个实验结果必须记录是否使用 Custom 特征。

metadata：

```json
{
  "uses_custom_features": true,
  "custom_feature_tag": "...",
  "custom_feature_path": "...",
  "custom_feature_names": [...],
  "custom_feature_count": 0,
  "custom_features_use_future_information": false,
  "custom_features_use_test_statistics": false
}
```

### 16.1 ST-MGPrompt on-the-fly historical features

ST-MGPrompt 的以下特征默认属于 on-the-fly historical features，不强制保存为全局 parquet：

```text
volatility
rolling_std
dynamic_patch_gate
x_fine
x_coarse
h_fine
h_coarse
macro_prompt
st_prompt
cross_attention_gate
fusion_gate
coupling_fusion_gate
```

这些特征必须在模型 forward 中仅由当前输入历史窗口 `x[:, :lookback]` 生成。禁止提前使用全时间序列、validation/test 统计量或预测窗口真实值生成。

metadata 增加：

```json
{
  "uses_on_the_fly_features": true,
  "on_the_fly_feature_names": [
    "volatility",
    "rolling_std",
    "dynamic_patch_gate",
    "x_fine",
    "x_coarse",
    "h_fine",
    "h_coarse",
    "macro_prompt",
    "st_prompt",
    "cross_attention_gate",
    "fusion_gate",
    "coupling_fusion_gate"
  ],
  "on_the_fly_features_history_only": true
}
```

---

## 17. 消融实验协议

ST-MGPrompt 至少支持以下消融：

```text
1. STMGPrompt_TemporalOnly
   只保留共享参数时间编码器，不使用动态分块、图、Prompt、交叉融合和 MS-MG-DWU。

2. STMGPrompt_FixedPatching
   使用固定窗口聚合，作为 w/o Dynamic Patching。

3. STMGPrompt_DynamicPatching
   启用 VADSP，验证波动感知动态语义分块。

4. STMGPrompt_NoSpatialGraph
   启用 VADSP，但关闭 A_micro/A_macro 图传播，用于验证空间拓扑模块贡献。

5. STMGPrompt_NoPhysicalPrior
   使用 adaptive graph，但不使用趋势先验约束，作为 w/o Physical Prior。

6. STMGPrompt_TrendPriorGraph
   启用趋势先验双图，但暂不启用完整 Coupling Block，仅验证双图空间增强。

7. STMGPrompt_NoPromptAlignment
   使用 concat 或 zero-fill 对齐，作为 w/o Prompt Alignment。

8. STMGPrompt_MacroPrompt
   启用 Macro-Trend Prompt 和 ST-Prompt，但必须记录 Prompt 是否由空间增强后的 h_coarse 生成。

9. STMGPrompt_SerialGraphThenFusion
   先独立图编码，再独立粗细粒度融合，作为“串行结构”反例，用于证明 ST-MG Coupling Block 的必要性。

10. STMGPrompt_NoSymmetricCoupling
   在 Coupling Block 内使用单向 cross-attention 或 concat，作为 w/o Symmetric Coupling。

11. STMGPrompt_CouplingCrossFusion
   启用 ST-MG Coupling Block：空间增强后的 h_fine/h_coarse 进入 MacroPrompt 与 Symmetric CrossFusion。

12. STMGPrompt_FairFull
    启用 VADSP + TrendPriorAdaptiveDualGraph + STMGPromptCouplingBlock + STPrompt Direct Decoder，但仍使用公平主表 loss。

13. STMGPrompt_Full_MSMGDWU
    在 FairFull 基础上启用 MS-MG-DWU，作为方法完整模型和 loss 消融。
```

每个消融必须：

1. 使用相同数据文件。
2. 使用相同基础 16 输入特征。
3. 使用相同 split。
4. 使用相同 scaler。
5. 使用相同 window sampling。
6. 使用相同公平主表 loss；只有 `STMGPrompt_Full_MSMGDWU` 可以改变 loss，并必须退出公平主表。
7. 使用相同 metrics。
8. 使用相同训练预算。
9. 保存相同 metadata 字段。

消融名称必须写入：

```json
{
  "ablation_family": "ST-MGPrompt",
  "ablation_name": "...",
  "ablation_group": "...",
  "enabled_modules": [
    "VADSP",
    "TrendPriorGraph",
    "STMGPromptCouplingBlock",
    "MacroPrompt",
    "STPrompt",
    "SymmetricCrossFusion",
    "MSMGDWU"
  ],
  "disabled_modules": [...]
}
```

---

## 17.1 Robust Missing-Data Evaluation 协议

该实验只用于鲁棒性分析，不进入正式主表排名。

设置：

```yaml
missing_ratios: [0.1, 0.2, 0.3]
```

遮蔽对象：

```text
只允许遮蔽 test 输入历史窗口中的输入特征；
不允许遮蔽 y_true；
不允许修改 valid_target_mask；
不允许遮蔽后重新 fit scaler。
```

推荐遮蔽策略：

```text
random_point_missing
block_missing
node_sensor_dropout
```

遮蔽值规则：

1. 默认在归一化空间将被遮蔽输入置为 0，即对应 train mean。
2. 可选 learned missing token，但必须作为模型内部参数，不能使用 `valid_target_mask` 或审计 mask 作为输入。
3. 不允许为了鲁棒性评估额外添加未来可见的缺失指示通道。

必须保存：

```text
robust_missing_metrics_r10.json
robust_missing_metrics_r20.json
robust_missing_metrics_r30.json
```

metadata：

```json
{
  "robust_missing_eval": true,
  "eligible_for_main_table": false,
  "missing_ratios": [0.1, 0.2, 0.3],
  "missing_strategies": [
    "random_point_missing",
    "block_missing",
    "node_sensor_dropout"
  ],
  "target_mask_modified": false,
  "scaler_refit_after_masking": false
}
```

---

## 18. 结果保存路径协议

推荐 checkpoint 路径：

```text
custom_models/checkpoints/{model}/{split_tag}/{protocol_variant}/max_pred_len_{max_pred_len}/best.pt
```

推荐 result 路径：

```text
custom_models/results/{model}/{split_tag}/{protocol_variant}/max_pred_len_{max_pred_len}/
```

每个模型至少保存：

```text
best.pt
run_manifest.json
prediction_metadata.json
predictions.npz
metrics_eval_h3.json
metrics_eval_h6.json
metrics_eval_h10.json
train_history.jsonl
train.log
eval.log
```

ST-MGPrompt 模型还应保存：

```text
diagnostics/
  vadsp_gate_hist.png
  volatility_hist.png
  macro_graph_heatmap.png
  micro_graph_heatmap.png
  prompt_norm.csv
  cross_attention_summary.csv
  coupling_block_summary.csv
  loss_weight_curve.csv
  challenging_period_metrics.json
  diagnostic_report.md
```

全局 summary：

```text
custom_models/results/metrics_all_custom.csv
custom_models/results/metrics_all_custom.json
```

也可额外保存：

```text
custom_models/results/{split_tag}/{protocol_variant}/metrics_all_custom.csv
custom_models/results/{split_tag}/{protocol_variant}/metrics_all_custom.json
```

---

## 19. Prediction 保存协议

`predictions.npz` 必须保存：

```text
y_true
y_pred
y_mask
timestamps
target_timestamps
turbine_ids
max_pred_len
eval_horizons
split_ratios
split_indices
feature_names
model_name
backend
result_protocol
protocol_variant
```

`prediction_metadata.json` 必须保存：

```text
num_parameters
num_trainable_parameters
model_size_mb
normalization_granularity
input_scaler_type
target_scaler_type
input_scaler_fit_split
target_scaler_fit_split
target_scaler_fit_mask
forecasting_setting
future_exogenous_available
decoder_input_strategy
temporal_input_mode
time_statistics
environment_info
resolved_config
ignored_config_keys
masked_metrics
target_mask_used_in_loss
target_col
target_mask_col
input_patv_col
input_feature_set
feature_names
num_features
valid_target_count
total_target_count
valid_target_ratio
uses_custom_features
custom_feature_protocol
custom_feature_path
uses_custom_graph
custom_graph_protocol
custom_graph_path
enabled_modules
disabled_modules
```

ST-MGPrompt 还必须追加以下诊断字段：

```text
vadsp_gate_mean
vadsp_gate_std
volatility_mean
volatility_std
macro_prompt_norm_mean
st_prompt_norm_mean
cross_attention_macro_entropy
cross_attention_fine_entropy
fusion_gate_mean
fusion_gate_std
macro_graph_sparsity
micro_graph_sparsity
adaptive_graph_prior_overlap
stmg_coupling_block_enabled
num_coupling_layers
macro_prompt_from_spatial_enhanced_coarse
cross_fusion_uses_spatial_enhanced_features
h_fine_spatial_delta_norm
h_coarse_spatial_delta_norm
coupling_fusion_gate_mean
coupling_fusion_gate_std
msmg_dwu_enabled
granularity_weight_h3
granularity_weight_h6
granularity_weight_h10
site_weight_mean
site_weight_max
site_weight_min
loss_changed_from_fair_protocol
eligible_for_fair_main_table
eligible_for_method_full_table
decoder_input_strategy
uses_autoregressive_decoder
teacher_forcing_used
target_mask_used_for_graph_statistics
target_mask_used_as_model_input
volatility_source_cols
auxiliary_granularity_heads_enabled
auxiliary_granularity_heads_used_in_fair_metrics
```

---

## 20. Per-horizon metrics 保存协议

每个 `metrics_eval_h{H}.json` 至少包含：

```text
model
backend
result_protocol
protocol_variant
custom_protocol_variant
split_tag
split_ratios
lookback
max_pred_len
eval_horizon

eval_horizon_unit
eval_horizon_minutes

mae_single_turbine_kw
rmse_single_turbine_kw
r2_single_turbine
smape
mape
official_align_score
score
lower_is_better

masked_metrics
official_score_masked
target_mask_used_in_loss
target_col
target_mask_col
input_patv_col
valid_target_count
total_target_count
valid_target_ratio
num_windows_total
num_windows_with_any_valid_target
num_windows_all_invalid_target

input_feature_set
feature_names
num_features
uses_custom_features
custom_feature_tag
custom_feature_path
uses_custom_graph
graph_type
graph_tag
graph_path

ablation_family
ablation_name
enabled_modules
disabled_modules

num_parameters
num_trainable_parameters
num_non_trainable_parameters
model_size_mb

epochs_configured
epochs_ran
best_epoch
configured_batch_size
actual_batch_size
base_learning_rate
actual_learning_rate
optimizer
weight_decay
loss_function
loss_changed_from_fair_protocol
eligible_for_fair_main_table
eligible_for_method_full_table
early_stopping_patience

normalization_granularity
input_scaler_type
target_scaler_type
target_scaler_fit_mask

forecasting_setting
future_exogenous_available
decoder_input_strategy
decoder_uses_future_ground_truth
teacher_forcing_used
uses_autoregressive_decoder

train_window_sampling
train_sample_stride
val_window_sampling
val_sample_stride
test_window_sampling

all_time_statistics_fields

checkpoint_path
prediction_path
python_executable
torch_version
cuda_available
resolved_config
ignored_config_keys
```

---

## 21. Summary 保存与排序协议

summary 必须包含所有 Custom 模型、所有 eval horizon 的结果。

排序规则：

1. 按 `eval_horizon` 分组。
2. 每个 `eval_horizon` 组内按 `score` 从小到大排序。
3. `score` 是误差型指标，越低越好。
4. `rank_score_asc_lower_better` 是主要论文排序字段。
5. 可保留 `rank_score_desc` 作为兼容字段，但不作为性能判断依据。
6. 如果 `eligible_for_fair_main_table=false`，该结果不能参与公平主表排名，但可以保留在 method-full 表和消融表中。

必须保存字段：

```text
score_sort_order = asc_lower_better
lower_is_better = true
rank_score_asc_lower_better
rank_score_desc
masked_metrics
valid_target_ratio
ablation_family
ablation_name
enabled_modules
uses_custom_features
uses_custom_graph
loss_function
loss_changed_from_fair_protocol
eligible_for_fair_main_table
eligible_for_method_full_table
```

---

## 22. Smoke 隔离协议

`--smoke` 只用于验证流程，不是正式实验。

smoke 结果只能保存到：

```text
custom_models/results_smoke/{timestamp}/...
custom_models/checkpoints_smoke/{timestamp}/...
custom_models/logs_smoke/{timestamp}/...
custom_models/reports/smoke_last_run.json
custom_models/reports/smoke_last_run.md
```

smoke 禁止写入：

```text
custom_models/results/metrics_all_custom.csv
custom_models/results/metrics_all_custom.json
custom_models/results/{model}/.../metrics_eval_h{H}.json
custom_models/results/{model}/.../predictions.npz
custom_models/checkpoints/{model}/.../best.pt
```

smoke metadata：

```json
{
  "run_mode": "smoke",
  "is_smoke": true,
  "eligible_for_summary": false,
  "eligible_for_paper_table": false,
  "result_isolation": "results_smoke_only"
}
```

---

## 23. CLI 参数协议

相关脚本建议包括：

```text
custom_models/src/train.py
custom_models/src/evaluate.py
custom_models/src/run_all_custom.py
custom_models/src/summarize_results.py
custom_models/src/check_custom_protocol.py
custom_models/src/st_mgprompt/run_st_mgprompt.py
custom_models/src/st_mgprompt/run_ablation.py
```

推荐通用参数：

```bash
--config custom_models/configs/custom_default.yaml
--model ModelName
--models ModelA ModelB ModelC

--model-input-path dataset/sdwpf_model_input_base.parquet
--eval-target-path dataset/sdwpf_eval_target.parquet
--target-col Patv_raw
--target-mask-col valid_target_mask
--input-patv-col Patv_clean_for_input

--split-ratios 0.8 0.1 0.1
--lookback 144
--max-pred-len 10
--eval-horizons 3 6 10

--fixed-base
--train-sample-stride 6
--val-sample-stride 3
--default-loss masked_score_aligned_hybrid
--masked-metrics

--device auto
--device cuda:0
--device cpu
--progress
--no-progress
--log-interval 50

--smoke
--full
--evaluate-only
--smoke-result-mode isolated
```

ST-MGPrompt 专属参数：

```bash
--model STMGPrompt_FairFull
--enable-dynamic-patching
--patching-mode fixed
--patching-mode dynamic
--volatility-source-cols Wspd Patv_clean_for_input

--enable-trend-prior-graph
--trend-method causal_moving_average
--trend-method causal_ema
--trend-method almon_smooth
--enable-adaptive-graph
--graph-fusion-mode multiply
--graph-fusion-mode add
--graph-fusion-mode mask
--graph-top-k 5

--enable-stmg-coupling-block
--num-coupling-layers 1
--coupling-mode graph_then_cross
--coupling-mode serial_graph_then_fusion
--coupling-mode cross_without_graph

--enable-macro-prompt
--num-prompt-tokens 4
--enable-st-prompt
--decoder-mode direct_multi_output_prompt_query
--decoder-mode autoregressive_no_teacher_forcing
--aux-granularity-horizons 3 6 10
--disable-auxiliary-granularity-heads

--fusion-mode add
--fusion-mode concat
--fusion-mode cross
--enable-symmetric-coupling
--disable-reverse-cross

--enable-msmg-dwu
--loss-protocol fair_main
--loss-protocol method_full

--enable-robust-missing-eval
--missing-ratios 0.1 0.2 0.3
--missing-strategy random_point_missing
--missing-strategy block_missing
--missing-strategy node_sensor_dropout
```

---

## 24. 自检清单

正式运行前必须检查：

1. `sdwpf_model_input_base.parquet` 存在。
2. `sdwpf_eval_target.parquet` 存在。
3. 坐标文件存在。
4. `model_input` 与 `eval_target` 的 `(Tmstamp, TurbID)` 完全对齐。
5. 每个 timestamp 都有 134 台风机。
6. 不存在重复 `(Tmstamp, TurbID)`。
7. 16 个基础输入特征全部存在。
8. `Patv_raw` 存在。
9. `Patv_clean_for_input` 存在。
10. `valid_target_mask` 存在。
11. `valid_target_mask` 为 bool 或 0/1。
12. 默认 feature_names 不包含：
    - `TurbID`
    - `Tmstamp`
    - `Patv_raw`
    - `valid_target_mask`
    - `Modification_Reason`
    - 审计 mask 列
13. split indices 正确。
14. 每个 split 内 window 完全 contained。
15. train/val sample stride 是样本起点 stride，不是窗口内部下采样。
16. scaler 只在 train split fit。
17. target scaler 只使用 train 且 `valid_target_mask=True`。
18. validation/test 没有使用前一 split 的 lookback。
19. 模型没有使用预测期真实外生变量。
20. decoder input 没有使用未来 ground truth。
21. decoder input 没有使用未来 target mask。
22. 所有模型输出 shape 为 `(B, N, max_pred_len)`。
23. test 使用 full windows。
24. 公平主表模型 loss 为 `masked_score_aligned_hybrid`。
25. 如果 loss 为 `msmg_dwu_loss`，确认 `eligible_for_fair_main_table=false`。
26. metrics 为 mask-aware metrics。
27. smoke 结果没有写入正式 results。
28. metrics 不含 `*_3sig`、`metrics_raw`、`*_percent`。
29. summary 按 score 升序给出真实性能排名。
30. predictions 保存了 `y_mask`。
31. metadata 保存了 mask-aware 字段。
32. 如果使用 custom features，确认未使用未来信息。
33. 如果使用 on-the-fly features，确认只由当前历史窗口生成。
34. 如果使用 custom graph，确认未使用 test 统计量。
35. 如果使用 trend prior graph，确认趋势统计量只来自 train split。
36. 如果使用 gate，确认未把 target mask 或审计标签作为输入。
37. 如果启用 Prompt，确认没有通过预测窗口真实值构造 prompt。
38. 如果启用 ST-Prompt，确认使用的是 `1..max_pred_len` 的 future-step embedding，而不是只对 `[3,6,10]` 建 embedding。
39. 如果启用 trend prior graph 且使用 `Patv_raw` 统计趋势，确认只使用 train split 内 `valid_target_mask=True` 的 target 点。
40. 如果启用自适应图，确认 `A_adaptive` 被先验图约束，且保存了 checkpoint 对应的邻接矩阵快照。
41. 如果启用 STMGPrompt_FairFull，确认 `ST-MG Coupling Block` 已启用，而不是只做串行 GraphEncoder -> CrossFusion。
42. 确认 Macro-Trend Prompt 由空间增强后的 `h_coarse` 生成，而不是由原始 `x_coarse` 生成。
43. 确认 CrossFusion 输入为空间增强后的 `h_fine/h_coarse`，而不是未经过图传播的 `x_fine/x_coarse`。
44. 确认 `num_coupling_layers` 已写入 metadata；若大于 1，每层都重复图增强与粗细粒度交叉更新。
45. 如果启用 auxiliary granularity heads，确认其不进入公平主表 metrics。
46. 如果启用 autoregressive decoder，确认未使用 teacher forcing，且该模型不进入默认公平主表。
47. 如果启用 robust missing eval，确认没有修改 y_true 和 valid_target_mask。
48. run_manifest 已保存。
49. 自检结果已保存到 `custom_models/reports/protocol_check.json`。

---

## 25. 推荐运行命令

### 25.1 Smoke test

```bash
python custom_models/src/run_all_custom.py ^
  --smoke ^
  --models STMGPrompt_FairFull ^
  --model-input-path dataset/sdwpf_model_input_base.parquet ^
  --eval-target-path dataset/sdwpf_eval_target.parquet ^
  --target-col Patv_raw ^
  --target-mask-col valid_target_mask ^
  --input-patv-col Patv_clean_for_input ^
  --split-ratios 0.8 0.1 0.1 ^
  --lookback 144 ^
  --max-pred-len 10 ^
  --eval-horizons 3 6 10 ^
  --fixed-base ^
  --train-sample-stride 6 ^
  --val-sample-stride 3 ^
  --default-loss masked_score_aligned_hybrid ^
  --masked-metrics ^
  --enable-dynamic-patching ^
  --enable-trend-prior-graph ^
  --enable-adaptive-graph ^
  --enable-stmg-coupling-block ^
  --num-coupling-layers 1 ^
  --enable-macro-prompt ^
  --enable-st-prompt ^
  --fusion-mode cross ^
  --loss-protocol fair_main ^
  --smoke-result-mode isolated
```

### 25.2 正式 fair full run

```bash
python custom_models/src/run_all_custom.py ^
  --full ^
  --models STMGPrompt_FairFull ^
  --model-input-path dataset/sdwpf_model_input_base.parquet ^
  --eval-target-path dataset/sdwpf_eval_target.parquet ^
  --target-col Patv_raw ^
  --target-mask-col valid_target_mask ^
  --input-patv-col Patv_clean_for_input ^
  --split-ratios 0.8 0.1 0.1 ^
  --lookback 144 ^
  --max-pred-len 10 ^
  --eval-horizons 3 6 10 ^
  --fixed-base ^
  --train-sample-stride 6 ^
  --val-sample-stride 3 ^
  --default-loss masked_score_aligned_hybrid ^
  --masked-metrics ^
  --enable-dynamic-patching ^
  --enable-trend-prior-graph ^
  --enable-adaptive-graph ^
  --enable-stmg-coupling-block ^
  --num-coupling-layers 1 ^
  --enable-macro-prompt ^
  --enable-st-prompt ^
  --fusion-mode cross ^
  --loss-protocol fair_main ^
  --smoke-result-mode isolated ^
  --no-progress
```

### 25.3 ST-MGPrompt 消融 full run

```bash
python custom_models/src/run_all_custom.py ^
  --full ^
  --models STMGPrompt_TemporalOnly STMGPrompt_FixedPatching STMGPrompt_DynamicPatching STMGPrompt_NoSpatialGraph STMGPrompt_NoPhysicalPrior STMGPrompt_TrendPriorGraph STMGPrompt_NoPromptAlignment STMGPrompt_MacroPrompt STMGPrompt_SerialGraphThenFusion STMGPrompt_NoSymmetricCoupling STMGPrompt_CouplingCrossFusion STMGPrompt_FairFull ^
  --model-input-path dataset/sdwpf_model_input_base.parquet ^
  --eval-target-path dataset/sdwpf_eval_target.parquet ^
  --target-col Patv_raw ^
  --target-mask-col valid_target_mask ^
  --input-patv-col Patv_clean_for_input ^
  --split-ratios 0.8 0.1 0.1 ^
  --lookback 144 ^
  --max-pred-len 10 ^
  --eval-horizons 3 6 10 ^
  --fixed-base ^
  --train-sample-stride 6 ^
  --val-sample-stride 3 ^
  --default-loss masked_score_aligned_hybrid ^
  --masked-metrics ^
  --smoke-result-mode isolated ^
  --no-progress
```

### 25.4 Method-full MS-MG-DWU run

```bash
python custom_models/src/run_all_custom.py ^
  --full ^
  --models STMGPrompt_Full_MSMGDWU ^
  --model-input-path dataset/sdwpf_model_input_base.parquet ^
  --eval-target-path dataset/sdwpf_eval_target.parquet ^
  --target-col Patv_raw ^
  --target-mask-col valid_target_mask ^
  --input-patv-col Patv_clean_for_input ^
  --split-ratios 0.8 0.1 0.1 ^
  --lookback 144 ^
  --max-pred-len 10 ^
  --eval-horizons 3 6 10 ^
  --fixed-base ^
  --train-sample-stride 6 ^
  --val-sample-stride 3 ^
  --enable-msmg-dwu ^
  --loss-protocol method_full ^
  --masked-metrics ^
  --smoke-result-mode isolated ^
  --no-progress
```

### 25.5 鲁棒性评估

```bash
python custom_models/src/run_all_custom.py ^
  --evaluate-only ^
  --models STMGPrompt_FairFull ^
  --model-input-path dataset/sdwpf_model_input_base.parquet ^
  --eval-target-path dataset/sdwpf_eval_target.parquet ^
  --target-col Patv_raw ^
  --target-mask-col valid_target_mask ^
  --input-patv-col Patv_clean_for_input ^
  --split-ratios 0.8 0.1 0.1 ^
  --lookback 144 ^
  --max-pred-len 10 ^
  --eval-horizons 3 6 10 ^
  --fixed-base ^
  --masked-metrics ^
  --enable-robust-missing-eval ^
  --missing-ratios 0.1 0.2 0.3 ^
  --missing-strategy block_missing
```

### 25.6 汇总已有结果

```bash
python custom_models/src/run_all_custom.py ^
  --evaluate-only ^
  --models STMGPrompt_FairFull STMGPrompt_Full_MSMGDWU ^
  --model-input-path dataset/sdwpf_model_input_base.parquet ^
  --eval-target-path dataset/sdwpf_eval_target.parquet ^
  --target-col Patv_raw ^
  --target-mask-col valid_target_mask ^
  --input-patv-col Patv_clean_for_input ^
  --split-ratios 0.8 0.1 0.1 ^
  --lookback 144 ^
  --max-pred-len 10 ^
  --eval-horizons 3 6 10 ^
  --fixed-base ^
  --masked-metrics
```

---

## 26. 最终交付要求

每次完成 Custom Model 实现或运行后，必须给出：

1. 修改文件列表。
2. 新增文件列表。
3. Custom model registry 摘要。
4. 使用的数据文件路径。
5. 基础输入特征列表。
6. Custom 增强特征列表。
7. on-the-fly historical feature 列表。
8. target 和 target mask 字段。
9. 每个模型的 `resolved_config`。
10. 每个模型的实际 batch size 与 learning rate。
11. 每个模型是否启用了 learning rate scaling。
12. 图构建配置与图文件路径。
13. `A_macro_trend`、`A_micro_local`、`A_adaptive` 路径。
13.1 `ST-MG Coupling Block` 是否启用、`num_coupling_layers`、Prompt 是否由空间增强后的 `h_coarse` 生成。
14. 特征生成配置与特征文件路径。
15. scaler 配置与归一化粒度。
16. target scaler 是否只使用 train valid target。
17. split indices 与每个 split 的有效 window 数量。
18. 每个 split 的 valid target ratio。
19. smoke test 结果。
20. full run 结果路径。
21. `metrics_all_custom.csv/json` 路径。
22. 每个模型是否成功。
23. 失败模型的错误原因与日志路径。
24. 自检清单结果。
25. ST-MGPrompt 诊断报告路径。
26. 如果启用 MS-MG-DWU，说明该结果是否进入 fair main table。
27. 如果启用鲁棒性评估，说明该结果是否进入 main table。
28. decoder_input_strategy、teacher_forcing_used 与 uses_autoregressive_decoder。
29. 如果启用辅助多粒度头，说明其目标构造方式以及是否进入公平主表指标。

---

## 27. Codex 执行优先级建议

为了避免一次实现过多模块导致返工，建议按以下顺序实现：

```text
Step 0：协议检查器与数据流 skeleton
Step 1：TemporalOnly baseline wrapper
Step 2：VADSP 动态语义分块
Step 3：Trend-Prior Adaptive Dual Graph 构图与诊断
Step 4：Graph-Temporal Encoder 单独 smoke，用于验证 h_fine/h_coarse shape
Step 5：ST-MG Coupling Block，把 Graph-Temporal Encoder、Macro-Trend Prompt、Symmetric Cross-Fusion 放入同一个耦合块
Step 6：ST-Prompt Direct Multi-output Decoder 与公平主表 STMGPrompt_FairFull
Step 7：串行结构与 Coupling Block 消融对比
Step 8：MS-MG-DWU method-full 版本
Step 9：鲁棒性评估与诊断报告
```

每一步必须先通过 smoke test，再进入 full run。
