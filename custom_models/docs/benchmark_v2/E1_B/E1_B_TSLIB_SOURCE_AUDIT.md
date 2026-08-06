# E1-B TSLib source audit

审计对象仅为本地 `D:\PaperProject\GyxPaper2\Time-Series-Library`。未修改上游源码，未使用自动扫描、TSLib dataset、TSLib Trainer 或网络 fallback。上游项目为 `THUML/Time-Series-Library`，许可证为 MIT；`LICENSE` content record 为 `<removed-content-record>`。

## 源码身份

| model | source | content record |
|---|---|---|
| DLinear | `Time-Series-Library/models/DLinear.py` | `<removed-content-record>` |
| LightTS | `Time-Series-Library/models/LightTS.py` | `<removed-content-record>` |
| TiDE | `Time-Series-Library/models/TiDE.py` | `<removed-content-record>` |
| SegRNN | `Time-Series-Library/models/SegRNN.py` | `<removed-content-record>` |

相关代码已逐文件核查：`layers/Autoformer_EncDec.py`、`exp/exp_basic.py`、`exp/exp_long_term_forecasting.py`、`run.py`、`LICENSE`。`exp_basic.py` 会扫描 `models/*.py`，本实现明确不使用该机制。`exp_long_term_forecasting.py` 在训练中访问 test loader，本实现明确不使用该训练路径。

## 真实接口审计

### DLinear

- 构造函数：`Model(configs, individual=False)`。
- forward：`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`。
- 读取字段：`task_name, seq_len, pred_len, moving_avg, enc_in`；classification 分支另读 `num_class`。
- forecast 输入/输出：`(B,seq_len,enc_in) -> (B,pred_len,enc_in)`。
- `c_out` 完全未读取，不能原生输出一通道。`x_mark_enc/x_dec/x_mark_dec` 在 forecast 中不使用。
- 无内部 normalization；使用 `series_decomp(moving_avg)`，奇数 kernel 25 时分解长度保持 144。
- `individual=False` 时所有变量共享两套时间线性层，但每个输出通道仍只处理自己的历史通道；因此最终功率通道不利用其他 15 个变量。`individual=True` 是每通道独立线性层，本轮采用 run.py 默认 false。

### LightTS

- 构造函数：`Model(configs, chunk_size=24)`。
- forward：`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`。
- 读取字段：`task_name, seq_len, pred_len, d_model, enc_in, dropout`；`c_out` 未读取。
- forecast 输出固定为 `(B,pred_len,enc_in)`，不能原生一通道；marks/decoder 输入不使用。
- 构造时实际 `chunk_size=min(pred_len,seq_len,argument)`。默认 argument 24 在本协议下变为 10，随后把内部 seq_len 从 144 padding 到 150。
- 为禁止 silent padding，本轮确定性选择 8：`144 % 8 == 0`，`num_chunks=18`。连续与间隔采样分支均保持完整 144 步。
- 无内部 normalization。模型在同一节点内部混合 16 个变量，但不接触节点维。

### TiDE

- 构造函数：`Model(configs, bias=True, feature_encode_dim=2)`。
- forward：`forward(x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None)`。
- 读取字段：`task_name, seq_len, label_len, pred_len, d_model, e_layers, d_layers, freq, c_out, d_ff, dropout`。
- 内部按 `for feature in range(x_enc.shape[-1])` 对每个历史变量分别 forecast，最终 `torch.stack(..., dim=-1)`，所以即使 `c_out=1`，原始输出仍为 `(B,pred_len,enc_in)`。`c_out` 是 decoder latent width，不是最终变量数。
- forecast 对每个通道做均值/方差 normalization 和 de-normalization。
- `x_dec` 与 `x_mark_enc` 数值在 `forecast` 中不使用；time feature 必须具有 `freq` 对应维度。上游在 `batch_y_mark=None` 时创建零张量。
- 本实现显式创建全零 historical/future time placeholders；不读取 target、mask、未来功率、未来风速、ERA5 或真实 calendar features。

### SegRNN

- 构造函数：`Model(configs)`。
- forward：`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)`。
- 读取字段：`seq_len, enc_in, d_model, dropout, task_name, pred_len, seg_len`；`c_out` 未读取。
- forecast 输出固定为 `(B,pred_len,enc_in)`；marks/decoder 输入不使用。
- 内部做 last-value subtraction/addition normalization。
- `seg_num_x=seq_len//seg_len`、`seg_num_y=pred_len//seg_len`，随后 reshape 假定两个长度均能整除。默认 `seg_len=96` 对 144 和 10 均不兼容，且 `seg_num_y=0`。
- 协议兼容配置为 `seg_len=2`，得到 `seg_num_x=72`、`seg_num_y=5`。详见 `E1_B_SEGRNN_SHAPE_TRACE.md`。

## 公平输出结论

四个本地源码都产生 16 个通道，均不能按源码真实行为直接得到一通道。四个 Adapter 都从冻结 `ordered_input_features` 动态解析 `Patv_clean_for_input` 的唯一索引，选择对应预测通道，并用 normalized `Patv_raw` 监督。没有硬编码索引、通道平均、裁剪、填充或广播。
