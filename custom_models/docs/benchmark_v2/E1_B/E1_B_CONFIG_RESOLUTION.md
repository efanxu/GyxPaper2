# E1-B config resolution

本轮没有超参数搜索。所有配置均为 `validation_search_performed=false`、`test_result_used=false`。共同协议为 `seq_len=144`、`pred_len=10`、`enc_in=16`、masked MSE、Adam `lr=0.001`、weight decay 0、scheduler none、AMP enabled、20 epochs、patience 6、min delta 0.01、seed 2026。

| model | upstream/default evidence | effective E1-B config | resolution |
|---|---|---|---|
| DLinear | `run.py`: moving_avg=25, individual=false | moving_avg=25, individual=false, raw c_out=16 | 完全采用上游默认；DLinear 不读取 c_out |
| LightTS | constructor argument chunk_size=24; run.py d_model=512, dropout=0.1 | chunk_size=8, num_chunks=18, d_model=512, dropout=0.1, raw c_out=16 | 上游在 horizon 10 下把默认 chunk 变为 10 并 padding 144→150；8 是不超过 10 且整除 144 的最大整数，纯 shape 决议 |
| TiDE | run.py: label_len=48, d_model=512, d_ff=2048, e_layers=2, d_layers=1, dropout=0.1, freq=h | 相同；c_out/decode_dim=1, feature_encode_dim=2, bias=true, zero placeholders | c_out=1 是 M→S decoder 最小宽度，但源码仍输出 16 通道；显式零 time placeholders 满足无未来信息协议 |
| SegRNN | run.py: seg_len=96, d_model=512, dropout=0.1 | seg_len=2, seg_num_x=72, seg_num_y=5, d_model=512, dropout=0.1 | `protocol_divisibility_constraint`；2 是大于 1 且同时整除 144 与 10 的最大整数 |

LightTS 的 chunk_size 和 SegRNN 的 seg_len 都在任何 smoke 指标产生前由 shape 约束冻结。TiDE/SegRNN full-shape OOM 后没有降低 d_model、batch、node、lookback 或 horizon。

正式候选配置的代码来源：

- `custom_models/src/benchmark_v2/configs/dlinear.py`
- `custom_models/src/benchmark_v2/configs/lightts.py`
- `custom_models/src/benchmark_v2/configs/tide.py`
- `custom_models/src/benchmark_v2/configs/segrnn.py`
