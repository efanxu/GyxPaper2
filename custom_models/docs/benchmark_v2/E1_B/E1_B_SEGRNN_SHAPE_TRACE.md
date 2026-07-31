# E1-B SegRNN shape trace

## Original failure reproduction

Command used the audited source with `B*N=8`, `seq_len=144`, `enc_in=16`, `pred_len=10`, `d_model=512`, `seg_len=96`.

Observed:

```text
input=(8,144,16)
seq_last=(8,1,16)
seg_num_x=144//96=1
seg_num_y=10//96=0
RuntimeError: The size of tensor a (0) must match the size of tensor b (8)
at SegRNN.py:81: y = y.permute(0, 2, 1) + seq_last
```

## Source-level trace

1. `seq_last=x[:,-1:,:]` gives `(8,1,16)`.
2. `(x-seq_last).permute(0,2,1)` gives `(8,16,144)`.
3. `reshape(-1,seg_num_x,seg_len)` with `(1,96)` cannot preserve the intended `B*C=128` leading dimension because 144 is not divisible by 96; `-1` silently becomes 192.
4. `seg_num_y=0`; `pos_emb` has shape `(0,256)`.
5. Decoder GRU and `predict` therefore produce zero prediction segments.
6. `view(-1,16,10)` has leading dimension 0; after permute, `y=(0,10,16)`.
7. Adding `seq_last=(8,1,16)` raises the recorded dimension-0 mismatch.

根因不是 permute 轴本身，而是源码用 floor division 计算 segment counts，同时后续 reshape 假定 `seq_len % seg_len == 0` 且 `pred_len % seg_len == 0`。默认 96 同时违反两项。

## Protocol-compatible resolution

固定协议不得改变：`seq_len=144`、`pred_len=10`。大于 1 且同时整除两者的最大整数是 2，因此：

```text
seg_len=2
seg_num_x=72
seg_num_y=5
embedded history=(B*N*16,72,512)
decoder segments=(B*N*16*5,1,512)
raw output=(B*N,10,16)
selected output=(B*N,10)
restored output=(B,N,10)
```

回归测试同时保留原配置 FAIL 与修复配置 PASS。未裁剪、padding、预测 12 再截断，也未修改 TSLib source；源码 SHA256 前后保持 `764d4ad950ee65162c70746248754942cfb4ba661162d515caab0a4eedd12252`。

## Full-shape result

shape 修复在 ordinary 和受限真实数据 smoke 均通过。正式 `(32,144,134,16)` AMP run 在上游第一段 GRU 处请求额外 4.71 GiB 并 OOM；没有降低 d_model=512 或 batch。最终 Registry 为 `BLOCKED_FULL_SHAPE_OOM`。
