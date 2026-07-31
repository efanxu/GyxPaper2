# E2-B MultiPatchFormer shape trace

The E0-A generic source path was reproduced at `MultiPatchFormer.py:245`:

`RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 18 but got size 19 for tensor number 2 in the list.`

For source input length 144, convolution token count is `floor((L + right_padding - patch_len) / stride) + 1`.

| branch | patch | source stride | source padding | source length | source tokens |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 0 | 144 | 18 |
| 2 | 16 | 8 | 8 | 152 | 18 |
| 3 | 24 | 7 | 7 | 151 | 19 |
| 4 | 32 | 6 | 6 | 150 | 20 |

The source concatenates all embeddings on feature dimension `-1`; token dimension must therefore match. Branch 3 is tensor number 2 (zero-based) and is the first mismatch, giving the recorded 18/19 error.

Final geometry is:

| branch | patch | stride | padding | effective length | tokens | embedding |
|---|---:|---:|---:|---:|---:|---|
| 1 | 8 | 8 | 0 | 144 | 18 | `(BN*16,18,64)` |
| 2 | 16 | 8 | 8 | 152 | 18 | `(BN*16,18,64)` |
| 3 | 24 | 7 | 0 | 144 | 18 | `(BN*16,18,64)` |
| 4 | 32 | 7 | 8 | 152 | 18 | `(BN*16,18,64)` |

The concatenated tensor is `(BN*16,18,256)`, matching positional encoding length 18 and `embedding_channel` input width `18*256`.

A second source defect appears only after patch concatenation is fixed: for `pred_len=10`, source layers 5–8 use algebra such as `pred_len//2` that does not equal the cumulative sizes of preceding `pred_len//8` chunks. E2-B rebuilds only these four small linear layers with exact inputs `d_model+4`, `+5`, `+6`, `+7`, producing chunks `1,1,1,1,1,1,1,3`.

There is no silent crop, token discard, broadcasting, branch disablement, equalized patch length, lookback change, or horizon change. Gradient tests prove all four patch embeddings participate. TSLib source is unchanged.
