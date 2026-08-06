# STID identity and timestamp trace

The four 32-dimensional components are:

1. a projection of each node's full `(144,16)` historical series;
2. a learned embedding indexed by canonical TurbID position `1..134`;
3. a 144-bin time-of-day embedding;
4. a 7-bin day-of-week embedding.

Only the final observed timestamp of the historical window is used. The source
naive timestamp is preserved without timezone conversion. Time-of-day is
`hour×6 + floor(minute/10)` and must be in `[0,143]`; day-of-week uses
Monday=0 and must be in `[0,6]`. No future timestamp, future calendar mark,
future observed covariate, target, or mask is accepted as a model input.

The concatenated width is 128, followed by three residual MLP blocks with
dropout 0.15. Cross-node isolation tests confirm that changing one node's
history cannot affect another node. Ordinary node-embedding record was
`<removed-content-record>`
and strict reload reproduced it. `learned_graph` is explicitly `null`.
