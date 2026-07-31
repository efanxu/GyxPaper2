# E2-D correlation fairness audit

This audit compares structural identity only. Smoke losses and real-data scores
were not used to rank or select models.

- Crossformer uses segment temporal attention, router-mediated
  cross-dimension attention, hierarchical segment merging, learned decoder
  tokens, decoder self/cross attention, and summed layer predictions.
- MSGNet discovers five periods with FFT; each period has its own adaptive
  16-variable graph and mix-hop convolution, period attention, and softmax
  scale aggregation.
- TimeFilter tokenizes variable-time patches, partitions S/T/ST regions, uses
  noisy three-region MoE routing in training, and filters a learned
  patch-specific graph.
- iTransformer represents variables as tokens and attends across them, but has
  no Crossformer segment hierarchy or learned segmented decoder.
- TimeXer separates endogenous/exogenous tokens; E2-D supplies no future
  exogenous data.
- TimesNet discovers periods and uses 2-D inception branches, but does not
  learn MSGNet's per-period 16-variable adaptive graphs.
- TimeMixer uses a fixed temporal downsampling pyramid with season/trend
  bottom-up/top-down mixing, not adaptive variable graphs.
- FreTS learns channel and temporal FFT-domain MLPs, not periods plus graph
  branches or S/T/ST filtering.
- ST-MGPrompt Fixed Dual retains its private fine/coarse, prompt, graph, and
  fusion paths. It is not assembled from Crossformer, MSGNet, or TimeFilter.

The three E2-D implementations import none of VADSP, ST-MGPrompt graph, Macro
Prompt, Reverse Cross, Cross Fusion, ST Prompt, or MS-MG-DWU. MSGNet and
TimeFilter graph structures are within one turbine and are not E3 turbine
graphs.
