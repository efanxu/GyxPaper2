# E2-C mixer and frequency fairness audit

This audit compares structural identity only. No smoke loss or real-data score
is interpreted as model quality.

| model | retained structural identity | explicit exclusions |
|---|---|---|
| TimeMixer | fixed 144/72/36/18 average-pooling pyramid; history normalization; moving-average trend/season decomposition; bottom-up season mixing; top-down trend mixing; four predictor sum | no graph, prompt, VADSP, Cross Fusion, or MS-MG-DWU |
| TSMixer | two residual blocks; temporal-axis MLP and 16-variable channel-axis MLP in every block; direct 144-to-10 projection | no explicit multiscale, frequency decomposition, attention, graph, or added normalization |
| FreTS | token embedding; channel rFFT learner enabled by string `"0"`; temporal rFFT learner; complex parameters; soft shrinkage; FC forecast | no graph, explicit temporal pyramid, prompt, or node mixing |
| WPMixer | db2 level-1 approximation/detail bands; two ResolutionBranches; wavelet inverse fusion | fixed wavelet bands rather than TimeMixer decomposition or FreTS learned complex FFT |
| MultiPatchFormer | four patch lengths 8/16/24/32, 18 tokens each, concatenation and shared attention | patch scales rather than trend/season or frequency learners |
| ST-MGPrompt Fixed Dual | project-private fine/coarse paths and private spatiotemporal fusion | does not reuse TimeMixer, TSMixer, FreTS, WPMixer, or MultiPatchFormer modules |

TimeMixer and Fixed Dual both expose more than one temporal resolution, but
their mechanisms and code are independent. TSMixer has no scale pyramid.
FreTS learns in temporal and feature frequency domains but performs no graph
propagation. All three external models remain node-shared comparisons and do
not import any proposed-method component.
