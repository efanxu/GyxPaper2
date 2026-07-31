# E2-B multiscale fairness and leakage audit

TimesNet retains five dynamically detected periods and all six inception kernel sizes in each of two TimesBlocks. Period evidence is derived solely from the representation produced by historical `x_enc`; dynamic periods are not frozen into the config hash. Node grouping prevents one turbine from influencing another turbine's batch-averaged spectrum.

MICN retains both convolution scales 12 and 16, both odd decomposition scales 13 and 17, both isometric scales 13 and 10, and the two-branch Conv2d fusion. Its full-history decoder is derived from observed history plus a zero future suffix; marks are zero.

WPMixer retains db2 level-1 approximation and detail bands, both independent ResolutionBranches, two Mixer blocks per branch, patching within both bands, and inverse-wavelet fusion. Gradients are present in both branches. WPMixer uses fixed wavelet frequency bands; it does not use ST-MGPrompt VADSP, dynamic gate, graph, prompt, or DWU.

MultiPatchFormer retains all four distinct patch lengths 8, 16, 24, and 32. Every branch produces 18 tokens, all four are concatenated, and every patch embedding receives gradients. It has fixed scales, shared temporal/channel attention, and a semi-autoregressive head. It has no dynamic scale selector, graph, or prompt.

Relative to ST-MGPrompt Fixed Dual, WPMixer and MultiPatchFormer share the broad idea of representing history at more than one fixed resolution. They differ materially: WPMixer uses wavelet bands plus patch mixers; MultiPatchFormer concatenates four fixed patch embeddings; Fixed Dual uses the project's own fine/coarse temporal paths and fusion. Neither external model is part of the proposed method and neither imports any ST-MGPrompt component.

No configuration was selected from ordinary loss, real-data loss, test metrics, or GTX 1060 OOM. No branch, layer, width, batch, node, lookback, horizon, feature, or scale was reduced after OOM. Future observed/calendar covariates remain disabled and non-zero supplied future tensors fail closed.
