# E3-B source identity audit

E3-B found no trusted local GCN, classic STGCN, or DCRNN implementation after
the required read-only workspace audit. No clone, download, package install, or
third-party source copy was performed. All three implementations are
primary-paper-guided clean-room PyTorch compatibility implementations located
only under `custom_models/src/benchmark_v2`.

## GCN

- Paper: Thomas N. Kipf and Max Welling, *Semi-Supervised Classification with
  Graph Convolutional Networks*, ICLR 2017,
  <https://arxiv.org/abs/1609.02907>.
- Author repository: <https://github.com/tkipf/gcn>.
- Repository license evidence: MIT (`LICENCE` in the author repository).
- Preserved identity: localized first-order graph propagation.
- Benchmark-only adaptation: a shared `144 -> 10` temporal projection and a
  shared `64 -> 1` output projection. The original paper did not propose this
  wind-power forecast head.

## STGCN

- Paper: Bing Yu, Haoteng Yin, and Zhanxing Zhu, *Spatio-Temporal Graph
  Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*,
  IJCAI 2018, <https://www.ijcai.org/Proceedings/2018/0505.pdf>.
- Author repository: <https://github.com/VeritasYin/STGCN_IJCAI-18>.
- Repository license evidence: BSD-2-Clause (`LICENSE` in the author
  repository).
- Classic identity gate: **PASS**. The paper and author repository establish a
  fully convolutional temporal gated convolution -> spectral graph convolution
  -> temporal gated convolution ST-Conv block. E3-B uses two such blocks,
  Chebyshev `Ks=3`, and frozen `L_tilde`.
- This resolves the E3-A canonical `stgcn` slot. `STCN` remains legacy metadata
  only and `stcn_stgcn_unresolved` remains a compatibility alias.

## DCRNN

- Paper: Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu, *Diffusion
  Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*,
  ICLR 2018, <https://openreview.net/pdf?id=SJiHXGWAZ>.
- Author repository: <https://github.com/liyaguang/DCRNN>.
- Repository license evidence: MIT (`LICENSE` in the author repository).
- Preserved identity: bidirectional random-walk diffusion convolution inside
  every GRU reset/update/candidate transform, stacked encoder-decoder, and
  autoregressive multi-step prediction.
- Benchmark contract adaptation: the original scheduled-sampling path is
  disabled because unified benchmark_v2 forbids target from entering forward.
  The decoder uses a zero GO token followed only by its own predictions.

The authoritative local closure records are in
`E3_B_source listing_MANIFEST.json`. Graph protocol and GraphBundle records are
separate identities and are not folded into model source records.
