# E3-B classic STGCN source resolution

The source-identity gate is **PASS**.

The IJCAI-2018 paper by Bing Yu, Haoteng Yin, and Zhanxing Zhu and the author
repository `VeritasYin/STGCN_IJCAI-18` identify STGCN as a fully convolutional
spatio-temporal graph network. Its core ST-Conv block is temporal gated
convolution -> spectral graph convolution -> temporal gated convolution. The
author repository is BSD-2-Clause.

E3-B therefore implements the canonical classic slot with:

- two T-G-T ST-Conv blocks;
- GLU temporal convolutions;
- Chebyshev spectral graph convolution using T0/T1/T2;
- frozen `L_tilde`, never `A_gcn`;
- no numerical lambda-max solve and no Laplacian reconstruction;
- a shared per-node temporal-collapse output block compatible with lookback 144
  and horizon 10.

No same-name but structurally different model was used. `STCN` is not promoted
as a separate model.
