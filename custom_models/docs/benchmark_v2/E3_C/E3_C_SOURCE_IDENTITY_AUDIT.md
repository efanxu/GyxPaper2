# E3-C source identity audit

## Conclusion

No trusted local implementation of Graph WaveNet, MTGNN, AGCRN, or STID was
present at task start. E3-C therefore uses clean-room PyTorch implementations
guided by the primary papers and official repositories. No repository was
cloned, no source file was copied, and no dependency was installed.

| Model | Primary architecture identity | Official repository | License evidence | Decision |
|---|---|---|---|---|
| Graph WaveNet | Wu et al., IJCAI 2019 | https://github.com/nnzhan/Graph-WaveNet | MIT `LICENSE` | clean-room |
| MTGNN | Wu et al., KDD 2020 | https://github.com/nnzhan/MTGNN | MIT `LICENSE` | clean-room |
| AGCRN | Bai et al., NeurIPS 2020 | https://github.com/LeiBAI/AGCRN | MIT `LICENSE` | clean-room |
| STID | Shao et al., CIKM 2022 | https://github.com/GestaltCogTeam/STID | Apache-2.0 `LICENSE` | clean-room |

The recursive local audit covered the project source, tests, configuration, and
documentation trees. Hits before implementation were registry/planning text
only. The closure of every implemented model is frozen in
`E3_C_SOURCE_CLOSURE_MANIFEST.json`; all closure records and aggregate hashes
were reverified after implementation.

`实验总Plan.md` and the requested root/workspace `HANDOFF.md` were not found
after read-only searches of both workspace roots. They were not reconstructed.
The project root has no Git metadata, so task-start unrelated changes are
reported as `UNKNOWN` and were not modified.
