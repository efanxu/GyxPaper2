# E9 Pairing / Initialization Audit

original26 controls are historical formal runs. Exact initial weights are claimed only when both artifacts expose matching initial-state identities.

| Model | Control | Transfer | Exact initial state | Allowed claim |
|---|---|---|---|---|
| lightts | LightTS_node_shared_chunk8_bs4_seed2026 | LightTS_node_shared_chunk8_bs4_msmg_dwu_seed2026 | NOT_VERIFIED | same seed, same source/config, same protocol |
| tide | TiDE_node_shared_d512_bs4_seed2026 | TiDE_node_shared_d512_bs4_msmg_dwu_seed2026 | NOT_VERIFIED | same seed, same source/config, same protocol |
| patchtst | PatchTST_node_shared_p16_s8_bs4_seed2026 | PatchTST_node_shared_p16_s8_bs4_msmg_dwu_seed2026 | NOT_VERIFIED | same seed, same source/config, same protocol |
| itransformer | iTransformer_node_shared_d512_bs4_seed2026 | iTransformer_node_shared_d512_bs4_msmg_dwu_seed2026 | NOT_VERIFIED | same seed, same source/config, same protocol |
| dcrnn | DCRNN_native_dualrw_k2_h64_l2_ar_bs4_seed2026 | DCRNN_native_dualrw_k2_h64_l2_ar_bs4_msmg_dwu_seed2026 | NOT_VERIFIED | same seed, same source/config, same protocol |
| mtgnn | MTGNN_native_adaptive_k20_gdep2_l3_bs4_seed2026 | MTGNN_native_adaptive_k20_gdep2_l3_bs4_msmg_dwu_seed2026 | NOT_VERIFIED | same seed, same source/config, same protocol |

All E9 transfers save loss state in checkpoints and preserve the sequential formal data order.
